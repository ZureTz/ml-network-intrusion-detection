"""表格 Transformer：数值逐字段 token、联合类别 token 和 CLS 分类头。"""

import torch
from torch import nn
from torch.nn import functional as F


class AttentionBlock(nn.Module):
    """Pre-Norm 自注意力块，显式 SDPA 避免依赖平台专属推理 fastpath。"""

    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.projection = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, width = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(batch, tokens, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        attention = attention.transpose(1, 2).reshape(batch, tokens, width)
        x = x + F.dropout(self.projection(attention), self.dropout, self.training)
        return x + self.ffn(self.norm2(x))


class TabularTransformer(nn.Module):
    """保留字段身份的数值嵌入；类别 One-Hot 联合投影为一个上下文 token。

    这不是原论文 FT-Transformer 的精确复现，也不将不同流量记录作为序列。
    """

    def __init__(
        self, input_dim: int, numeric_dim: int, width: int, heads: int, layers: int, dropout: float
    ) -> None:
        super().__init__()
        if width < 1 or heads < 1 or width % heads or layers < 1:
            raise ValueError("Transformer 维度必须为正且 d_model 能被 n_heads 整除")
        if numeric_dim < 1 or input_dim <= numeric_dim:
            raise ValueError("Transformer 输入必须包含数值字段与类别编码")
        self.numeric_dim = numeric_dim
        self.numeric_weight = nn.Parameter(torch.empty(numeric_dim, width))
        self.numeric_bias = nn.Parameter(torch.empty(numeric_dim, width))
        self.category_projection = nn.Linear(input_dim - numeric_dim, width)
        self.cls = nn.Parameter(torch.empty(1, 1, width))
        nn.init.normal_(self.numeric_weight, std=0.02)
        nn.init.normal_(self.numeric_bias, std=0.02)
        nn.init.normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList([AttentionBlock(width, heads, dropout) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        numeric = x[:, : self.numeric_dim, None] * self.numeric_weight + self.numeric_bias
        categorical = self.category_projection(x[:, self.numeric_dim :]).unsqueeze(1)
        tokens = torch.cat((self.cls.expand(x.shape[0], -1, -1), numeric, categorical), dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(tokens[:, 0]).squeeze(-1)
