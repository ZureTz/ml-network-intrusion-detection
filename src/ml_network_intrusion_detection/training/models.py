"""训练集拟合的 sklearn 预处理与 Torch 二分类器。"""

from __future__ import annotations

import numpy as np
import torch
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from torch import nn

from ml_network_intrusion_detection.data.schema import CATEGORICAL, NUMERIC
from ml_network_intrusion_detection.device import device_info, resolve_device, synchronize
from ml_network_intrusion_detection.training.transformer import TabularTransformer


class MajorityNetwork(nn.Module):
    """多数类基线以 buffer 持久化；并列时选择 0。"""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("probability", torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.probability.expand(x.shape[0])


def build_network(
    input_dim: int,
    family: str,
    hidden_dims: tuple[int, ...],
    dropout: float = 0.0,
    d_model: int = 32,
    n_heads: int = 4,
    n_layers: int = 2,
) -> nn.Module:
    """按已记录架构构建网络；神经分类器输出 logits，基线直接输出概率。"""
    if family == "dummy":
        return MajorityNetwork()
    if not 0 <= dropout < 1:
        raise ValueError("dropout 必须在 [0, 1) 内")
    if family == "transformer":
        return TabularTransformer(input_dim, len(NUMERIC), d_model, n_heads, n_layers, dropout)
    if family not in ("linear", "mlp"):
        raise ValueError(f"未知模型: {family}")
    widths = (input_dim,) + (hidden_dims if family == "mlp" else ()) + (1,)
    layers: list[nn.Module] = []
    for index in range(len(widths) - 1):
        layers.append(nn.Linear(widths[index], widths[index + 1]))
        if index < len(widths) - 2:
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
    layers.append(nn.Flatten(start_dim=0))
    return nn.Sequential(*layers)


class TorchClassifier(ClassifierMixin, BaseEstimator):
    """兼容 Pipeline 的 float32 二分类器；数据按小批量传入 GPU。

    输入是预处理后的二维稠密/稀疏矩阵，标签必须同时包含 0 和 1。
    非有限特征或训练损失抛出 ValueError，设备错误不会静默重试 CPU。
    """

    def __init__(
        self,
        family: str = "mlp",
        hidden_dims: tuple[int, ...] = (64, 32),
        balanced: bool = False,
        seed: int = 42,
        threads: int = 4,
        device: str = "auto",
        epochs: int = 20,
        batch_size: int = 512,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        dropout: float = 0.0,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
    ) -> None:
        self.family = family
        self.hidden_dims = hidden_dims
        self.balanced = balanced
        self.seed = seed
        self.threads = threads
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

    def initialize(self, input_dim: int) -> TorchClassifier:
        """分配网络；加载权重和训练共用架构，默认在 CPU 初始化再转设备。"""
        self.device_ = resolve_device(self.device)
        torch.set_num_threads(self.threads)
        self.n_features_in_ = input_dim
        self.classes_ = np.array([0, 1])
        self.network_ = build_network(
            input_dim,
            self.family,
            tuple(self.hidden_dims),
            self.dropout,
            self.d_model,
            self.n_heads,
            self.n_layers,
        ).to(self.device_)
        return self

    def network_batch_size(self) -> int:
        """限制注意力的二次方内存；CSV 批量可大于网络批量。"""
        return min(self.batch_size, 256) if self.family == "transformer" else self.batch_size

    def _tensor(self, x) -> torch.Tensor:
        if sparse.issparse(x):
            x = x.toarray()
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(x, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError("预处理结果超出 float32 范围或包含非有限值")
        return torch.from_numpy(np.ascontiguousarray(array)).to(self.device_)

    def fit(self, x, y) -> TorchClassifier:
        """仅拟合训练集，固定轮数；训练历史记录每轮平均损失。"""
        if self.epochs < 1 or self.batch_size < 1 or self.threads < 1:
            raise ValueError("epochs、batch_size、threads 必须为正整数")
        labels = np.asarray(y, dtype=np.float32)
        if labels.shape != (x.shape[0],) or set(labels.tolist()) != {0, 1}:
            raise ValueError("训练标签必须同时含 0 和 1，且与样本数量一致")
        torch.manual_seed(self.seed)
        self.initialize(x.shape[1])
        if self.device_.type == "cuda":
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.cuda.reset_peak_memory_stats(self.device_)
        self.loss_history_ = []
        self.positive_weight_ = (
            float((labels == 0).sum() / (labels == 1).sum()) if self.balanced else 1.0
        )
        if self.family == "dummy":
            self.network_.probability.fill_(float(labels.mean() > 0.5))
        else:
            optimizer = torch.optim.AdamW(
                self.network_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(self.positive_weight_, device=self.device_)
            )
            rng = np.random.default_rng(self.seed)
            self.network_.train()
            for _ in range(self.epochs):
                indices = rng.permutation(len(labels))
                total = torch.zeros((), device=self.device_)
                batch_size = self.network_batch_size()
                for start in range(0, len(indices), batch_size):
                    batch = indices[start : start + batch_size]
                    features = self._tensor(x[batch])
                    target = torch.from_numpy(labels[batch]).to(self.device_)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(self.network_(features), target)
                    loss.backward()
                    optimizer.step()
                    total += loss.detach() * len(batch)
                value = float(total.cpu()) / len(labels)
                if not np.isfinite(value):
                    raise ValueError("训练损失为非有限值；请检查输入或减小学习率")
                self.loss_history_.append(value)
                print(
                    f"  epoch {len(self.loss_history_)}/{self.epochs}: loss={value:.6f}", flush=True
                )
        self.network_.eval()
        synchronize(self.device_)
        self.runtime_ = device_info(self.device_)
        self.runtime_["network_batch_size"] = self.network_batch_size()
        self.runtime_["cuda_peak_allocated_bytes"] = (
            torch.cuda.max_memory_allocated(self.device_) if self.device_.type == "cuda" else None
        )
        return self

    def predict_proba(self, x) -> np.ndarray:
        """返回 [P(normal), P(attack)]；返回前等待 GPU 并将结果复制到 CPU。"""
        check_is_fitted(self, "network_")
        if x.shape[1] != self.n_features_in_:
            raise ValueError("预处理后的特征维度与模型不符")
        self.network_.eval()
        batches = []
        with torch.inference_mode():
            batch_size = self.network_batch_size()
            for start in range(0, x.shape[0], batch_size):
                values = self.network_(self._tensor(x[start : start + batch_size]))
                scores = values if self.family == "dummy" else torch.sigmoid(values)
                batches.append(scores.cpu().numpy())
        synchronize(self.device_)
        scores = np.concatenate(batches) if batches else np.empty(0, dtype=np.float32)
        if not np.isfinite(scores).all():
            raise ValueError("模型输出非有限概率")
        return np.column_stack((1 - scores, scores))

    def predict(self, x) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


def candidates(profile: str) -> list[dict]:
    """smoke 比较三个模型族；full 比较 7 个固定候选。"""
    if profile == "simulation":
        return [{"name": "mlp-64x32", "family": "mlp", "params": {"hidden_dims": (64, 32)}}]
    if profile in ("tune", "transformer_smoke"):
        from ml_network_intrusion_detection.training.search import search_candidates

        return search_candidates(profile)
    result = [{"name": "dummy", "family": "dummy", "params": {}}]
    for balanced in [False] if profile == "smoke" else [False, True]:
        result.append(
            {
                "name": f"linear-balanced{balanced}",
                "family": "linear",
                "params": {"balanced": balanced},
            }
        )
        for widths in [(64, 32)] if profile == "smoke" else [(64, 32), (128, 64)]:
            result.append(
                {
                    "name": f"mlp-{'x'.join(map(str, widths))}-balanced{balanced}",
                    "family": "mlp",
                    "params": {"hidden_dims": widths, "balanced": balanced},
                }
            )
    return result


def build_pipeline(spec: dict, seed: int, threads: int, **training_options) -> Pipeline:
    """全部数值标准化；只在训练数据拟合；稀疏编码在传入 GPU 前按批稠密化。"""
    transform = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                CATEGORICAL,
            ),
        ],
        sparse_threshold=1.0,
    )
    options = training_options | spec["params"]
    model = TorchClassifier(family=spec["family"], seed=seed, threads=threads, **options)
    return Pipeline([("preprocess", transform), ("classifier", model)])
