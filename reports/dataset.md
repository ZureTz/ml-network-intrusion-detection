# 数据、特征与来源

## 使用范围

任务为 UNSW-NB15 二分类：`label=0` 正常，`label=1` 攻击。`attack_cat` 仅用于事后错误分析；不作为输入，也不从它派生特征。数据是实验室网络流量基准，包含正常活动与构造的攻击流量，不能当作真实生产网络的代表性抽样。

[官方页面](https://research.unsw.edu.au/projects/unsw-nb15-dataset)说明预定义训练集有 175,341 行、测试集有 82,332 行。原始全量采集的 49 字段说明与这两个划分 CSV 的 45 列布局不同：划分文件包含 42 个输入特征及 `id`、`attack_cat`、`label`。

## 本次获取与核验

2026-11-03 检查时，官方 SharePoint 跳转到登录页。经用户允许，使用 [Mireu-Lab 镜像](https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15/tree/3eb02c4a0a29866b7abcb5ef77c45cf4fcc8f6b0)，固定提交 `3eb02c4a0a29866b7abcb5ef77c45cf4fcc8f6b0`。镜像声明其 CSV 来自官方划分，但命名相反；按照官方行数、标签分布及额外交叉核验建立以下映射。

| 镜像文件 | 本项目角色及本地文件 | 行数 | SHA-256 |
|---|---|---:|---|
| `test.csv` | 训练：`UNSW_NB15_training-set.csv` | 175341 | `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa` |
| `train.csv` | 测试：`UNSW_NB15_testing-set.csv` | 82332 | `734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559` |

训练文件哈希与 [Mouwiya 镜像的固定版本训练文件](https://huggingface.co/datasets/Mouwiya/UNSW-NB15/blob/8ad6b19373f143086a66f52b0d6035a6959d83a9/UNSW_NB15_training-set.csv)公开的 SHA-256 一致。这是镜像间的交叉核验，**没有与官方原文件完成逐字节核验**，因此来源真实性仍有这一限制。下载字节保持不变，只重命名本地文件；全部清洗另存。

官方允许学术研究使用，要求引用其列出的五篇文献；商业使用需与作者协商。镜像卡片另有 GPL-3.0 标记，本项目不据此改变原始数据的授权范围，也不分发完整数据。参考文献见[实验报告](README.md)。

## 输入字段

`proto`、`service`、`state` 为类别字段，其余 39 个为数值字段。下表按网络含义归组；严格列名由 `data/schema.py` 的白名单定义。

| 字段 | 含义 |
|---|---|
| `dur` | 流持续时间 |
| `proto`, `service`, `state` | 协议、应用服务、连接状态 |
| `spkts`, `dpkts`, `sbytes`, `dbytes` | 源/目的方向包数及字节数 |
| `rate`, `sload`, `dload` | 数据集导出的流速率及双向负载 |
| `sttl`, `dttl`, `sloss`, `dloss` | 双向 TTL 和丢包/重传统计 |
| `sinpkt`, `dinpkt`, `sjit`, `djit` | 双向包间到达时间和抖动 |
| `swin`, `dwin`, `stcpb`, `dtcpb` | TCP 窗口和初始序列号 |
| `tcprtt`, `synack`, `ackdat` | TCP 建连时间及组成部分 |
| `smean`, `dmean` | 双向平均包大小 |
| `trans_depth`, `response_body_len` | HTTP 事务深度与响应体长度 |
| `ct_srv_src`, `ct_srv_dst` | 相同服务与源/目的地址的邻近连接计数 |
| `ct_dst_ltm`, `ct_src_ltm`, `ct_dst_src_ltm` | 相同目的、源、源目的组合的邻近连接计数 |
| `ct_src_dport_ltm`, `ct_dst_sport_ltm` | 地址与对端端口组合的邻近连接计数 |
| `ct_state_ttl` | 状态与 TTL 范围组合统计 |
| `ct_flw_http_mthd` | HTTP 方法相关流计数 |
| `is_ftp_login`, `ct_ftp_cmd` | FTP 登录指示及命令相关统计 |
| `is_sm_ips_ports` | 源目的地址与端口相同的指示 |

特征释义参考官方页面所列字典及其[固定版本镜像](https://huggingface.co/datasets/Mouwiya/UNSW-NB15/blob/8ad6b19373f143086a66f52b0d6035a6959d83a9/NUSW-NB15_features.csv)。原始字典中的 `smeansz`、`dmeansz`、`res_bdy_len` 在划分文件中分别对应 `smean`、`dmean`、`response_body_len`。新 CSV 必须使用同样的提取定义与单位；系统不负责从原始报文计算这些特征。

## 特征保留与泄漏边界

删除行标识 `id`，排除标签 `label` 和攻击类别 `attack_cat`。保留网络协议、流统计和已提供的连接统计字段，因为它们不是标签派生字段；`source_file`、`source_row` 仅用于追溯，永不进入 Pipeline。

TCP 序列号、TTL 和连接聚合特征可能体现实验室设备、采集环境或邻近流量特征。所用 CSV 缺少原始 IP、会话键与可靠时间字段，无法证明主机/会话/时间层面的隔离，也无法重建聚合特征计算窗口。这些是明确的有效性限制；不能把重复组隔离等同于消除了全部泄漏。

特征按固定顺序、统一类型生成不含标签的 pandas 64 位指纹，用于重复审计和分组，极低概率的哈希碰撞仍是方法限制。测试特征仅用于重叠检查，不用于选择模型或阈值；这种审计使用了测试输入分布信息，应与完全不接触测试输入的实验区分。
