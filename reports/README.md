# 基于 Torch 的网络流量入侵检测系统

最新追加实验见 [Transformer 与当前 Torch 模型调参](transformer-tuning-20261105.md)。该轮 Transformer 测试 F1 为 90.54%，但按验证集整体选出的调参 MLP 测试效果下降；下文保留原 7 候选实验，避免混淆两轮结果。

2026-11-05 已完成完整 Torch 实验：7 个候选训练、验证选优、82,332 行测试集独立评估，以及完整测试集的 CPU 无标签推理验证。获选 MLP 128/64 的测试 Accuracy 为 **87.05%**、攻击类 F1 为 **89.16%**、误报率为 **24.70%**。正式证据保存在 [results/torch-full-20261105](results/torch-full-20261105/)。

本次使用 Apple MPS；CUDA 路径已实现，但没有 NVIDIA 硬件实测。[原 scikit-learn 实验报告](sklearn-20261103.md) 与 [历史证据](results/full-20261103/) 保留供比较，不将旧模型指标描述为 Torch 结果。

2026-11-06 新增 CLI 启动的本地 Web Dashboard 持续学习模拟，见[操作说明](../README.md#持续学习模拟与-web-dashboard)。独立实验只使用原训练/验证数据：少量样本训练初始模型，后续记录先预测、延迟揭示原始标签，再去重加入训练池并周期重训。官方测试集不进入此流程。真实 NB15 开发数据演示已观察到候选拒绝和候选接受两条路径，属于功能验证；没有将回放指标加入下文正式测试效果比较。固定样本测试覆盖标签时序、数据分组隔离、重训闭环及暂停/重启恢复。

## 1. 问题定义与数据集

使用 UNSW-NB15 的 42 个流量特征做离线二分类，`0 = normal`、`1 = attack`，攻击为正类。来源、镜像固定版本、SHA-256、许可说明、标签与特征含义见 [dataset.md](dataset.md)。提供本机模拟 Web 控制台，不包含实时抓包或生产线上部署。

数据处理沿用原实现：优先遵守官方训练/测试角色，隔离训练侧 940 条冲突标签记录，移除与测试集完全重叠的 9,768 条开发记录，再按特征组分层划分训练/验证。同组不跨训练/验证；所有预处理仅在训练集拟合，标签、攻击类型与 ID 不进入特征。原始 CSV 不被修改，移除操作保留来源行号清单。

| 集合 | 行数 | 正常 | 攻击 | 用途 |
|---|---:|---:|---:|---|
| 训练 | 131706 | 43237 | 88469 | 拟合预处理和分类器 |
| 验证 | 32927 | 10809 | 22118 | 候选选择 |
| 测试 | 82332 | 37000 | 45332 | 冻结选择后评估 |

三个划分的 CSV SHA-256 与 2026-11-03 实验完全相同，见 [本次清单](results/torch-full-20261105/data/manifest.json) 和 [历史清单](results/full-20261103/data/manifest.json)。缺少时间、主机和会话字段，不能证明这些层面的隔离；重复记录的权重偏差和公开镜像未经官方字节核验的限制仍存在。

## 2. 系统设计与实现

命令为 `prepare → train → evaluate / predict`，完整用法见 [根目录 README](../README.md)。CPU 上的 `ColumnTransformer` 对数值进行中位数填充与标准化，对类别做 One-Hot；Pipeline 最后一步是 Torch 分类器。预处理保留稀疏矩阵，每次只将一批转成 float32 稠密 tensor，网络、梯度和优化器在所选设备执行。

设备选择支持 `auto`（CUDA → MPS → CPU）、`cpu`、`mps`、`cuda` 和 `cuda:N`。显式设备不可用时抛错；算子、内存或非有限数值错误会记录失败，不自动重跑其他设备。固定种子控制初始化与样本打乱，跨设备和版本不保证逐位一致。

Torch 实现多数类基线、单层线性分类器和 ReLU MLP。训练采用 AdamW 与 BCEWithLogitsLoss，固定轮数并记录逐轮损失。类别加权只使用训练集计数，`pos_weight = normal / attack`。模型保存为 `preprocess.joblib`、CPU `weights.pt` 和包含架构/类别/字段/阈值/哈希/依赖版本的 `metadata.json`。推理先校验，再用 `weights_only=True` 加载权重，重建完整 Pipeline 并移到所选设备。预处理 joblib 仍只允许可信来源。

新格式版本为 2。旧 `pipeline.joblib` 不能直接加载到 Torch 流程，应重新训练；历史结果需要对应历史代码及锁文件。数据契约、无标签输入、额外列告警、未知类别统计、分批输出和行顺序保持原有语义。

## 3. 环境、配置与测量方法

正式实验使用 Python 3.12.13、PyTorch 2.15.0、macOS 27.2 arm64、8 逻辑 CPU、16 GiB 内存。训练和测试评估使用 MPS、4 个 CPU 算子线程；CPU 独立推理也使用 4 线程。完整依赖、设备和构建记录见 [environment.json](results/torch-full-20261105/environment.json)。

训练代码为已推送提交 `40ae60151fe68049b9a7a7ae2f754a37a169224a`，训练开始时工作区干净。配置来自 [full.yaml](../configs/full.yaml)，命令行显式覆盖 `--device mps`：种子 42，20 轮，训练批量 512，学习率 0.001，权重衰减 0.0001，测试预测批量 10,000。实际完整配置及源码哈希见 [training/run.json](results/torch-full-20261105/training/run.json)。报告与导出材料在实验完成后补充，没有回改训练代码或根据测试结果调参。

训练耗时包含预处理拟合、网络初始化和优化，结束前同步 GPU。评估预热一批后测量三次完整测试集预测，包含 CPU 预处理、数据传输与已同步的 GPU 计算，加载耗时单独记录。进程 RSS 每 50 ms 采样，不是 GPU 显存；CUDA 训练额外支持分配峰值，本次 MPS 显存峰值未测量。测量未包括原始网络特征提取，也不代表实时链路吞吐。

## 4. 候选训练与验证选择

7 个候选均成功，无训练告警，整次训练运行耗时 **59.92 秒**（包括读取、候选训练、验证和模型保存）。单个候选拟合时间另列。多数类基线不做梯度训练；其余候选均执行预先设定的 20 轮，不声称已保证收敛。

只使用验证集攻击 F1、较低误报率、较高 Recall 和候选顺序选优，阈值固定 0.5。不合并训练/验证重新拟合，不用测试指标返回调参。

| 候选 | 类别加权 | 验证 F1 | 验证 Recall | 验证误报率 | 拟合秒数 |
|---|---|---:|---:|---:|---:|
| 多数类 | 不适用 | 0.8036 | 1.0000 | 1.0000 | 0.99 |
| 线性 | 否 | 0.9515 | 0.9887 | 0.1831 | 7.81 |
| MLP 64/32 | 否 | 0.9587 | 0.9654 | 0.0995 | 10.60 |
| **MLP 128/64** | **否** | **0.9590** | **0.9596** | **0.0854** | **10.86** |
| 线性 | 是 | 0.9471 | 0.9534 | 0.1228 | 6.50 |
| MLP 64/32 | 是 | 0.9509 | 0.9261 | 0.0443 | 10.45 |
| MLP 128/64 | 是 | 0.9483 | 0.9200 | 0.0414 | 10.88 |

最终获选 `mlp-128x64-balancedFalse`。加权模型在验证集降低误报的同时降低攻击召回，未赢得预定的 F1 排序。MLP 两种不加权宽度的验证 F1 差距很小，单种子结果不能证明更宽网络稳定优于较窄网络。完整逐轮损失与指标见 [candidates.json](results/torch-full-20261105/training/candidates.json)，冻结选择见 [selection.json](results/torch-full-20261105/training/selection.json)。

## 5. 完整测试集结果

冻结选择后，仅评估每个模型族的验证获胜候选。Precision、Recall、F1 均针对攻击正类，误报率为 `FP/(FP+TN)`。数据见 [comparison.csv](results/torch-full-20261105/comparison.csv)。

| 模型族获选候选 | Accuracy | Precision | Recall | F1 | 误报率 | 宏平均 F1 |
|---|---:|---:|---:|---:|---:|---:|
| 多数类 | 55.06% | 55.06% | 100.00% | 71.02% | 100.00% | 35.51% |
| Torch 线性、不加权 | 80.23% | 74.70% | 96.90% | 84.36% | 40.20% | 78.73% |
| **MLP 128/64、不加权** | **87.05%** | **82.74%** | **96.65%** | **89.16%** | **24.70%** | **86.55%** |

| 模型族 | ROC-AUC | Average Precision |
|---|---:|---:|
| 多数类 | 0.5000 | 0.5506 |
| Torch 线性 | 0.9478 | 0.9615 |
| MLP | 0.9754 | 0.9819 |

ROC-AUC 和 AP 均基于连续概率。AP 使用召回增量加权精度，与 PR 曲线梯形面积不同。多数类基线从不预测正常，正常类 Precision 和宏平均 Precision 无定义，JSON 中保留 null 与原因，未填造数值。

获选 MLP 的混淆矩阵（行是真实类别，列是预测类别）为：

| 真实 / 预测 | Normal | Attack |
|---|---:|---:|
| Normal | 27862（TN） | 9138（FP） |
| Attack | 1520（FN） | 43812（TP） |

对应 [混淆矩阵图](results/torch-full-20261105/evaluation/mlp/confusion_matrix.png)、[ROC 曲线](results/torch-full-20261105/evaluation/mlp/roc.png)、[PR 曲线](results/torch-full-20261105/evaluation/mlp/precision_recall.png) 和 [完整指标](results/torch-full-20261105/evaluation/mlp/metrics.json)。正常类 Recall 为 75.30%、F1 为 83.94%，正常流量误报仍明显。

攻击类别细分见 [category_breakdown.csv](results/torch-full-20261105/evaluation/mlp/category_breakdown.csv)。其中 Fuzzers 的二分类检出率为 77.25%，弱于其他攻击类；该表只表示正常/攻击判别是否正确，不是多分类识别能力。Worms 仅 44 条，不能据此作稳定泛化判断。

### 与历史随机森林比较

划分文件相同，可对照该固定基准上的结果；模型训练方法、硬件执行路径和运行时间不同，不能将差异归因于 Torch 框架本身。

| 指标 | 历史随机森林 | 本次 MLP |
|---|---:|---:|
| Accuracy | 86.83% | 87.05% |
| Precision | 81.31% | 82.74% |
| Recall | 98.78% | 96.65% |
| F1 | 89.20% | 89.16% |
| 误报率 | 27.82% | 24.70% |

MLP 少误报 1,156 条正常记录，同时多漏检 968 条攻击记录；误报率降低 3.12 个百分点，攻击召回降低 2.14 个百分点，F1 基本持平。不能宣称全面超越旧模型，也没有用本次测试结果重选候选或阈值。历史数值来源为 [旧比较表](results/full-20261103/comparison.csv)。

## 6. 性能与独立推理

| 模型族 | 候选拟合秒数 | 模型加载秒数 | 82,332 行预测中位秒数 | 吞吐（行/秒） |
|---|---:|---:|---:|---:|
| 多数类 | 0.99 | 0.0281 | 0.1883 | 437339 |
| Torch 线性 | 7.81 | 0.0250 | 0.1756 | 468902 |
| MLP | 10.86 | 0.0291 | 0.1865 | 441533 |

获选 MLP 三次预测耗时为 0.2917、0.1748、0.1865 秒；预测阶段峰值进程 RSS 约 628.2 MiB。时间波动与首次图执行、设备状态等因素可能有关，未做归因实验；本次数据不证明 GPU 对相同模型的加速比。各候选顺序运行，进程缓存也使 RSS 不等同于单模型净内存。详细测量范围和原值见 [performance.json](results/torch-full-20261105/evaluation/mlp/performance.json)。

另在独立 CPU 进程中加载 MPS 训练的最终模型，将全部 82,332 行测试记录去掉标签和辅助列后分批预测。概率与直接 CPU 加载结果在 `atol=1e-6, rtol=1e-5` 内一致，标签和行顺序完全一致。5 条训练未见的 `state` 值被统计并正常预测。该验证不重复训练，也不将测试标签作为输入；见 [verification.json](results/torch-full-20261105/verification/verification.json) 与 [预测性能记录](results/torch-full-20261105/verification/predictions.csv.json)。这验证了 GPU 训练产物的 CPU 可移植性，不声称 CPU/MPS 所有结果逐位一致。

## 7. 工程验证与复现

Torch 重构的完整测试记录为 **39 passed、1 skipped**，跳过项是无 NVIDIA 硬件的 CUDA 实测；真实 MPS 测试通过。覆盖实际梯度学习、CPU 种子复现、设备选择、稀疏输入、分批一致性、GPU 权重转 CPU、独立进程预测、数据隔离、预处理不泄漏、输入校验、哈希与失败记录。合成数据仅用于工程测试。2026-11-05 提交前的 `uv sync --locked`、Ruff 检查与格式检查通过；本次 MPS 环境检查、正式实验与独立推理均成功。

先前 2026-11-04 的 5,000 行开发集小样本验证证据保留于 [torch-smoke-20261104-v2/evidence.json](results/torch-smoke-20261104-v2/evidence.json)。其验证指标与本次完整测试集指标分开保存。

本次正式命令如下，重跑必须使用新的输出目录：

```bash
uv run ml-network-intrusion-detection prepare --config configs/full.yaml --output data/processed/torch-full-20261105
uv run ml-network-intrusion-detection train --config configs/full.yaml --data data/processed/torch-full-20261105 --output models/torch-full-20261105 --device mps
uv run python scripts/evaluate_selected.py --training models/torch-full-20261105 --data data/processed/torch-full-20261105 --output outputs/torch-full-20261105 --device mps
uv run python scripts/check_environment.py --device mps > outputs/torch-full-20261105/environment.json
uv run python scripts/verify_inference.py --model models/torch-full-20261105/best --input data/processed/torch-full-20261105/test.csv --output outputs/torch-full-20261105-cpu-verification --device cpu
uv run python scripts/export_results.py --training models/torch-full-20261105 --evaluations outputs/torch-full-20261105 --data data/processed/torch-full-20261105 --environment outputs/torch-full-20261105/environment.json --output reports/results/torch-full-20261105
```

最终模型在 `models/torch-full-20261105/best`；Git 只保存精简证据，不提交数据集、模型二进制或逐行预测。原始数据来源清单、配置、代码/数据/模型哈希、各候选验证指标、三个模型族测试指标与图表均已导出。导出器对 JSON 的本机绝对路径进行规范化，原文件哈希保留在 [export_manifest.json](results/torch-full-20261105/export_manifest.json)。

独立验证的两个 JSON 另用 `scripts.export_results.portable` 规范化后保存到 `verification/`，同时将原文件路径与 SHA-256 补入导出清单；没有复制无标签 CSV 或预测 CSV。报告证据来源的所有哈希均可由清单定位核验。

## 8. 局限与后续实验

当前为单数据集、单种子、单设备的离线实验，不包含多 GPU、混合精度、早停、概率校准、时间/主机隔离或实时网络验证。测试集已用于多个历史方案的比较，继续实验应使用新的独立数据验证，而不能反复根据同一测试集调参后仍称其为未见数据。

训练数据与独立评估数据仍需放入 CPU 内存，只有推理 CSV 分批读取。NVIDIA、AMD 和 Windows 加速尚无本机实测。误报率 24.70% 对部署仍然偏高，应将结果视为课程基准实验；若需要研究误报约束下的阈值选择，应在新的预定方案中只使用验证数据，并以独立测试数据确认。

本次完成了正式训练和冻结测试评估；多种子统计、同模型 CPU/CUDA/MPS 性能对比、概率校准和独立数据泛化仍未完成。固定轮数与有限候选不能证明全局最优，较高 ROC-AUC/AP 也不能替代对当前阈值误报与漏报的分析。

## 9. 参考资料

数据集论文与来源见 [数据说明](dataset.md) 和 [历史报告参考文献](sklearn-20261103.md#11-参考文献与复现说明)。重构所依据的官方资料：

- [PyTorch MPS backend](https://docs.pytorch.org/docs/2.15/notes/mps.html)：设备可用性与 MPS tensor。
- [PyTorch Serialization semantics](https://docs.pytorch.org/docs/2.15/notes/serialization.html)：state_dict 与 weights_only 加载。
- [uv 与 PyTorch 集成](https://docs.astral.sh/uv/guides/integration/pytorch/)：不同平台的构建与索引。
