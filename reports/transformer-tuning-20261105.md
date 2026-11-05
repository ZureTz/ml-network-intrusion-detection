# Transformer 与现有 Torch 模型调参实验（2026-11-05）

本次按用户确认的范围优化当前 Torch 线性模型与 MLP，不重新调参历史随机森林。新增表格 Transformer，完整训练 11 个候选；仅依据验证集选择各模型族候选，再在同一 82,332 行测试集评估。所有候选训练成功，完整训练、验证与保存耗时 **465.74 秒**。完整证据见 [结果目录](results/torch-tune-20261105/) 与 [比较表](results/torch-tune-20261105/comparison.csv)。

**结论：Transformer 在本次测试集上取得 F1 90.54%、误报率 23.06%；线性调参有小幅改善；MLP 验证指标略升但测试表现下降，不能将调参一概称为优化成功。** 本轮根据验证集选出的 `best/` 仍是调参 MLP；没有在查看测试结果后将 Transformer 改为整体获选模型，也没有继续依据测试分数调整参数。

## 1. 数据和实验边界

复用 `data/processed/torch-full-20261105` 的完整划分：训练 131,706 行、验证 32,927 行、测试 82,332 行。来源、标签映射、特征、重复隔离与冲突处理见 [数据说明](dataset.md) 和 [上一轮报告](README.md)。预处理仍仅在训练集拟合，标签/攻击名称/ID 不参与特征。

固定种子 42、攻击正类 1、阈值 0.5。选择顺序仍是验证 F1 高、误报率低、Recall 高、候选顺序靠前。没有阈值调优、验证集合并重训或按测试表现重选。测试集已在多轮历史实验中使用，因此这是同一基准上的探索性比较，不是全新独立测试，也不能证明统计显著的泛化提升。

## 2. 模型和搜索空间

Transformer 将 39 个数值字段分别转换成带字段身份的可学习 token，类别 One-Hot 联合投影为一个上下文 token，再加入 CLS token，共 41 个 token。Pre-Norm 自注意力、GELU 前馈层与 CLS 分类头学习同一行的特征交互，不在不同流量记录之间做注意力。推理时关闭 Dropout。这个轻量实现并非原论文 FT-Transformer 的精确复现，类别 token 分组方式不同。

使用现有 AdamW + BCEWithLogitsLoss，不添加外部依赖。搜索在运行前固定并保存为 [search_plan.json](results/torch-tune-20261105/training/search_plan.json)：

- 线性模型 3 个候选：原学习率 0.001、20 轮；学习率 0.003、40 轮；学习率 0.01、权重衰减 0.001、40 轮。
- MLP 5 个候选：原 128/64、20 轮；128/64 + 学习率 0.0005 + 权重衰减 0.001、40 轮；128/64 + Dropout 0.1、40 轮；256/128 + Dropout 0.1、40 轮；64/32 + 学习率 0.003、40 轮。
- Transformer 2 个候选：16 维、2 头、1 层、学习率 0.001；32 维、4 头、2 层、学习率 0.0003；均为 Dropout 0.1、15 轮、网络批量 256。
- 多数类基线 1 个，不做梯度训练。

未特别覆盖的参数为学习率 0.001、权重衰减 0.0001、无类别加权，线性/MLP 训练批量 512。原始对照的已保存权重与上一轮对应权重逐元素比较，最大绝对差均为 0；验证 F1 也一致。候选参数覆盖 YAML 通用值，实际完整参数在各模型的元数据中记录。

这是有限预设搜索，各模型族预算不同，不是等预算架构排名。固定轮数不等于已保证收敛。未根据测试分数追加候选。

## 3. 环境与验证选择

Apple MPS，PyTorch 2.15.0、Python 3.12.13、macOS arm64、4 个 CPU 算子线程；float32。训练时工作区有未提交修改，Git 基线为 `fe58aba02978ee7dc7bdb3757e192ad551991982`，实际代码以记录的源码 SHA-256 为准。见 [训练运行](results/torch-tune-20261105/training/run.json) 和 [环境](results/torch-tune-20261105/environment.json)。

| 候选 | 验证 F1 | 验证 Recall | 验证误报率 | 拟合秒数 |
|---|---:|---:|---:|---:|
| `dummy` | 0.8036 | 1.0000 | 1.0000 | 1.01 |
| `linear-original` | 0.9515 | 0.9887 | 0.1831 | 10.09 |
| `linear-lr003-e40` | 0.9524 | 0.9876 | 0.1766 | 17.17 |
| `linear-lr01-wd001-e40` | 0.9529 | 0.9904 | 0.1806 | 17.17 |
| `mlp-original` | 0.9590 | 0.9596 | 0.0854 | 14.79 |
| `mlp128-lr0005-wd001-e40` | 0.9588 | 0.9858 | 0.1441 | 32.36 |
| `mlp128-drop01-e40` | 0.9603 | 0.9842 | 0.1343 | 21.25 |
| `mlp256-drop01-e40` | 0.9597 | 0.9816 | 0.1310 | 27.11 |
| `mlp64-lr003-e40` | 0.9599 | 0.9768 | 0.1194 | 21.88 |
| `transformer16-l1` | 0.9571 | 0.9692 | 0.1148 | 114.61 |
| `transformer32-l2` | 0.9584 | 0.9549 | 0.0774 | 182.98 |

冻结的获选候选为：线性 `linear-lr01-wd001-e40`、MLP `mlp128-drop01-e40`、Transformer `transformer32-l2`。整体获选为 MLP，验证 F1 为 96.03%，略高于 Transformer 的 95.84%。见 [selection.json](results/torch-tune-20261105/training/selection.json)。

## 4. 测试结果与解释

只评估每个模型族的验证获胜候选。原模型指标引用 [上一轮证据](results/torch-full-20261105/comparison.csv)，数据划分完全相同；本轮对照重训权重也一致。Precision/Recall/F1 均针对攻击正类，误报率为 FP/(FP+TN)。

| 模型 | Accuracy | Precision | Recall | F1 | 误报率 |
|---|---:|---:|---:|---:|---:|
| 原 Torch 线性 | 80.23% | 74.70% | 96.90% | 84.36% | 40.20% |
| 调参后线性 | 80.75% | 75.00% | 97.55% | 84.80% | 39.84% |
| 原 MLP 128/64 | 87.05% | 82.74% | 96.65% | 89.16% | 24.70% |
| 调参后 MLP 128/64 | 84.45% | 78.55% | 98.71% | 87.49% | 33.02% |
| Transformer 32维/2层 | 88.69% | 83.93% | 98.28% | 90.54% | 23.06% |

- 线性模型测试 F1 提高约 **0.44 个百分点**，误报率略降，但其检测效果仍落后于两个更复杂的模型。
- 调参 MLP 的验证 F1 提高约 **0.13 个百分点**，测试 F1 却降低约 **1.67 个百分点**，误报率升高约 **8.32 个百分点**。验证集上的微小优势没有转化为测试改进；这不足以单独证明原因是过拟合、分布差异或概率校准问题，需额外实验区分。
- Transformer 相比原 MLP，测试 F1 提高约 **1.39 个百分点**，误报率降低约 **1.64 个百分点**，Recall 提高约 **1.64 个百分点**。同一测试集上少误报 607 条正常记录、少漏检 742 条攻击记录，但仍误报约 23% 的正常记录。

Transformer 测试混淆矩阵为 `[[28469, 8531], [778, 44554]]`，行是真实 `[Normal, Attack]`，列是预测类别；ROC-AUC **0.9851**，Average Precision **0.9888**，宏平均 F1 **88.24%**。AP 采用召回增量加权精度，不等同于 PR 曲线梯形面积。调参 MLP 混淆矩阵为 `[[24784, 12216], [584, 44748]]`，其更高攻击 Recall 伴随更高误报。

详见 Transformer 的 [指标](results/torch-tune-20261105/evaluation/transformer/metrics.json)、[混淆矩阵](results/torch-tune-20261105/evaluation/transformer/confusion_matrix.png)、[ROC](results/torch-tune-20261105/evaluation/transformer/roc.png)、[PR](results/torch-tune-20261105/evaluation/transformer/precision_recall.png)。多数类的未定义指标以 null 和原因保存，没有填充伪造数值。

## 5. 性能、保存加载和测试

| 获选候选 | 拟合时间 | 完整测试集预测中位时间 | 吞吐（行/秒） |
|---|---:|---:|---:|
| 调参线性 | 17.17 秒 | 0.1676 秒 | 491289 |
| 调参 MLP | 21.25 秒 | 0.1815 秒 | 453667 |
| Transformer 32维/2层 | 182.98 秒 | 3.2356 秒 | 25445 |

预测先预热一批，再测量三次完整测试集，模型加载另计。时间包含 CPU 预处理、传输和已同步的 GPU 计算；CSV 外层批量均为 10,000，但 Transformer 内部网络批量上限 256，MLP/线性为 10,000。因此表格反映当前系统配置的实际成本，不是相同批量下的纯算子或 CPU/GPU 加速比。RSS 每 50 ms 采样，不等于 GPU 显存，MPS 显存峰值未测。

Transformer 和调参 MLP 均已在独立 CPU 进程对 82,332 行无标签输入完成预测验证，概率在 atol=1e-6、rtol=1e-5 内一致，标签和行顺序一致。未知 `state` 类别 5 条已统计。见 [Transformer 验证](results/torch-tune-20261105/verification/transformer/verification.json) 和 [MLP 验证](results/torch-tune-20261105/verification/mlp/verification.json)。训练前先完成真实 4,000/1,000 训练/验证小样本检查，精简记录见结果目录 `smoke/`。

新增测试验证注意力权重得到非零梯度、CPU/MPS 真正学习、样本之间不串扰、GPU 权重在 CPU 恢复、旧网络权重键兼容、注意力批量上限及环境报告 JSON 输出。最终完整测试为 **43 passed、2 skipped**；两项 CUDA 测试仅因本机无对应硬件跳过，真实 MPS 测试通过。Ruff 与格式检查通过。

工程中发现新增训练进度混入环境检查 stdout，导致首次证据导出失败；已将环境检查的进度重定向到 stderr，加入 JSON 解析回归测试并成功重新导出。失败材料保留于 `outputs/torch-tune-20261105-export-failed`，未写成成功记录。此修正只涉及环境检查脚本，没有改动已经训练的网络或指标。

## 6. 复现与产物

```bash
# 先按 README 准备完整数据；这里复用上一轮已冻结划分。
uv run ml-network-intrusion-detection train --config configs/tune.yaml --data data/processed/torch-full-20261105 --output models/torch-tune-20261105 --device mps
uv run python scripts/evaluate_selected.py --training models/torch-tune-20261105 --data data/processed/torch-full-20261105 --output outputs/torch-tune-20261105 --device mps
uv run python scripts/check_environment.py --device mps > outputs/torch-tune-20261105/environment-v2.json
uv run python scripts/verify_inference.py --model models/torch-tune-20261105/transformer32-l2 --input data/processed/torch-full-20261105/test.csv --output outputs/torch-tune-20261105-transformer-verification --device cpu
uv run python scripts/verify_inference.py --model models/torch-tune-20261105/best --input data/processed/torch-full-20261105/test.csv --output outputs/torch-tune-20261105-mlp-verification --device cpu
uv run python scripts/export_results.py --training models/torch-tune-20261105 --evaluations outputs/torch-tune-20261105 --data data/processed/torch-full-20261105 --environment outputs/torch-tune-20261105/environment-v2.json --output reports/results/torch-tune-20261105
```

输出目录必须是新的。Transformer 模型为 `models/torch-tune-20261105/transformer32-l2`；本轮验证整体获选为 `models/torch-tune-20261105/best`（MLP）。上一轮模型保持在 `models/torch-full-20261105/best`，未覆盖。旧配置 `full/smoke` 的候选不变，新配置为 `tune/transformer_smoke`。

精简证据由导出脚本生成，独立验证与小样本 JSON 通过 `portable` 规范化后补入结果目录及 [export_manifest.json](results/torch-tune-20261105/export_manifest.json)。清单记录原始路径与哈希；不提交数据、模型二进制或逐行预测。模型目录保存完整输入契约、预处理器、架构与 CPU state_dict，加载不需要重新训练。

## 7. 局限与参考

这是单种子、有限参数范围且重复使用历史测试集的实验；不能证明 Transformer 普遍优于 MLP，或这套参数就是最优。新类别、主机/时间分组、外部数据、误报约束下的验证阈值选择和多种子统计仍需后续独立实验。新模型增加了计算成本，当前误报率仍不适合直接据此宣布可上线。

架构背景参考 [Gorishniy 等，Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959)；注意力调用遵循 [PyTorch SDPA 官方文档](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html)。数据来源与历史实验引用见 [数据说明](dataset.md)。
