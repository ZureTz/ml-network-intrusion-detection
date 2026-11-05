# 基于 Torch 的网络流量入侵检测系统

2026-11-04 完成 PyTorch 重构及小样本验证。当前已验证 Apple MPS 与 CPU，CUDA 设备路径已实现，尚无 NVIDIA 硬件实测。本文明确区分工程验证和正式检测实验；[原 scikit-learn 实验报告](sklearn-20261103.md) 与 [历史证据](results/full-20261103/) 保留，不将旧模型指标用于描述 Torch 模型。

## 1. 问题定义与数据集

使用 UNSW-NB15 的 42 个流量特征做离线二分类，`0 = normal`、`1 = attack`，攻击为正类。来源、镜像固定版本、SHA-256、许可说明、标签与特征含义见 [dataset.md](dataset.md)。不包含实时抓包、Web 服务或线上部署。

数据处理沿用原实现：优先遵守官方训练/测试角色，隔离训练侧冲突标签组，移除与测试集完全重叠的开发记录，按特征组分层划分训练/验证。同组不跨训练/验证；所有预处理仅在训练集拟合，标签、攻击类型与 ID 不进入特征。缺少时间、主机和会话字段，不能证明这些层面的隔离。

## 2. 系统设计与实现

命令仍为 `prepare → train → evaluate / predict`，完整用法见 [根目录 README](../README.md)。CPU 上的 `ColumnTransformer` 对数值进行中位数填充与标准化，对类别做 One-Hot；Pipeline 最后一步是 Torch 分类器。预处理保留稀疏矩阵，每次只将一批转成 float32 稠密 tensor，网络、梯度和优化器在所选设备执行。

设备选择支持 `auto`（CUDA → MPS → CPU）、`cpu`、`mps`、`cuda` 和 `cuda:N`。显式设备不可用时抛错；算子、内存或非有限数值错误会记录失败，不自动重跑其他设备。固定种子控制初始化与样本打乱，跨设备和版本不保证逐位一致。

Torch 实现多数类基线、单层线性分类器和 ReLU MLP。训练采用 AdamW 与 BCEWithLogitsLoss，固定轮数并记录逐轮损失。类别加权只使用训练集计数。模型保存为 `preprocess.joblib`、CPU `weights.pt` 和包含架构/类别/字段/阈值/哈希/依赖版本的 `metadata.json`。推理先校验，再用 `weights_only=True` 加载权重，重建完整 Pipeline 并移到本机选择的设备。预处理 joblib 仍只允许可信来源。

新格式版本为 2。旧 `pipeline.joblib` 不能直接加载到 Torch 流程，应重新训练；历史结果需要对应历史代码及锁文件。数据契约、无标签输入、额外列告警、未知类别统计、分批输出和行顺序保持原有语义。

## 3. 环境与测量方法

实际小样本运行环境为 Python 3.12.13、PyTorch 2.15.0、macOS 27.2 arm64、8 逻辑 CPU、16 GiB 内存；训练及独立验证评估使用 MPS、2 个 CPU 算子线程，训练批量 512、3 轮，独立评估批量 256。完整依赖、实际设备、种子、配置、代码与锁文件哈希见 [evidence.json](results/torch-smoke-20261104-v2/evidence.json)。Git 状态为有未提交修改，基线提交不等于重构后的完整代码，以记录的源码哈希为准。

训练耗时包含预处理拟合、网络初始化和优化，结束前同步 GPU。评估预热一批后测量三次，包含 CPU 预处理、数据传输与已同步的 GPU 预测，加载耗时单独记录。进程 RSS 每 50 ms 采样，不是 GPU 显存；CUDA 训练额外支持分配峰值，MPS 显存峰值未测量。小样本首次初始化开销明显，这些数据不能证明 GPU 加速比。

## 4. 实验方案与当前结果

正式方案为 [full.yaml](../configs/full.yaml) 的 7 个候选：多数类基线、线性分类器（加权/不加权）、MLP 64/32 和 128/64（各加权/不加权），20 轮。仅以验证集攻击 F1、较低误报率、较高 Recall 和候选顺序选优；阈值固定 0.5。正式训练、完整测试集评估、多种子和 CPU/CUDA/MPS 对比仍待完成。

本次只运行 [smoke.yaml](../configs/smoke.yaml)：从真实开发数据抽取 5,000 条，得到 4,000 条训练、1,000 条验证；三个候选均成功。未用正式测试指标调参，未运行 Torch 正式测试集评估。以下只是参与选优的验证集结果，不是独立泛化性能：

| Torch 候选 | Accuracy | Precision | Recall | F1 | 误报率 |
|---|---:|---:|---:|---:|---:|
| 多数类 | 0.6720 | 0.6720 | 1.0000 | 0.8038 | 1.0000 |
| 线性 | 0.7610 | 0.9017 | 0.7232 | 0.8026 | 0.1616 |
| MLP 64/32 | 0.9010 | 0.8952 | 0.9658 | 0.9291 | 0.2317 |

选择 `mlp-64x32-balancedFalse`。其验证混淆矩阵（行是真实 `[normal, attack]`，列是预测）为 `[[252, 76], [23, 649]]`；ROC-AUC 为 0.9199，Average Precision 为 0.9492。误报率为 `FP/(FP+TN)`；AP 使用召回增量加权精度，与 PR 曲线梯形面积不同。所有候选的类别指标、宏平均和未定义指标原因保存在证据中。

MLP 的验证 F1 高于本次其他候选，但只有三轮和一次划分，正常类仍有明显误报。不能据此与历史随机森林完整测试结果直接比较，也不能推断已达到生产部署要求。

## 5. 工程验证

本机完整测试为 **39 passed、1 skipped**，跳过项为缺少 NVIDIA 硬件的 CUDA 实测；MPS 实测通过。覆盖实际梯度学习、CPU 固定种子复现、自动设备优先级和显式不可用设备报错、稀疏输入、分批结果一致性、GPU 权重转 CPU、独立进程无标签预测、数据隔离、预处理不泄漏、输入校验、产物哈希与失败记录。合成样本仅作工程检查。

`uv sync --locked`、Ruff 检查、格式检查及 Torch MPS 环境检查通过。沙箱内 MPS 不可见时测试为 38 passed、2 skipped；本机权限下才实际执行 MPS 测试，未将跳过视为 GPU 验证成功。

真实验证集 1,000 行移除标签后，在独立进程中用 CPU 加载 MPS 训练的获选模型。概率与直接 CPU 加载结果在 `atol=1e-6, rtol=1e-5` 内一致，标签和行顺序完全一致。17 条未知 `proto` 记录被统计并正常预测。结果见证据的 `cpu_verification` 和 `cpu_prediction`。跨设备数值接近性另由 MPS 自动化测试验证，不承诺边界附近的标签在所有设备逐位一致。

## 6. 复现与证据生成

本次运行目录均为新建目录，重跑时更换名称：

```bash
uv run ml-network-intrusion-detection prepare --config configs/smoke.yaml --output data/processed/torch-smoke-20261104
uv run ml-network-intrusion-detection train --config configs/smoke.yaml --data data/processed/torch-smoke-20261104 --output models/torch-smoke-20261104-v2 --device mps
uv run ml-network-intrusion-detection evaluate --model models/torch-smoke-20261104-v2/best --input data/processed/torch-smoke-20261104/validation.csv --output outputs/torch-smoke-20261104-v2-validation --batch-size 256 --device mps
uv run python scripts/verify_inference.py --model models/torch-smoke-20261104-v2/best --input data/processed/torch-smoke-20261104/validation.csv --output outputs/torch-smoke-20261104-v2-cpu-verification --device cpu
```

精简证据的 `sources` 映射逐项记录原 JSON 路径和 SHA-256；`records` 对应这些 JSON 的内容，使用 `scripts.export_results.portable` 将本机路径转为相对路径。没有复制数据或模型二进制。可按下列方法重建到新文件，并检查原始文件哈希：

```bash
uv run python - <<'PY'
from pathlib import Path
from ml_network_intrusion_detection.common import read_json, sha256, write_json
from scripts.export_results import portable
root = Path.cwd()
record = read_json(Path('reports/results/torch-smoke-20261104-v2/evidence.json'))
for source in record['sources'].values():
    assert sha256(root / source['path']) == source['sha256']
record['records'] = {
    key: portable(read_json(root / source['path']), root)
    for key, source in record['sources'].items()
}
write_json(Path('outputs/torch-smoke-evidence-rebuilt.json'), record)
PY
```

重训会产生新的时间、权重和环境哈希，须生成新的来源清单，不应强行匹配旧运行哈希。正式完整实验的导出使用根目录 README 中的 `scripts/export_results.py`，本次验证子集未冒充正式测试导出。

## 7. 局限与后续实验

当前仍是离线、单设备、单种子实验，不包含多 GPU、混合精度、阈值调优、早停、概率校准或实时网络验证。训练数据与独立评估数据仍需放入 CPU 内存；只有推理 CSV 分批读取。当前默认依赖在 Apple Silicon 与 Linux CUDA 上可使用对应后端，其他构建需重新配置官方索引；NVIDIA、AMD 和 Windows 加速尚无本机实测。

GPU 是否优于 CPU 需要同模型、相同批量及测量范围的受控实验。正式报告应完成七候选训练与冻结测试评估，再分析误报、资源使用和泛化限制；不得沿用旧 sklearn 指标，也不得用当前验证分数代替测试分数。

## 8. 参考资料

数据集论文与来源见 [数据说明](dataset.md) 和 [历史报告参考文献](sklearn-20261103.md#11-参考文献与复现说明)。重构所依据的官方资料：

- [PyTorch MPS backend](https://docs.pytorch.org/docs/2.15/notes/mps.html)：设备可用性与 MPS tensor。
- [PyTorch Serialization semantics](https://docs.pytorch.org/docs/2.15/notes/serialization.html)：state_dict 与 weights_only 加载。
- [uv 与 PyTorch 集成](https://docs.astral.sh/uv/guides/integration/pytorch/)：不同平台的构建与索引。
