# 基于机器学习的网络流量入侵检测系统的设计与实现

基于 Python 3.12、uv 和 PyTorch 的课程项目。使用 UNSW-NB15 网络流量特征进行离线二分类：`0 = normal`，`1 = attack`。支持 Apple GPU（MPS）、NVIDIA GPU（CUDA）和 CPU，提供数据审计、训练比较、独立评估和无标签 CSV 分批预测。scikit-learn 继续用于预处理、数据划分和指标计算，分类器由 Torch 实现。

## 当前进度

- 新增常驻 NB15 特征回放与定时训练服务：支持 CLI 启动本地 Web Dashboard、延迟标签、累计新样本重训、验证后切换和轮转日志。它模拟已提取的流特征到达，不执行实时抓包。

- 已完成 [Transformer 与现有模型调参实验](reports/transformer-tuning-20261105.md)：测试 F1 为 Transformer 90.54%、调参线性 84.80%、调参 MLP 87.49%。MLP 验证集略升但测试下降，结果没有被写成全面提升；旧模型保留。本轮整体 `best/` 仍按预先规定的验证集选择为 MLP。

- 已实现 `prepare`、`train`、`evaluate`、`predict`，固定样本测试覆盖防泄漏、输入校验及独立进程预测。
- 已迁移为 Torch 多数类基线、线性分类器和 MLP，支持自动设备选择、批量训练及设备无关权重保存。
- 已在 Apple MPS 完成 7 个 Torch 候选的完整实验及 82,332 行测试评估；验证集选出的 MLP 128/64 测试 Accuracy 87.05%、攻击类 F1 89.16%、误报率 24.70%。完整测试集的 CPU 独立无标签预测验证通过，见 [正式结果](reports/results/torch-full-20261105/comparison.csv)。
- Torch 的验证记录与正式实验进度见 [实验报告](reports/README.md)；[原 scikit-learn 报告](reports/sklearn-20261103.md) 及精简证据保留，旧指标不代表 Torch 模型效果。
- 不包含实时抓包、多分类或生产部署；CSV 必须包含相同语义和单位的网络流量特征，不能直接输入任意日志。

## 环境

```bash
uv sync --locked
uv run python scripts/check_environment.py
uv run ml-network-intrusion-detection --help
```

受限环境若不允许写默认缓存，可在命令前设置 `UV_CACHE_DIR=/tmp/ml-ids-uv-cache` 和 `MPLCONFIGDIR=/tmp/ml-ids-mpl`。普通本地环境不需要。所有命令在项目根目录执行。

### 加速设备

`device: auto` 按 CUDA → MPS → CPU 选择；训练可用 `--device` 覆盖 YAML，评估和推理各自独立选择设备。显式请求不可用的设备会报错；训练中出现算子不支持或显存不足会记录失败，不会静默切换 CPU。单次使用一张 GPU，可通过 `cuda:N` 指定编号。

```bash
# Apple Silicon；检查实际 GPU 运算及保存后在 CPU 恢复。
uv run python scripts/check_environment.py --device mps
# NVIDIA；需要可用的 CUDA 构建与兼容驱动。
uv run python scripts/check_environment.py --device cuda
# 无 GPU 或需要固定 CPU 运行时。
uv run python scripts/check_environment.py --device cpu
```

当前锁定 PyTorch 2.15.0。默认 PyPI 安装在 Apple Silicon macOS 提供 MPS，在 Linux 提供 CUDA 构建；Windows 默认 PyPI 构建为 CPU。Windows CUDA、其他 CUDA 版本及 ROCm 需要按 [uv 官方 PyTorch 指南](https://docs.astral.sh/uv/guides/integration/pytorch/) 配置对应索引并重新锁定 `pyproject.toml`/`uv.lock`，不能只添加 `--device cuda`。相同 Torch 发布版本的不同构建后缀（如 `+cpu`/`+cuXXX`）允许交换模型；ROCm 构建使用 Torch 的 `cuda` 接口，但本仓库尚未在 AMD 硬件验证。当前锁文件没有 Intel Mac Torch wheel。

设备是否可用以环境检查输出为准；容器/沙箱可能屏蔽宿主 GPU。CSV、填充、标准化和编码仍在 CPU，网络前向、反向和优化器在所选设备运行。小表格模型不保证 GPU 比 CPU 快。

## 数据来源与获取

参见 [数据说明](reports/dataset.md)。优先从 [UNSW 官方页面](https://research.unsw.edu.au/projects/unsw-nb15-dataset) 获取训练及测试 CSV，分别放在：

```text
data/raw/UNSW_NB15_training-set.csv   # 175341 行
data/raw/UNSW_NB15_testing-set.csv    # 82332 行
```

本次官方 SharePoint 需要登录，经用户允许使用固定版本公开镜像。显式选择该镜像时运行：

```bash
uv run python scripts/download_data.py --source mireu-mirror
```

脚本固定提交和 SHA-256，核验字段与行数，生成 `source_manifest.json`；拒绝覆盖原始文件。镜像的文件名与官方角色相反，脚本已明确映射，不能直接按镜像文件名划分。若已用浏览器或 curl 下载镜像中的 `train.csv`、`test.csv`，可加 `--cache-dir 下载目录`，仍强制校验哈希。不会自动切换来源。

## 从小样本到正式实验

每次使用全新的输出目录，避免覆盖已有证据。下列命令的 `my-smoke`、`my-full` 是新运行名称示例。

```bash
# 真实训练数据抽样 5000 行，三个轻量候选；不评估正式测试集。
uv run ml-network-intrusion-detection prepare --config configs/smoke.yaml --output data/processed/my-smoke
uv run ml-network-intrusion-detection train --config configs/smoke.yaml --data data/processed/my-smoke --output models/my-smoke

# 正式划分与 7 个 Torch 候选，完成选择后才访问测试指标。
uv run ml-network-intrusion-detection prepare --config configs/full.yaml --output data/processed/my-full
uv run ml-network-intrusion-detection train --config configs/full.yaml --data data/processed/my-full --output models/my-full
uv run python scripts/evaluate_selected.py --training models/my-full --data data/processed/my-full --output outputs/my-full
```

训练比较多数类基线、2 个 Torch 线性分类器（单层 Linear，相当于用梯度优化的逻辑回归）和 4 个 MLP（隐藏层 64/32 或 128/64，ReLU）。线性和 MLP 分别比较不加权及训练类别计数计算的 `pos_weight = normal / attack`。smoke 每类仅一个不加权配置。使用 AdamW、BCEWithLogitsLoss、固定轮数，记录每轮训练损失；不会声称固定轮数已保证收敛。全部数值特征标准化，类别 One-Hot；预处理仅在训练集拟合。验证集按攻击类 F1、较低误报率、较高 Recall、候选顺序选优，阈值固定 `0.5`。`best/` 是最终交付模型；不将验证集合并后重新拟合，不用测试指标返回调参。

例如显式用 Apple GPU 训练：

```bash
uv run ml-network-intrusion-detection train --config configs/smoke.yaml --data data/processed/my-smoke --output models/my-smoke-mps --device mps
```

### Transformer 与现有 Torch 模型参数搜索

新增 `configs/transformer-smoke.yaml`（4 个模型族的小样本检查）与 `configs/tune.yaml`（11 个候选的完整搜索）。默认 `smoke/full` 仍保留原有 3/7 个候选，用于复现历史结果。

Transformer 将 39 个标准化数值字段分别映射成带字段身份的 token，将类别 One-Hot 联合投影为一个上下文 token，再加入 CLS token，共 41 个 token。使用 Pre-Norm 自注意力、GELU 前馈层和 CLS 分类头；注意力作用于同一行的特征，不跨流量记录。这是轻量表格 Transformer，并非原论文 FT-Transformer 的精确复现。网络批量最多 256，避免大的 CSV 读取批量直接放大注意力内存。

固定搜索空间见 `training/search.py`：原线性与 MLP 128/64 配置作对照，线性搜索学习率/权重衰减/轮数，MLP 搜索 64/32、128/64、256/128 宽度、学习率、权重衰减、Dropout 和轮数；Transformer 比较 16 维/2 头/1 层和 32 维/4 头/2 层，各 15 轮。候选参数覆盖 YAML 中的同名通用训练参数，实际参数保存到模型元数据；训练开始前将固定列表写入 `search_plan.json`。以验证 F1 选优，阈值仍为 0.5，不用测试指标调参。

```bash
# 首先按原小样本数据准备步骤创建 data/processed/my-smoke。
uv run ml-network-intrusion-detection train --config configs/transformer-smoke.yaml --data data/processed/my-smoke --output models/my-transformer-smoke --device mps
# 复用完整的防泄漏划分，训练并冻结各模型族的获选候选。
uv run ml-network-intrusion-detection train --config configs/tune.yaml --data data/processed/my-full --output models/my-tune --device mps
uv run python scripts/evaluate_selected.py --training models/my-tune --data data/processed/my-full --output outputs/my-tune --device mps
```

旧 Torch 模型的无 Dropout 网络结构和权重键保持兼容，仍可独立加载。新网络超参数由候选定义控制：`dropout` 默认 0，`d_model` 默认 32，`n_heads` 默认 4，`n_layers` 默认 2；维度必须能被头数整除。完整搜索为有限预设试验，不保证优于旧结果。旧测试集已多次使用，应将结果视为同一基准上的探索性比较，不能代替新独立测试集。

### 通用配置字段与默认值

YAML 使用安全读取，拒绝未知字段。文件路径相对配置文件所在目录；命令行路径相对工作目录。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `seed` | `42` | 非负整数随机种子 |
| `threads` | `4` | 正整数，上限为机器逻辑核数；控制 BLAS/OpenMP 与 Torch CPU 算子线程 |
| `device` | `auto` | `auto`、`cpu`、`mps`、`cuda`、`cuda:N`；训练 CLI 可覆盖 |
| `epochs` | `20` | 固定训练轮数；smoke 配置显式为 3 |
| `train_batch_size` | `512` | 训练小批量及训练时验证预测批量；最后不足一批仍参与训练 |
| `learning_rate` | `0.001` | AdamW 有限正学习率 |
| `weight_decay` | `0.0001` | AdamW 有限非负权重衰减 |
| `batch_size` | `10000` | 实验约定批量；评估/预测实际 CSV 和网络批量由 `--batch-size` 指定，默认同为 10000 |
| `sample_size` | `null` | 开发集可选抽样行数，最少 20；正式配置不抽样 |
| `profile` | `full` | `full` 为 7 个配置，`smoke` 为 3 个；`tune` 为 11 个，`transformer_smoke` 为 4 个 |
| `conflict_policy` | `error` | 默认遇到同特征冲突标签就停止；`quarantine_development` 隔离开发集冲突组，完整保留测试集并记录冲突 |
| `train_csv` | `../data/raw/UNSW_NB15_training-set.csv` | 原始训练文件 |
| `test_csv` | `../data/raw/UNSW_NB15_testing-set.csv` | 原始测试文件 |

提供的两套配置显式启用 `quarantine_development`，因为本数据确实包含冲突标签。准备阶段进一步移除开发集中与测试集特征完全重复的记录，再按特征组分层划出约 20% 验证集；同组不能跨训练/验证。所有移除操作都有来源行号清单，不修改原始 CSV。

固定种子控制网络初始化与每轮样本顺序，但不同设备、Torch 版本及批量可能产生浮点差异，不承诺逐位复现。网络统一使用 float32，不启用混合精度；超出 float32 范围的预处理输入会报错。准备和训练仍将各 CPU 数据划分载入内存，GPU 每次仅接收一批；`predict` 分批读取 CSV，`evaluate` 为绘图和错误分析仍将完整评估集载入 CPU 内存。

## 独立评估和无标签推理

```bash
# 只评估已冻结的可信模型，输入需要 label。
uv run ml-network-intrusion-detection evaluate --model models/my-full/best --input data/processed/my-full/test.csv --output outputs/my-evaluation --batch-size 10000

# 新进程直接加载；输入不需要 label 或 attack_cat。
uv run ml-network-intrusion-detection predict --model models/my-full/best --input data/new_flows.csv --output outputs/new_predictions.csv --batch-size 10000

# GPU 训练的同一模型也可以直接在 CPU 上预测。
uv run ml-network-intrusion-detection predict --model models/my-full/best --input data/new_flows.csv --output outputs/cpu_predictions.csv --device cpu
```

- 输入需要 [42 个固定特征](reports/dataset.md)，程序按模型顺序重排列；多余列忽略并提示，缺少必需列、重复列、字段数量错误、非法数值会报错。
- 数值空单元格和无穷值使用训练阶段保存的填充值，类别空单元格使用固定标记；保留 `service` 的 `-` 类别。未知类别允许预测并计数，原始数值异常也记录在审计中。
- 输出 `row_index`（从 1 开始的数据记录序号，不含表头）、`predicted_label`、`attack_probability`。概率表示模型给出的攻击类概率，未做概率校准。
- 分批预测成功后才原子发布最终 CSV，失败清理临时结果；空表输出只有表头。拒绝覆盖已存在的输出。
- 同名 `.csv.json` 记录实际设备、构建、批量、线程、未知类别、缺失/无穷计数和耗时。`evaluate` 输出完整指标、曲线、错误行清单及类别细分；预热一次后计时三次。计时包含 CPU 预处理、数据传输及同步后的 GPU 计算，模型加载单独测量。RSS 是进程内存，不能等同于显存；训练另记录 CUDA 分配峰值，MPS 显存峰值未测量。
- 模型目录为 `preprocess.joblib`、`weights.pt`、`metadata.json`，元数据记录架构、实际设备、特征契约、阈值、依赖和两个产物哈希。权重只保存 CPU `state_dict`，使用 `torch.load(weights_only=True, map_location="cpu")` 读取，再移至指定设备。加载后重建完整 Pipeline，训练与推理共用同一预处理器。
- `joblib` 可能执行代码，仅加载可信来源模型。校验和不等于信任验证；加载前检查格式、依赖版本和全部产物哈希。恢复版本使用对应的 `uv.lock`。旧 v1 `pipeline.joblib` 不兼容，需要重新训练 Torch 模型；复现旧结果应恢复历史代码及锁文件。

## 持续学习模拟与 Web Dashboard

```bash
uv sync --locked
uv run ml-network-intrusion-detection serve --config configs/simulation.yaml
```

访问终端打印的 `http://127.0.0.1:8000`。首次启动自动初始化独立实验和起始 MLP，之后持续模拟；关闭浏览器不影响后台，终端 Ctrl+C 保存并退出，再次执行相同命令恢复。无需双击脚本或登录启动任务。macOS/Linux 支持此入口，Windows 尚未验证（实验锁使用 fcntl）。

先按前文执行 `prepare`，并将配置 `data` 指向准备后的目录。默认引用本机已有 `data/processed/torch-full-20261105`；无需已有模型。服务只读其中的训练/验证文件并核验哈希，**不读取正式 test.csv**。按标签分层抽取完整特征组，约 10,000 条训练池中约 2,000 条用于初始训练；固定验证集约 2,000 条。保留重复组可能使实际数量略超目标，来源映射保存在实验 pool.csv。历史模型已见过完整训练集，因此不作为模拟起点。

默认 MLP 64/32、CPU 2 线程、5 epochs。每秒模拟到达 20 条记录，先使用当前模型预测，10 秒模拟时间后才揭示原始真实标签。未见且已确认的来源记录才入训练池；每 60 秒检查一次，新增至少 500 条则冻结累计样本快照，在独立进程中重新训练。验证 F1 严格上升且误报率不增加才加载候选并切换；训练失败保留旧模型，30 秒后允许重试。最长训练 2 小时。暂停冻结模拟时间及新任务调度，已启动训练允许完成。停止服务会终止子进程，下次恢复将其记录为中断。

页面支持暂停/恢复、手动触发有新样本的训练、速率调整、原始比例/攻击升高/突发流量三个场景、创建新实验、事件与攻击告警查看、模型决策对比和导出摘要。新实验表单可设置样本规模、轮次、标签延迟和训练间隔；旧实验保留，`active.json` 记录当前实验，再次启动恢复它。要从配置建立全新实验，请指定新的 output。

攻击升高场景在两类都有未见记录时以 80% 攻击概率抽样，类别耗尽后实际比例随剩余数据变化；突发场景每 30 秒中前 5 秒提高到 4 倍速率。所有输入来自真实记录，不扰动字段，不声称生成了新的独立网络数据。随机状态和事件游标持久化，但操作系统调度影响批次大小，不保证两次运行逐事件完全一致。剩余数据释放完后继续循环检测，来源去重后不再新增训练样本，不触发无数据的重复训练。

配置字段（所有路径相对于 YAML）：`data`、`output`；`seed=42`；`pool_size=10000`、`initial_size=2000`、`validation_size=2000`；`rate=20`（1–200）；`label_delay=10`；`training_interval=60`；`min_new_samples=500`；`epochs=5`、`threads=2`；`scenario=original`；`port=8000`。实验开始后划分与训练参数固定；改变配置需指定新的 output，场景/速率可在页面改变。训练 `profile: simulation` 仅使用 MLP 64/32，不进行全模型搜索。

运行输出：

- `service.log`：终端同款进度和训练日志，10 MB 轮转、5 份备份。
- `experiment.json`、`pool.csv`、`validation.csv`：配置、源清单、哈希与固定实验数据。
- `state.sqlite`：模拟时间、随机状态、来源去重、待反馈标签、模型指针、事件及训练决策。事件保留最近 100,000 条已确认记录，待反馈事件不会清理；累计混淆矩阵独立持久化。
- `training/<编号>/`：冻结样本清单、训练配置、模型、逐 epoch 日志和验证决策。这些实验依据不自动删除，长时间运行需归档旧版本。

API：`GET /api/status`、`GET /api/events?after=0`（100 条游标分页，支持 `attacks`、`model`、`recent`）、`GET /api/export`；`POST /api/control`（pause/resume/train/settings）；`POST /api/experiments` 创建独立实验。仅监听回环地址并限制同源控制请求，不提供远程多用户访问。

界面吞吐量由浏览器观察事件计数估算。事件延迟字段是整批模型预测耗时（包含 Pipeline 预处理），不是端到端网络延迟。引擎按模拟时钟尽力产生事件，最多积压 1,000 个待生成事件额度；超过容量的额度计入 dropped，未生成的记录仍可后续选取。回放指标基于当时模型的预测，在标签揭示后累计，包含重复回放，不能作为独立测试分数。正式报告继续通过离线 evaluate 完成。旧 `service` 命令保留兼容，它是固定测试集循环回放与固定训练集重训，不是此持续学习实验。

## 报告证据导出

```bash
uv run python scripts/check_environment.py > outputs/my-full/environment.json
uv run python scripts/export_results.py --training models/my-full --evaluations outputs/my-full --data data/processed/my-full --environment outputs/my-full/environment.json --output reports/results/my-full
uv run python scripts/verify_inference.py --model models/my-full/best --input data/processed/my-full/test.csv --output outputs/my-verification
```

导出比较表、指标、图表、运行配置、数据与模型哈希、代码文件哈希。JSON 中本机绝对路径转为相对路径，原始文件哈希保留在导出清单；不会复制数据集或模型二进制。正式模型本身位于被忽略的 `models/`，重现时重新训练，或单独交付可信模型目录。

## 验证

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run python scripts/check_environment.py
uv run pytest
```

测试用固定合成样本，不要求下载真实数据，测试分数不作为论文证据。测试包含训练/验证/测试重复隔离、冲突标签策略、预处理不泄漏、未知类别、指标未定义情况、Torch 实际学习与种子复现、设备选择、保存加载和独立子进程预测。真实 MPS/CUDA 测试只在对应硬件可用时运行，否则明确跳过。float32 概率一致性使用 `atol=1e-6, rtol=1e-5`，不套用原 float64 模型的 `1e-12` 要求。

## 目录职责

| 目录 | 内容 |
|---|---|
| `src/ml_network_intrusion_detection/data/` | 字段契约、读取、审计与划分 |
| `src/ml_network_intrusion_detection/training/` | 模型、验证选择、指标、图表及评估 |
| `src/ml_network_intrusion_detection/inference/` | 可信模型加载和分批推理 |
| `configs/`、`scripts/`、`tests/` | 配置、辅助命令和测试 |
| `data/raw/`、`data/processed/` | 原始数据及可追溯划分，不提交 |
| `models/`、`outputs/` | 模型和完整实验输出，不提交 |
| `reports/` | 中文报告、数据说明及可提交的精简结果 |
