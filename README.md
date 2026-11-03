# 基于机器学习的网络流量入侵检测系统的设计与实现

基于 Python 3.12、uv 和 scikit-learn 的课程项目。使用 UNSW-NB15 网络流量特征进行离线二分类：`0 = normal`，`1 = attack`。提供数据审计、训练比较、独立评估和无标签 CSV 分批预测。

## 当前进度

- 已实现 `prepare`、`train`、`evaluate`、`predict`，固定样本测试覆盖防泄漏、输入校验及独立进程预测。
- 已完成真实数据小样本闭环、11 个正式候选训练、完整测试集评估与独立无标签预测；结果和限制见 [实验报告](reports/README.md)。
- 原生 arm64，NumPy 使用 Apple Accelerate；随机森林使用 CPU 线程，不使用 Metal GPU。
- 不包含实时抓包、Web 服务、多分类或线上部署；CSV 必须包含相同语义和单位的网络流量特征，不能直接输入任意日志。

## 环境

```bash
uv sync --locked
uv run python scripts/check_environment.py
uv run ml-network-intrusion-detection --help
```

受限环境若不允许写默认缓存，可在命令前设置 `UV_CACHE_DIR=/tmp/ml-ids-uv-cache` 和 `MPLCONFIGDIR=/tmp/ml-ids-mpl`。普通本地环境不需要。所有命令在项目根目录执行。

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

# 正式划分与 11 个候选，完成选择后才访问测试指标。
uv run ml-network-intrusion-detection prepare --config configs/full.yaml --output data/processed/my-full
uv run ml-network-intrusion-detection train --config configs/full.yaml --data data/processed/my-full --output models/my-full
uv run python scripts/evaluate_selected.py --training models/my-full --data data/processed/my-full --output outputs/my-full
```

训练依次比较 Dummy、6 个逻辑回归和 4 个随机森林配置。预处理仅在训练集拟合，验证集按攻击类 F1、较低误报率、较高 Recall、候选顺序选优，阈值固定 `0.5`。`best/` 是最终交付模型；不将验证集合并后重新拟合，不用测试指标返回调参。

### 配置字段与默认值

YAML 使用安全读取，拒绝未知字段。文件路径相对配置文件所在目录；命令行路径相对工作目录。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `seed` | `42` | 非负整数随机种子 |
| `threads` | `4` | 正整数，上限为机器逻辑核数；控制 BLAS/OpenMP 和森林线程 |
| `batch_size` | `10000` | 实验约定批量；评估/预测实际批量由 `--batch-size` 指定，默认同为 10000 |
| `sample_size` | `null` | 开发集可选抽样行数，最少 20；正式配置不抽样 |
| `profile` | `full` | `full` 为 11 个配置；`smoke` 为 3 个轻量配置 |
| `conflict_policy` | `error` | 默认遇到同特征冲突标签就停止；`quarantine_development` 隔离开发集冲突组，完整保留测试集并记录冲突 |
| `train_csv` | `../data/raw/UNSW_NB15_training-set.csv` | 原始训练文件 |
| `test_csv` | `../data/raw/UNSW_NB15_testing-set.csv` | 原始测试文件 |

提供的两套配置显式启用 `quarantine_development`，因为本数据确实包含冲突标签。准备阶段进一步移除开发集中与测试集特征完全重复的记录，再按特征组分层划出约 20% 验证集；同组不能跨训练/验证。所有移除操作都有来源行号清单，不修改原始 CSV。

## 独立评估和无标签推理

```bash
# 只评估已冻结的可信模型，输入需要 label。
uv run ml-network-intrusion-detection evaluate --model models/my-full/best --input data/processed/my-full/test.csv --output outputs/my-evaluation --batch-size 10000

# 新进程直接加载；输入不需要 label 或 attack_cat。
uv run ml-network-intrusion-detection predict --model models/my-full/best --input data/new_flows.csv --output outputs/new_predictions.csv --batch-size 10000
```

- 输入需要 [42 个固定特征](reports/dataset.md)，程序按模型顺序重排列；多余列忽略并提示，缺少必需列、重复列、字段数量错误、非法数值会报错。
- 数值空单元格和无穷值使用训练阶段保存的填充值，类别空单元格使用固定标记；保留 `service` 的 `-` 类别。未知类别允许预测并计数，原始数值异常也记录在审计中。
- 输出 `row_index`（从 1 开始的数据记录序号，不含表头）、`predicted_label`、`attack_probability`。概率表示模型给出的攻击类概率，未做概率校准。
- 分批预测成功后才原子发布最终 CSV，失败清理临时结果；空表输出只有表头。拒绝覆盖已存在的输出。
- 同名 `.csv.json` 记录批量、线程、未知类别、缺失/无穷计数和耗时。`evaluate` 输出完整指标、曲线、错误行清单及类别细分。
- `joblib` 可能执行代码，仅加载可信来源模型。校验和不等于信任验证；加载前检查格式、依赖版本和内容哈希。恢复版本使用对应的 `uv.lock`。

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

测试用固定合成样本，不要求下载真实数据，测试分数不作为论文证据。测试包含训练/验证/测试重复隔离、冲突标签策略、填充器不泄漏、未知类别、指标未定义情况、保存加载和独立子进程预测。

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
