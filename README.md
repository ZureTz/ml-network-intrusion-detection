# 基于机器学习的网络流量入侵检测系统的设计与实现

机器学习系统课程作业，使用 Python 和 scikit-learn 实现网络流量入侵检测，提供命令行训练和批量推理，不开发图形界面。

## 当前进度

已创建仓库、配置 uv 环境并建立项目目录。数据集选型、数据处理、模型训练、推理及正式实验报告尚未完成。环境检查中的合成数据仅用于验证依赖可用，不作为入侵检测实验结果。

## 环境配置

- Python：3.12（`.python-version` 固定主次版本）
- 包管理：uv，使用 `uv.lock` 固定依赖版本
- 运行方式：CPU；后续根据数据规模选择笔记本或服务器
- 建模：scikit-learn、NumPy、pandas、SciPy
- 可视化与持久化：Matplotlib、Seaborn、joblib
- 配置与资源采集：PyYAML、psutil
- 开发工具：pytest、Ruff

```bash
git clone https://github.com/ZureTz/ml-network-intrusion-detection.git
cd ml-network-intrusion-detection
uv sync --locked
uv run python scripts/check_environment.py
uv run ruff check .
```

仓库为私有仓库，克隆需要 GitHub 访问权限。`uv sync` 会创建本地 `.venv`，不需要手动激活；运行命令时使用 `uv run`。

## 目录结构

```text
configs/             训练与推理配置（待添加）
data/raw/            原始数据（不提交）
data/processed/      预处理数据（不提交）
models/              保存的模型及预处理器（不提交）
outputs/             推理结果、实验指标及图表（不提交）
reports/             实验报告与实验记录
scripts/             环境检查等辅助脚本
src/ml_network_intrusion_detection/
  data/              数据读取与预处理（待实现）
  training/          模型训练与评估（待实现）
  inference/         模型加载与批量预测（待实现）
tests/               自动化测试（待添加）
```

## 后续实现要求

1. 选择公开入侵检测数据集，记录来源、授权、特征含义、标签分布及划分方式。
2. 建立训练、验证、测试集；仅在训练集拟合预处理器，避免数据泄漏。
3. 比较基线分类器与候选模型，固定随机种子，并记录参数和训练耗时。
4. 将预处理器与分类器作为同一个 Pipeline 保存，实现新流量特征 CSV 的批量推理。
5. 统计 Precision、Recall、F1、混淆矩阵、误报率及适用的 ROC-AUC/PR-AUC，记录推理吞吐和资源消耗。
6. 根据实际运行结果完成实验报告，讨论类别不平衡、数据泄漏和部署局限。

实验报告提纲见 [reports/README.md](reports/README.md)。
