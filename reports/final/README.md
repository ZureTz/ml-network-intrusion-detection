# 最终实验报告构建说明

正文的唯一维护入口是 [reports/README.md](../README.md)。公开 PDF 不填写姓名或学号，所有可编辑正文和图表使用常规字重。PPT 目录和扩展名被 Git 忽略，不属于此报告交付。

## 文件

- `report.tex`：Pandoc 从正文生成的完整 LaTeX 文件，不需要 Pandoc 即可直接重新编译。
- `report.pdf`：XeLaTeX 编译的最终报告。
- `template.tex`：封面、目录、字体、页码和表格样式。
- `report-filter.lua`：将 PDF 内部的相对文档链接转换为公开仓库链接，优先选用矢量图。
- `comparison.csv`：从三个真实实验批次合并，保留每行的来源路径。
- `figures/`：结果比较、MLP 调参比较、结构/流程图及经过可见隐私检查的真实前端截图。

## 重建

系统需要 Pandoc 3.x 和 TeX Live（包含 XeLaTeX、ctex、Fandol 字体、fvextra、booktabs 等）。本次使用 Pandoc 3.13、XeTeX / TeX Live 2026。Python 依赖沿用项目 uv 环境，不新增训练依赖。

从仓库根目录运行：

```bash
uv sync --locked
uv run python scripts/plot_report.py
uv run python scripts/build_report.py
```

第一步图表构建读取已提交的 CSV，不重新训练或调用正式测试集。正文表格以原始证据为准；若更新实验，需同步正文表格与解释，不能仅重新绘图。第二步生成 `report.tex`，连续执行三遍 XeLaTeX 稳定目录和表格，最终 PDF 保存到此目录，中间文件留在被忽略的 `outputs/report-build/`。

只编译已交付的 LaTeX 源文件时，在 `reports/` 目录执行以下命令三遍：

```bash
xelatex -interaction=nonstopmode -halt-on-error -output-directory=final final/report.tex
```

LaTeX 内的图路径相对于 `reports/`，分享源文件时需要同时提供 `final/figures/`。编译不会更新模型或实验结果。发布前应检查日志中的缺字、越界和未定义引用，渲染每页后目视检查。

## 结果来源和模拟证据

离线指标来自 `reports/results/full-20261103/`、`torch-full-20261105/` 和 `torch-tune-20261105/`。报告表格与图表不包含被撤回论文版 FT-Transformer 的数据。

模拟证据来自历史 `outputs/simulation/state.sqlite` 的只读 state/jobs 快照。每次任务的样本计数与训练 `data/manifest.json` 中 `source_ids` 长度核对，后续切换决策与 `model/promotion.json` 核对；导出时递归去除工作区绝对路径前缀。精简文件位于 [simulation-20261106](../results/simulation-20261106/)，未复制 SQLite、原始数据、模型或逐行预测。来源哈希清单记录原始证据哈希与导出方式。

界面截图来自同一历史演示的本地留存。快照累计 20,412 条事件，18,907 条已有反馈、1,505 条待反馈，8 次训练任务；来源去重训练池为 10,007 条。界面暂停展示与进程退出后数据库 paused 字段不作为严格同步状态，只对齐累计计数和训练任务。回放指标包含重复事件及多个模型版本，不能代替独立测试指标。

## 发布检查

本次更新不修改训练或推理实现，不重复完整训练。检查报告本地链接、跨批次划分哈希、表格与原始数值的一致性、模拟决策，以及 PDF 元数据和正文中是否出现姓名或本机路径。暂存时明确列出报告和脚本路径，禁止使用包含私人材料的整目录无差别添加。
