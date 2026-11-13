"""从已提交的真实实验证据重建最终报告图表，不训练或重新评估模型。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports/final/figures"
SPECS = [
    ("Dummy", "torch-tune-20261105", "dummy"),
    ("Logistic regression", "full-20261103", "logistic_regression"),
    ("Random forest", "full-20261103", "random_forest"),
    ("Torch linear (original)", "torch-full-20261105", "linear"),
    ("Torch linear (tuned)", "torch-tune-20261105", "linear"),
    ("MLP (original)", "torch-full-20261105", "mlp"),
    ("MLP (tuned)", "torch-tune-20261105", "mlp"),
    ("Transformer", "torch-tune-20261105", "transformer"),
]


def comparison() -> pd.DataFrame:
    """读取模型族的验证获选候选；保留证据相对路径。"""
    rows = []
    for label, experiment, family in SPECS:
        path = Path("reports/results") / experiment / "comparison.csv"
        data = pd.read_csv(ROOT / path).set_index("family").loc[family].to_dict()
        rows.append({"model": label, "source": str(path), "family": family, **data})
    return pd.DataFrame(rows)


def save(fig: plt.Figure, name: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def diagram(name: str, rows: list[list[str]]) -> None:
    """用可读的并列方框表达顺序，图中文字使用常规字重。"""
    fig, ax = plt.subplots(figsize=(10, len(rows) * 1.35))
    ax.set(xlim=(0, 10), ylim=(0, len(rows) * 1.4))
    ax.axis("off")
    for r, labels in enumerate(rows):
        y = (len(rows) - r - 1) * 1.4 + 0.25
        width = 9.6 / len(labels)
        for i, label in enumerate(labels):
            column = i if r % 2 == 0 else len(labels) - 1 - i
            x = 0.2 + width * column
            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    width - 0.35,
                    0.9,
                    boxstyle="round,pad=0.06",
                    facecolor="#eef4f8",
                    edgecolor="#52758a",
                    linewidth=1,
                )
            )
            ax.text(
                x + (width - 0.35) / 2,
                y + 0.45,
                label,
                ha="center",
                va="center",
                fontsize=10,
                color="#183b52",
            )
            if i < len(labels) - 1:
                start = x + width - 0.25 if r % 2 == 0 else x - 0.1
                end = x + width - 0.1 if r % 2 == 0 else x - 0.25
                ax.add_patch(
                    FancyArrowPatch(
                        (start, y + 0.45),
                        (end, y + 0.45),
                        arrowstyle="->",
                        mutation_scale=12,
                        color="#52758a",
                    )
                )
            elif r < len(rows) - 1:
                center = x + (width - 0.35) / 2
                ax.add_patch(
                    FancyArrowPatch(
                        (center, y - 0.08),
                        (center, y - 0.42),
                        arrowstyle="->",
                        mutation_scale=12,
                        color="#52758a",
                    )
                )
    save(fig, name)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    data = comparison()
    data.to_csv(ROOT / "reports/final/comparison.csv", index=False)
    selected = data.iloc[1:]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    y = np.arange(len(selected))
    for ax, field, title, color in zip(
        axes,
        ["f1", "false_positive_rate"],
        ["Attack F1 (%) - higher is better", "False positive rate (%) - lower is better"],
        ["#286080", "#b56840"],
        strict=True,
    ):
        values = selected[field].to_numpy() * 100
        ax.barh(y, values, height=0.58, color=color)
        for yi, v in zip(y, values, strict=True):
            ax.text(v + 0.6, yi, f"{v:.2f}", va="center", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 105 if field == "f1" else 48)
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, selected.model)
    axes[0].invert_yaxis()
    fig.tight_layout()
    save(fig, "model-comparison")

    original = data.loc[data.model == "MLP (original)"].iloc[0]
    tuned = data.loc[data.model == "MLP (tuned)"].iloc[0]
    fields = ["validation_f1", "f1", "false_positive_rate"]
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    for offset, row, label, color in [
        (-0.18, original, "Original MLP", "#286080"),
        (0.18, tuned, "Tuned MLP", "#b56840"),
    ]:
        vals = np.array([row[k] * 100 for k in fields])
        bars = ax.bar(np.arange(3) + offset, vals, width=0.34, label=label, color=color)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], padding=3, fontsize=9)
    ax.set(
        xticks=np.arange(3),
        xticklabels=["Validation F1", "Test F1", "Test FPR"],
        ylabel="Percent (%)",
        ylim=(0, 113),
    )
    ax.legend(loc="upper center", ncol=2, frameon=False)
    save(fig, "mlp-tuning")
    diagram(
        "transformer-architecture",
        [
            [
                "39 numeric + 1 category + CLS\n41 tokens x 32 dimensions",
                "2 Pre-Norm blocks\n4 heads; FFN 32-128-32",
                "CLS: LayerNorm + Linear\nSigmoid; threshold 0.5",
            ],
        ],
    )
    diagram(
        "simulation-flow",
        [
            [
                "NB15 development pool",
                "Resident model\nPredict first",
                "Delayed original labels\n10 simulation seconds",
            ],
            [
                "Deduplicate source IDs",
                "Freeze accumulated samples\nRetrain in a child process",
                "Fixed validation gate\nPromote or retain model",
            ],
        ],
    )


if __name__ == "__main__":
    main()
