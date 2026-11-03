"""指标定义、未定义情况及静态图表。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    """攻击为正类；分母为零返回 None，AP 与梯形 PR-AUC 不混用。"""
    labels, scores = np.asarray(labels), np.asarray(scores)
    if len(labels) == 0 or len(labels) != len(scores):
        raise ValueError("评估需要非空且长度一致的标签与概率")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("标签或预测概率非法")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("概率必须位于 [0, 1]")
    predicted = (scores >= threshold).astype(int)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in matrix.ravel())
    reasons = {}

    def ratio(name: str, numerator: int, denominator: int) -> float | None:
        if denominator == 0:
            reasons[name] = "zero denominator"
            return None
        return numerator / denominator

    per_class = {}
    for name, true_positive, false_positive, false_negative, support in (
        ("normal", tn, fn, fp, tn + fp),
        ("attack", tp, fp, fn, tp + fn),
    ):
        per_class[name] = {
            "precision": ratio(f"{name}.precision", true_positive, true_positive + false_positive),
            "recall": ratio(f"{name}.recall", true_positive, support),
            "f1": ratio(
                f"{name}.f1",
                2 * true_positive,
                2 * true_positive + false_positive + false_negative,
            ),
            "support": support,
        }
    macro = {}
    for name in ("precision", "recall", "f1"):
        values = [per_class[key][name] for key in per_class]
        macro[name] = sum(values) / 2 if all(v is not None for v in values) else None
        if macro[name] is None:
            reasons[f"macro.{name}"] = "one or more class metrics undefined"
    both = len(np.unique(labels)) == 2
    if not both:
        reasons.update(roc_auc="requires both classes", average_precision="requires both classes")
    return {
        "rows": len(labels),
        "accuracy": (tp + tn) / len(labels),
        "precision": per_class["attack"]["precision"],
        "recall": per_class["attack"]["recall"],
        "f1": per_class["attack"]["f1"],
        "false_positive_rate": ratio("false_positive_rate", fp, fp + tn),
        "roc_auc": float(roc_auc_score(labels, scores)) if both else None,
        "average_precision": float(average_precision_score(labels, scores)) if both else None,
        "average_precision_definition": "sum of recall increments weighted by precision (AP)",
        "confusion_matrix": matrix.tolist(),
        "class_order": ["normal", "attack"],
        "per_class": per_class,
        "macro": macro,
        "threshold": threshold,
        "undefined_reasons": reasons,
    }


def save_figures(labels: np.ndarray, scores: np.ndarray, output: Path) -> None:
    """输出可随报告提交的 PNG；单类别不绘制 ROC/PR。"""
    predicted = (scores >= 0.5).astype(int)
    display = ConfusionMatrixDisplay.from_predictions(
        labels,
        predicted,
        labels=[0, 1],
        display_labels=["Normal", "Attack"],
        colorbar=False,
        values_format="d",
    )
    display.figure_.savefig(output / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(display.figure_)
    if len(np.unique(labels)) == 2:
        for name, cls in (("roc", RocCurveDisplay), ("precision_recall", PrecisionRecallDisplay)):
            display = cls.from_predictions(labels, scores)
            display.figure_.savefig(output / f"{name}.png", dpi=160, bbox_inches="tight")
            plt.close(display.figure_)
