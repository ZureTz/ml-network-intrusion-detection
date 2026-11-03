import pytest

from ml_network_intrusion_detection.training.metrics import binary_metrics


def test_known_confusion_matrix():
    metrics = binary_metrics([0, 0, 0, 1, 1], [0.1, 0.2, 0.8, 0.4, 0.9])
    assert metrics["confusion_matrix"] == [[2, 1], [1, 1]]
    assert metrics["accuracy"] == 0.6
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["false_positive_rate"] == pytest.approx(1 / 3)
    assert metrics["roc_auc"] == pytest.approx(5 / 6)


def test_undefined_metrics_not_fabricated():
    metrics = binary_metrics([0, 0], [0.1, 0.2])
    assert metrics["precision"] is None
    assert metrics["recall"] is None
    assert metrics["roc_auc"] is None
    assert metrics["average_precision"] is None
    assert metrics["macro"]["precision"] is None
    assert metrics["undefined_reasons"]


def test_continuous_scores_used_for_auc():
    first = binary_metrics([0, 1], [0.1, 0.4])
    second = binary_metrics([0, 1], [0.4, 0.1])
    assert first["confusion_matrix"] == second["confusion_matrix"]
    assert first["roc_auc"] == 1
    assert second["roc_auc"] == 0
