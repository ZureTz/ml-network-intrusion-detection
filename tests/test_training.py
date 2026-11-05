import numpy as np
import pytest
import torch

from ml_network_intrusion_detection.common import read_json
from ml_network_intrusion_detection.data.prepare import prepare
from ml_network_intrusion_detection.data.schema import validate_features
from ml_network_intrusion_detection.training.artifacts import restore_pipeline, save_pipeline
from ml_network_intrusion_detection.training.models import build_pipeline, candidates
from ml_network_intrusion_detection.training.train import selection_key, train

from .conftest import fixed_frame


@pytest.mark.parametrize("candidate_index", [1, 2])
def test_full_pipeline_predictions_survive_serialization(tmp_path, candidate_index):
    frame = fixed_frame(100)
    x, _ = validate_features(frame)
    model = build_pipeline(candidates("smoke")[candidate_index], 42, 1, device="cpu", epochs=3)
    model.fit(x.iloc[:80], frame.label.iloc[:80])
    probabilities = model.predict_proba(x.iloc[80:])
    labels = model.predict(x.iloc[80:])
    metadata = save_pipeline(model, tmp_path)
    weights = torch.load(tmp_path / "weights.pt", map_location="cpu", weights_only=True)
    assert all(tensor.device.type == "cpu" for tensor in weights.values())
    restored = restore_pipeline(tmp_path, metadata, "cpu", 7)
    np.testing.assert_allclose(restored.predict_proba(x.iloc[80:]), probabilities, atol=1e-6)
    np.testing.assert_array_equal(restored.predict(x.iloc[80:]), labels)


def test_selection_ties_are_deterministic():
    def result(f1, fpr, recall, order):
        return {
            "validation": {"f1": f1, "false_positive_rate": fpr, "recall": recall},
            "order": order,
        }

    choices = [
        result(0.8, 0.1, 0.9, 0),
        result(0.9, 0.2, 0.8, 1),
        result(0.9, 0.1, 0.8, 2),
        result(0.9, 0.1, 0.9, 3),
        result(0.9, 0.1, 0.9, 4),
    ]
    assert min(choices, key=selection_key)["order"] == 3


def test_failed_candidates_recorded_and_not_selected(raw_data, monkeypatch):
    config, root = raw_data
    prepare(config, root / "data")

    def failing_pipeline(*args, **kwargs):
        raise ValueError("deliberate test failure")

    monkeypatch.setattr(
        "ml_network_intrusion_detection.training.train.build_pipeline", failing_pipeline
    )
    with pytest.raises(ValueError, match="全部候选失败"):
        train(config, root / "data", root / "failed")
    assert read_json(root / "failed" / "run.json")["status"] == "failed"
    results = read_json(root / "failed" / "candidates.json")
    assert len(results) == 3
    assert all(item["status"] == "failed" for item in results)
    assert not (root / "failed" / "best").exists()
