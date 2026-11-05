"""Transformer 注意力学习、批量隔离、旧权重兼容和真实设备回归。"""

import numpy as np
import pytest
import torch

from ml_network_intrusion_detection.data.schema import validate_features
from ml_network_intrusion_detection.training.artifacts import restore_pipeline, save_pipeline
from ml_network_intrusion_detection.training.models import build_network, build_pipeline, candidates

from .conftest import fixed_frame


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_transformer_learning_and_portable_weights(tmp_path, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    frame = fixed_frame(64)
    x, _ = validate_features(frame)
    spec = candidates("transformer_smoke")[-1]
    model = build_pipeline(spec, 42, 1, device=device, epochs=8, batch_size=16, learning_rate=0.01)
    model.fit(x, frame.label)
    classifier = model["classifier"]
    assert classifier.loss_history_[-1] < classifier.loss_history_[0]
    attention_grad = classifier.network_.blocks[0].qkv.weight.grad
    assert attention_grad is not None and attention_grad.abs().sum().item() > 0
    expected = model.predict_proba(x)
    assert (model.predict(x) == frame.label).mean() > 0.8
    assert next(classifier.network_.parameters()).device.type == device
    metadata = save_pipeline(model, tmp_path)
    restored = restore_pipeline(tmp_path, metadata, "cpu", 7)
    np.testing.assert_allclose(restored.predict_proba(x), expected, atol=1e-6, rtol=1e-5)
    # 另一个流量记录不能改变当前记录的预测。
    np.testing.assert_allclose(restored.predict_proba(x.iloc[[0]]), expected[[0]], atol=1e-6)
    classifier.batch_size = 10000
    assert classifier.network_batch_size() == 256


def test_search_and_legacy_architecture():
    plan = candidates("tune")
    assert len(plan) == 11 and len({s["name"] for s in plan}) == 11
    assert {s["family"] for s in plan} == {"dummy", "linear", "mlp", "transformer"}
    # 无 Dropout 时保留旧 Sequential key，使旧 Torch state_dict 可直接加载。
    assert list(build_network(42, "mlp", (128, 64)).state_dict()) == [
        "0.weight",
        "0.bias",
        "2.weight",
        "2.bias",
        "4.weight",
        "4.bias",
    ]
    with pytest.raises(ValueError, match="整除"):
        build_network(42, "transformer", (), d_model=15, n_heads=4)
    with pytest.raises(ValueError, match="dropout"):
        build_network(42, "mlp", (32,), dropout=1)


def test_environment_report_remains_machine_readable():
    import json
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts/check_environment.py"
    result = subprocess.run(
        [sys.executable, str(script), "--device", "cpu"],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["runtime"]["device"] == "cpu"
    assert report["smoke_check"].startswith("passed:")
