"""Torch 实际学习、设备选择和跨设备持久化回归测试。"""

import numpy as np
import pytest
import torch
from scipy import sparse

from ml_network_intrusion_detection.data.schema import validate_features
from ml_network_intrusion_detection.device import resolve_device
from ml_network_intrusion_detection.training.artifacts import restore_pipeline, save_pipeline
from ml_network_intrusion_detection.training.models import (
    TorchClassifier,
    build_pipeline,
    candidates,
)

from .conftest import fixed_frame


def test_device_auto_priority_and_explicit_errors(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert str(resolve_device()) == "cuda:0"
    assert str(resolve_device("cuda:1")) == "cuda:1"
    assert str(resolve_device("cpu")) == "cpu"
    with pytest.raises(ValueError, match="编号"):
        resolve_device("cuda:2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert str(resolve_device()) == "mps"
    with pytest.raises(ValueError, match="CUDA 不可用"):
        resolve_device("cuda")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert str(resolve_device()) == "cpu"
    with pytest.raises(ValueError, match="MPS 不可用"):
        resolve_device("mps")
    for name in ("metal", "cuda:-1", "cpu:0", "xpu", "cuda:abc"):
        with pytest.raises(ValueError):
            resolve_device(name)


@pytest.mark.parametrize("family", ["linear", "mlp"])
def test_training_learns_reproducibly_and_batches_sparse_input(family):
    rng = np.random.default_rng(17)
    x = rng.normal(size=(96, 4)).astype(np.float32)
    y = (x[:, 0] > 0).astype(int)
    params = dict(
        family=family,
        device="cpu",
        threads=1,
        epochs=12,
        batch_size=13,
        learning_rate=0.02,
        hidden_dims=(16,),
    )
    first = TorchClassifier(**params).fit(sparse.csr_matrix(x), y)
    second = TorchClassifier(**params).fit(x, y)
    assert first.loss_history_[-1] < first.loss_history_[0] * 0.5
    assert (first.predict(x) == y).mean() > 0.9
    np.testing.assert_array_equal(first.predict_proba(x), second.predict_proba(x))
    first.batch_size = 1
    np.testing.assert_allclose(first.predict_proba(x), second.predict_proba(x), atol=1e-6)
    assert first.predict_proba(x[:0]).shape == (0, 2)
    assert all(parameter.device.type == "cpu" for parameter in first.network_.parameters())


def test_balancing_uses_training_class_counts_and_rejects_bad_values():
    x = np.arange(40, dtype=float).reshape(10, 4)
    y = np.array([0] * 8 + [1] * 2)
    model = TorchClassifier(balanced=True, device="cpu", threads=1, epochs=1).fit(x, y)
    assert model.positive_weight_ == 4.0
    with pytest.raises(ValueError, match="float32"):
        model.predict_proba(np.full_like(x, 1e300))
    with pytest.raises(ValueError, match="维度"):
        model.predict_proba(x[:, :2])
    with pytest.raises(ValueError, match="标签"):
        model.fit(x, np.zeros(10))


@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_real_accelerator_training_and_cpu_reload(tmp_path, device):
    available = torch.backends.mps.is_available() if device == "mps" else torch.cuda.is_available()
    if not available:
        pytest.skip(f"本机 {device} 不可用；不视为已验证该硬件")
    frame = fixed_frame(48)
    x, _ = validate_features(frame)
    pipeline = build_pipeline(candidates("smoke")[2], 42, 1, device=device, epochs=2, batch_size=11)
    pipeline.fit(x, frame.label)
    classifier = pipeline.named_steps["classifier"]
    assert next(classifier.network_.parameters()).device.type == device
    probabilities = pipeline.predict_proba(x)
    metadata = save_pipeline(pipeline, tmp_path)
    restored = restore_pipeline(tmp_path, metadata, "cpu", 7)
    np.testing.assert_allclose(restored.predict_proba(x), probabilities, atol=1e-6, rtol=1e-5)
