"""设备无关模型产物：CPU state_dict、已拟合预处理与架构元数据。"""

from pathlib import Path

import joblib
import torch
from sklearn.pipeline import Pipeline

from ml_network_intrusion_detection.common import sha256
from ml_network_intrusion_detection.device import synchronize
from ml_network_intrusion_detection.training.models import TorchClassifier


def save_pipeline(model: Pipeline, directory: Path) -> dict:
    """保存完整推理所需状态，返回架构和内容哈希；不保存 GPU 对象或优化器。"""
    classifier = model.named_steps["classifier"]
    joblib.dump(model.named_steps["preprocess"], directory / "preprocess.joblib")
    torch.save(
        {name: value.detach().cpu() for name, value in classifier.network_.state_dict().items()},
        directory / "weights.pt",
    )
    return {
        "format": "torch-state-dict-v2",
        "classifier": classifier.get_params(deep=False),
        "input_dim": classifier.n_features_in_,
        "artifacts": {
            name: sha256(directory / name) for name in ("preprocess.joblib", "weights.pt")
        },
    }


def restore_pipeline(directory: Path, metadata: dict, device: str, batch_size: int) -> Pipeline:
    """哈希验证由调用者完成；预处理 joblib 只允许来自可信目录。"""
    params = metadata["classifier"] | {"device": device, "batch_size": batch_size}
    classifier = TorchClassifier(**params).initialize(metadata["input_dim"])
    # 永远先加载 CPU tensor，再由 load_state_dict 复制到实际推理设备。
    state = torch.load(directory / "weights.pt", map_location="cpu", weights_only=True)
    classifier.network_.load_state_dict(state, strict=True)
    classifier.network_.eval()
    preprocess = joblib.load(directory / "preprocess.joblib")
    synchronize(classifier.device_)
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])
