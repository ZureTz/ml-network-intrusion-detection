"""检查 Torch 后端、实际训练和跨设备加载；合成数据不用于正式实验。"""

import argparse
import contextlib
import importlib
import json
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import make_classification
from threadpoolctl import threadpool_info

from ml_network_intrusion_detection.common import environment
from ml_network_intrusion_detection.device import device_info, resolve_device
from ml_network_intrusion_detection.training.models import TorchClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto/cpu/mps/cuda/cuda:N")
    args = parser.parse_args()
    dependencies = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "sklearn",
        "scipy": "scipy",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "joblib": "joblib",
        "pyyaml": "yaml",
        "psutil": "psutil",
        "torch": "torch",
    }
    for module in dependencies.values():
        importlib.import_module(module)
    device = resolve_device(args.device)
    features, labels = make_classification(n_samples=100, n_features=8, random_state=42)
    classifier = TorchClassifier(device=str(device), threads=2, epochs=2, batch_size=32)
    # stdout 是可重定向的 JSON 报告；训练进度单独写入 stderr。
    with contextlib.redirect_stdout(sys.stderr):
        classifier.fit(features[:80], labels[:80])
    probabilities = classifier.predict_proba(features[80:])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "weights.pt"
        torch.save({k: v.cpu() for k, v in classifier.network_.state_dict().items()}, path)
        restored = TorchClassifier(device="cpu", threads=2).initialize(features.shape[1])
        restored.network_.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        np.testing.assert_allclose(
            probabilities, restored.predict_proba(features[80:]), atol=1e-6, rtol=1e-5
        )
    print(
        json.dumps(
            {
                **environment(),
                "accelerators": {
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_device_count": torch.cuda.device_count(),
                    "mps_built": torch.backends.mps.is_built(),
                    "mps_available": torch.backends.mps.is_available(),
                },
                "runtime": device_info(device),
                "numpy_build_config": np.show_config(mode="dicts"),
                "threadpools": threadpool_info(),
                "dependencies": {name: version(name) for name in dependencies},
                "smoke_check": (
                    "passed: Torch fit, predict, CPU state_dict save/load, probability consistency"
                ),
                "note": "合成数据仅用于环境检查，不代表入侵检测性能；不保证跨设备逐位一致。",
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
