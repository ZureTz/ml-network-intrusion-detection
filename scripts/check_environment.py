"""检查依赖、CPU 信息和模型序列化；合成数据不用于正式实验。"""

import importlib
import json
import platform
import tempfile
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import psutil
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
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
    }
    for module in dependencies.values():
        importlib.import_module(module)

    features, labels = make_classification(n_samples=100, n_features=8, random_state=42)
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
    pipeline.fit(features[:80], labels[:80])
    predictions = pipeline.predict(features[80:])
    with tempfile.TemporaryDirectory() as temporary_directory:
        model_path = Path(temporary_directory) / "smoke-model.joblib"
        joblib.dump(pipeline, model_path)
        restored = joblib.load(model_path)
        np.testing.assert_array_equal(predictions, restored.predict(features[80:]))

    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "architecture": platform.machine(),
                "logical_cpu_count": psutil.cpu_count(),
                "memory_gib": round(psutil.virtual_memory().total / 1024**3, 2),
                "dependencies": {name: version(name) for name in dependencies},
                "smoke_check": "passed: fit, predict, save, load, prediction consistency",
                "note": "合成数据仅用于环境检查，不代表入侵检测性能。",
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
