"""固定合成样本仅验证程序正确性，不提供检测效果证据。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_network_intrusion_detection.data.schema import CATEGORICAL, FEATURES


def fixed_frame(rows: int = 160, offset: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(42 + offset)
    labels = np.arange(rows) % 2
    frame = pd.DataFrame({name: rng.random(rows) for name in FEATURES if name not in CATEGORICAL})
    frame["dur"] = np.arange(offset, offset + rows, dtype=float)
    frame["sbytes"] = labels * 100 + rng.random(rows)
    frame["proto"] = np.where(labels, "tcp", "udp")
    frame["service"] = "-"
    frame["state"] = "FIN"
    frame["label"] = labels
    frame["id"] = np.arange(1, rows + 1)
    frame["attack_cat"] = np.where(labels, "Generic", "Normal")
    return frame


@pytest.fixture
def raw_data(tmp_path: Path) -> tuple[dict, Path]:
    train = fixed_frame()
    # 一个训练内部重复记录和一个训练/测试重叠记录。
    train = pd.concat([train, train.iloc[[0]]], ignore_index=True)
    test = pd.concat([fixed_frame(40, 1000), train.iloc[[1]]], ignore_index=True)
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    return {
        "seed": 42,
        "threads": 1,
        "device": "cpu",
        "epochs": 3,
        "train_batch_size": 32,
        "batch_size": 17,
        "sample_size": None,
        "profile": "smoke",
        "train_csv": str(tmp_path / "train.csv"),
        "test_csv": str(tmp_path / "test.csv"),
    }, tmp_path
