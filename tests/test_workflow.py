import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from ml_network_intrusion_detection.common import read_json, write_json
from ml_network_intrusion_detection.data.prepare import prepare
from ml_network_intrusion_detection.data.schema import FEATURES, read_frame, validate_features
from ml_network_intrusion_detection.inference.predict import load_model, predict
from ml_network_intrusion_detection.training.evaluate import evaluate
from ml_network_intrusion_detection.training.train import train


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    from .conftest import fixed_frame

    root = tmp_path_factory.mktemp("workflow")
    fixed_frame().to_csv(root / "train.csv", index=False)
    fixed_frame(40, 1000).to_csv(root / "test.csv", index=False)
    config = {
        "seed": 42,
        "threads": 1,
        "device": "cpu",
        "epochs": 4,
        "train_batch_size": 32,
        "batch_size": 17,
        "sample_size": None,
        "profile": "smoke",
        "train_csv": str(root / "train.csv"),
        "test_csv": str(root / "test.csv"),
    }
    prepare(config, root / "data")
    train(config, root / "data", root / "models")
    return root


def test_end_to_end_subprocess_and_chunk_consistency(trained):
    root = trained
    frame = read_frame(root / "test.csv")[list(reversed(FEATURES))]
    frame.to_csv(root / "unlabeled.csv", index=False)
    selection = read_json(root / "models" / "selection.json")
    model_dir = root / "models" / selection["family_winners"]["mlp"]
    model, _ = load_model(model_dir, "cpu")
    expected = model.predict_proba(validate_features(frame)[0])[:, 1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml_network_intrusion_detection",
            "predict",
            "--model",
            str(model_dir),
            "--input",
            str(root / "unlabeled.csv"),
            "--output",
            str(root / "predictions.csv"),
            "--batch-size",
            "7",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr
    actual = pd.read_csv(root / "predictions.csv")
    np.testing.assert_allclose(actual.attack_probability, expected, atol=1e-6, rtol=1e-5)
    np.testing.assert_array_equal(actual.predicted_label, expected >= 0.5)
    assert actual.row_index.tolist() == list(range(1, 41))
    predict(model_dir, root / "unlabeled.csv", root / "whole.csv", 1000, "cpu")
    pd.testing.assert_frame_equal(actual, pd.read_csv(root / "whole.csv"), atol=1e-6, rtol=1e-5)
    evaluate(model_dir, root / "test.csv", root / "evaluation", 7, "cpu")
    assert read_json(root / "evaluation" / "metrics.json")["rows"] == 40
    assert (root / "evaluation" / "confusion_matrix.png").exists()


def test_late_invalid_chunk_does_not_publish(trained, tmp_path):
    frame = read_frame(trained / "test.csv")
    frame["dur"] = frame.dur.astype(object)
    frame.loc[len(frame) - 1, "dur"] = "bad"
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        predict(trained / "models" / "best", path, tmp_path / "result.csv", 7)
    assert not (tmp_path / "result.csv").exists()
    assert not list(tmp_path.glob(".prediction-*"))


def test_empty_and_unknown_input(trained, tmp_path):
    frame = read_frame(trained / "test.csv")[FEATURES]
    path = tmp_path / "input.csv"
    frame.iloc[:0].to_csv(path, index=False)
    report = predict(trained / "models" / "best", path, tmp_path / "empty.csv")
    assert report["rows"] == 0
    assert len(pd.read_csv(tmp_path / "empty.csv")) == 0
    frame["proto"] = "NEW_PROTOCOL"
    frame.to_csv(path, index=False)
    with pytest.warns(UserWarning, match="未知类别"):
        report = predict(trained / "models" / "best", path, tmp_path / "unknown.csv", 11)
    assert report["unknown_categories"]["proto"] == len(frame)
    with pytest.raises(FileExistsError):
        predict(trained / "models" / "best", path, tmp_path / "unknown.csv")


def test_model_metadata_version_and_hash(trained, tmp_path):
    import shutil

    target = tmp_path / "model"
    shutil.copytree(trained / "models" / "best", target)
    metadata = read_json(target / "metadata.json")
    metadata["schema_version"] = 99
    write_json(target / "metadata.json", metadata)
    with pytest.raises(ValueError, match="不兼容"):
        load_model(target)
    metadata["schema_version"] = 2
    write_json(target / "metadata.json", metadata)
    with (target / "weights.pt").open("ab") as handle:
        handle.write(b"corrupted")
    with pytest.raises(ValueError, match="校验和"):
        load_model(target)


def test_manifest_tampering_rejected(trained, tmp_path):
    import shutil

    shutil.copytree(trained / "data", tmp_path / "data")
    with (tmp_path / "data" / "train.csv").open("a") as handle:
        handle.write("\n")
    config = read_json(trained / "models" / "run.json")["config"]
    with pytest.raises(ValueError, match="校验和"):
        train(config, tmp_path / "data", tmp_path / "model")
    assert read_json(tmp_path / "model" / "run.json")["status"] == "failed"
