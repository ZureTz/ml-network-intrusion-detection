from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_network_intrusion_detection.common import load_config, read_json
from ml_network_intrusion_detection.data.prepare import prepare
from ml_network_intrusion_detection.data.schema import (
    FEATURES,
    NUMERIC,
    check_header,
    fingerprints,
    read_frame,
    validate_features,
    validate_labels,
)
from ml_network_intrusion_detection.training.models import build_pipeline, candidates

from .conftest import fixed_frame


def test_split_reproducible_and_disjoint(raw_data):
    config, root = raw_data
    prepare(config, root / "a")
    prepare(config, root / "b")
    hashes = {}
    for name in ("train", "validation", "test"):
        a = read_frame(root / "a" / f"{name}.csv")
        b = read_frame(root / "b" / f"{name}.csv")
        pd.testing.assert_frame_equal(a, b)
        hashes[name] = set(fingerprints(validate_features(a)[0]))
    assert not hashes["train"] & hashes["validation"]
    assert not (hashes["train"] | hashes["validation"]) & hashes["test"]
    assert read_json(root / "a" / "audit.json")["removed_cross_split_rows"] == 1
    with pytest.raises(FileExistsError):
        prepare(config, root / "a")


def test_conflicting_labels_fail_and_record(raw_data):
    config, root = raw_data
    frame = read_frame(Path(config["train_csv"]))
    frame.loc[len(frame) - 1, "label"] = 1
    frame.to_csv(config["train_csv"], index=False)
    with pytest.raises(ValueError, match="冲突"):
        prepare(config, root / "conflict")
    assert read_json(root / "conflict" / "run.json")["status"] == "failed"
    config["conflict_policy"] = "quarantine_development"
    prepare(config, root / "quarantined")
    audit = read_json(root / "quarantined" / "audit.json")
    assert audit["development"]["conflicting_rows"] == 2
    kept = pd.concat(
        [
            read_frame(root / "quarantined" / "train.csv"),
            read_frame(root / "quarantined" / "validation.csv"),
        ]
    )
    assert not kept.source_row.isin([1, len(frame)]).any()


def test_schema_labels_and_normalization(tmp_path):
    raw = fixed_frame(10)
    x, _ = validate_features(raw[list(reversed(raw.columns))])
    assert list(x.columns) == FEATURES
    assert validate_labels(raw).tolist() == [0, 1] * 5
    raw.loc[0, "dur"] = np.inf
    x, audit = validate_features(raw)
    assert np.isnan(x.loc[0, "dur"])
    assert audit["infinite"]["dur"] == 1
    with pytest.raises(ValueError, match="缺失"):
        validate_features(raw.drop(columns="dur"))
    raw["dur"] = raw.dur.astype(object)
    raw.loc[0, "dur"] = "oops"
    with pytest.raises(ValueError, match="dur"):
        validate_features(raw)
    raw.loc[0, "label"] = 2
    with pytest.raises(ValueError, match="label"):
        validate_labels(raw)
    path = tmp_path / "duplicate.csv"
    path.write_text("dur,dur\n1,2\n")
    with pytest.raises(ValueError, match="重复"):
        check_header(path)


def test_preprocessing_fits_training_only():
    raw = fixed_frame(40)
    raw["rate"] = np.nan  # 全空数值列须保留。
    x, _ = validate_features(raw)
    pipeline = build_pipeline(candidates("smoke")[1], seed=42, threads=1, device="cpu", epochs=2)
    pipeline.fit(x, raw.label)
    numeric = pipeline["preprocess"].named_transformers_["numeric"]
    before = numeric["impute"].statistics_.copy()
    validation = x.copy()
    validation["dur"] = 1e12
    validation["proto"] = "never_seen"
    probabilities = pipeline.predict_proba(validation)
    np.testing.assert_array_equal(numeric["impute"].statistics_, before)
    assert before[NUMERIC.index("dur")] == np.median(x.dur)
    assert before[NUMERIC.index("rate")] == 0
    assert np.isfinite(probabilities).all()


@pytest.mark.parametrize(
    "content",
    [
        "[]",
        "seed: -1",
        "threads: 0",
        "typo: 1",
        "profile: other",
        "device: metal",
        "device: null",
        "epochs: 0",
        "epochs: true",
        "train_batch_size: -1",
        "learning_rate: 0",
        "learning_rate: .nan",
        "weight_decay: -1",
        "seed: 4294967296",
    ],
)
def test_invalid_configuration(tmp_path, content):
    path = tmp_path / "bad.yaml"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_config(path)


def test_config_paths_relative_to_config(tmp_path):
    path = tmp_path / "valid.yaml"
    path.write_text("train_csv: train.csv\n")
    assert load_config(path)["train_csv"] == str(tmp_path / "train.csv")


@pytest.mark.parametrize("row", ["\n", "1,2,3\n"])
def test_malformed_csv_not_silently_accepted(tmp_path, row):
    path = tmp_path / "malformed.csv"
    path.write_text(",".join(FEATURES) + "\n" + row)
    with pytest.raises(ValueError, match="字段数量"):
        check_header(path)
