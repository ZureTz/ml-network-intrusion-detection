"""常驻模拟：标签隔离、过载、真实定时训练和停止/重启恢复。"""

import json
import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd
import pytest
import yaml

from ml_network_intrusion_detection.common import read_json, write_json
from ml_network_intrusion_detection.data.schema import FEATURES, validate_features
from ml_network_intrusion_detection.inference.predict import load_model
from ml_network_intrusion_detection.inference.service import DEFAULTS, Replay, load_service_config
from ml_network_intrusion_detection.training.refresh import should_promote

from .conftest import fixed_frame


def wait_for(check, timeout=45):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.05)
    raise AssertionError("等待后台服务超时")


def events(path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    result = []
    for line in lines:
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            # 写入中的最后一行留待下一轮观察。
            break
    return result


def test_replay_isolates_labels_and_counts_overload(tmp_path):
    path = tmp_path / "input.csv"
    fixed_frame(4).to_csv(path, index=False)
    records = queue.Queue(maxsize=1)
    stop = threading.Event()
    replay = Replay(path, 1000, records, stop)
    replay.start()
    try:
        wait_for(lambda: replay.cycle >= 2 and replay.dropped > 0, timeout=3)
    finally:
        stop.set()
        replay.join(timeout=3)
    record = records.get_nowait()
    assert set(record["features"]) == set(FEATURES)
    assert "label" not in record and "attack_cat" not in record
    assert record["cycle"] == 1 and record["row_index"] == 1
    assert replay.error is None


@pytest.mark.parametrize("bad", [{"rate": True}, {"queue_size": 0}, {"unknown": 1}])
def test_service_config_rejects_invalid(tmp_path, bad):
    path = tmp_path / "service.yaml"
    path.write_text(yaml.safe_dump({"model": "m", "data": "d", "training_config": "t"} | bad))
    with pytest.raises(ValueError):
        load_service_config(path)


@pytest.mark.parametrize(
    "f1,fpr,expected",
    [(0.9, 0.1, True), (0.8, 0.1, False), (0.9, 0.3, False), (None, 0.1, False)],
)
def test_promotion_requires_improvement_without_more_false_alarms(f1, fpr, expected):
    assert (
        should_promote(
            {"f1": 0.8, "false_positive_rate": 0.2},
            {"f1": f1, "false_positive_rate": fpr},
        )
        is expected
    )


@pytest.mark.parametrize("training_ok", [True, False])
def test_service_training_replay_and_restart(trained, tmp_path, training_ok):
    training = read_json(trained / "models/run.json")["config"]
    training_path = tmp_path / "training.yaml"
    # 非法配置用于验证失败训练不会中止推理或替换当前模型。
    training_path.write_text(yaml.safe_dump(training if training_ok else {"invalid": True}))
    config = DEFAULTS | {
        "model": str(trained / "models/best"),
        "data": str(trained / "data"),
        "training_config": str(training_path),
        "output": str(tmp_path / "service"),
        "rate": 50,
        "max_wait_ms": 20,
        "training_interval_seconds": 3600,
    }
    path = tmp_path / "service.yaml"
    path.write_text(yaml.safe_dump(config))
    output = tmp_path / "service"
    output.mkdir()
    write_json(
        output / "state.json",
        {
            "identity": {key: config[key] for key in ("model", "data", "training_config")},
            "model": config["model"],
            "next_training_at": 0,
        },
    )
    command = [
        sys.executable,
        "-m",
        "ml_network_intrusion_detection",
        "service",
        "--config",
        str(path),
    ]
    with (tmp_path / "process.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, env=os.environ.copy())
        try:
            wanted = "training_finished" if training_ok else "training_failed"
            wait_for(lambda: any(e.get("event") == wanted for e in events(output / "events.jsonl")))
            wait_for(lambda: len(events(output / "predictions.jsonl")) >= 40)
            assert process.poll() is None
            # 独占锁防止第二实例启动重复回放和训练。
            duplicate = subprocess.run(command, capture_output=True, text=True, timeout=20)
            assert duplicate.returncode != 0 and "已有服务" in duplicate.stderr
        finally:
            process.terminate()
            process.wait(timeout=15)
    assert process.returncode == 0, (tmp_path / "process.log").read_text()
    status = read_json(output / "status.json")
    assert status["status"] == "stopped"
    predictions = events(output / "predictions.jsonl")[:40]
    model, _ = load_model(trained / "models/best", "cpu", 32)
    frame = pd.read_csv(trained / "data/test.csv", keep_default_na=False, na_values=[""])
    expected = model.predict_proba(validate_features(frame)[0])[:, 1]
    np.testing.assert_allclose([p["attack_probability"] for p in predictions], expected, atol=1e-6)
    assert [p["row_index"] for p in predictions] == list(range(1, 41))
    assert all("label" not in p and p["latency_ms"] >= 0 for p in predictions)
    state = read_json(output / "state.json")
    assert state["next_training_at"] > time.time()
    assert state["model"] == config["model"]  # 相同训练数据/配置不会被当作改善。
    if training_ok:
        decisions = list((output / "training").glob("*/promotion.json"))
        assert len(decisions) == 1 and not read_json(decisions[0])["accepted"]
    before = len(events(output / "predictions.jsonl"))
    with (tmp_path / "restart.log").open("w") as log:
        restarted = subprocess.Popen(command, stdout=log, stderr=log)
        try:
            wait_for(lambda: len(events(output / "predictions.jsonl")) > before)
        finally:
            restarted.terminate()
            restarted.wait(timeout=15)
    assert restarted.returncode == 0
    assert sum(e.get("event") == "training_started" for e in events(output / "events.jsonl")) == 1
    assert read_json(output / "state.json") == state
