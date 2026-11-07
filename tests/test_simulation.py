"""模拟系统的真实训练闭环与标签时序检查。"""

import sqlite3
import time

import pandas as pd
from fastapi.testclient import TestClient

from ml_network_intrusion_detection.data.schema import fingerprints
from ml_network_intrusion_detection.simulation.engine import DEFAULTS, Engine, initialize
from ml_network_intrusion_detection.simulation.web import create_app


def wait_for(client, predicate, timeout=45):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        status = client.get("/api/status").json()
        assert status["phase"] != "failed", status
        if predicate(status):
            return status
        time.sleep(0.1)
    raise AssertionError(client.get("/api/status").json())


def test_split_groups_and_no_test_access(trained, tmp_path):
    cfg = DEFAULTS | dict(
        data=str(trained / "data"),
        output=str(tmp_path),
        pool_size=100,
        initial_size=30,
        validation_size=30,
    )
    initialize(cfg, tmp_path)
    pool = pd.read_csv(tmp_path / "pool.csv")
    validation = pd.read_csv(tmp_path / "validation.csv")
    assert not set(fingerprints(pool)) & set(fingerprints(validation))
    assert not set(fingerprints(pool[pool.initial])) & set(fingerprints(pool[~pool.initial]))
    assert pool[pool.initial].label.nunique() == 2


def test_web_learning_and_restart(trained, tmp_path):
    cfg = DEFAULTS | dict(
        data=str(trained / "data"),
        output=str(tmp_path),
        pool_size=120,
        initial_size=30,
        validation_size=30,
        rate=20,
        label_delay=2,
        training_interval=3,
        min_new_samples=10,
        epochs=2,
        threads=1,
    )
    engine = Engine(cfg)
    with TestClient(create_app(engine)) as client:
        assert client.get("/").status_code == 200
        assert (
            client.post("/api/control", json={"action": "settings", "rate": 201}).status_code == 422
        )
        assert (
            client.post(
                "/api/control", headers={"Origin": "http://evil.example"}, json={"action": "pause"}
            ).status_code
            == 403
        )
        early = wait_for(client, lambda s: s.get("processed", 0) > 0)
        assert early["new_samples"] == 0
        assert all(e["truth"] is None for e in client.get("/api/events").json())
        result = wait_for(client, lambda s: len(s.get("jobs", [])) >= 2 and not s["training"])
        assert result["known"] > early["known"]
        assert result["processed"] > early["processed"]
        assert result["job"]["status"] in ("accepted", "rejected")
        assert client.post("/api/control", json={"action": "pause"}).status_code == 202
        paused = wait_for(client, lambda s: s["phase"] == "paused")
        time.sleep(0.2)
        assert client.get("/api/status").json()["processed"] == paused["processed"]
        assert client.get("/api/export").json()["summary"]["model"]
    with sqlite3.connect(tmp_path / "state.sqlite") as db:
        assert db.execute("select count(*) from samples").fetchone()[0] <= 125
        assert db.execute("select count(*) from events where truth is not null").fetchone()[0] > 0
    restored = Engine(cfg)
    with TestClient(create_app(restored)) as client:
        current = wait_for(client, lambda s: s["phase"] == "paused")
        assert current["processed"] == paused["processed"]
        assert current["model"] == paused["model"]


def test_create_experiment_and_resume_pointer(trained, tmp_path):
    from ml_network_intrusion_detection.common import read_json

    cfg = DEFAULTS | dict(
        data=str(trained / "data"),
        output=str(tmp_path),
        pool_size=100,
        initial_size=30,
        validation_size=30,
        epochs=1,
    )
    engine = Engine(cfg)
    with TestClient(create_app(engine)) as client:
        wait_for(client, lambda s: s.get("model") is not None)
        bad = client.post("/api/experiments", json={"pool_size": 40, "initial_size": 50})
        assert bad.status_code == 422
        response = client.post(
            "/api/experiments",
            json={"pool_size": 100, "initial_size": 30, "validation_size": 30, "epochs": 1},
        )
        assert response.status_code == 202
        current = wait_for(client, lambda s: s.get("model") is not None)
        assert current["experiment"] == response.json()["output"]
        pointer = read_json(tmp_path / "active.json")
        assert pointer["output"] == current["experiment"]
        assert client.get("/api/events?recent=true").status_code == 200
    assert not engine.thread.is_alive()
