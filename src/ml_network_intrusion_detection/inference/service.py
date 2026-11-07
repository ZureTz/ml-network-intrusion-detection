"""单机 NB15 常驻回放服务；训练子进程与推理解耦，不进行在线自标注。"""

from __future__ import annotations

import csv
import fcntl
import json
import logging
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from ml_network_intrusion_detection.common import read_json, sha256, write_json
from ml_network_intrusion_detection.data.schema import FEATURES, check_header, validate_features
from ml_network_intrusion_detection.inference.predict import load_model, unknown_categories

DEFAULTS = {
    "model": None,
    "data": None,
    "training_config": None,
    "output": "../outputs/service",
    "rate": 100,
    "batch_size": 32,
    "max_wait_ms": 100,
    "queue_size": 1000,
    "training_interval_seconds": 86400,
    "training_timeout_seconds": 7200,
    "log_max_bytes": 10_000_000,
    "log_backups": 5,
}


def load_service_config(path: Path) -> dict:
    """读取服务配置；所有路径相对于 YAML，拒绝未知字段和非正参数。"""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise ValueError("服务配置字段错误")
    config = DEFAULTS | raw
    for key in ("model", "data", "training_config", "output"):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"{key} 必须为非空路径")
        config[key] = str((path.resolve().parent / config[key]).resolve())
    for key in set(DEFAULTS) - {"model", "data", "training_config", "output"}:
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} 必须为有限正数")
        if key in ("batch_size", "queue_size", "log_max_bytes", "log_backups"):
            if type(value) is not int:
                raise ValueError(f"{key} 必须为正整数")
    return config


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    write_json(temporary, value)
    os.replace(temporary, path)


def make_logger(path: Path, config: dict) -> logging.Logger:
    logger = logging.Logger(str(path), level=logging.INFO)
    handler = RotatingFileHandler(
        path, maxBytes=config["log_max_bytes"], backupCount=config["log_backups"], encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def log_event(logger: logging.Logger, **fields: object) -> None:
    logger.info(json.dumps({"time": datetime.now(UTC).isoformat(), **fields}, ensure_ascii=False))


class Replay(threading.Thread):
    """按人工速率循环回放；只向推理队列发送白名单特征，不发送标签。"""

    def __init__(self, path: Path, rate: float, records: queue.Queue, stop: threading.Event):
        super().__init__(daemon=True)
        self.path, self.rate, self.records, self.stop = path, rate, records, stop
        self.sent = self.dropped = self.cycle = 0
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            while not self.stop.is_set():
                self.cycle += 1
                rows = 0
                due = time.monotonic()
                with self.path.open(encoding="utf-8-sig", newline="") as handle:
                    for index, raw in enumerate(csv.DictReader(handle), start=1):
                        if self.stop.wait(max(0, due - time.monotonic())):
                            return
                        due = max(due, time.monotonic()) + 1 / self.rate
                        rows += 1
                        record = {
                            "cycle": self.cycle,
                            "row_index": index,
                            "arrived_at": time.time(),
                            "arrival_clock": time.monotonic(),
                            "features": {
                                key: raw[key] if raw[key] != "" else None for key in FEATURES
                            },
                        }
                        try:
                            self.records.put_nowait(record)
                            self.sent += 1
                        except queue.Full:
                            self.dropped += 1
                if not rows:
                    raise ValueError("回放文件为空")
        except Exception as error:
            self.error = error
            self.stop.set()


def run_service(config: dict, stop: threading.Event | None = None) -> None:
    """持续回放和定时训练，SIGTERM/SIGINT 可停止；同一输出目录只允许一个实例。"""
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "service.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("同一输出目录已有服务运行") from error
        _run_service(config, stop or threading.Event())


def _run_service(config: dict, stop: threading.Event) -> None:
    output, data = Path(config["output"]), Path(config["data"])
    session = uuid.uuid4().hex
    state_path = output / "state.json"
    identity = {key: config[key] for key in ("model", "data", "training_config")}
    state = read_json(state_path) if state_path.exists() else {}
    if state and state["identity"] != identity:
        raise ValueError("服务数据或初始模型配置改变，请使用新的 output 目录")
    state = state or {
        "identity": identity,
        "model": config["model"],
        "next_training_at": time.time() + config["training_interval_seconds"],
    }
    events = make_logger(output / "events.jsonl", config)
    predictions = make_logger(output / "predictions.jsonl", config)
    worker = None
    training_log = None
    replay = None
    old_handlers = {}
    processed = 0
    training_output = None
    worker_started = 0.0
    last_status = 0.0
    failure = None
    try:
        model, metadata = load_model(Path(state["model"]), "cpu", config["batch_size"])
        model_hash = sha256(Path(state["model"]) / "metadata.json")
        manifest = read_json(data / "manifest.json")
        replay_path = data / "test.csv"
        if sha256(replay_path) != manifest["splits"]["test"]["sha256"]:
            raise ValueError("回放测试集与划分清单不一致")
        check_header(replay_path)
        records: queue.Queue = queue.Queue(maxsize=config["queue_size"])
        replay = Replay(replay_path, config["rate"], records, stop)
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                old_handlers[signum] = signal.signal(signum, lambda *_: stop.set())
        atomic_json(state_path, state)
        replay.start()
        log_event(events, event="started", session=session, config=config)
        with threadpool_limits(limits=metadata["config"]["threads"]):
            while not stop.is_set():
                now = time.time()
                if worker is not None and (
                    worker.poll() is not None
                    or time.monotonic() - worker_started > config["training_timeout_seconds"]
                ):
                    if worker.poll() is None:
                        worker.terminate()
                        try:
                            worker.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            worker.kill()
                            worker.wait()
                    training_log.close()
                    try:
                        if worker.returncode != 0:
                            raise RuntimeError(f"训练子进程退出码: {worker.returncode}")
                        decision = read_json(training_output / "promotion.json")
                        if decision["accepted"]:
                            candidate = training_output / "best"
                            new_model, new_metadata = load_model(
                                candidate, "cpu", config["batch_size"]
                            )
                            new_model.predict_proba(
                                validate_features(
                                    pd.read_csv(
                                        data / "validation.csv",
                                        nrows=1,
                                        keep_default_na=False,
                                        na_values=[""],
                                    )
                                )[0]
                            )
                            new_state = state | {"model": str(candidate)}
                            new_hash = sha256(candidate / "metadata.json")
                            atomic_json(state_path, new_state)
                            model, metadata, state = new_model, new_metadata, new_state
                            model_hash = new_hash
                        log_event(
                            events,
                            event="training_finished",
                            output=str(training_output),
                            accepted=decision["accepted"],
                        )
                    except Exception as error:
                        log_event(
                            events,
                            event="training_failed",
                            error=str(error),
                            output=str(training_output),
                        )
                    worker = None
                if worker is None and now >= state["next_training_at"]:
                    training_output = (
                        output
                        / "training"
                        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8])
                    )
                    training_output.parent.mkdir(exist_ok=True)
                    state["next_training_at"] = now + config["training_interval_seconds"]
                    atomic_json(state_path, state)
                    training_log = training_output.with_suffix(".log").open("w")
                    try:
                        worker = subprocess.Popen(
                            [
                                sys.executable,
                                "-m",
                                "ml_network_intrusion_detection.training.refresh",
                                "--config",
                                config["training_config"],
                                "--data",
                                str(data),
                                "--current",
                                state["model"],
                                "--output",
                                str(training_output),
                            ],
                            stdout=training_log,
                            stderr=subprocess.STDOUT,
                        )
                        worker_started = time.monotonic()
                        log_event(events, event="training_started", output=str(training_output))
                    except OSError as error:
                        training_log.close()
                        log_event(events, event="training_failed", error=str(error))
                batch = []
                deadline = time.monotonic() + config["max_wait_ms"] / 1000
                while len(batch) < config["batch_size"] and not stop.is_set():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        batch.append(records.get(timeout=min(remaining, 0.1)))
                    except queue.Empty:
                        continue
                if batch:
                    features, audit = validate_features(
                        pd.DataFrame([r["features"] for r in batch])
                    )
                    unknown = unknown_categories(features, metadata)
                    scores = model.predict_proba(features)[:, 1]
                    finished = time.time()
                    clock = time.monotonic()
                    for record, score in zip(batch, scores):
                        log_event(
                            predictions,
                            session=session,
                            cycle=record["cycle"],
                            row_index=record["row_index"],
                            arrived_at=record["arrived_at"],
                            predicted_at=finished,
                            latency_ms=(clock - record["arrival_clock"]) * 1000,
                            predicted_label=int(score >= metadata["threshold"]),
                            attack_probability=float(score),
                            model_sha256=model_hash,
                        )
                    processed += len(batch)
                    if (
                        any(unknown.values())
                        or any(audit["missing"].values())
                        or any(audit["infinite"].values())
                    ):
                        log_event(
                            events, event="input_audit", unknown_categories=unknown, audit=audit
                        )
                if now - last_status >= 1:
                    atomic_json(
                        output / "status.json",
                        {
                            "status": "running",
                            "pid": os.getpid(),
                            "session": session,
                            "updated_at": now,
                            "processed": processed,
                            "dropped": replay.dropped,
                            "cycle": replay.cycle,
                            "queued": records.qsize(),
                            "model": state["model"],
                            "training": worker is not None,
                            "next_training_at": state["next_training_at"],
                        },
                    )
                    last_status = now
            if replay.error:
                raise replay.error
    except Exception as error:
        failure = str(error)
        log_event(events, event="failed", error=str(error))
        raise
    finally:
        stop.set()
        if replay is not None:
            replay.join(timeout=2)
        if worker is not None and worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
            log_event(events, event="training_interrupted", output=str(training_output))
        if training_log is not None:
            training_log.close()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        atomic_json(
            output / "status.json",
            {
                "status": "failed" if failure is not None else "stopped",
                "error": failure,
                "session": session,
                "processed": processed,
                "updated_at": time.time(),
                "model": state["model"],
                "dropped": replay.dropped if replay is not None else 0,
            },
        )
        log_event(events, event="stopped", session=session)
        for logger in (events, predictions):
            for handler in logger.handlers:
                handler.close()
