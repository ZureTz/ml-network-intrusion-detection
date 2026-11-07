"""配置、运行记录与资源测量。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

import psutil
import yaml

DEFAULTS = {
    "seed": 42,
    "threads": 4,
    "batch_size": 10000,
    "device": "auto",
    "epochs": 20,
    "train_batch_size": 512,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "sample_size": None,
    "profile": "full",
    "conflict_policy": "error",
    "train_csv": "../data/raw/UNSW_NB15_training-set.csv",
    "test_csv": "../data/raw/UNSW_NB15_testing-set.csv",
}


def load_config(path: Path) -> dict[str, Any]:
    """读取安全 YAML；未知字段及非法参数抛出 ValueError。路径相对配置文件。"""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise ValueError("配置必须为映射，且只能包含文档列出的字段")
    config = DEFAULTS | raw
    for key in ("seed", "threads", "batch_size", "epochs", "train_batch_size"):
        value = config[key]
        minimum = 0 if key == "seed" else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"{key} 必须为 >= {minimum} 的整数")
    if config["seed"] > 2**32 - 1:
        raise ValueError("seed 必须小于 2**32")
    from ml_network_intrusion_detection.device import validate_device_name

    validate_device_name(config["device"])
    for key in ("learning_rate", "weight_decay"):
        value = config[key]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
            or (key == "learning_rate" and value == 0)
        ):
            raise ValueError(f"{key} 必须为有限{'正' if key == 'learning_rate' else '非负'}数")
    size = config["sample_size"]
    if size is not None and (type(size) is not int or size < 20):
        raise ValueError("sample_size 必须为空或 >= 20 的整数")
    if config["profile"] not in ("smoke", "full", "tune", "transformer_smoke", "simulation"):
        raise ValueError("profile 必须为 smoke、full、tune 、transformer_smoke 或 simulation")
    if config["conflict_policy"] not in ("error", "quarantine_development"):
        raise ValueError("conflict_policy 必须为 error 或 quarantine_development")
    config["threads"] = min(config["threads"], os.cpu_count() or 1)
    for key in ("train_csv", "test_csv"):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"{key} 必须为非空路径")
        config[key] = str((path.resolve().parent / config[key]).resolve())
    return config


def write_json(path: Path, value: Any) -> None:
    """写入严格 JSON，不允许 NaN/Infinity。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    packages = ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "psutil", "pyyaml", "torch")
    try:
        memory = psutil.virtual_memory().total
    except (OSError, PermissionError):
        memory = None
    root = Path(__file__).resolve().parents[2]
    code_files = sorted((root / "src").rglob("*.py"))
    source_hashes = {str(path.relative_to(root)): sha256(path) for path in code_files}
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "memory_bytes": memory,
        "versions": {name: importlib.metadata.version(name) for name in packages},
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "source_sha256": source_hashes,
        "lock_sha256": sha256(root / "uv.lock") if (root / "uv.lock").exists() else None,
    }


@contextmanager
def run_directory(path: Path, kind: str, config: dict | None = None) -> Iterator[Path]:
    """创建唯一运行目录；无论成功或失败均记录状态，拒绝覆盖。"""
    path.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record = {
        "run_id": path.name,
        "kind": kind,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "config": config,
        "environment": environment(),
    }
    write_json(path / "run.json", record)
    try:
        yield path
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    else:
        record["status"] = "completed"
    finally:
        record["elapsed_seconds"] = time.perf_counter() - started
        write_json(path / "run.json", record)


class ResourceMonitor:
    """每 50 ms 采样当前进程及子进程 RSS；共享页可能重复计数。"""

    def __enter__(self) -> ResourceMonitor:
        self.peak = 0
        self.unavailable = False
        self.stop = threading.Event()
        self.started = time.perf_counter()
        self.worker = threading.Thread(target=self._sample, daemon=True)
        self.worker.start()
        return self

    def _sample(self) -> None:
        while True:
            try:
                process = psutil.Process()
                total = process.memory_info().rss
                try:
                    children = process.children(recursive=True)
                except (psutil.Error, PermissionError, OSError):
                    children = []
                    self.unavailable = True
                for child in children:
                    try:
                        total += child.memory_info().rss
                    except psutil.Error:
                        pass
                self.peak = max(self.peak, total)
            except (psutil.Error, PermissionError, OSError):
                self.unavailable = True
            if self.stop.wait(0.05):
                return

    def __exit__(self, *args: Any) -> None:
        self.stop.set()
        self.worker.join()
        self.result = {
            "seconds": time.perf_counter() - self.started,
            "peak_rss_bytes": self.peak or None,
            "memory_sampling_incomplete": self.unavailable,
            "memory_method": (
                "process RSS sampled every 50ms; child enumeration unavailable"
                if self.unavailable
                else "process + children RSS sampled every 50ms; shared pages may repeat"
            ),
        }
