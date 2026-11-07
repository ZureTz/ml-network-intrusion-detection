"""持久化的先预测后反馈模拟；单个所有者线程管理 SQLite 与训练子进程。"""

from __future__ import annotations

import json
import logging
import queue
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ml_network_intrusion_detection.common import read_json, sha256, write_json
from ml_network_intrusion_detection.data.schema import fingerprints, read_frame, validate_features
from ml_network_intrusion_detection.inference.predict import load_model

LOG = logging.getLogger("simulation")
DEFAULTS = dict(
    data="../data/processed/torch-full-20261105",
    output="../outputs/simulation",
    seed=42,
    pool_size=10000,
    initial_size=2000,
    validation_size=2000,
    rate=20,
    label_delay=10,
    training_interval=60,
    min_new_samples=500,
    epochs=5,
    threads=2,
    scenario="original",
    port=8000,
)


def config_from(path: Path) -> dict:
    """安全加载配置，路径相对于 YAML；配置错误抛出 ValueError。"""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise ValueError("未知模拟配置字段")
    cfg = DEFAULTS | raw
    for key in ("data", "output"):
        if not isinstance(cfg[key], str) or not cfg[key]:
            raise ValueError(f"{key} 必须是非空路径")
        cfg[key] = str((path.resolve().parent / cfg[key]).resolve())
    for key in set(DEFAULTS) - {"data", "output", "scenario"}:
        if type(cfg[key]) is not int or cfg[key] < (0 if key == "seed" else 1):
            raise ValueError(f"{key} 必须是有效整数")
    if cfg["initial_size"] >= cfg["pool_size"] or cfg["rate"] > 200 or cfg["port"] > 65535:
        raise ValueError("初始样本必须少于总样本，rate <= 200，port <= 65535")
    if cfg["scenario"] not in ("original", "attack", "burst"):
        raise ValueError("未知场景")
    return cfg


def group_sample(frame: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """按标签分层随机选择完整特征组；样本数近似目标值。"""
    groups = fingerprints(frame)
    representatives = frame.assign(group=groups).drop_duplicates("group")
    rng = np.random.default_rng(seed)
    chosen = []
    for label, part in representatives.groupby("label"):
        keys = rng.permutation(part.group.to_numpy())
        target = max(1, round(size * (frame.label == label).mean()))
        counts = groups.value_counts()
        total = 0
        for key in keys:
            chosen.append(key)
            total += int(counts[key])
            if total >= target:
                break
    return frame.loc[groups.isin(chosen)].copy()


def initialize(cfg: dict, root: Path) -> None:
    """构建独立实验数据；仅访问训练和验证，不读取官方测试文件。"""
    source = Path(cfg["data"])
    manifest = read_json(source / "manifest.json")
    frames = {}
    for name in ("train", "validation"):
        path = source / f"{name}.csv"
        if sha256(path) != manifest["splits"][name]["sha256"]:
            raise ValueError(f"{name} 校验和不匹配")
        frames[name] = read_frame(path)
    if set(fingerprints(frames["train"])) & set(fingerprints(frames["validation"])):
        raise ValueError("训练与验证存在重复组")
    pool = group_sample(frames["train"], cfg["pool_size"], cfg["seed"])
    initial = group_sample(pool, cfg["initial_size"], cfg["seed"] + 1)
    pool["initial"] = fingerprints(pool).isin(set(fingerprints(initial)))
    pool = pool.reset_index(drop=True)
    if pool.initial.all():
        raise ValueError("没有剩余到达样本，请增加 pool_size 或数据规模")
    validation = group_sample(frames["validation"], cfg["validation_size"], cfg["seed"])
    for frame in (initial, validation):
        if frame.label.nunique() != 2:
            raise ValueError("初始训练与验证必须同时包含两个类别")
    pool.to_csv(root / "pool.csv", index=False)
    validation.to_csv(root / "validation.csv", index=False)
    write_json(
        root / "experiment.json",
        {
            "config": cfg,
            "source_manifest": manifest,
            "hashes": {n: sha256(root / n) for n in ("pool.csv", "validation.csv")},
        },
    )


class Engine:
    """后台引擎；API 通过命令队列写入，通过受锁保护的快照读取。"""

    def __init__(self, cfg: dict):
        self.cfg, self.root = cfg, Path(cfg["output"])
        self.stop = threading.Event()
        self.commands: queue.Queue = queue.Queue(maxsize=20)
        self.lock = threading.Lock()
        self.view = {"phase": "initializing", "message": "准备独立实验数据"}
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.worker = None
        self.log_file = None

    def snapshot(self) -> dict:
        with self.lock:
            return json.loads(json.dumps(self.view))

    def publish(self, **values) -> None:
        with self.lock:
            self.view.update(values)

    def start_training(self, db, state, pool) -> None:
        ids = [r[0] for r in db.execute("select id from samples where revealed=1")]
        run = self.root / "training" / f"{time.time_ns()}"
        data = run / "data"
        data.mkdir(parents=True)
        pool.loc[ids].drop(columns="initial").to_csv(data / "train.csv", index=False)
        shutil.copyfile(self.root / "validation.csv", data / "validation.csv")
        write_json(
            data / "manifest.json",
            {
                "seed": self.cfg["seed"],
                "source_ids": ids,
                "feedback_cutoff_clock": state["clock"],
                "splits": {
                    n: {"sha256": sha256(data / f"{n}.csv")} for n in ("train", "validation")
                },
            },
        )
        training_cfg = dict(
            seed=self.cfg["seed"],
            threads=self.cfg["threads"],
            device="cpu",
            epochs=self.cfg["epochs"],
            profile="simulation",
            train_batch_size=256,
        )
        (run / "config.yaml").write_text(yaml.safe_dump(training_cfg))
        cmd = [sys.executable, "-m"]
        if state["model"]:
            cmd += ["ml_network_intrusion_detection.training.refresh", "--current", state["model"]]
        else:
            cmd += ["ml_network_intrusion_detection", "train"]
        cmd += [
            "--config",
            str(run / "config.yaml"),
            "--data",
            str(data),
            "--output",
            str(run / "model"),
        ]
        self.log_file = (run / "train.log").open("w")
        self.worker = subprocess.Popen(cmd, stdout=self.log_file, stderr=subprocess.STDOUT)
        state["job"] = {
            "path": str(run),
            "count": len(ids),
            "started": time.time(),
            "status": "running",
        }
        state["last_check"] = state["clock"]
        LOG.info("训练启动：%s 条已确认样本，%s", len(ids), run.name)

    def run(self) -> None:
        import fcntl

        self.root.mkdir(parents=True, exist_ok=True)
        lockfile = (self.root / "engine.lock").open("w")
        db = None
        try:
            fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not (self.root / "experiment.json").exists():
                initialize(self.cfg, self.root)
            experiment = read_json(self.root / "experiment.json")
            immutable = set(self.cfg) - {"port"}
            if any(experiment["config"][k] != self.cfg[k] for k in immutable):
                raise ValueError("实验配置改变，请指定新的 output；运行场景可在页面调整")
            for name, digest in experiment["hashes"].items():
                if sha256(self.root / name) != digest:
                    raise ValueError("实验数据已改变")
            pool = pd.read_csv(self.root / "pool.csv", keep_default_na=False, na_values=[""])
            features, _ = validate_features(pool)
            db = sqlite3.connect(self.root / "state.sqlite")
            db.executescript("""PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, value TEXT);
                CREATE TABLE IF NOT EXISTS samples(
                    id INTEGER PRIMARY KEY, sent INTEGER, revealed INTEGER);
                CREATE TABLE IF NOT EXISTS events(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, source INTEGER,
                    due REAL, score REAL, prediction INTEGER, truth INTEGER,
                    model TEXT, latency REAL);
                CREATE TABLE IF NOT EXISTS jobs(value TEXT);
                CREATE INDEX IF NOT EXISTS pending_labels ON events(truth,due);""")
            saved = db.execute("select value from state where id=1").fetchone()
            state = (
                json.loads(saved[0])
                if saved
                else dict(
                    model=None,
                    clock=0,
                    last_check=0,
                    trained=0,
                    job=None,
                    paused=False,
                    scenario=self.cfg["scenario"],
                    rate=self.cfg["rate"],
                    rng=np.random.default_rng(self.cfg["seed"]).bit_generator.state,
                    processed=0,
                    confusion=[0, 0, 0, 0],
                )
            )
            if not saved:
                db.executemany(
                    "insert into samples values(?,?,?)",
                    [(i, int(row.initial), int(row.initial)) for i, row in pool.iterrows()],
                )
            if state["job"] and state["job"]["status"] == "running":
                state["job"]["status"] = "interrupted"
                db.execute("insert into jobs values(?)", (json.dumps(state["job"]),))
            rng = np.random.default_rng()
            rng.bit_generator.state = state["rng"]
            model = load_model(Path(state["model"]), "cpu", 32)[0] if state["model"] else None
            previous, credit = time.monotonic(), 0.0
            retry_at = 0.0
            last_progress = 0.0
            logged_lines = 0
            while not self.stop.wait(0.05):
                now = time.monotonic()
                elapsed, previous = min(now - previous, 1), now
                force = False
                while not self.commands.empty():
                    command = self.commands.get_nowait()
                    action = command["action"]
                    if action in ("pause", "resume"):
                        state["paused"] = action == "pause"
                    elif action == "train":
                        force = True
                    elif action == "settings":
                        state.update({k: command[k] for k in ("rate", "scenario") if k in command})
                    LOG.info("控制操作：%s", command)
                if self.worker is not None:
                    if time.time() - state["job"]["started"] > 7200:
                        self.worker.terminate()
                        if time.time() - state["job"]["started"] > 7205:
                            self.worker.kill()
                    if self.worker.poll() is not None:
                        self.log_file.close()
                        job = state["job"]
                        try:
                            if self.worker.returncode:
                                raise RuntimeError("训练失败，请查看训练日志")
                            target = Path(job["path"]) / "model"
                            decision = (
                                read_json(target / "promotion.json")
                                if model
                                else {"accepted": True}
                            )
                            if decision["accepted"]:
                                candidate = load_model(target / "best", "cpu", 32)[0]
                                candidate.predict_proba(features.iloc[:1])
                                model = candidate
                                state["model"] = str(target / "best")
                            job.update(
                                status="accepted" if decision["accepted"] else "rejected",
                                decision=decision,
                            )
                            state["trained"] = job["count"]
                        except Exception as error:
                            job.update(status="failed", error=str(error))
                        LOG.info("训练完成：%s", job["status"])
                        db.execute("insert into jobs values(?)", (json.dumps(job),))
                        self.worker = None
                        retry_at = time.monotonic() + 30 if job["status"] == "failed" else 0
                if model is not None and not state["paused"]:
                    state["clock"] += elapsed
                    rate = state["rate"] * (
                        4 if state["scenario"] == "burst" and int(state["clock"]) % 30 < 5 else 1
                    )
                    offered = credit + rate * elapsed
                    state["dropped"] = state.get("dropped", 0) + max(0, int(offered - 1000))
                    credit = min(offered, 1000)
                    count = min(32, int(credit))
                    credit -= count
                    available = np.array(
                        [r[0] for r in db.execute("select id from samples where sent=0")], dtype=int
                    )
                    if not len(available):
                        available = np.flatnonzero(~pool.initial.to_numpy())
                    if count:
                        count = min(count, len(available))
                        weights = None
                        if state["scenario"] == "attack":
                            labels = pool.loc[available, "label"].to_numpy()
                            if len(set(labels)) == 2:
                                weights = np.where(
                                    labels == 1, 0.8 / sum(labels == 1), 0.2 / sum(labels == 0)
                                )
                        ids = rng.choice(available, count, replace=False, p=weights)
                        started = time.monotonic()
                        scores = model.predict_proba(features.loc[ids])[:, 1]
                        latency = (time.monotonic() - started) * 1000
                        for i, score in zip(ids, scores):
                            db.execute(
                                "insert into events(source,due,score,prediction,model,latency) "
                                "values(?,?,?,?,?,?)",
                                (
                                    int(i),
                                    state["clock"] + self.cfg["label_delay"],
                                    float(score),
                                    int(score >= 0.5),
                                    state["model"],
                                    latency,
                                ),
                            )
                            db.execute("update samples set sent=1 where id=?", (int(i),))
                        state["processed"] += count
                    for event, source, pred in db.execute(
                        "select id,source,prediction from events where truth is null and due<=?",
                        (state["clock"],),
                    ).fetchall():
                        truth = int(pool.loc[source, "label"])
                        db.execute("update events set truth=? where id=?", (truth, event))
                        db.execute("update samples set revealed=1 where id=?", (source,))
                        state["confusion"][truth * 2 + pred] += 1
                known = db.execute("select count(*) from samples where revealed=1").fetchone()[0]
                unseen = db.execute("select count(*) from samples where sent=0").fetchone()[0]
                if self.worker is None and not state["paused"] and now >= retry_at:
                    due = state["clock"] - state["last_check"] >= self.cfg["training_interval"]
                    if model is None or (
                        (force or due)
                        and known > state["trained"]
                        and (force or known - state["trained"] >= self.cfg["min_new_samples"])
                    ):
                        self.start_training(db, state, pool)
                        logged_lines = 0
                    elif due:
                        state["last_check"] = state["clock"]
                state["rng"] = rng.bit_generator.state
                db.execute("insert or replace into state values(1,?)", (json.dumps(state),))
                db.execute(
                    "delete from events where truth is not null and id < "
                    "(select coalesce(max(id),0)-100000 from events)"
                )
                db.commit()
                logs = ""
                if state["job"]:
                    logs = (Path(state["job"]["path"]) / "train.log").read_text()[-12000:]
                lines = logs.splitlines()
                for line in lines[logged_lines:]:
                    LOG.info("训练 · %s", line)
                logged_lines = len(lines)
                if now - last_progress >= 5:
                    LOG.info(
                        "模拟检测 %s 条 · 已确认 %s · 未见 %s", state["processed"], known, unseen
                    )
                    last_progress = now
                tn, fp, fn, tp = state["confusion"]
                total = tn + fp + fn + tp
                metrics = {
                    "accuracy": (tn + tp) / total if total else None,
                    "precision": tp / (tp + fp) if tp + fp else None,
                    "recall": tp / (tp + fn) if tp + fn else None,
                    "confusion_matrix": [[tn, fp], [fn, tp]],
                    "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                    "false_positive_rate": fp / (fp + tn) if fp + tn else None,
                }
                self.publish(
                    experiment=str(self.root),
                    phase="paused" if state["paused"] else "running" if model else "initializing",
                    message="新数据已耗尽，继续检测；重复来源不再次入池"
                    if not unseen
                    else "数据持续到达，标签延迟确认",
                    queued=int(credit),
                    dropped=state.get("dropped", 0),
                    processed=state["processed"],
                    known=known,
                    unseen=unseen,
                    new_samples=known - state["trained"],
                    pending=db.execute(
                        "select count(*) from events where truth is null"
                    ).fetchone()[0],
                    model=state["model"],
                    job=state["job"],
                    training=self.worker is not None,
                    logs=logs,
                    scenario=state["scenario"],
                    rate=state["rate"],
                    clock=state["clock"],
                    metrics=metrics,
                    next_training=max(
                        0, self.cfg["training_interval"] - (state["clock"] - state["last_check"])
                    ),
                    jobs=[
                        json.loads(r[0])
                        for r in db.execute("select value from jobs order by rowid desc limit 30")
                    ],
                )
        except Exception as error:
            LOG.exception("模拟引擎停止")
            self.publish(phase="failed", message=str(error))
        finally:
            if self.worker is not None and self.worker.poll() is None:
                self.worker.terminate()
                try:
                    self.worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.worker.kill()
                    self.worker.wait()
            if self.log_file:
                self.log_file.close()
            if db:
                db.close()
            lockfile.close()

    def events(
        self, after: int = 0, attacks: bool = False, model: str | None = None, recent: bool = False
    ) -> list:
        path = self.root / "state.sqlite"
        if not path.exists():
            return []
        with sqlite3.connect(path) as db:
            db.row_factory = sqlite3.Row
            query = "select * from events where id>?"
            args: list = [after]
            if attacks:
                query += " and prediction=1"
            if model:
                query += " and model=?"
                args.append(model)
            try:
                return [
                    dict(r)
                    for r in db.execute(
                        query
                        + (" order by id desc limit 100" if recent else " order by id limit 100"),
                        args,
                    )
                ]
            except sqlite3.OperationalError:
                return []
