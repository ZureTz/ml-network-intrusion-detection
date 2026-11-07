"""仅监听本机的模拟控制台。"""

import logging
import queue
import threading
import time
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ml_network_intrusion_detection.common import read_json
from ml_network_intrusion_detection.inference.service import atomic_json
from ml_network_intrusion_detection.simulation.engine import Engine, config_from


class Control(BaseModel):
    action: Literal["pause", "resume", "train", "settings"]
    rate: int | None = Field(default=None, ge=1, le=200)
    scenario: Literal["original", "attack", "burst"] | None = None


class Experiment(BaseModel):
    pool_size: int = Field(default=10000, ge=40, le=100000)
    initial_size: int = Field(default=2000, ge=20)
    validation_size: int = Field(default=2000, ge=20)
    epochs: int = Field(default=5, ge=1, le=100)
    label_delay: int = Field(default=10, ge=1, le=3600)
    training_interval: int = Field(default=60, ge=1, le=86400)
    min_new_samples: int = Field(default=500, ge=1)


def create_app(engine: Engine) -> FastAPI:
    """提供状态快照、游标事件、控制指令与实验摘要。"""

    creation_lock = threading.Lock()
    launch_root = Path(engine.cfg.pop("_launch_root", str(engine.root)))

    @asynccontextmanager
    async def lifespan(app):
        engine.thread.start()
        yield
        engine.stop.set()
        engine.thread.join(timeout=15)

    app = FastAPI(title="NB15 持续学习模拟", lifespan=lifespan)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"]
    )

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and origin != f"http://{request.headers.get('host')}":
            from fastapi.responses import JSONResponse

            return JSONResponse({"detail": "仅允许同源请求"}, status_code=403)
        return await call_next(request)

    @app.get("/")
    def dashboard():
        return FileResponse(Path(__file__).parent / "static/index.html")

    @app.get("/api/status")
    def status():
        return engine.snapshot()

    @app.get("/api/events")
    def events(
        after: int = Query(0, ge=0),
        attacks: bool = False,
        model: str | None = None,
        recent: bool = False,
    ):
        return engine.events(after, attacks, model, recent)

    @app.post("/api/control", status_code=202)
    def control(command: Control):
        if engine.snapshot()["phase"] == "failed":
            raise HTTPException(409, "引擎已失败，请修复日志中的问题并重启")
        state = engine.snapshot()
        if command.action == "train":
            if state.get("training") or state["phase"] in ("paused", "initializing"):
                raise HTTPException(409, "请先恢复数据流并等待当前训练完成")
            if state.get("new_samples", 0) <= 0:
                raise HTTPException(409, "没有新增已确认样本，跳过重复训练")
        try:
            engine.commands.put_nowait(command.model_dump(exclude_none=True))
        except queue.Full:
            raise HTTPException(429, "控制队列已满，请稍后重试") from None
        return {"accepted": True}

    @app.post("/api/experiments", status_code=202)
    def new_experiment(options: Experiment):
        nonlocal engine
        if options.initial_size >= options.pool_size:
            raise HTTPException(422, "初始样本必须少于总样本")
        if not creation_lock.acquire(blocking=False):
            raise HTTPException(409, "正在切换实验")
        try:
            engine.stop.set()
            engine.thread.join(timeout=15)
            if engine.thread.is_alive():
                raise HTTPException(409, "旧实验尚未停止")
            cfg = engine.cfg | options.model_dump()
            cfg["output"] = str(launch_root / "experiments" / f"experiment-{time.time_ns()}")
            atomic_json(launch_root / "active.json", cfg)
            engine = Engine(cfg)
            engine.thread.start()
            return {"output": cfg["output"]}
        finally:
            creation_lock.release()

    @app.get("/api/export")
    def export():
        return {
            "config": engine.cfg,
            "summary": engine.snapshot(),
            "notice": "人工时序模拟；实时指标含重复事件，不是独立测试效果",
        }

    return app


def serve(path: Path) -> None:
    cfg = config_from(path)
    root = Path(cfg["output"])
    root.mkdir(parents=True, exist_ok=True)
    if (root / "active.json").exists():
        port = cfg["port"]
        cfg = read_json(root / "active.json")
        cfg["port"] = port
    cfg["_launch_root"] = str(root)
    logger = logging.getLogger("simulation")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handlers = [
        logging.StreamHandler(),
        RotatingFileHandler(
            root / "service.log", maxBytes=10_000_000, backupCount=5, encoding="utf-8"
        ),
    ]
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.info("Dashboard: http://127.0.0.1:%s · Ctrl+C 保存并退出", cfg["port"])
    try:
        uvicorn.run(create_app(Engine(cfg)), host="127.0.0.1", port=cfg["port"], access_log=False)
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
