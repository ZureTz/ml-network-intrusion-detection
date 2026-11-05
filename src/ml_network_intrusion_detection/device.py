"""Torch 设备选择与异步执行同步；仅 auto 允许无加速器时回退 CPU。"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def validate_device_name(name: str) -> None:
    """校验配置语法，不初始化 GPU；非法值抛出 ValueError。"""
    if not isinstance(name, str) or not re.fullmatch(r"auto|cpu|mps|cuda(?::\d+)?", name):
        raise ValueError("device 必须为 auto、cpu、mps、cuda 或 cuda:N")


def resolve_device(requested: str = "auto") -> torch.device:
    """自动按 CUDA、MPS、CPU 选择；明确请求的设备不可用时直接报错。"""
    import torch

    validate_device_name(requested)
    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA 不可用；检查 Torch CUDA 构建和 NVIDIA 驱动，或使用 --device cpu")
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA 设备编号越界: {index}")
        device = torch.device("cuda", index)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Apple MPS 不可用；检查 macOS、硬件和 Torch 构建，或使用 --device cpu")
    return device


def synchronize(device: torch.device) -> None:
    """等待设备工作完成，防止 GPU 异步执行低估计时。"""
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def device_info(device: torch.device) -> dict:
    """记录实际设备及构建；不把进程 RSS 当成 GPU 显存。"""
    import torch

    return {
        "device": str(device),
        "name": torch.cuda.get_device_name(device) if device.type == "cuda" else device.type,
        "torch_version": str(torch.__version__),
        "cuda_build": torch.version.cuda,
        "hip_build": torch.version.hip,
        "dtype": "float32",
        "torch_threads": torch.get_num_threads(),
        "memory_note": "RSS measures host process memory, not accelerator peak allocation",
    }
