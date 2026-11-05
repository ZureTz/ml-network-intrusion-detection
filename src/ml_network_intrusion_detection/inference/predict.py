"""可信模型加载及事务式分批预测。"""

import importlib.metadata
import os
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ml_network_intrusion_detection.common import (
    ResourceMonitor,
    read_json,
    sha256,
    write_json,
)
from ml_network_intrusion_detection.data.schema import (
    CATEGORICAL,
    FEATURES,
    check_header,
    validate_features,
)
from ml_network_intrusion_detection.device import device_info
from ml_network_intrusion_detection.training.artifacts import restore_pipeline


def load_model(directory: Path, device: str = "auto", batch_size: int = 10000) -> tuple:
    """加载可信 v2 模型并选择本机设备；先核对元数据、依赖和全部产物哈希。"""
    if batch_size < 1:
        raise ValueError("batch_size 必须为正整数")
    metadata = read_json(directory / "metadata.json")
    if (
        metadata.get("schema_version") != 2
        or metadata.get("format") != "torch-state-dict-v2"
        or metadata.get("features") != FEATURES
    ):
        raise ValueError("模型元数据格式不兼容；旧 sklearn 模型请使用原版本或重新训练 Torch 模型")
    if metadata.get("classes") != [0, 1] or metadata.get("threshold") != 0.5:
        raise ValueError("模型类别或阈值不兼容")
    for package in ("numpy", "scipy", "scikit-learn", "joblib", "torch"):
        trained_version = metadata["environment"]["versions"][package]
        current_version = importlib.metadata.version(package)
        # 相同 Torch 发布版的 CPU/CUDA/ROCm 构建可以互换 state_dict。
        if package == "torch":
            trained_version = trained_version.split("+")[0]
            current_version = current_version.split("+")[0]
        if trained_version != current_version:
            raise ValueError(f"{package} 版本不一致，请恢复模型的训练环境（uv sync --locked）")
    if set(metadata.get("artifacts", {})) != {"weights.pt", "preprocess.joblib"}:
        raise ValueError("模型产物清单不兼容")
    for name, expected in metadata["artifacts"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"模型文件校验和不一致: {name}")
    model = restore_pipeline(directory, metadata, device, batch_size)
    if list(model.classes_) != [0, 1]:
        raise ValueError("模型类别顺序错误")
    return model, metadata


def unknown_categories(frame: pd.DataFrame, metadata: dict) -> dict[str, int]:
    return {
        name: int((~frame[name].isin(metadata["categories"][name])).sum()) for name in CATEGORICAL
    }


def predict(
    directory: Path, input_csv: Path, output: Path, batch_size: int = 10000, device: str = "auto"
) -> dict:
    """分批无标签预测。错误不发布部分结果，已有目标或性能记录拒绝覆盖。"""
    if batch_size < 1:
        raise ValueError("batch_size 必须为正整数")
    report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or report_path.exists():
        raise FileExistsError(f"输出已存在: {output}")
    header = check_header(input_csv)
    extras = sorted(set(header) - set(FEATURES))
    if extras:
        warnings.warn(f"忽略额外列: {extras}", stacklevel=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=".prediction-", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    report = {"batch_size": batch_size, "rows": 0, "extra_columns": extras}
    try:
        with ResourceMonitor() as monitor:
            started = time.perf_counter()
            model, metadata = load_model(directory, device, batch_size)
            report["model_load_seconds"] = time.perf_counter() - started
            report["runtime"] = device_info(model.named_steps["classifier"].device_)
            report["network_batch_size"] = model.named_steps["classifier"].network_batch_size()
            report["requested_device"] = device
            report["prediction_scope"] = (
                "CPU preprocessing + transfers + synchronized Torch inference; no warmup"
            )
            report["threads"] = metadata["config"]["threads"]
            report["prediction_seconds"] = 0.0
            report["unknown_categories"] = dict.fromkeys(CATEGORICAL, 0)
            report["missing"] = dict.fromkeys(FEATURES, 0)
            report["infinite"] = {}
            pd.DataFrame(columns=["row_index", "predicted_label", "attack_probability"]).to_csv(
                temporary, index=False
            )
            with threadpool_limits(limits=report["threads"]):
                reader = pd.read_csv(
                    input_csv, chunksize=batch_size, keep_default_na=False, na_values=[""]
                )
                with reader:
                    for raw in reader:
                        if raw.empty:
                            continue
                        x, audit = validate_features(raw)
                        for name, count in unknown_categories(x, metadata).items():
                            report["unknown_categories"][name] += count
                        for key in ("missing", "infinite"):
                            for name, count in audit[key].items():
                                report[key][name] = report[key].get(name, 0) + count
                        started = time.perf_counter()
                        scores = model.predict_proba(x)[:, 1]
                        report["prediction_seconds"] += time.perf_counter() - started
                        pd.DataFrame(
                            {
                                "row_index": np.arange(
                                    report["rows"] + 1, report["rows"] + len(x) + 1
                                ),
                                "predicted_label": (scores >= metadata["threshold"]).astype(int),
                                "attack_probability": scores,
                            }
                        ).to_csv(temporary, mode="a", header=False, index=False)
                        report["rows"] += len(x)
        report["performance"] = monitor.result
        report["io_validation_seconds"] = (
            monitor.result["seconds"] - report["prediction_seconds"] - report["model_load_seconds"]
        )
        report["score"] = metadata["score"]
        report["model_artifacts"] = metadata["artifacts"]
        report["model_sha256"] = sha256(directory / "metadata.json")
        report["rows_per_second"] = (
            report["rows"] / report["prediction_seconds"] if report["rows"] else None
        )
        report["status"] = "completed"
        if any(report["unknown_categories"].values()):
            warnings.warn(f"未知类别计数: {report['unknown_categories']}", stacklevel=2)
        # link 原子发布且不覆盖并发创建的目标，临时文件与目标位于同一文件系统。
        os.link(temporary, output)
        write_json(report_path, report)
        return report
    finally:
        temporary.unlink(missing_ok=True)
