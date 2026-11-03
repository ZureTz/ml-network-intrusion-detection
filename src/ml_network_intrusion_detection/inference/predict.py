"""可信模型加载及事务式分批预测。"""

import importlib.metadata
import os
import tempfile
import time
import warnings
from pathlib import Path

import joblib
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


def load_model(directory: Path) -> tuple:
    """仅用于可信 joblib；先核对元数据版本、依赖和内容哈希。"""
    metadata = read_json(directory / "metadata.json")
    if metadata.get("schema_version") != 1 or metadata.get("features") != FEATURES:
        raise ValueError("模型元数据格式不兼容，请使用对应训练环境")
    if metadata.get("classes") != [0, 1] or metadata.get("threshold") != 0.5:
        raise ValueError("模型类别或阈值不兼容")
    for package in ("numpy", "scipy", "scikit-learn", "joblib"):
        if metadata["environment"]["versions"][package] != importlib.metadata.version(package):
            raise ValueError(f"{package} 版本不一致，请恢复模型的训练环境（uv sync --locked）")
    if sha256(directory / "pipeline.joblib") != metadata["pipeline_sha256"]:
        raise ValueError("模型文件校验和不一致")
    model = joblib.load(directory / "pipeline.joblib")
    if list(model.classes_) != [0, 1]:
        raise ValueError("模型类别顺序错误")
    return model, metadata


def unknown_categories(frame: pd.DataFrame, metadata: dict) -> dict[str, int]:
    return {
        name: int((~frame[name].isin(metadata["categories"][name])).sum()) for name in CATEGORICAL
    }


def predict(directory: Path, input_csv: Path, output: Path, batch_size: int = 10000) -> dict:
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
            model, metadata = load_model(directory)
            report["model_load_seconds"] = time.perf_counter() - started
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
        report["model_sha256"] = metadata["pipeline_sha256"]
        report["status"] = "completed"
        if any(report["unknown_categories"].values()):
            warnings.warn(f"未知类别计数: {report['unknown_categories']}", stacklevel=2)
        # link 原子发布且不覆盖并发创建的目标，临时文件与目标位于同一文件系统。
        os.link(temporary, output)
        write_json(report_path, report)
        return report
    finally:
        temporary.unlink(missing_ok=True)
