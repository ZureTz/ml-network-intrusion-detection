"""固定模型的独立评估与性能测量。"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ml_network_intrusion_detection.common import ResourceMonitor, run_directory, sha256, write_json
from ml_network_intrusion_detection.data.schema import (
    read_frame,
    validate_features,
    validate_labels,
)
from ml_network_intrusion_detection.device import device_info
from ml_network_intrusion_detection.inference.predict import load_model, unknown_categories
from ml_network_intrusion_detection.training.metrics import binary_metrics, save_figures


def evaluate(
    model_dir: Path, input_csv: Path, output: Path, batch_size: int = 10000, device: str = "auto"
) -> None:
    """标签用于指标计算；不拟合任何参数。预测预热一次、测量三次取中位数。"""
    if batch_size < 1:
        raise ValueError("batch_size 必须为正整数")
    with run_directory(output, "evaluate", {"batch_size": batch_size, "device": device}):
        with ResourceMonitor() as total:
            started = time.perf_counter()
            model, metadata = load_model(model_dir, device, batch_size)
            load_seconds = time.perf_counter() - started
            started = time.perf_counter()
            frame = read_frame(input_csv)
            x, audit = validate_features(frame)
            labels = validate_labels(frame).to_numpy()
            if not len(x):
                raise ValueError("评估数据为空")
            read_validate_seconds = time.perf_counter() - started
            with threadpool_limits(limits=metadata["config"]["threads"]):
                model.predict_proba(x.iloc[:batch_size])
                timings = []
                with ResourceMonitor() as prediction_resources:
                    for _ in range(3):
                        started = time.perf_counter()
                        scores = np.concatenate(
                            [
                                model.predict_proba(x.iloc[i : i + batch_size])[:, 1]
                                for i in range(0, len(x), batch_size)
                            ]
                        )
                        timings.append(time.perf_counter() - started)
            metrics = binary_metrics(labels, scores)
            write_json(output / "metrics.json", metrics)
            save_figures(labels, scores, output)
            predicted = (scores >= metadata["threshold"]).astype(int)
            errors = frame.loc[labels != predicted].copy()
            # 不复制全部特征，保留可追溯错误记录及类别。
            error_columns = [
                c for c in ("source_file", "source_row", "id", "attack_cat") if c in errors
            ]
            errors = errors[error_columns]
            errors["row_index"] = np.flatnonzero(labels != predicted) + 1
            errors["label"] = labels[labels != predicted]
            errors["prediction"] = predicted[labels != predicted]
            errors["attack_probability"] = scores[labels != predicted]
            errors.to_csv(output / "errors.csv", index=False)
            if "attack_cat" in frame:
                breakdown = (
                    pd.DataFrame(
                        {
                            "category": frame.attack_cat,
                            "correct": labels == predicted,
                        }
                    )
                    .groupby("category")
                    .correct.agg(["count", "mean"])
                )
                breakdown.rename(columns={"mean": "binary_accuracy"}).to_csv(
                    output / "category_breakdown.csv"
                )
        median = float(np.median(timings))
        write_json(
            output / "performance.json",
            {
                "model_load_seconds": load_seconds,
                "read_validate_seconds": read_validate_seconds,
                "prediction_seconds_trials": timings,
                "prediction_seconds_median": median,
                "rows_per_second": len(x) / median,
                "batch_size": batch_size,
                "network_batch_size": model.named_steps["classifier"].network_batch_size(),
                "threads": metadata["config"]["threads"],
                "warmup": "one batch, once",
                "prediction_resources": prediction_resources.result,
                "end_to_end": total.result,
                "input_sha256": sha256(input_csv),
                "model_sha256": sha256(model_dir / "metadata.json"),
                "model_artifacts": metadata["artifacts"],
                "runtime": device_info(model.named_steps["classifier"].device_),
                "prediction_scope": "CPU preprocessing + transfers + synchronized Torch inference",
                "audit": audit,
                "unknown_categories": unknown_categories(x, metadata),
                "spec": metadata["spec"],
            },
        )
