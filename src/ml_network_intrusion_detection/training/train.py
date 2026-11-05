"""仅使用训练/验证数据选择模型，测试集不参与训练。"""

import shutil
import warnings
from pathlib import Path

from threadpoolctl import threadpool_info, threadpool_limits

from ml_network_intrusion_detection.common import (
    DEFAULTS,
    ResourceMonitor,
    environment,
    read_json,
    run_directory,
    sha256,
    write_json,
)
from ml_network_intrusion_detection.data.schema import (
    CATEGORICAL,
    FEATURES,
    NUMERIC,
    read_frame,
    validate_features,
    validate_labels,
)
from ml_network_intrusion_detection.device import resolve_device
from ml_network_intrusion_detection.training.artifacts import save_pipeline
from ml_network_intrusion_detection.training.metrics import binary_metrics
from ml_network_intrusion_detection.training.models import build_pipeline, candidates


def selection_key(result: dict) -> tuple:
    metrics = result["validation"]
    return (-metrics["f1"], metrics["false_positive_rate"], -metrics["recall"], result["order"])


def train(config: dict, data_dir: Path, output: Path) -> None:
    """比较候选模型并保存获选 Pipeline；失败候选记入 JSON。"""
    config = DEFAULTS | config
    with run_directory(output, "train", config), threadpool_limits(limits=config["threads"]):
        actual_device = resolve_device(config["device"])
        print(f"Torch 训练设备: {actual_device}", flush=True)
        manifest = read_json(data_dir / "manifest.json")
        if manifest["seed"] != config["seed"]:
            raise ValueError("训练种子与数据划分种子不一致")
        data = {}
        for name in ("train", "validation"):
            path = data_dir / f"{name}.csv"
            if sha256(path) != manifest["splits"][name]["sha256"]:
                raise ValueError(f"{name} 文件校验和与清单不一致")
            frame = read_frame(path)
            x, _ = validate_features(frame)
            y = validate_labels(frame)
            if y.nunique() != 2:
                raise ValueError("训练和验证必须同时含两个类别")
            data[name] = (x, y)
        write_json(output / "data_manifest.json", manifest)
        write_json(output / "threadpools.json", threadpool_info())
        results = []
        for order, spec in enumerate(candidates(config["profile"])):
            print(f"训练 {order + 1}: {spec['name']}", flush=True)
            result = {"order": order, "spec": spec}
            model_dir = output / spec["name"]
            model_dir.mkdir()
            try:
                model = build_pipeline(
                    spec,
                    config["seed"],
                    config["threads"],
                    device=str(actual_device),
                    epochs=config["epochs"],
                    batch_size=config["train_batch_size"],
                    learning_rate=config["learning_rate"],
                    weight_decay=config["weight_decay"],
                )
                with ResourceMonitor() as monitor, warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    model.fit(*data["train"])
                result["training"] = monitor.result
                classifier = model.named_steps["classifier"]
                result["runtime"] = classifier.runtime_
                result["loss_history"] = classifier.loss_history_
                result["positive_weight"] = classifier.positive_weight_
                result["warnings"] = [str(w.message) for w in captured]
                scores = model.predict_proba(data["validation"][0])[:, 1]
                result["validation"] = binary_metrics(data["validation"][1], scores)
                artifact = save_pipeline(model, model_dir)
                encoder = model.named_steps["preprocess"].named_transformers_["categorical"]
                metadata = {
                    "schema_version": 2,
                    **artifact,
                    "runtime": classifier.runtime_,
                    "features": FEATURES,
                    "numeric": NUMERIC,
                    "categorical": CATEGORICAL,
                    "dtypes": {
                        name: "float64" if name in NUMERIC else "string" for name in FEATURES
                    },
                    "categories": {
                        name: values.tolist()
                        for name, values in zip(CATEGORICAL, encoder.categories_)
                    },
                    "classes": [0, 1],
                    "label_mapping": {"0": "normal", "1": "attack"},
                    "score": "P(label=1 | features); uncalibrated model probability",
                    "threshold": 0.5,
                    "config": config,
                    "spec": spec,
                    "environment": environment(),
                    "data_manifest_sha256": sha256(output / "data_manifest.json"),
                }
                write_json(model_dir / "metadata.json", metadata)
                result["status"] = "completed"
            except Exception as error:
                result.update(status="failed", error=f"{type(error).__name__}: {error}")
            write_json(model_dir / "result.json", result)
            results.append(result)
            write_json(output / "candidates.json", results)
        successful = [r for r in results if r["status"] == "completed"]
        if not successful:
            raise ValueError("全部候选失败；查看 candidates.json")
        winners = {}
        for family in ("dummy", "linear", "mlp"):
            choices = [r for r in successful if r["spec"]["family"] == family]
            if not choices:
                raise ValueError(f"模型族 {family} 全部失败，正式比较未完成")
            winners[family] = min(choices, key=selection_key)["spec"]["name"]
        best = min(successful, key=selection_key)["spec"]["name"]
        shutil.copytree(output / best, output / "best")
        write_json(
            output / "selection.json",
            {
                "best": best,
                "family_winners": winners,
                "criterion": "validation attack F1 desc, FPR asc, recall desc, candidate order asc",
                "threshold": 0.5,
                "refit_train_validation": False,
            },
        )
