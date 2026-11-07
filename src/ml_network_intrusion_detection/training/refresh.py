"""后台重训：仅通过固定验证集决定是否替换当前模型。"""

import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits

from ml_network_intrusion_detection.common import load_config, read_json, sha256, write_json
from ml_network_intrusion_detection.data.schema import (
    read_frame,
    validate_features,
    validate_labels,
)
from ml_network_intrusion_detection.inference.predict import load_model
from ml_network_intrusion_detection.training.metrics import binary_metrics
from ml_network_intrusion_detection.training.train import train


def should_promote(old: dict, new: dict) -> bool:
    """仅在两项指标有定义且 F1 严格改善、误报率不升高时接受候选。"""
    return (
        new["f1"] is not None
        and old["f1"] is not None
        and new["false_positive_rate"] is not None
        and old["false_positive_rate"] is not None
        and new["f1"] > old["f1"]
        and new["false_positive_rate"] <= old["false_positive_rate"]
    )


def refresh(config: dict, data: Path, current: Path, output: Path) -> dict:
    """重训并比较相同验证集；严格改善 F1 且误报率不增加才建议切换。"""
    train(config, data, output)
    expected = read_json(output / "data_manifest.json")["splits"]["validation"]["sha256"]
    if sha256(data / "validation.csv") != expected:
        raise ValueError("训练后验证集发生变化，拒绝模型切换")
    frame = read_frame(data / "validation.csv")
    features, _ = validate_features(frame)
    labels = validate_labels(frame)
    metrics = {}
    with threadpool_limits(limits=config["threads"]):
        for name, directory in (("current", current), ("candidate", output / "best")):
            model, metadata = load_model(directory, "cpu", 256)
            metrics[name] = binary_metrics(
                labels, model.predict_proba(features)[:, 1], metadata["threshold"]
            )
    decision = {
        "accepted": should_promote(metrics["current"], metrics["candidate"]),
        "criterion": "validation F1 strictly increases and false positive rate does not increase",
        "current_model": str(current.resolve()),
        "candidate_model": str((output / "best").resolve()),
        "validation_sha256": expected,
        "metrics": metrics,
    }
    write_json(output / "promotion.json", decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="后台训练与验证，不读取回放测试集")
    for name in ("config", "data", "current", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    refresh(load_config(args.config), args.data, args.current, args.output)


if __name__ == "__main__":
    main()
