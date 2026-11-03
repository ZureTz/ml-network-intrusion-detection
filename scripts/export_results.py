"""导出报告所需的精简证据；不复制数据集或模型二进制。"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

from ml_network_intrusion_detection.common import read_json, sha256, write_json


def portable(value, root: Path):
    """公开材料中的项目绝对路径改为相对路径，其他本机库路径仅保留文件名。"""
    if isinstance(value, dict):
        return {key: portable(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item, root) for item in value]
    if isinstance(value, str) and value.startswith(str(root) + "/"):
        return value[len(str(root)) + 1 :]
    if isinstance(value, str) and value.startswith("/"):
        return Path(value).name
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--evaluations", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--environment", required=True, type=Path)
    parser.add_argument("--source", type=Path, default=Path("data/raw/source_manifest.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {}

    def export_json(source: Path, relative: str) -> None:
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json(target, portable(read_json(source), root))
        sources[relative] = {
            "source": portable(str(source.resolve()), root),
            "sha256": sha256(source),
        }

    for filename in ("run.json", "candidates.json", "selection.json", "threadpools.json"):
        export_json(args.training / filename, f"training/{filename}")
    for filename in ("audit.json", "manifest.json", "run.json"):
        export_json(args.data / filename, f"data/{filename}")
    for filename in ("run.json", "frozen_selection.json"):
        export_json(args.evaluations / filename, f"evaluation/{filename}")
    export_json(args.source, "source_manifest.json")
    export_json(args.environment, "environment.json")
    selection = read_json(args.training / "selection.json")
    candidates = read_json(args.training / "candidates.json")
    by_name = {candidate["spec"]["name"]: candidate for candidate in candidates}
    rows = []
    for family, name in selection["family_winners"].items():
        evaluation = args.evaluations / family
        if read_json(evaluation / "run.json")["status"] != "completed":
            raise ValueError(f"评估未完成: {family}")
        metrics = read_json(evaluation / "metrics.json")
        performance = read_json(evaluation / "performance.json")
        candidate = by_name[name]
        for filename in ("metrics.json", "performance.json", "run.json"):
            export_json(evaluation / filename, f"evaluation/{family}/{filename}")
        export_json(
            args.training / name / "metadata.json", f"evaluation/{family}/model_metadata.json"
        )
        for filename in (
            "confusion_matrix.png",
            "roc.png",
            "precision_recall.png",
            "category_breakdown.csv",
        ):
            source = evaluation / filename
            if source.exists():
                target = args.output / "evaluation" / family / filename
                shutil.copyfile(source, target)
                sources[str(target.relative_to(args.output))] = {
                    "source": portable(str(source.resolve()), root),
                    "sha256": sha256(source),
                }
        rows.append(
            {
                "family": family,
                "candidate": name,
                "selected": name == selection["best"],
                "validation_f1": candidate["validation"]["f1"],
                **{
                    key: metrics[key]
                    for key in (
                        "accuracy",
                        "precision",
                        "recall",
                        "f1",
                        "false_positive_rate",
                        "roc_auc",
                        "average_precision",
                    )
                },
                "macro_f1": metrics["macro"]["f1"],
                "training_seconds": candidate["training"]["seconds"],
                "training_peak_rss_bytes": candidate["training"]["peak_rss_bytes"],
                "model_load_seconds": performance["model_load_seconds"],
                "prediction_seconds_median": performance["prediction_seconds_median"],
                "rows_per_second": performance["rows_per_second"],
            }
        )
    pd.DataFrame(rows).to_csv(args.output / "comparison.csv", index=False)
    write_json(
        args.output / "export_manifest.json",
        {
            "note": "JSON local absolute paths normalized; hashes refer to original source files",
            "sources": sources,
        },
    )
    print(f"已导出报告证据: {args.output}")


if __name__ == "__main__":
    main()
