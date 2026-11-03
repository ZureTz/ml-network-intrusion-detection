"""原始数据审计与无重复组泄漏的数据划分。"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from ml_network_intrusion_detection.common import run_directory, sha256, write_json
from ml_network_intrusion_detection.data.schema import (
    FEATURES,
    fingerprints,
    read_frame,
    validate_features,
    validate_labels,
)


def prepare(config: dict, output: Path) -> None:
    """审计官方划分，隔离重复特征组，输出带原始行来源的 CSV 与清单。"""
    with run_directory(output, "prepare", config):
        frames, hashes, audit = {}, {}, {}
        for name, key in (("development", "train_csv"), ("test", "test_csv")):
            path = Path(config[key])
            raw = read_frame(path)
            x, details = validate_features(raw)
            y = validate_labels(raw)
            if len(x) == 0 or y.nunique() < 2:
                raise ValueError(f"{name} 必须包含正常和攻击两个类别")
            group = fingerprints(x)
            conflicts = pd.DataFrame({"group": group, "label": y}).groupby("group").label.nunique()
            details["conflicting_groups"] = int((conflicts > 1).sum())
            conflict_mask = group.isin(conflicts[conflicts > 1].index)
            details["conflicting_rows"] = int(conflict_mask.sum())
            details["duplicate_feature_rows"] = int(group.duplicated().sum())
            details["class_counts"] = {str(k): int(v) for k, v in y.value_counts().items()}
            details["source"] = {"filename": path.name, "sha256": sha256(path)}
            audit[name] = details
            write_json(output / "audit.json", audit)
            if details["conflicting_groups"] and config.get("conflict_policy", "error") == "error":
                raise ValueError(f"{name} 存在同特征冲突标签，请检查 audit.json")
            x["label"] = y
            x["source_file"] = path.name
            x["source_row"] = range(1, len(x) + 1)
            if "attack_cat" in raw:
                x["attack_cat"] = raw["attack_cat"]
            if details["conflicting_groups"]:
                x.loc[conflict_mask, ["source_file", "source_row", "label"]].to_csv(
                    output / f"{name}_conflicting_rows.csv", index=False
                )
                if name == "development":
                    x = x.loc[~conflict_mask]
                    group = group.loc[~conflict_mask]
            frames[name], hashes[name] = x, group

        overlap = hashes["development"].isin(hashes["test"])
        audit["removed_cross_split_rows"] = int(overlap.sum())
        frames["development"].loc[overlap, ["source_file", "source_row", "label"]].to_csv(
            output / "removed_cross_split_rows.csv", index=False
        )
        development = frames["development"].loc[~overlap].reset_index(drop=True)
        if config["sample_size"] and len(development) > config["sample_size"]:
            indices, _ = train_test_split(
                range(len(development)),
                train_size=config["sample_size"],
                random_state=config["seed"],
                stratify=development.label,
            )
            development = development.iloc[sorted(indices)].reset_index(drop=True)
        groups = fingerprints(development)
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config["seed"])
        train_idx, val_idx = next(splitter.split(development[FEATURES], development.label, groups))
        splits = {
            "train": development.iloc[train_idx],
            "validation": development.iloc[val_idx],
            "test": frames["test"],
        }
        if set(groups.iloc[train_idx]) & set(groups.iloc[val_idx]):
            raise RuntimeError("训练/验证重复组泄漏")
        manifest = {"seed": config["seed"], "config": config, "audit": audit, "splits": {}}
        for name, frame in splits.items():
            if frame.label.nunique() != 2:
                raise ValueError(f"{name} 划分缺少类别；不能用于正式二分类训练")
            path = output / f"{name}.csv"
            frame.to_csv(path, index=False)
            manifest["splits"][name] = {
                "rows": len(frame),
                "class_counts": {str(k): int(v) for k, v in frame.label.value_counts().items()},
                "sha256": sha256(path),
            }
        write_json(output / "audit.json", audit)
        write_json(output / "manifest.json", manifest)
