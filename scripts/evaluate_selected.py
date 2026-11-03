"""冻结验证选择后，在独立进程中评估每个模型族的最佳候选。"""

import argparse
import subprocess
import sys
from pathlib import Path

from ml_network_intrusion_detection.common import read_json, run_directory, sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=10000)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size 必须为正整数")
    if read_json(args.training / "run.json")["status"] != "completed":
        raise ValueError("训练运行尚未完整完成")
    selection = read_json(args.training / "selection.json")
    manifest = read_json(args.training / "data_manifest.json")
    test_path = args.data / "test.csv"
    if sha256(test_path) != manifest["splits"]["test"]["sha256"]:
        raise ValueError("测试数据与训练记录中的冻结划分不一致")
    with run_directory(args.output, "final_evaluation", {"batch_size": args.batch_size}):
        write_json(
            args.output / "frozen_selection.json",
            {
                "selection": selection,
                "selection_sha256": sha256(args.training / "selection.json"),
                "test_sha256": sha256(test_path),
            },
        )
        for family, name in selection["family_winners"].items():
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ml_network_intrusion_detection",
                    "evaluate",
                    "--model",
                    str(args.training / name),
                    "--input",
                    str(test_path),
                    "--output",
                    str(args.output / family),
                    "--batch-size",
                    str(args.batch_size),
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
