"""在真实样本上验证独立进程无标签预测与保存模型的一致性。"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ml_network_intrusion_detection.common import sha256, write_json
from ml_network_intrusion_detection.data.schema import FEATURES, read_frame, validate_features
from ml_network_intrusion_detection.inference.predict import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="auto", help="auto/cpu/mps/cuda/cuda:N")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frame = read_frame(args.input)[FEATURES]
    unlabeled = args.output / "unlabeled.csv"
    frame.to_csv(unlabeled, index=False)
    predictions = args.output / "predictions.csv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ml_network_intrusion_detection",
            "predict",
            "--model",
            str(args.model),
            "--input",
            str(unlabeled),
            "--output",
            str(predictions),
            "--device",
            args.device,
        ],
        check=True,
    )
    model, metadata = load_model(args.model, args.device)
    expected = model.predict_proba(validate_features(frame)[0])[:, 1]
    actual = pd.read_csv(predictions)
    np.testing.assert_allclose(actual.attack_probability, expected, atol=1e-6, rtol=1e-5)
    np.testing.assert_array_equal(actual.predicted_label, expected >= metadata["threshold"])
    np.testing.assert_array_equal(actual.row_index, np.arange(1, len(frame) + 1))
    write_json(
        args.output / "verification.json",
        {
            "status": "passed",
            "rows": len(frame),
            "input_has_labels": False,
            "independent_process": True,
            "probability_atol": 1e-6,
            "probability_rtol": 1e-5,
            "labels_identical": True,
            "row_order_identical": True,
            "model_sha256": sha256(args.model / "metadata.json"),
            "model_artifacts": metadata["artifacts"],
            "device": str(model.named_steps["classifier"].device_),
            "input_sha256": sha256(unlabeled),
            "prediction_sha256": sha256(predictions),
        },
    )
    print(f"独立无标签预测验证通过: {len(frame)} 行")


if __name__ == "__main__":
    main()
