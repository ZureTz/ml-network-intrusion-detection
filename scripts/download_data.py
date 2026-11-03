"""下载用户明确选择的固定版本公开镜像，不自动切换来源。"""

import argparse
import shutil
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ml_network_intrusion_detection.common import sha256, write_json
from ml_network_intrusion_detection.data.schema import check_header

REVISION = "3eb02c4a0a29866b7abcb5ef77c45cf4fcc8f6b0"
BASE = f"https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15/resolve/{REVISION}"
OFFICIAL = "https://research.unsw.edu.au/projects/unsw-nb15-dataset"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["mireu-mirror"], required=True)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--cache-dir", type=Path, help="复用已下载的 train.csv/test.csv，仍强制校验哈希"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "official_reference": OFFICIAL,
        "mirror": "https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15",
        "revision": REVISION,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "verification_limit": "Official SharePoint requires login; no official bytewise comparison",
        "filename_mapping": "Mirror test.csv is the 175341-row official training partition; "
        "mirror train.csv is the 82332-row official testing partition",
        "mirror_license_note": (
            "Mirror card says GPL-3.0; original academic-use terms take precedence"
        ),
        "usage": "Academic research; cite the five papers requested by the official dataset page",
        "files": [],
    }
    manifest_path = args.output / "source_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"来源清单已存在: {manifest_path}")
    # 在临时目录内验证两个文件，全部通过才移动到目标目录。
    with tempfile.TemporaryDirectory(prefix="unsw-download-") as temporary:
        staged = []
        for remote, local, rows, expected_hash in (
            (
                "test.csv",
                "UNSW_NB15_training-set.csv",
                175341,
                "bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa",
            ),
            (
                "train.csv",
                "UNSW_NB15_testing-set.csv",
                82332,
                "734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559",
            ),
        ):
            target = args.output / local
            if target.exists():
                raise FileExistsError(f"原始数据已存在，拒绝覆盖: {target}")
            url = f"{BASE}/{remote}"
            path = Path(temporary) / local
            print(f"下载 {url}", flush=True)
            if args.cache_dir:
                shutil.copyfile(args.cache_dir / remote, path)
            else:
                request = urllib.request.Request(url, headers={"User-Agent": "ml-ids-course/0.1"})
                with (
                    urllib.request.urlopen(request, timeout=120) as response,
                    path.open("wb") as out,
                ):
                    shutil.copyfileobj(response, out)
            if sha256(path) != expected_hash:
                raise ValueError(f"{remote} 校验和与固定版本记录不同")
            check_header(path, labeled=True)
            actual_rows = sum(len(chunk) for chunk in pd.read_csv(path, chunksize=10000))
            if actual_rows != rows:
                raise ValueError(f"{remote} 行数 {actual_rows} 与官方说明 {rows} 不同")
            manifest["files"].append(
                {
                    "filename": local,
                    "url": url,
                    "sha256": sha256(path),
                    "rows": actual_rows,
                }
            )
            staged.append((path, target))
        for path, target in staged:
            # 'xb' 拒绝覆盖其他进程刚创建的文件。
            with path.open("rb") as source, target.open("xb") as out:
                shutil.copyfileobj(source, out)
    write_json(manifest_path, manifest)
    print(f"下载及结构核验完成: {manifest_path}")


if __name__ == "__main__":
    main()
