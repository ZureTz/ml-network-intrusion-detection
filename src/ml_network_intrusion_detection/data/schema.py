"""UNSW-NB15 官方划分 CSV 的输入契约。"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORICAL = ["proto", "service", "state"]
FEATURES = (
    "dur proto service state spkts dpkts sbytes dbytes rate sttl dttl sload dload "
    "sloss dloss sinpkt dinpkt sjit djit swin stcpb dtcpb dwin tcprtt synack ackdat "
    "smean dmean trans_depth response_body_len ct_srv_src ct_state_ttl ct_dst_ltm "
    "ct_src_dport_ltm ct_dst_sport_ltm ct_dst_src_ltm is_ftp_login ct_ftp_cmd "
    "ct_flw_http_mthd ct_src_ltm ct_srv_dst is_sm_ips_ports"
).split()
NUMERIC = [name for name in FEATURES if name not in CATEGORICAL]


def check_header(path: Path, labeled: bool = False) -> list[str]:
    """检查原始表头，避免 pandas 自动改名掩盖重复列。"""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader, [])
        for row_number, row in enumerate(reader, start=1):
            if len(row) != len(header):
                raise ValueError(f"CSV 第 {row_number} 条数据字段数量不匹配或为空行")
    if len(set(header)) != len(header):
        raise ValueError("CSV 含重复列名")
    required = FEATURES + (["label"] if labeled else [])
    missing = sorted(set(required) - set(header))
    if missing:
        raise ValueError(f"缺少必需列: {', '.join(missing)}")
    return header


def validate_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """按固定特征顺序规范化；非法数值报错，缺失/无穷转为 NaN 并计数。"""
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing or frame.columns.duplicated().any():
        raise ValueError(f"输入列缺失或重复: {missing}")
    result = pd.DataFrame(index=frame.index)
    audit: dict = {"rows": len(frame), "extra_columns": sorted(set(frame) - set(FEATURES))}
    audit["missing"] = {}
    audit["infinite"] = {}
    for name in FEATURES:
        values = frame[name]
        if name in NUMERIC:
            converted = pd.to_numeric(values, errors="coerce").astype(float)
            bad = values.notna() & converted.isna()
            if bad.any():
                examples = list(frame.index[bad][:5])
                raise ValueError(f"数值列 {name} 存在非法值，行索引 {examples}")
            infinite = np.isinf(converted)
            audit["infinite"][name] = int(infinite.sum())
            result[name] = converted.mask(infinite, np.nan)
        else:
            # 保留 '-'（UNSW service 的真实类别），只把空字符串识别为缺失。
            result[name] = values.map(lambda x: str(x) if pd.notna(x) else "__MISSING__")
        audit["missing"][name] = int(values.isna().sum())
    return result[FEATURES], audit


def validate_labels(frame: pd.DataFrame) -> pd.Series:
    if "label" not in frame:
        raise ValueError("缺少 label 列")
    labels = pd.to_numeric(frame["label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("label 只能为 0（正常）或 1（攻击），不允许缺失")
    return labels.astype(int)


def read_frame(path: Path, labeled: bool = True) -> pd.DataFrame:
    check_header(path, labeled)
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def fingerprints(features: pd.DataFrame) -> pd.Series:
    """固定列序、规范类型后的 64 位特征指纹；仅用于分组，不含标签。"""
    return pd.util.hash_pandas_object(features[FEATURES], index=False)
