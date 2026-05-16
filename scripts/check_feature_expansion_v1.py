#!/usr/bin/env python3
"""Feature Expansion v1 离线检查：从 snapshot 构建 features 并输出可附报告摘要.

不读取打印 .env / 密钥；仅使用本地 snapshot parquet + meta.json。
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantmind.data.snapshot import load_snapshot
from quantmind.features.expansion import (
    CATEGORICAL_EXPOSURE_FIELDS,
    EXPANSION_FACTORS,
    MARKET_LEVEL_FEATURES,
)
from quantmind.features.fundamental import FUNDAMENTAL_FACTORS
from quantmind.features.pipeline import FeaturePipeline, list_all_factor_names
from quantmind.features.sentiment import SENTIMENT_FACTORS
from quantmind.features.technical import TECHNICAL_FACTORS


def _load_snapshot_dir(snapshot_dir: Path) -> dict[str, Any]:
    meta_path = snapshot_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"no meta.json in {snapshot_dir}")
    meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    out: dict[str, Any] = {"meta": meta}
    for fname in meta.get("files", []):
        key = str(fname).removesuffix(".parquet")
        pq = snapshot_dir / fname
        if pq.is_file():
            out[key] = pd.read_parquet(pq)
    return out


def _baseline_41_names() -> set[str]:
    out: set[str] = set()
    for block in (FUNDAMENTAL_FACTORS, TECHNICAL_FACTORS, SENTIMENT_FACTORS):
        out.update(n for n, _ in block)
    return out


def _expansion_names() -> set[str]:
    return {n for n, _ in EXPANSION_FACTORS}


def _fmt_pct(x: float) -> str:
    if math.isnan(x):
        return "nan"
    return f"{100.0 * x:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description="Feature Expansion v1 snapshot report")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--snapshot-dir", type=Path, help="指向 data/snapshots/<as_of>/ 目录")
    g.add_argument("--as-of", type=lambda s: date.fromisoformat(s), help="ISO 日期（用 settings.data.dir 加载）")
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="仅 raw 因子（不做 winsorize/zscore/中性化）",
    )
    args = parser.parse_args()

    if args.snapshot_dir is not None:
        snap = _load_snapshot_dir(args.snapshot_dir.resolve())
        meta = snap.get("meta") or {}
        as_of = args.as_of
        if as_of is None:
            raw = meta.get("as_of")
            if not raw:
                raise SystemExit("需指定 --as-of，或在 meta.json 中提供 as_of")
            as_of = date.fromisoformat(str(raw)[:10])
    else:
        assert args.as_of is not None
        as_of = args.as_of
        snap = load_snapshot(as_of)

    pipe = FeaturePipeline(do_standardize=not args.no_standardize)
    df = pipe.run_single_from_snapshot(snap, as_of)

    base41 = _baseline_41_names()
    new_names = _expansion_names()
    colnames = set(df.columns)
    legacy_ok = base41 <= colnames
    expansion_in = sorted(colnames & new_names)

    numeric_cols = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_integer_dtype(df[c])
    ]

    miss_rates = {
        c: float(df[c].isna().mean())
        for c in sorted(new_names)
        if c in df.columns
    }

    mkt_preservation: dict[str, str] = {}
    for c in sorted(MARKET_LEVEL_FEATURES):
        if c not in df.columns:
            mkt_preservation[c] = "missing_column"
        elif c in CATEGORICAL_EXPOSURE_FIELDS:
            mkt_preservation[c] = "n/a (categorical)"
        elif not pd.api.types.is_numeric_dtype(df[c]):
            mkt_preservation[c] = "non_numeric"
        else:
            sub = df[c].dropna()
            if sub.empty:
                mkt_preservation[c] = "all_nan"
            else:
                uq = sub.unique()
                mkt_preservation[c] = (
                    "cross_section_constant (ok)"
                    if len(uq) == 1
                    else f"WARNING unique={len(uq)}"
                )

    cat_cov: dict[str, str] = {}
    for c in sorted(CATEGORICAL_EXPOSURE_FIELDS):
        if c not in df.columns:
            cat_cov[c] = "missing"
            continue
        s = df[c].astype("string")
        ok = s.notna() & (s.str.len() > 0)
        cat_cov[c] = _fmt_pct(float(ok.mean())) + " non-empty"

    all_null = [c for c in df.columns if df[c].isna().all()]
    inf_cols: list[str] = []
    for c in numeric_cols:
        v = np.asarray(df[c], dtype="float64")
        if np.isneginf(v).any() or np.isposinf(v).any():
            inf_cols.append(c)

    total_registry = list_all_factor_names()

    lines = [
        "=== QuantMind Feature Expansion v1 — Check Report ===",
        f"as_of: {as_of.isoformat()}",
        f"mode: {'raw' if args.no_standardize else 'standardized'}",
        f"panel_shape: {df.shape[0]} tickers × {df.shape[1]} columns",
        "",
        "--- counts ---",
        f"registry_total_factors (all groups): {len(total_registry)}",
        f"baseline_factor_count (fundamental+technical+sentiment): {len(base41)}",
        f"expansion_factor_count (designed): {len(new_names)}",
        f"baseline_present_in_output: {legacy_ok}",
        f"expansion_columns_present: {len(expansion_in)}",
        "",
        "--- legacy missing (if any) ---",
        "none" if legacy_ok else ", ".join(sorted(base41 - colnames)),
        "",
        "--- expansion missing rates ---",
    ]
    for k in sorted(miss_rates.keys()):
        lines.append(f"  {k}: {_fmt_pct(miss_rates[k])}")

    lines += [
        "",
        "--- market_level_features (should be one value per as_of when standardized & excluded from z) ---",
    ]
    for k, v in mkt_preservation.items():
        lines.append(f"  {k}: {v}")

    lines += [
        "",
        "--- categorical_exposure_fields coverage (non-empty string) ---",
    ]
    for k, v in cat_cov.items():
        lines.append(f"  {k}: coverage={v}")

    lines += [
        "",
        "--- quality ---",
        f"all_null_columns: {all_null if all_null else 'none'}",
        f"inf_or_neginf_columns: {inf_cols if inf_cols else 'none'}",
        "",
        "--- column groups (names only) ---",
        f"market_level_features: {sorted(MARKET_LEVEL_FEATURES & colnames)}",
        f"categorical_exposure_fields: {sorted(CATEGORICAL_EXPOSURE_FIELDS & colnames)}",
        f"expansion_feature_names: {expansion_in}",
    ]
    report = "\n".join(lines)
    print(report)
    if not legacy_ok or inf_cols:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
