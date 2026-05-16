#!/usr/bin/env python3
"""训练用 panel Parquet 审计：维度、expansion 列、标签、缺失与 inf、按日分布.

只读本地 parquet，不访问 .env / 密钥。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantmind.features.expansion import CATEGORICAL_EXPOSURE_FIELDS, EXPANSION_FACTORS, MARKET_LEVEL_FEATURES

def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit training panel parquet")
    parser.add_argument("panel", type=Path, help="panel parquet 路径")
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="expansion 缺失率排名显示前 K 列",
    )
    args = parser.parse_args()
    path = args.panel.resolve()
    if not path.is_file():
        print(f"# Error: file not found: `{path}`")
        return 1

    df = pd.read_parquet(path)

    # MultiIndex or columns
    if isinstance(df.index, pd.MultiIndex):
        if "as_of" in df.index.names:
            dates = df.index.get_level_values("as_of")
        else:
            dates = df.index.get_level_values(0)
        n_dates = int(dates.nunique())
        n_tickers = (
            int(df.index.get_level_values("ticker").nunique())
            if "ticker" in df.index.names
            else None
        )
    else:
        if "as_of" in df.columns:
            dates = pd.to_datetime(df["as_of"])
            n_dates = int(dates.nunique())
        else:
            dates = pd.Series([])
            n_dates = 0
        n_tickers = int(df["ticker"].nunique()) if "ticker" in df.columns else None

    n_rows = len(df)
    label_pat = "forward_return_"
    label_cols = [c for c in df.columns if str(c).startswith(label_pat)]
    feat_cols = [c for c in df.columns if not str(c).startswith(label_pat)]

    exp_cols = [c for c in df.columns if c in {n for n, _ in EXPANSION_FACTORS}]
    miss_rank: list[tuple[str, float]] = []
    for c in exp_cols:
        miss_rank.append((c, float(df[c].isna().mean())))
    miss_rank.sort(key=lambda x: -x[1])

    all_null = [c for c in df.columns if df[c].isna().all()]

    inf_cols: list[str] = []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        v = np.asarray(df[c], dtype="float64")
        if np.isinf(v).any():
            inf_cols.append(c)

    mkt = [c for c in MARKET_LEVEL_FEATURES if c in df.columns]
    cat = [c for c in CATEGORICAL_EXPOSURE_FIELDS if c in df.columns]

    # per-date counts
    per_date_rows: list[list[Any]] = []
    per_date_labels: list[list[Any]] = []
    if isinstance(df.index, pd.MultiIndex) and df.index.nlevels >= 2:
        sub = df.reset_index()
        if "as_of" in sub.columns:
            g = sub.groupby(pd.to_datetime(sub["as_of"]).dt.date)
            for d, part in g:
                per_date_rows.append([str(d), len(part)])
                if label_cols:
                    lc = label_cols[0]
                    nn = int(part[lc].notna().sum()) if lc in part.columns else 0
                    per_date_labels.append([str(d), nn])
    elif "as_of" in df.columns:
        g = df.groupby(pd.to_datetime(df["as_of"]).dt.date)
        for d, part in g:
            per_date_rows.append([str(d), len(part)])
            if label_cols:
                lc = label_cols[0]
                nn = int(part[lc].notna().sum()) if lc in part.columns else 0
                per_date_labels.append([str(d), nn])

    print("# Training Panel Audit\n")
    print(f"_path_: `{path}`\n")
    print("## Summary\n")
    print(f"- **n_rows**: {n_rows}")
    print(f"- **n_dates** (as_of): {n_dates}")
    print(f"- **n_tickers**: {n_tickers}")
    print(f"- **n_feature_columns**: {len(feat_cols)}")
    print(f"- **label_columns**: {label_cols}")
    print(f"- **expansion_columns_found**: {len(exp_cols)}")
    print(f"- **market_level_features present**: {mkt}")
    print(f"- **categorical_exposure_fields present**: {cat}")
    print(f"- **all_null columns**: {all_null if all_null else 'none'}")
    print(f"- **inf columns**: {inf_cols if inf_cols else 'none'}")
    print()

    print("## Expansion missing-rate rank (highest first)\n")
    top = miss_rank[: max(1, args.top_k)]
    print(_md_table(["column", "missing_rate"], [[a, f"{100*b:.2f}%"] for a, b in top]))
    print()

    if per_date_rows:
        print("## Rows per as_of date\n")
        print(_md_table(["as_of_date", "n_rows"], per_date_rows[:50]))
        if len(per_date_rows) > 50:
            print(f"\n_… truncated, total {len(per_date_rows)} dates_")
        print()
    if per_date_labels:
        print("## Label non-null count (first label column)\n")
        lc0 = label_cols[0] if label_cols else ""
        print(_md_table([f"as_of_date", f"non_null_{lc0}"], per_date_labels[:50]))
        print()

    return 0 if not inf_cols else 1


if __name__ == "__main__":
    raise SystemExit(main())
