#!/usr/bin/env python3
"""从 factor_ic_analysis.csv 筛选稳定因子，写入 stable_factors.txt."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ic-csv",
        type=Path,
        default=_ROOT / "reports/model_training/factor_ic_analysis.csv",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "reports/model_training/stable_factors.txt",
    )
    p.add_argument("--ic-abs-min", type=float, default=0.02)
    p.add_argument("--pos-rate-high", type=float, default=0.55)
    p.add_argument("--pos-rate-low", type=float, default=0.45)
    p.add_argument("--max-factors", type=int, default=30)
    p.add_argument("--min-factors", type=int, default=20)
    args = p.parse_args()

    df = pd.read_csv(args.ic_csv)
    col_ic = "IC_mean" if "IC_mean" in df.columns else df.columns[0]
    col_pr = "IC_positive_rate" if "IC_positive_rate" in df.columns else None
    name_col = "factor" if "factor" in df.columns else df.columns[0]

    ic = pd.to_numeric(df[col_ic], errors="coerce")
    sel = ic.abs() > args.ic_abs_min
    if col_pr and col_pr in df.columns:
        pr = pd.to_numeric(df[col_pr], errors="coerce")
        sel = sel & ((pr > args.pos_rate_high) | (pr < args.pos_rate_low))

    picked = df.loc[sel].copy()
    picked["_abs_ic"] = picked[col_ic].abs()
    picked = picked.sort_values("_abs_ic", ascending=False)

    names = picked[name_col].astype(str).tolist()
    if len(names) < args.min_factors:
        extra = df[~df[name_col].astype(str).isin(names)].copy()
        extra["_abs_ic"] = extra[col_ic].abs()
        extra = extra.sort_values("_abs_ic", ascending=False)
        for _, row in extra.iterrows():
            nm = str(row[name_col])
            if nm not in names:
                names.append(nm)
            if len(names) >= args.min_factors:
                break

    names = names[: args.max_factors]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"wrote {len(names)} factors → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
