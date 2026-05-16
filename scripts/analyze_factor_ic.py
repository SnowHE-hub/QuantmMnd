#!/usr/bin/env python3
"""截面 Rank IC（Spearman）因子分析 — train.parquet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

EXCLUDED_FEATURE_COLS = frozenset({"exposure_industry", "exposure_area"})
DEFAULT_LABEL_SKIP = frozenset({"forward_return_21d", "forward_return_63d"})
MIN_CROSS_SECTION = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel-file", type=Path, default=Path("data/panel/train.parquet"))
    p.add_argument("--label", default="forward_return_21d")
    p.add_argument("--output-dir", type=Path, default=Path("reports/model_training"))
    p.add_argument("--min-cross-section", type=int, default=MIN_CROSS_SECTION)
    return p.parse_args()


def numeric_feature_columns(panel: pd.DataFrame, label: str) -> list[str]:
    skip = EXCLUDED_FEATURE_COLS | DEFAULT_LABEL_SKIP | {label}
    cols: list[str] = []
    for c in panel.columns:
        if c in skip:
            continue
        if pd.api.types.is_numeric_dtype(panel[c]):
            cols.append(c)
    return sorted(cols)


def cross_section_ic(
    panel: pd.DataFrame,
    feat: str,
    label: str,
    dates: list[pd.Timestamp],
    min_n: int,
) -> dict[pd.Timestamp, float]:
    out: dict[pd.Timestamp, float] = {}
    for d in dates:
        if d not in panel.index.get_level_values("as_of"):
            continue
        xs = panel.xs(d, level="as_of")[[feat, label]].dropna()
        if len(xs) < min_n:
            out[d] = float("nan")
            continue
        rho, _ = stats.spearmanr(xs[feat].values, xs[label].values)
        out[d] = float(rho) if rho == rho else float("nan")
    return out


def ic_summary(ic_by_date: dict[pd.Timestamp, float]) -> tuple[float, float, float, float]:
    vals = [v for v in ic_by_date.values() if v == v]  # not nan
    if not vals:
        return float("nan"), float("nan"), float("nan"), float("nan")
    arr = np.array(vals)
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=1)) if len(vals) > 1 else float("nan")
    ir = m / s if s == s and s > 1e-12 else float("nan")
    win = float(np.mean([v > 0 for v in vals]))
    return m, s, ir, win


def run_ic_analysis(
    panel: pd.DataFrame,
    label: str,
    *,
    min_cross_section: int = MIN_CROSS_SECTION,
) -> tuple[pd.DataFrame, dict[str, dict[pd.Timestamp, float]]]:
    """对 panel 全体数值因子做截面 IC 汇总（供脚本与单元测试）."""
    if label not in panel.columns:
        raise ValueError(f"label column '{label}' missing")

    feature_cols = numeric_feature_columns(panel, label)
    dates = sorted(panel.index.get_level_values("as_of").unique())

    ic_matrix: dict[str, dict[pd.Timestamp, float]] = {}
    rows_summary: list[dict[str, float | str]] = []

    for feat in feature_cols:
        ibd = cross_section_ic(panel, feat, label, dates, min_cross_section)
        ic_matrix[feat] = ibd
        ic_m, ic_s, ic_ir, ic_pos = ic_summary(ibd)
        rows_summary.append({
            "factor": feat,
            "IC_mean": ic_m,
            "IC_std": ic_s,
            "ICIR": ic_ir,
            "IC_positive_rate": ic_pos,
            "abs_ICIR": abs(ic_ir) if ic_ir == ic_ir else float("nan"),
        })

    df = pd.DataFrame(rows_summary).sort_values("abs_ICIR", ascending=False).drop(columns=["abs_ICIR"])
    return df, ic_matrix


def main() -> None:
    args = parse_args()
    if not args.panel_file.is_file():
        raise SystemExit(f"panel not found: {args.panel_file}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(args.panel_file)

    df, ic_matrix = run_ic_analysis(
        panel,
        args.label,
        min_cross_section=args.min_cross_section,
    )
    feature_cols = df["factor"].tolist()
    dates = sorted(panel.index.get_level_values("as_of").unique())
    print(f"panel shape={panel.shape}  numeric_features={len(feature_cols)}  periods={len(dates)}")

    csv_path = args.output_dir / "factor_ic_analysis.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")

    icir_gt_03 = int((df["ICIR"].abs() > 0.3).sum())
    icir_neg = int((df["ICIR"] < 0).sum())
    print(f"\nTop 15 (|ICIR|):\n{df.head(15).to_string(index=False)}")
    print(f"\nICIR > 0.3 的因子数量: {icir_gt_03}")
    print(f"ICIR < 0（反向因子）数量: {icir_neg}")
    print(f"\nCSV → {csv_path.resolve()}")

    # Heatmap MD: rows=factors sorted by ICIR magnitude, cols=dates
    dstr = [str(d)[:10] for d in dates]
    fac_order = df["factor"].tolist()
    lines = ["# Factor IC Heatmap（Rank IC 按截面 Spearman）", ""]
    lines.append("| factor | " + " | ".join(dstr) + " |")
    lines.append("| --- | " + " | ".join(["---"] * len(dstr)) + " |")
    for f in fac_order:
        cells = []
        for d in dates:
            v = ic_matrix[f].get(d, float("nan"))
            cells.append(f"{v:+.4f}" if v == v else "")
        lines.append("| " + f + " | " + " | ".join(cells) + " |")
    heat_path = args.output_dir / "factor_ic_heatmap.md"
    heat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Heatmap MD → {heat_path.resolve()}")


if __name__ == "__main__":
    main()
