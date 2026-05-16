#!/usr/bin/env python3
"""月频技术因子 + 季末基本面前填合并 → monthly_panel.parquet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quantmind.core.logger import get_logger, setup_logger
from quantmind.features.expansion import CATEGORICAL_EXPOSURE_FIELDS
from quantmind.features.neutralize import neutralize_cross_section
from quantmind.features.pipeline import ALL_FACTOR_GROUPS
from quantmind.features.technical import TECHNICAL_FACTORS

log = get_logger(__name__)

MONTHLY_TECH = [
    "momentum_1m",
    "momentum_3m",
    "momentum_6m",
    "vol_1m",
    "vol_3m",
    "reversal_1w",
    "amihud_1m",
]
TECH_NAMES_QUARTERLY = {n for n, _ in TECHNICAL_FACTORS}
CAT_EXPOSURE = set(CATEGORICAL_EXPOSURE_FIELDS)


def _quarter_fundamental_like_columns() -> list[str]:
    names: list[str] = []
    for grp in ("fundamental", "sentiment", "expansion"):
        for n, _ in ALL_FACTOR_GROUPS[grp]:
            if n in CAT_EXPOSURE:
                continue
            names.append(n)
    return names


def _load_wide_adj_close(path: Path) -> pd.DataFrame:
    px = pd.read_parquet(path)
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    if px.index.has_duplicates:
        px = px[~px.index.duplicated(keep="last")]
    return px


def _load_amount_wide(ohlcv_path: Path, tickers: list[str], cal: pd.DatetimeIndex) -> pd.DataFrame:
    if not ohlcv_path.is_file():
        return pd.DataFrame(index=cal, columns=tickers, dtype=float)
    df = pd.read_parquet(ohlcv_path)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    if "amount" not in df.columns:
        return pd.DataFrame(index=cal, columns=tickers, dtype=float)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    code_col = "ts_code" if "ts_code" in df.columns else "ticker"
    w = df.pivot(index="trade_date", columns=code_col, values="amount").sort_index()
    return w.reindex(index=cal, columns=tickers)


def _month_end_dates(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    ser = pd.Series(np.arange(len(idx)), index=idx)
    pos = ser.groupby(idx.to_period("M")).transform("max")
    picks = sorted(idx[np.unique(pos.astype(int))])
    return pd.DatetimeIndex(picks)


def _forward_return_next_calendar_month(
    px_m: pd.DataFrame,
    cal: pd.DatetimeIndex,
    me: pd.Timestamp,
    tickers: list[str],
) -> pd.Series:
    """下月首个交易日收盘买入，同月最后一个交易日收盘卖出的持有期收益（PIT）。"""
    me = pd.Timestamp(me).normalize()
    next_m_start = (me.to_period("M") + 1).to_timestamp(how="start")
    entry_candidates = cal[cal.normalize() >= next_m_start.normalize()]
    if len(entry_candidates) == 0:
        return pd.Series(np.nan, index=tickers, dtype=float)
    entry_dt = pd.Timestamp(entry_candidates[0]).normalize()
    ep = entry_dt.to_period("M")
    in_month = cal[cal.to_period("M") == ep]
    exit_candidates = in_month[in_month.normalize() >= entry_dt.normalize()]
    if len(exit_candidates) == 0:
        return pd.Series(np.nan, index=tickers, dtype=float)
    exit_dt = pd.Timestamp(exit_candidates[-1]).normalize()
    if exit_dt < entry_dt:
        return pd.Series(np.nan, index=tickers, dtype=float)
    try:
        pe = px_m.loc[entry_dt]
        pxexit = px_m.loc[exit_dt]
    except KeyError:
        return pd.Series(np.nan, index=tickers, dtype=float)
    out = pxexit / pe.replace(0, np.nan) - 1.0
    return out.reindex(tickers)


def compute_monthly_forward_return(
    price_panel: pd.DataFrame,
    as_of_dates: list[str] | pd.DatetimeIndex | list[pd.Timestamp],
) -> pd.DataFrame:
    """
    正确的月频标签：每个截面日 as_of（通常为月末交易日），
    entry = 下月第一个交易日收盘价，exit = 该月最后一个交易日收盘价，
    forward_return_1m = exit/entry - 1。
    """
    px_m = price_panel.sort_index()
    if px_m.index.has_duplicates:
        px_m = px_m[~px_m.index.duplicated(keep="last")]
    px_m.index = pd.to_datetime(px_m.index)
    cal = px_m.index
    tickers = [str(c) for c in px_m.columns]
    rows: list[dict[str, object]] = []
    for me in as_of_dates:
        me = pd.Timestamp(me).normalize()
        fwd = _forward_return_next_calendar_month(px_m, cal, me, tickers)
        for tkr in tickers:
            rows.append({
                "month_end_date": me,
                "ticker": tkr,
                "forward_return_1m": float(fwd.get(tkr, np.nan)),
            })
    return pd.DataFrame(rows)


def _resolve_split_suffix(output: Path, explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    stem = output.stem
    if stem.endswith("_v2"):
        return "_v2"
    return ""


def _merge_asof_by_ticker(
    monthly_rows: pd.DataFrame,
    quarterly_long: pd.DataFrame,
    merge_cols: list[str],
) -> pd.DataFrame:
    out_parts: list[pd.DataFrame] = []
    monthly_rows = monthly_rows.copy()
    monthly_rows["month_end_date"] = pd.to_datetime(monthly_rows["month_end_date"]).astype(
        "datetime64[ns]"
    )
    q_long = quarterly_long.copy()
    q_long["as_of"] = pd.to_datetime(q_long["as_of"]).astype("datetime64[ns]")
    q_keep = ["as_of", "ticker", *[c for c in merge_cols if c in q_long.columns]]
    q_base = q_long[q_keep].drop_duplicates(subset=["as_of", "ticker"], keep="last")

    for tkr, ml in monthly_rows.groupby("ticker"):
        ml = ml.sort_values("month_end_date")
        ql = q_base[q_base["ticker"] == tkr].sort_values("as_of").drop(columns=["ticker"], errors="ignore")
        if ql.empty:
            merged = ml.copy()
            for c in merge_cols:
                if c not in merged.columns:
                    merged[c] = np.nan
            out_parts.append(merged)
            continue
        merged = pd.merge_asof(
            ml,
            ql,
            left_on="month_end_date",
            right_on="as_of",
            direction="backward",
            allow_exact_matches=True,
        )
        merged = merged.drop(columns=["as_of"], errors="ignore")
        out_parts.append(merged)
    return pd.concat(out_parts, ignore_index=True)


def main() -> int:
    setup_logger()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--adj-close",
        type=Path,
        default=_ROOT / "data/prices/csi300_daily_adj_close.parquet",
    )
    p.add_argument(
        "--ohlcv",
        type=Path,
        default=_ROOT / "data/prices/csi300_daily_ohlcv.parquet",
    )
    p.add_argument(
        "--quarterly-panels",
        type=Path,
        nargs="*",
        default=[
            _ROOT / "data/panel/train.parquet",
            _ROOT / "data/panel/val.parquet",
            _ROOT / "data/panel/test.parquet",
        ],
    )
    p.add_argument("--output", type=Path, default=_ROOT / "data/panel/monthly_panel.parquet")
    p.add_argument("--log-dir", type=Path, default=_ROOT / "reports/monthly")
    p.add_argument(
        "--label-method",
        choices=("forward_21d", "next_month"),
        default="forward_21d",
        help="next_month → forward_return_1m（下整月持有收益）；forward_21d 为原 21 日标签",
    )
    p.add_argument(
        "--split-suffix",
        default=None,
        help="train/val/test / meta 文件名后缀；默认从 output 推导（*_v2 → _v2）",
    )
    p.add_argument("--neutralize", action="store_true")
    p.add_argument("--industry-col", default="exposure_industry")
    args = p.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / "build_monthly_panel.log"

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)

        def flush(self):
            for s in self.streams:
                s.flush()

    tee_f = open(log_path, "w", encoding="utf-8")
    old_stdout = sys.stdout
    sys.stdout = _Tee(sys.__stdout__, tee_f)

    try:
        if not args.adj_close.is_file():
            log.error("missing adj close parquet: {}", args.adj_close)
            return 1

        px_m = _load_wide_adj_close(args.adj_close)
        tickers = [str(c) for c in px_m.columns]
        cal = px_m.index

        ret_daily = px_m.pct_change(fill_method=None)
        amt_wide = _load_amount_wide(args.ohlcv, tickers, cal)
        amihud_num = ret_daily.abs() / (amt_wide + 1.0e-12)

        month_ends = _month_end_dates(cal)
        vol21 = ret_daily.rolling(21, min_periods=15).std() * np.sqrt(252.0)
        vol63 = ret_daily.rolling(63, min_periods=45).std() * np.sqrt(252.0)

        rows: list[dict[str, object]] = []
        idx_pos = {pd.Timestamp(t).normalize(): i for i, t in enumerate(cal)}

        for me in month_ends:
            me = pd.Timestamp(me).normalize()
            pos = idx_pos.get(me)
            if pos is None:
                continue

            def _px_month(me_ts: pd.Timestamp, periods_back: int) -> pd.Series:
                tgt = me_ts.to_period("M") - periods_back
                hits = month_ends[month_ends.to_period("M") == tgt]
                if len(hits) == 0:
                    return pd.Series(np.nan, index=tickers)
                dt = pd.Timestamp(hits[-1]).normalize()
                return px_m.loc[dt]

            px_me = px_m.loc[me]
            mom1 = px_me / _px_month(me, 1) - 1.0
            mom3 = px_me / _px_month(me, 3) - 1.0
            mom6 = px_me / _px_month(me, 6) - 1.0

            pos5 = max(0, pos - 5)
            dt5 = cal[pos5]
            rev5 = px_me / px_m.loc[dt5] - 1.0

            v1 = vol21.loc[me]
            v3 = vol63.loc[me]
            am_mean = amihud_num.iloc[max(0, pos - 20) : pos + 1].mean(axis=0)

            if args.label_method == "next_month":
                fwd_ser = _forward_return_next_calendar_month(px_m, cal, me, tickers)
                label_col_name = "forward_return_1m"
                fwd_val = fwd_ser
            else:
                nm_start = (me.to_period("M") + 1).to_timestamp(how="start")
                future_idx = cal[cal.normalize() >= nm_start.normalize()]
                if len(future_idx) > 21:
                    base_dt = future_idx[0]
                    end_dt = future_idx[21]
                    fwd_val = px_m.loc[end_dt] / px_m.loc[base_dt].replace(0, np.nan) - 1.0
                else:
                    fwd_val = pd.Series(np.nan, index=tickers)
                label_col_name = "forward_return_21d"

            for tkr in tickers:
                row_out: dict[str, object] = {
                    "month_end_date": me,
                    "ticker": tkr,
                    "momentum_1m": float(mom1.get(tkr, np.nan)),
                    "momentum_3m": float(mom3.get(tkr, np.nan)),
                    "momentum_6m": float(mom6.get(tkr, np.nan)),
                    "vol_1m": float(v1.get(tkr, np.nan)),
                    "vol_3m": float(v3.get(tkr, np.nan)),
                    "reversal_1w": float(rev5.get(tkr, np.nan)),
                    "amihud_1m": float(am_mean.get(tkr, np.nan)),
                }
                row_out[label_col_name] = float(fwd_val.get(tkr, np.nan))
                rows.append(row_out)

        tech_df = pd.DataFrame(rows)
        tech_df = tech_df.drop_duplicates(subset=["month_end_date", "ticker"], keep="last")

        q_frames = []
        for path in args.quarterly_panels:
            if path.is_file():
                q_frames.append(pd.read_parquet(path))
        if not q_frames:
            log.error("no quarterly panel parquet found")
            return 1

        q_panel = pd.concat(q_frames, axis=0)
        if not isinstance(q_panel.index, pd.MultiIndex):
            q_panel = q_panel.set_index(["as_of", "ticker"]).sort_index()
        q_panel.index.set_names(["as_of", "ticker"], inplace=True)
        q_panel = q_panel[~q_panel.index.duplicated(keep="last")]

        want_cols = _quarter_fundamental_like_columns()
        quarter_cols = [
            c for c in want_cols if c in q_panel.columns and c not in TECH_NAMES_QUARTERLY
        ]
        merge_cols = list(quarter_cols)
        if args.neutralize and args.industry_col in q_panel.columns:
            merge_cols.append(args.industry_col)

        q_long = q_panel.reset_index()
        q_long["as_of"] = pd.to_datetime(q_long["as_of"]).dt.normalize()

        merged = _merge_asof_by_ticker(tech_df, q_long, merge_cols)

        feature_cols = MONTHLY_TECH + [c for c in quarter_cols if c in merged.columns]
        label_col = "forward_return_1m" if args.label_method == "next_month" else "forward_return_21d"

        panel = merged.set_index(["month_end_date", "ticker"]).sort_index()
        panel.index.set_names(["as_of", "ticker"], inplace=True)
        ordered = feature_cols + [label_col]
        extra_ind = [args.industry_col] if args.neutralize else []
        build_cols = feature_cols + extra_ind + [label_col]
        for c in build_cols:
            if c not in panel.columns:
                panel[c] = np.nan
        panel = panel[[c for c in build_cols if c in panel.columns]]

        if args.neutralize and args.industry_col in panel.columns:
            parts: list[pd.DataFrame] = []
            for dt in sorted(panel.index.get_level_values(0).unique()):
                sub = panel.xs(dt, level=0).copy()
                work = sub.rename(columns={args.industry_col: "industry"})
                neu = neutralize_cross_section(
                    work,
                    industry_col="industry",
                    factors=[c for c in MONTHLY_TECH if c in work.columns],
                    neutralize_mcap=False,
                ).rename(columns={"industry": args.industry_col})
                neu.index.name = "ticker"
                neu = neu.reset_index()
                neu["as_of"] = dt
                parts.append(neu.set_index(["as_of", "ticker"]))
            panel = pd.concat(parts).sort_index()
            panel = panel.drop(columns=[args.industry_col], errors="ignore")

        panel = panel[[c for c in ordered if c in panel.columns]]
        for c in ordered:
            if c not in panel.columns:
                panel[c] = np.nan
        panel = panel[ordered]
        if panel.index.duplicated().any():
            log.warning("dropping {} duplicate panel rows", int(panel.index.duplicated().sum()))
            panel = panel[~panel.index.duplicated(keep="last")]
        panel.to_parquet(args.output)

        meta_out = {
            "panel_file": str(args.output.resolve()),
            "label_column": label_col,
            "feature_columns": feature_cols,
            "monthly_technical": MONTHLY_TECH,
            "quarterly_forward_filled_columns": quarter_cols,
        }
        split_suffix = _resolve_split_suffix(args.output, args.split_suffix)
        meta_path = args.output.parent / f"monthly_split_meta{split_suffix}.json"
        meta_path.write_text(json.dumps(meta_out, indent=2, ensure_ascii=False), encoding="utf-8")

        idx = panel.index.get_level_values("as_of")
        train = panel.loc[(idx.year >= 2019) & (idx.year <= 2022)]
        val = panel.loc[idx.year == 2023]
        test = panel.loc[idx.year == 2024]

        suf = split_suffix
        train.to_parquet(args.output.parent / f"monthly_train{suf}.parquet")
        val.to_parquet(args.output.parent / f"monthly_val{suf}.parquet")
        test.to_parquet(args.output.parent / f"monthly_test{suf}.parquet")

        lab_series = panel[label_col]
        nan_ratio = float(lab_series.isna().mean()) if len(lab_series) else 1.0
        lab_clean = lab_series.dropna()
        lab_mean = float(lab_clean.mean()) if len(lab_clean) else float("nan")
        lab_std = float(lab_clean.std()) if len(lab_clean) else float("nan")
        print(f"monthly_panel rows={len(panel)} cols={len(panel.columns)} → {args.output}")
        print(f"train={len(train)} val={len(val)} test={len(test)}")
        print(f"split_meta → {meta_path}")
        print(
            f"label [{label_col}]: mean={lab_mean:.4f} ({lab_mean*100:.2f}%) "
            f"std={lab_std:.4f} ({lab_std*100:.2f}%) nan_ratio={nan_ratio:.2%}"
        )
        return 0
    finally:
        sys.stdout = old_stdout
        tee_f.close()


if __name__ == "__main__":
    raise SystemExit(main())
