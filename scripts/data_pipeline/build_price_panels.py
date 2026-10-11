"""scripts/data_pipeline/build_price_panels.py — 由 data/lake 生成带日期的价格面板新文件并核验（T-015）.

不覆盖任何已有文件（AGENTS.md §3.6；p2_build_v6 对 v5 价格文件有 md5 守卫）。输出：
  data/raw/alpha_prices_panel_v6_<END>.parquet   与 p2_build_v6 --prices 同口径：全市场（lake 全部代码），adj_close = close × adj_factor（后复权）
  data/raw/alpha_prices_panel_<END>.parquet      与原 alpha_prices_panel 同 schema、同口径：1374 票池，
                                                 adj_close = close × adj_factor ÷ 该票最新 adj_factor（前复权，基准为本次数据的最后一天）
  data/raw/price_panels_<END>.validation.json    核验结果

核验：
  1. 交易日与 data/lake/trade_cal_sse.parquet 在区间内完全一致（无缺日、无多余日）
  2. 与旧文件重叠的 (ts_code, trade_date)：原始列逐值相等；v6 的 adj_close 逐值相等；v5 的 adj_close 日收益相等（前复权基准不同）
  3. pct_chg 与 close/pre_close 自洽的占比
  4. 主键无重复；adj_factor 无缺失

用法：python scripts/data_pipeline/build_price_panels.py --end 2026-10-09 [--data-root ~/projects/quantmind]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW_COLS = ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount", "adj_factor"]
COLS = ["ts_code", "trade_date", *RAW_COLS, "adj_close"]


def load_lake_prices(root: Path, end: pd.Timestamp) -> pd.DataFrame:
    daily = pd.read_parquet(root / "data/lake/daily.parquet")
    adj = pd.read_parquet(root / "data/lake/adj_factor.parquet", columns=["ts_code", "trade_date", "adj_factor"])
    df = daily.merge(adj, on=["ts_code", "trade_date"], how="left")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df[df["trade_date"] <= end]


def build_v6(df: pd.DataFrame) -> pd.DataFrame:
    if df["adj_factor"].isna().any():
        bad = df.loc[df["adj_factor"].isna(), ["ts_code", "trade_date"]].head(5).to_dict("records")
        raise SystemExit(f"adj_factor 缺口 {int(df['adj_factor'].isna().sum())} 行，例 {bad}")
    out = df.copy()
    out["adj_close"] = out["close"] * out["adj_factor"]
    return out[COLS].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def build_v5(v6: pd.DataFrame, universe: set[str]) -> pd.DataFrame:
    out = v6[v6["ts_code"].isin(universe)].copy()
    last_af = out.sort_values("trade_date").groupby("ts_code")["adj_factor"].transform("last")
    out["adj_close"] = out["close"] * out["adj_factor"] / last_af
    return out[COLS].sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def validate(new: pd.DataFrame, old_path: Path, cal: pd.DatetimeIndex, adj_mode: str) -> dict:
    rep: dict = {}
    dates = pd.DatetimeIndex(sorted(new["trade_date"].unique()))
    want = cal[(cal >= dates[0]) & (cal <= dates[-1])]
    rep["n_dates"] = int(len(dates))
    rep["first"], rep["last"] = str(dates[0].date()), str(dates[-1].date())
    rep["missing_vs_trade_cal"] = [str(d.date()) for d in want.difference(dates)]
    rep["extra_vs_trade_cal"] = [str(d.date()) for d in dates.difference(want)]
    rep["duplicate_keys"] = int(new.duplicated(["ts_code", "trade_date"]).sum())
    rep["adj_factor_na"] = int(new["adj_factor"].isna().sum())
    ok = new["pre_close"] > 0
    implied = (new.loc[ok, "close"] / new.loc[ok, "pre_close"] - 1) * 100
    rep["pct_chg_consistent_frac"] = float(((implied - new.loc[ok, "pct_chg"]).abs() < 0.01).mean())
    rep["n_rows"], rep["n_tickers"] = int(len(new)), int(new["ts_code"].nunique())

    old = pd.read_parquet(old_path, columns=COLS)
    old["trade_date"] = pd.to_datetime(old["trade_date"])
    m = old.merge(new, on=["ts_code", "trade_date"], how="left", suffixes=("_old", "_new"), indicator=True)
    rep["old_rows"] = int(len(old))
    rep["old_rows_missing_in_new"] = int((m["_merge"] == "left_only").sum())
    both = m[m["_merge"] == "both"]
    diffs = {}
    for c in RAW_COLS:
        a, b = both[f"{c}_old"], both[f"{c}_new"]
        neq = ~((a == b) | (a.isna() & b.isna()))
        diffs[c] = int(neq.sum())
    if adj_mode == "exact":
        a, b = both["adj_close_old"], both["adj_close_new"]
        diffs["adj_close"] = int((~((a - b).abs() < 1e-9 * a.abs().clip(lower=1))).sum())
    else:  # 前复权基准不同：比较日收益
        s = both.sort_values(["ts_code", "trade_date"])
        r_old = s.groupby("ts_code")["adj_close_old"].pct_change()
        r_new = s.groupby("ts_code")["adj_close_new"].pct_change()
        diffs["adj_close_daily_return"] = int(((r_old - r_new).abs() > 1e-9).sum())
    rep["overlap_value_mismatches"] = diffs
    rep["new_rows_beyond_old"] = int(len(new) - len(both))
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--end", required=True)
    ap.add_argument("--data-root", type=Path, default=REPO)
    a = ap.parse_args(argv)
    root, end = a.data_root.resolve(), pd.Timestamp(a.end)
    tag = end.strftime("%Y%m%d")
    out6 = root / f"data/raw/alpha_prices_panel_v6_{tag}.parquet"
    out5 = root / f"data/raw/alpha_prices_panel_{tag}.parquet"
    rep_path = root / f"data/raw/price_panels_{tag}.validation.json"
    for p in (out6, out5):
        if p.exists():
            print(f"refusing to overwrite {p}")
            return 1
    cal = pd.DatetimeIndex(pd.read_parquet(root / "data/lake/trade_cal_sse.parquet")["cal_date"])
    universe = {ln.strip() for ln in (root / "data/alpha_universe/alpha_universe.txt").read_text().splitlines()
                if ln.strip()}

    v6 = build_v6(load_lake_prices(root, end))
    v5 = build_v5(v6, universe)
    report = {
        "end": a.end,
        "v6": validate(v6, root / "data/raw/alpha_prices_panel_v6.parquet", cal, "exact"),
        "v5": validate(v5, root / "data/raw/alpha_prices_panel.parquet", cal, "returns"),
        "v5_universe_size": len(universe),
        "v5_universe_tickers_with_rows": int(v5["ts_code"].nunique()),
    }
    report["v5"]["tickers_with_bar_on_last_day"] = int((v5["trade_date"] == v5["trade_date"].max()).sum())
    v6.to_parquet(out6, index=False)
    v5.to_parquet(out5, index=False)
    rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"wrote {out6}\nwrote {out5}\nwrote {rep_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
