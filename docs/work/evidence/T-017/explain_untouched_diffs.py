"""解释 v6_1 核验里"价格未缺日窗口"中的因子差异（只读）。

假设：差异集中在依赖指数序列的因子（beta、相对强弱、市场动量 / 波动），原因是旧 index_daily 本身也缺日
（含 2019-01～02），而核验脚本的"touched"只按旧价格表的缺日划分。
做法：列出旧 index_daily（_backup_20261010）各指数相对 trade_cal 的缺日；把共同 as_of 上的差异按
"该 as_of 前 LOOKBACK 个交易日内是否含旧指数缺日或旧价格缺日"重新分组（默认 300：beta_252d 对齐后实际回看超过 252）。
用法：PYTHONPATH=. python explain_untouched_diffs.py <data_root> <out.json> [LOOKBACK]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root, out = Path(sys.argv[1]), Path(sys.argv[2])
D = root / "data"
cal = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(D / "lake/trade_cal_sse.parquet")["cal_date"].astype(str))).sort_values()
old_idx = pd.read_parquet(D / "lake/_backup_20261010/index_daily.parquet", columns=["ts_code", "trade_date"])
old_idx["trade_date"] = pd.to_datetime(old_idx["trade_date"].astype(str))
cal_old = cal[cal <= old_idx["trade_date"].max()]
missing = {c: sorted(set(cal_old) - set(g["trade_date"])) for c, g in old_idx.groupby("ts_code")}
res = {"old_index_missing_days": {c: {"n": len(v), "first": str(v[0].date()) if v else None,
                                      "last": str(v[-1].date()) if v else None} for c, v in missing.items()}}
idx_missing = set().union(*missing.values())

cols = ["beta_252d", "beta_60d", "market_volatility_60d", "relative_strength_vs_csi300_60d",
        "relative_strength_vs_csi500_60d", "market_momentum_60d"]
new = pd.read_parquet(D / "panel/alpha_panel_weekly_v6_1.parquet", columns=cols).reset_index()  # 索引为 (as_of, ticker)
old = pd.read_parquet(D / "panel/alpha_panel_weekly_v6.parquet", columns=cols).reset_index()
common = sorted(set(new.as_of) & set(old.as_of))
m = new[new.as_of.isin(common)].merge(old[old.as_of.isin(common)], on=["ticker", "as_of"], suffixes=("_n", "_o"))
PRICE_MISSING = {pd.Timestamp(x) for x in (
    "2021-03-05", "2021-03-08", "2021-03-09", "2021-03-10", "2021-03-11", "2021-03-15", "2021-06-10",
    "2023-03-16", "2023-05-31", "2023-06-01", "2023-06-06", "2023-06-07", "2023-06-16", "2023-06-20",
    "2023-09-04", "2023-10-20", "2026-03-03", "2026-03-04")}
LOOKBACK = int(sys.argv[3]) if len(sys.argv) > 3 else 300   # 交易日；beta_252d 对齐后实际回看可超过 252
rows = []
for a, g in m.groupby("as_of"):
    pos = cal.searchsorted(pd.Timestamp(a), side="right")
    window = set(cal[max(0, pos - LOOKBACK):pos])
    touched_idx = bool(window & (idx_missing | PRICE_MISSING))
    for c in cols:
        x, y = g[f"{c}_n"].to_numpy(float), g[f"{c}_o"].to_numpy(float)
        both = ~(np.isnan(x) & np.isnan(y))
        same = np.isclose(x, y, rtol=1e-9, atol=1e-12, equal_nan=True)
        rows.append({"as_of": str(pd.Timestamp(a).date()), "touched_any_gap": touched_idx, "factor": c,
                     "n": int(both.sum()), "diff": int((~same & both).sum())})
f = pd.DataFrame(rows)
by = f.groupby("touched_any_gap")[["n", "diff"]].sum()
res["diff_share_by_touched_any_gap"] = {str(k): round(v["diff"] / max(v["n"], 1), 6) for k, v in by.iterrows()}
res["diff_counts_by_touched_any_gap"] = {str(k): {"n": int(v["n"]), "diff": int(v["diff"])} for k, v in by.iterrows()}
res["as_of_with_diff_not_touching_any_gap"] = sorted(f[(~f.touched_any_gap) & (f["diff"] > 0)]["as_of"].unique().tolist())[:20]
out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
