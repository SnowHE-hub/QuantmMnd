"""T-018：Alpha158 v6_1 核验（只读）。新旧两版在共同 (as_of, ticker) 上逐列比较，按"as_of 前 60 个交易日内是否含旧价格表缺失的日子"分组。
预期：窗口不含缺失日的位置应几乎相同（同一价格、同一 qlib 表达式），含缺失日的位置才有差异。
用法：python validate_alpha158_v6_1.py <data_root> <out.json>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root, out = Path(sys.argv[1]), Path(sys.argv[2])
D = root / "data"
new = pd.read_parquet(D / "bakeoff/alpha158_asof_v6_1.parquet")
old = pd.read_parquet(D / "bakeoff/alpha158_asof_v6.parquet")
cal = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(D / "lake/trade_cal_sse.parquet")["cal_date"])).sort_values()
old_days = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(D / "raw/alpha_prices_panel_v6.parquet",
                                                           columns=["trade_date"])["trade_date"].unique()))
missing = cal[(cal >= old_days.min()) & (cal <= old_days.max())].difference(old_days)
common_asof = pd.DatetimeIndex(new.index.get_level_values(0).unique()).intersection(
    pd.DatetimeIndex(old.index.get_level_values(0).unique()))
cols = [c for c in old.columns if c in new.columns]
rows = []
for a in common_asof:
    pos = cal.searchsorted(a)
    touched = bool(len(cal[max(0, pos - 65):pos + 1].intersection(missing)))
    o, n = old.xs(a, level=0), new.xs(a, level=0)
    idx = o.index.intersection(n.index)
    x, y = o.loc[idx, cols].to_numpy(float), n.loc[idx, cols].to_numpy(float)
    same = (np.abs(x - y) <= 1e-6 * (1 + np.abs(x))) | (np.isnan(x) & np.isnan(y))
    rows.append({"as_of": a, "touched": touched, "cells": int(same.size), "diff": int((~same).sum())})
r = pd.DataFrame(rows)
res = {"new_shape": list(new.shape), "old_shape": list(old.shape), "columns_equal": list(new.columns) == list(old.columns),
       "new_as_of": [str(new.index.get_level_values(0).min().date()), str(new.index.get_level_values(0).max().date()),
                     int(new.index.get_level_values(0).nunique())],
       "new_nan_frac": round(float(new.isna().mean().mean()), 4), "old_nan_frac": round(float(old.isna().mean().mean()), 4),
       "common_as_of": int(len(common_asof)),
       "diff_share_untouched": float(r[~r.touched]["diff"].sum() / max(r[~r.touched]["cells"].sum(), 1)),
       "diff_share_touched": float(r[r.touched]["diff"].sum() / max(r[r.touched]["cells"].sum(), 1)),
       "n_as_of_touched": int(r.touched.sum())}
out.write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))
