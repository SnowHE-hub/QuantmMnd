"""T-017 产物核验（只读）：alpha_panel_weekly_v6_1.parquet.

1 网格：as_of = build_asof_grid(trade_cal 在价格区间内的部分)
2 标签：独立写法——目标日 = trade_cal 上基准日后第 h 个交易日，label = adj_close(目标)/adj_close(基准) − 1；
  只比较未退市、基准日与目标日都有价的票（delist-aware 分支另由单元测试覆盖）；抽 40 个截面
3 因子：与旧 v6 在两版都有的 as_of 上逐列比较；按"该 as_of 前 260 个交易日内是否含旧价格表缺失的日子"分组
4 概况：形状、as_of 区间、票数、各列非空率（新 vs 旧）
用法：PYTHONPATH=. python validate_v6_1.py <data_root> <out.json>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quantmind.features.weekly_panel import build_asof_grid

root, out = Path(sys.argv[1]), Path(sys.argv[2])
D = root / "data"
new = pd.read_parquet(D / "panel/alpha_panel_weekly_v6_1.parquet")
old = pd.read_parquet(D / "panel/alpha_panel_weekly_v6.parquet")
px = pd.read_parquet(D / "raw/alpha_prices_panel_v6_20261009.parquet", columns=["ts_code", "trade_date", "adj_close"])
px["trade_date"] = pd.to_datetime(px["trade_date"])
cal = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(D / "lake/trade_cal_sse.parquet")["cal_date"])).sort_values()
cal = cal[(cal >= px.trade_date.min()) & (cal <= px.trade_date.max())]
res = {}

# 1 网格
grid = pd.DatetimeIndex(build_asof_grid(cal))
got = pd.DatetimeIndex(sorted(new.index.get_level_values("as_of").unique()))
res["grid"] = {"expected": len(grid), "got": len(got), "equal": bool(got.equals(grid)),
               "panel_dates_not_on_grid": int(len(got.difference(grid))),
               "first": str(got.min().date()), "last": str(got.max().date()),
               "missing_from_panel": [str(d.date()) for d in grid.difference(got)][:10]}

# 2 标签（独立写法）
adj = px.drop_duplicates(["trade_date", "ts_code"]).pivot(index="trade_date", columns="ts_code", values="adj_close")
adj = adj.reindex(cal)
sb = pd.read_parquet(D / "lake/stock_basic_full_merged_20261009.parquet")
delisted = set(sb.loc[sb["list_status"] == "D", "ts_code"].astype(str))
rng = np.random.default_rng(17)
sample = sorted(rng.choice(got, size=min(40, len(got)), replace=False))
lab = {}
for h in (12, 21, 63):
    col, n_cmp, bad = f"forward_return_{h}d", 0, 0
    for a in sample:
        bi = cal.get_indexer([a])[0]
        if bi < 0 or bi + h >= len(cal):
            continue
        base, tgt = adj.iloc[bi], adj.iloc[bi + h]
        ind = (tgt / base - 1).dropna()
        ind = ind[~ind.index.isin(delisted)]
        st = new.xs(a, level="as_of")[col].reindex(ind.index).dropna()
        d = (st - ind.reindex(st.index)).abs()
        n_cmp += len(d)
        bad += int((d > 1e-9).sum())
    lab[col] = {"compared": n_cmp, "mismatch_gt_1e-9": bad}
res["labels_independent"] = lab | {"n_as_of_sampled": len(sample)}

# 3 因子 vs 旧 v6
old_px_dates = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(D / "raw/alpha_prices_panel_v6.parquet",
                                                               columns=["trade_date"])["trade_date"].unique()))
missing = cal[cal <= old_px_dates.max()].difference(old_px_dates)
common_asof = got.intersection(pd.DatetimeIndex(old.index.get_level_values("as_of").unique()))
factors = [c for c in old.columns if c in new.columns and not c.startswith("forward_return")]
rows = []
for a in common_asof:
    pos = cal.get_indexer([a])[0]
    window = cal[max(0, pos - 260):pos + 1]
    touched = bool(len(window.intersection(missing)))
    o, n = old.xs(a, level="as_of"), new.xs(a, level="as_of")
    idx = o.index.intersection(n.index)
    for c in factors:
        if c in ("exposure_industry", "exposure_area"):
            same = (o.loc[idx, c].astype(object) == n.loc[idx, c].astype(object)) | (o.loc[idx, c].isna() & n.loc[idx, c].isna())
        else:
            x, y = o.loc[idx, c].astype(float), n.loc[idx, c].astype(float)
            same = ((x - y).abs() <= 1e-9 * (1 + x.abs())) | (x.isna() & y.isna())
        rows.append({"as_of": a, "touched": touched, "factor": c, "n": len(idx), "diff": int((~same).sum())})
f = pd.DataFrame(rows)
res["factors_vs_old"] = {
    "common_as_of": int(len(common_asof)), "old_as_of": int(old.index.get_level_values("as_of").nunique()),
    "missing_days_in_old_prices": [d.strftime("%Y%m%d") for d in missing],
    "diff_share_untouched_windows": float(f[~f.touched]["diff"].sum() / max(f[~f.touched]["n"].sum(), 1)),
    "diff_share_touched_windows": float(f[f.touched]["diff"].sum() / max(f[f.touched]["n"].sum(), 1)),
    "worst_factors_untouched": f[~f.touched].groupby("factor")["diff"].sum().sort_values(ascending=False).head(8).to_dict(),
}

# 4 概况
res["shape"] = {"new": list(new.shape), "old": list(old.shape),
                "new_tickers": int(new.index.get_level_values("ticker").nunique()),
                "old_tickers": int(old.index.get_level_values("ticker").nunique()),
                "columns_equal": list(new.columns) == list(old.columns),
                "nonnull_new": {c: round(float(new[c].notna().mean()), 4) for c in new.columns},
                "nonnull_old": {c: round(float(old[c].notna().mean()), 4) for c in old.columns}}
out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "shape"}, ensure_ascii=False, indent=1, default=str))
print({k: res["shape"][k] for k in ("new", "old", "new_tickers", "old_tickers", "columns_equal")})
