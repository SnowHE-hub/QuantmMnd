"""主线 B 现有样本外预测的 RankIC：旧 v6 标签 vs 完整交易日历重算的标签（同一 delist-aware 规则、同一截止日 2026-05-11）。只读。"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/lenovo/projects/quantmind")
from scripts.survivorship.p2_build_v6 import compute_forward_returns_delist_aware  # noqa: E402

Q = "/home/lenovo/projects/quantmind/data"
CUTOFF = pd.Timestamp("2026-05-11")
px = pd.read_parquet(f"{Q}/raw/alpha_prices_panel_v6_20261009.parquet", columns=["ts_code", "trade_date", "adj_close"])
px["trade_date"] = pd.to_datetime(px["trade_date"])
px = px[px["trade_date"] <= CUTOFF].dropna(subset=["adj_close"])
adj = (px.drop_duplicates(["trade_date", "ts_code"], keep="last")
       .pivot(index="trade_date", columns="ts_code", values="adj_close").sort_index())
ffill = adj.ffill()
last_trade = adj.apply(lambda s: s.last_valid_index())
sb = pd.read_parquet(f"{Q}/lake/stock_basic_full.parquet")
dmap = sb.set_index(sb["ts_code"].astype(str))
has_delist = pd.Series(False, index=adj.columns)
common = has_delist.index.intersection(dmap.index)
has_delist.loc[common] = ((dmap.loc[common, "list_status"] == "D") & dmap.loc[common, "delist_date"].notna()).values

out = {}
for name, h, path in [("ridge_full_fnd_63d_v6_seed", 63, "ridge_full_fnd_63d_quarterly_v6.parquet"),
                      ("ridge_full_12d_v6_seed", 12, "ridge_full_12d_quarterly_v6.parquet")]:
    preds = pd.read_parquet(f"{Q}/bakeoff/preds/{path}")
    preds["as_of"] = pd.to_datetime(preds["as_of"])
    col = f"forward_return_{h}d"
    old = pd.read_parquet(f"{Q}/panel/alpha_panel_weekly_v6.parquet", columns=[col])[col]
    rows = []
    for a, g in preds.groupby("as_of"):
        s = g.set_index("ticker")["score"]
        try:
            o = old.xs(a, level="as_of")
        except KeyError:
            continue
        n = compute_forward_returns_delist_aware(adj, ffill, a, (h,), last_trade, has_delist)[col]
        df = pd.DataFrame({"s": s, "old": o.reindex(s.index), "new": n.reindex(s.index)}).dropna()
        if len(df) < 50:
            continue
        rows.append({"as_of": a, "n": len(df), "ic_old": df["s"].rank().corr(df["old"].rank()),
                     "ic_new": df["s"].rank().corr(df["new"].rank()),
                     "label_bad05": int(((df["old"] - df["new"]).abs() > 0.005).sum())})
    r = pd.DataFrame(rows)
    aff = r[r.label_bad05 > 0]
    out[name] = {"n_as_of": int(len(r)), "ic_old_mean": float(r.ic_old.mean()), "ic_new_mean": float(r.ic_new.mean()),
                 "icir_old": float(r.ic_old.mean() / r.ic_old.std()), "icir_new": float(r.ic_new.mean() / r.ic_new.std()),
                 "n_as_of_with_label_changes": int(len(aff)),
                 "affected_ic_old_mean": float(aff.ic_old.mean()) if len(aff) else None,
                 "affected_ic_new_mean": float(aff.ic_new.mean()) if len(aff) else None,
                 "max_abs_ic_change": float((r.ic_new - r.ic_old).abs().max())}
    print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[name].items()}), flush=True)
json.dump(out, open("/home/lenovo/c3_scratch/b_ic_label_fix.json", "w"), indent=1)
