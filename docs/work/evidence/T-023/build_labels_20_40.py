"""T-023：在 v6_1 网格上加 20d / 40d 标签（与 v6_1 面板同一 delist-aware 函数、同一价格表与日历）.

同时用同一函数重算 12 / 21 / 63d，与 alpha_panel_weekly_v6_1.parquet 中的标签逐值比较——一致才说明口径相同。
输出 data/panel/labels_extra_v6_1.parquet（index = (as_of, ticker)，列 forward_return_20d / forward_return_40d），拒绝覆盖。
用法：PYTHONPATH=. python build_labels_20_40.py <data_root> <out_json>
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

from scripts.survivorship.p2_build_v6 import compute_forward_returns_delist_aware

root, out_json = Path(sys.argv[1]), Path(sys.argv[2])
D = root / "data"
out = D / "panel/labels_extra_v6_1.parquet"
if out.exists():
    raise SystemExit(f"refusing to overwrite {out}")
t0 = time.time()
panel = pd.read_parquet(D / "panel/alpha_panel_weekly_v6_1.parquet",
                        columns=["forward_return_12d", "forward_return_21d", "forward_return_63d"])
px = pd.read_parquet(D / "raw/alpha_prices_panel_v6_20261009.parquet", columns=["ts_code", "trade_date", "adj_close"])
px["trade_date"] = pd.to_datetime(px["trade_date"])
sub = px.dropna(subset=["adj_close"]).rename(columns={"ts_code": "ticker"})
adj = (sub.drop_duplicates(["trade_date", "ticker"], keep="last")
       .pivot(index="trade_date", columns="ticker", values="adj_close").sort_index())
ffill = adj.ffill()
last_trade = adj.apply(lambda s: s.last_valid_index())
sb = pd.read_parquet(D / "lake/stock_basic_full_merged_20261009.parquet")
sb = sb.rename(columns={"ts_code": "ticker"}) if "ticker" not in sb.columns else sb
dmap = sb.drop_duplicates("ticker").set_index("ticker")
has_delist = pd.Series(False, index=adj.columns)
common = has_delist.index.intersection(dmap.index)
has_delist.loc[common] = ((dmap.loc[common, "list_status"] == "D") & dmap.loc[common, "delist_date"].notna()).values
print(f"prep {adj.shape} ({time.time() - t0:.0f}s)", flush=True)

H = (12, 20, 21, 40, 63)
rows = []
as_ofs = panel.index.get_level_values("as_of").unique().sort_values()
for i, a in enumerate(as_ofs):
    fr = compute_forward_returns_delist_aware(adj, ffill, a, H, last_trade, has_delist)
    tick = panel.xs(a, level="as_of").index
    fr = fr.reindex(tick)
    fr.index = pd.MultiIndex.from_product([[a], fr.index], names=["as_of", "ticker"])
    rows.append(fr)
    if i % 50 == 0:
        print(f"  {i}/{len(as_ofs)} {a.date()} ({time.time() - t0:.0f}s)", flush=True)
lab = pd.concat(rows)
res = {"n_rows": int(len(lab)), "n_as_of": int(len(as_ofs)), "check_vs_panel": {}}
for h in (12, 21, 63):
    a, b = lab[f"forward_return_{h}d"], panel[f"forward_return_{h}d"].reindex(lab.index)
    both = a.notna() & b.notna()
    res["check_vs_panel"][f"{h}d"] = {"compared": int(both.sum()), "max_abs_diff": float((a[both] - b[both]).abs().max()),
                                      "nan_mismatch": int((a.isna() != b.isna()).sum())}
res["coverage"] = {f"{h}d": round(float(lab[f"forward_return_{h}d"].notna().mean()), 4) for h in H}
ok = all(v["max_abs_diff"] < 1e-12 and v["nan_mismatch"] == 0 for v in res["check_vs_panel"].values())
res["consistent_with_panel"] = ok
if not ok:
    out_json.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    raise SystemExit("recomputed 12/21/63d labels differ from the panel; not writing")
lab[["forward_return_20d", "forward_return_40d"]].to_parquet(out)
res["output"] = str(out)
out_json.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
