"""v6 周频面板标签审计（只读）：用 T-015 补齐 18 个交易日后的价格表、同一 delist-aware 规则、同样的数据截止日，
重算 alpha_panel_weekly_v6 现有网格上的 12d/21d/63d 标签，与存储值比较。两份价格表在重叠行上逐值相同（T-015 已验证），
因此差异只来自旧价格表缺失的 18 个交易日（horizon 按价格表自身的日期计数）。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/lenovo/projects/quantmind")
from scripts.survivorship.p2_build_v6 import compute_forward_returns_delist_aware  # noqa: E402

Q = "/home/lenovo/projects/quantmind/data"
H = (12, 21, 63)
old_px_dates = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(f"{Q}/raw/alpha_prices_panel_v6.parquet",
                                                               columns=["trade_date"])["trade_date"].unique()))
cutoff = old_px_dates.max()
px = pd.read_parquet(f"{Q}/raw/alpha_prices_panel_v6_20261009.parquet", columns=["ts_code", "trade_date", "adj_close"])
px["trade_date"] = pd.to_datetime(px["trade_date"])
px = px[px["trade_date"] <= cutoff].dropna(subset=["adj_close"])
adj = (px.drop_duplicates(["trade_date", "ts_code"], keep="last")
       .pivot(index="trade_date", columns="ts_code", values="adj_close").sort_index())
missing_days = adj.index.difference(old_px_dates)
print(f"cutoff {cutoff.date()}; complete calendar {len(adj.index)} days; old price table lacks {len(missing_days)}: "
      f"{[d.strftime('%Y%m%d') for d in missing_days]}")
ffill = adj.ffill()
last_trade = adj.apply(lambda s: s.last_valid_index())
sb = pd.read_parquet(f"{Q}/lake/stock_basic_full.parquet")
dmap = sb.set_index(sb["ts_code"].astype(str))
has_delist = pd.Series(False, index=adj.columns)
common = has_delist.index.intersection(dmap.index)
has_delist.loc[common] = ((dmap.loc[common, "list_status"] == "D") & dmap.loc[common, "delist_date"].notna()).values

panel = pd.read_parquet(f"{Q}/panel/alpha_panel_weekly_v6.parquet", columns=[f"forward_return_{h}d" for h in H])
as_ofs = sorted(panel.index.get_level_values("as_of").unique())
rows = []
for a in as_ofs:
    fr = compute_forward_returns_delist_aware(adj, ffill, pd.Timestamp(a), H, last_trade, has_delist)
    st = panel.xs(a, level="as_of")
    for h in H:
        c = f"forward_return_{h}d"
        s, n = st[c], fr[c].reindex(st.index) if c in fr else pd.Series(np.nan, index=st.index)
        both = s.notna() & n.notna()
        d = (s[both] - n[both]).abs()
        rows.append({"as_of": a, "h": h, "n_both": int(both.sum()), "bad05": int((d > 0.005).sum()),
                     "bad10": int((d > 0.10).sum()), "spearman": float(s[both].rank().corr(n[both].rank()))
                     if both.sum() > 5 else np.nan, "stored_only": int((s.notna() & n.isna()).sum()),
                     "new_only": int((s.isna() & n.notna()).sum())})
r = pd.DataFrame(rows)
summary = {}
for h in H:
    x = r[r.h == h]
    summary[f"{h}d"] = {"n_as_of": int(len(x)), "n_both": int(x.n_both.sum()), "bad05": int(x.bad05.sum()),
                        "bad05_frac": float(x.bad05.sum() / max(x.n_both.sum(), 1)), "bad10": int(x.bad10.sum()),
                        "as_of_with_bad05": int((x.bad05 > 0).sum()), "spearman_min": float(x.spearman.min()),
                        "stored_only": int(x.stored_only.sum()), "new_only": int(x.new_only.sum())}
print(json.dumps(summary, indent=1))
worst = r.sort_values("bad05", ascending=False).head(12)
print(worst.to_string(index=False))
out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
r.to_csv(out_dir / "v6_label_audit_by_asof.csv", index=False)
(out_dir / "v6_label_audit.json").write_text(json.dumps(
    {"cutoff": str(cutoff.date()), "missing_days": [d.strftime("%Y%m%d") for d in missing_days], "summary": summary},
    indent=1), encoding="utf-8")
