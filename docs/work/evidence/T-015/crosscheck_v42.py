"""独立核验 alpha_panel_v4_2 标签：用 alpha_prices_panel_20261009（close×adj_factor 后复权，交易日完整）重算."""
import sys

import pandas as pd

sys.path.insert(0, "/home/lenovo/projects/quantmind")
from quantmind.features.panel import compute_forward_returns  # noqa: E402

Q = "/home/lenovo/projects/quantmind"
px = pd.read_parquet(f"{Q}/data/raw/alpha_prices_panel_20261009.parquet", columns=["ts_code", "trade_date", "close", "adj_factor"])
px["hfq"] = px["close"] * px["adj_factor"]
wide = px.pivot(index="trade_date", columns="ts_code", values="hfq").sort_index()
for name in ("alpha_panel_v4_2", "alpha_panel_v4_1"):
    p = pd.read_parquet(f"{Q}/data/panel/{name}.parquet", columns=["forward_return_21d", "forward_return_63d"])
    rows = []
    for a in sorted(p.index.get_level_values(0).unique()):
        fr = compute_forward_returns(wide, a.date(), (21, 63))
        if fr.empty:
            continue
        x = p.xs(a, level=0)
        for h in (21, 63):
            c = f"forward_return_{h}d"
            d = pd.DataFrame({"s": x[c], "t": fr[c].reindex(x.index)}).dropna()
            if len(d) < 5:
                continue
            rows.append({"h": h, "as_of": a.date(), "n": len(d), "bad": int(((d.s - d.t).abs() > 0.005).sum()),
                         "sp": d.s.rank().corr(d.t.rank())})
    r = pd.DataFrame(rows)
    for h in (21, 63):
        s = r[r.h == h]
        print(f"{name} h={h}: 截面 {len(s)} 样本 {s.n.sum()} 偏差>0.5% {s.bad.sum()} ({s.bad.sum() / s.n.sum():.3%}) "
              f"Spearman 最低 {s.sp.min():.4f}")
