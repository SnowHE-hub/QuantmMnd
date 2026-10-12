"""T-025 补充：调仓相位敏感性（只读）.

同一批分数、同一起点（第一个决策日建仓），只改变之后调仓日落在周频网格的哪个相位：
  决策日 = [as_of[0]] + as_of[k::step]，k = 1 … step（k = step 即主结果的相位）。
对每个 B 策略、两种口径（研究口径 Top 300 零碎股；散户修正版 Top 20 一手买得起 + 整手），报告 1–9 月总收益与相对选股池等权的超额的分布。
用法：PYTHONPATH=. python phases_2026.py <data_root> <out.json>
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_p = Path(__file__).with_name("simulate_2026.py")
_spec = importlib.util.spec_from_file_location("sim2026", _p)
sim = importlib.util.module_from_spec(_spec)
sys.modules["sim2026"] = sim
_spec.loader.exec_module(sim)

from quantmind.execution.nav_engine import ExecutableNavEngine, NavConfig  # noqa: E402

root, out = Path(sys.argv[1]), Path(sys.argv[2])
D = root / "data"
cols = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "adj_factor", "amount"]
prices = pd.read_parquet(D / "raw/alpha_prices_panel_v6_20261009.parquet", columns=cols)
prices = prices[pd.to_datetime(prices["trade_date"]) <= sim.END]
am = pd.read_parquet(D / "panel/alpha_panel_weekly_v6_1.parquet", columns=["amihud_illiquidity"])
res = {}
for s, step in (("12d", 3), ("20d", 4), ("40d", 8), ("63d", 13)):
    p = pd.read_parquet(D / f"bakeoff/preds_v6_1/ridge_{s}_v6_1_C.parquet")
    p["as_of"] = pd.to_datetime(p["as_of"])
    p = p[(p.as_of >= sim.START) & (p.as_of <= sim.END)]
    dates = sorted(p.as_of.unique())
    for mode in ("research", "retail_affordable"):
        rows = []
        for k in range(1, step + 1):
            if mode == "research":
                eng = ExecutableNavEngine(p, prices, NavConfig(horizon_td=int(s[:-1]), rebalance_step=step, top_n=300,
                                                               universe_size=1500, initial_nav=sim.CAPITAL), amihud=am)
            else:
                eng = sim.RetailNavEngine(p, prices, NavConfig(horizon_td=int(s[:-1]), rebalance_step=step, top_n=sim.TOP_N,
                                                               universe_size=1500, initial_nav=sim.CAPITAL),
                                          amihud=am, drift_band=sim.DRIFT)
                eng.affordable = True
            eng.rebalance_dates = [pd.Timestamp(dates[0])] + [pd.Timestamp(x) for x in dates[k::step]]
            r = eng.run()
            nav = r["nav_daily"].set_index("date")
            nav = nav[nav.index <= sim.END]
            tot = float(nav["nav_net"].iloc[-1] / sim.CAPITAL - 1)
            ex = float((nav["nav_net"] / nav["nav_bench"]).iloc[-1] / (nav["nav_net"] / nav["nav_bench"]).iloc[0] - 1)
            rows.append({"k": k, "total_return": round(tot, 4), "excess_vs_pool": round(ex, 4),
                         "second_decision": str(pd.Timestamp(dates[k]).date()) if k < len(dates) else None})
            print(s, mode, rows[-1], flush=True)
        t = np.array([x["total_return"] for x in rows])
        e = np.array([x["excess_vs_pool"] for x in rows])
        res[f"B_{s}_{mode}"] = {"n_phases": len(rows), "total_min": float(t.min()), "total_median": float(np.median(t)),
                                "total_max": float(t.max()), "excess_min": float(e.min()),
                                "excess_median": float(np.median(e)), "excess_max": float(e.max()), "phases": rows}
out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: {x: v[x] for x in v if x != "phases"} for k, v in res.items()}, ensure_ascii=False, indent=1))
