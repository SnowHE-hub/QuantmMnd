"""T-052 验证 2：模拟器的研究口径（mode = adj）与冻结的 ExecutableNavEngine 在 W6 上对照（ENS20 前 300、契约调仓）.

判定（登记）：逐日净值收益差的平均绝对值 ≤ 1 bp，且区间收益差 ≤ 0.3 个百分点。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_data as sd  # noqa: E402

from quantmind.backtest.wf_costs import SlippageTiers  # noqa: E402
from quantmind.execution.nav_engine import ExecutableNavEngine, NavConfig  # noqa: E402
from quantmind.line_a.sim_account import AccountSimulator, SimConfig  # noqa: E402

W6 = (pd.Timestamp("2026-04-23"), pd.Timestamp("2026-10-23"))
OUT = sd.ROOT / "data/line_a_v2/t052_sim"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    start, end = W6
    sc, info = sd.ens_scores("20d", start)
    px = sd.prices(start)
    ami = sd.amihud(start)
    tiers = SlippageTiers()
    eng = ExecutableNavEngine(sc, px, NavConfig(horizon_td=20, rebalance_step=4, top_n=300, universe_size=1500,
                                                commission_bp=3.0, transfer_bp=0.2, tiers=tiers), amihud=ami).run()
    en = eng["nav_daily"].set_index("date")
    cfg = SimConfig(aum=1.0, top_n=300, rebalance_step=4, universe_size=1500, buffer_mult=None, drift_band=0.0,
                    commission_rate=3e-4, min_commission=0.0, transfer_rate=2e-5, tiers=tiers, participation=None,
                    mode="adj", dividend_tax=False)  # drift_band=0.0：每次调平，与引擎相同
    sim = AccountSimulator(sc, px, cfg, start=start, end=end, amihud=ami).run()
    sn = sim["nav"]
    j = en.join(sn, how="inner", rsuffix="_sim")
    re, rs = j["nav_net"].pct_change().dropna(), j["equity"].pct_change().dropna()
    mad = float((re - rs).abs().mean() * 1e4)
    tot_e, tot_s = float(j["nav_net"].iloc[-1] / j["nav_net"].iloc[0] - 1), float(j["equity"].iloc[-1] - 1)
    tot_e_full = float(en["nav_net"].iloc[-1] - 1)
    res = {"days": int(len(j)), "mean_abs_daily_diff_bp": mad, "engine_total": tot_e_full, "sim_total": tot_s,
           "total_diff_pp": (tot_s - tot_e_full) * 100, "bench_engine": float(en["nav_bench"].iloc[-1] - 1),
           "bench_sim": float(sn["bench_pool_ew"].iloc[-1] - 1), "pass": bool(mad <= 1.0 and abs(tot_s - tot_e_full) * 100 <= 0.3),
           "engine_trades": int(len(eng["trades"])), "sim_trades": int(sim["summary"]["n_trades"]),
           "max_daily_diff_bp": float((re - rs).abs().max() * 1e4), "signal_info": info}
    worst = (re - rs).abs().sort_values(ascending=False).head(5)
    res["worst_days"] = {str(k.date()): float(v * 1e4) for k, v in worst.items()}
    (OUT / "validate_engine.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "signal_info"}, ensure_ascii=False, indent=1, default=str))
    print("signal:", json.dumps({k: info[k] for k in ("hist_dates", "forward_dates", "forward_missing",
                                                      "dates_without_scores")}))
    _ = np


if __name__ == "__main__":
    main()
