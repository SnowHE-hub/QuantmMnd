"""T-052 事后稳健性检查（登记里没有，结果出来之后为了解释"每 4 期调仓明显更好"而加，如实标注）.

同样"每 4 个周频网格调一次仓"，起点分别推后 0 / 1 / 2 / 3 个网格点（错开调仓相位），三档资金的主配置各跑一遍；
再看调仓周期 1 / 2 期时的结果。相位之间差别很大，就说明"每 4 期更好"主要是调仓日碰巧赶上，而不是规律。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_data as sd  # noqa: E402
from sim_grid import ALL, SMALL  # noqa: E402

from quantmind.line_a.sim_account import AccountSimulator, Market, SimConfig  # noqa: E402

END = pd.Timestamp("2026-10-23")


def main() -> None:
    start = pd.Timestamp("2026-04-23")
    px = sd.prices(start)
    m = Market(px, start, END)
    sc, info = sd.ens_scores("20d", start)
    ami, dv, st = sd.amihud(start), sd.dividends(), sd.st_intervals()
    grid = sorted(sc["as_of"].unique())
    rows = []
    for aum, n in ((1e5, 10), (1e6, 30), (1e7, 100)):
        for step in (1, 2, 4):
            for off in range(step):
                s0 = pd.Timestamp(grid[off])
                cfg = SimConfig(aum=aum, top_n=n, rebalance_step=step, boards=SMALL if aum < 5e5 else ALL)
                r = AccountSimulator(sc, m, cfg, start=s0, end=END, amihud=ami, dividends=dv, st_intervals=st).run()
                s = r["summary"]
                adj = s["total_return"] - (1 - s["avg_cash_share"]) * s["bench_pool_ew_return"]
                rows.append({"aum": aum, "n": n, "step": step, "offset": off, "start": str(s0.date()),
                             "total_return": s["total_return"], "bench": s["bench_pool_ew_return"],
                             "excess_exposure_adj": adj, "turnover_x": s["turnover_x"],
                             "cost_share": s["cost_share_of_aum"], "avg_cash": s["avg_cash_share"]})
                print(json.dumps(rows[-1]), flush=True)
    pd.DataFrame(rows).to_parquet(sd.ROOT / "data/line_a_v2/t052_sim/phase_check.parquet")


if __name__ == "__main__":
    main()
