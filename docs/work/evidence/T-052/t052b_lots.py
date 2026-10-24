"""T-052b：小账户的取整（floor / nearest）与买不起一手（skip / next），2 档 × 4 组合 × 2 窗口 × 4 相位（登记见 T-052.md）."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_data as sd  # noqa: E402
from sim_grid import SMALL, ALL  # noqa: E402

from quantmind.line_a.sim_account import AccountSimulator, Market, SimConfig  # noqa: E402

import os
WIN = {"W6": pd.Timestamp("2026-04-23"), "W3": pd.Timestamp("2026-07-23")}
if os.environ.get("T052B_LONG"):                                  # v2：2022-01 起的完整研究期
    WIN = {"L5": pd.Timestamp("2022-01-01")}
END = pd.Timestamp("2026-10-23")
VARIANTS = {"floor+skip": ("floor", False), "floor+next": ("floor", True), "nearest+skip": ("nearest", False),
            "nearest+next": ("nearest", True)}


def main() -> None:
    start = WIN["W6"]
    px = sd.prices(start)
    mk = Market(px, start, END)
    sc, _ = sd.ens_scores("20d", start)
    ami, dv, st = sd.amihud(start), sd.dividends(), sd.st_intervals()
    grid = sorted(sc["as_of"].unique())
    rows = []
    for aum, n, boards in ((1e5, 10, SMALL), (1e6, 30, ALL)):
        for w, ws in WIN.items():
            g = [d for d in grid if d >= ws]
            for off in range(4):
                s0 = pd.Timestamp(g[off])
                for vname, (lr, fill) in VARIANTS.items():
                    c = SimConfig(aum=aum, top_n=n, rebalance_step=4, boards=boards, lot_round=lr, fill_unaffordable=fill)
                    r = AccountSimulator(sc, mk, c, start=s0, end=END, amihud=ami, dividends=dv, st_intervals=st).run()
                    s = r["summary"]
                    rows.append({"aum": aum, "window": w, "offset": off, "variant": vname, "total_return": s["total_return"],
                                 "avg_cash": s["avg_cash_share"], "cost_share": s["cost_share_of_aum"],
                                 "bench": s["bench_pool_ew_return"], "n_cannot_afford": s["n_cannot_afford_one_lot"]})
                    print(json.dumps(rows[-1]), flush=True)
    df = pd.DataFrame(rows)
    tag = "_long" if os.environ.get("T052B_LONG") else ""
    out = sd.ROOT / f"data/line_a_v2/t052_sim/t052b_lots{tag}.parquet"
    df.to_parquet(out)
    base = df[df["variant"] == "floor+skip"].set_index(["aum", "window", "offset"])
    res = {}
    for v in VARIANTS:
        x = df[df["variant"] == v].set_index(["aum", "window", "offset"])
        d = x["total_return"] - base["total_return"]
        res[v] = {str(int(a)): {"avg_cash": float(x.loc[a, "avg_cash"].mean()), "cash_drop_pp": float((base.loc[a, "avg_cash"] - x.loc[a, "avg_cash"]).mean() * 100),
                                "ret_diff_median_pp": float(d.loc[a].median() * 100),
                                "ann_ret_diff_median_pp": float(((1 + x.loc[a, "total_return"]) ** (252 / 1150) - (1 + base.loc[a, "total_return"]) ** (252 / 1150)).median() * 100), "ret_diff_min_pp": float(d.loc[a].min() * 100),
                                "cost_increase_pp": float((x.loc[a, "cost_share"] - base.loc[a, "cost_share"]).mean() * 100)}
                  for a in (1e5, 1e6)}
        r5 = res[v]["100000"]
        key = "ann_ret_diff_median_pp" if tag else "ret_diff_median_pp"
        res[v]["adopt"] = bool(v != "floor+skip" and r5["cash_drop_pp"] >= 10 and r5[key] >= -0.5
                               and r5["cost_increase_pp"] <= 0.5)
    (sd.ROOT / f"data/line_a_v2/t052_sim/t052b_result{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                                                       encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
