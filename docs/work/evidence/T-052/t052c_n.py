"""T-052c：10 万档持股数 N ∈ {5, 6, 8, 10}（floor + skip），2022–2026 × 4 个相位（登记见 T-052.md）."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_data as sd  # noqa: E402
from sim_grid import SMALL  # noqa: E402

from quantmind.line_a.sim_account import AccountSimulator, Market, SimConfig  # noqa: E402

START, END = pd.Timestamp("2022-01-01"), pd.Timestamp("2026-10-23")


def main() -> None:
    px = sd.prices(START)
    mk = Market(px, START, END)
    sc, _ = sd.ens_scores("20d", START)
    ami, dv, st = sd.amihud(START), sd.dividends(), sd.st_intervals()
    grid = sorted(sc["as_of"].unique())
    rows = []
    for off in range(4):
        s0 = pd.Timestamp(grid[off])
        for n in (5, 6, 8, 10):
            r = AccountSimulator(sc, mk, SimConfig(aum=1e5, top_n=n, rebalance_step=4, boards=SMALL), start=s0, end=END,
                                 amihud=ami, dividends=dv, st_intervals=st).run()
            s = r["summary"]
            rows.append({"offset": off, "n": n, "total_return": s["total_return"], "avg_cash": s["avg_cash_share"],
                         "max_dd": s["max_drawdown"], "cost_share": s["cost_share_of_aum"], "days": s["days"]})
            print(json.dumps(rows[-1]), flush=True)
    df = pd.DataFrame(rows)
    df["ann"] = (1 + df["total_return"]) ** (252 / df["days"]) - 1
    df.to_parquet(sd.ROOT / "data/line_a_v2/t052_sim/t052c_n.parquet")
    b = df[df["n"] == 10].set_index("offset")
    res = {}
    for n in (5, 6, 8):
        x = df[df["n"] == n].set_index("offset")
        res[n] = {"avg_cash": float(x["avg_cash"].mean()), "cash_drop_pp": float((b["avg_cash"] - x["avg_cash"]).mean() * 100),
                  "ann_diff_median_pp": float((x["ann"] - b["ann"]).median() * 100),
                  "maxdd_diff_median_pp": float((x["max_dd"] - b["max_dd"]).median() * 100),
                  "cost_diff_pp": float((x["cost_share"] - b["cost_share"]).mean() * 100)}
        r = res[n]
        r["adopt"] = bool(r["cash_drop_pp"] >= 5 and r["ann_diff_median_pp"] >= -0.5 and r["maxdd_diff_median_pp"] >= -5)
    res["base_n10"] = {"avg_cash": float(b["avg_cash"].mean()), "ann_median": float(b["ann"].median()),
                       "max_dd_median": float(b["max_dd"].median())}
    (sd.ROOT / "data/line_a_v2/t052_sim/t052c_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
