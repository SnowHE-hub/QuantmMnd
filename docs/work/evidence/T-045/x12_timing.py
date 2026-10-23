"""T-045 X12：择时仓位（趋势、波动）与拥挤时降小盘配额（规则在 docs/work/tasks/T-045.md 登记，先于本脚本运行）.

R 与 T-042 相同。T1 / T2 用 R 的日收益按调仓日仓位缩放（仓位变动部分收 0.25% 单边成本）；S1 用引擎原样跑。
用法：PYTHONPATH=. python docs/work/evidence/T-045/x12_timing.py <data_root> <20d|63d> <out_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
for p in (REPO, REPO / "docs/work/evidence/T-030", REPO / "docs/work/evidence/T-042", REPO / "docs/work/evidence/T-043"):
    sys.path.insert(0, str(p))

from eval_primary import ENS_MEMBERS, NAV, load_family, nav_metrics  # noqa: E402
from x1_filters import wide_at  # noqa: E402
from x2_extended import N_BOOT, SEED, boot_idx, mrwd  # noqa: E402

from quantmind.execution.nav_engine import ExecutableNavEngine, NavConfig  # noqa: E402
from quantmind.line_a.m_market import market_state  # noqa: E402
from quantmind.research.gpu.ensemble import rank_average_frames  # noqa: E402
from quantmind.research.gpu.folds import calendar_and_grid, make_folds  # noqa: E402

LOW_EXPOSURE, SWITCH_COST = 0.7, 0.0025
PRICES = "raw/alpha_prices_panel_v6_20261009.parquet"


def mdd(r: np.ndarray) -> np.ndarray:
    w = np.cumprod(1 + r, axis=-1)
    return (1 - w / np.maximum.accumulate(w, axis=-1)).max(axis=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("horizon", choices=["20d", "63d"])
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    root, h = a.root, a.horizon
    D = root / "data"
    a.out.mkdir(parents=True, exist_ok=True)
    cal, grid = calendar_and_grid(root)
    test = sorted({d for f in make_folds(cal, grid, h) for d in f.test_asof})
    meta = pd.concat([pd.read_parquet(p, columns=["as_of", "ticker", "top1500"])
                      for p in sorted((D / "gpu_v1/model/v1").glob("*.meta.parquet"))], ignore_index=True)
    meta["as_of"] = pd.to_datetime(meta["as_of"])
    meta = meta[meta["as_of"].isin(set(test))].set_index(["as_of", "ticker"])
    fams = {f: load_family(root, f, h) for f in ENS_MEMBERS}
    ens = rank_average_frames({m: s.reindex(meta.index) for m, s in fams.items()}).reindex(meta.index).dropna()
    hd, step, blk = NAV[h]
    cols = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "adj_factor", "amount"]
    prices = pd.read_parquet(D / PRICES, columns=cols)
    amihud = pd.read_parquet(D / "panel/alpha_panel_weekly_v6_1.parquet", columns=["amihud_illiquidity"])
    dates = pd.DatetimeIndex(sorted(ens.index.get_level_values("as_of").unique()))
    rebal = list(dates[::step])                                  # 与引擎相同的调仓日历
    state = {}
    for d in rebal:
        m = market_state(root, d, prices_path=D / PRICES)
        t = m["trend"]["沪深300"]
        v = m["volatility_hs300"]
        s = m.get("style") or {}
        crowd_known = s.get("amount_ratio_pct3y") is not None and s.get("rs60_pct3y") is not None
        state[d] = {"trend": bool(t.get("vs_ma60", 0) < 0), "vol": bool((v.get("realized_vol_20d_pct3y") or 0) > 0.8),
                    "crowd": bool(crowd_known and (s["amount_ratio_pct3y"] > 0.9 or s["rs60_pct3y"] > 0.9)),
                    "crowd_known": crowd_known}
    st = pd.DataFrame(state).T
    st.to_csv(a.out / f"states_{h}.csv")
    res: dict = {"horizon": h, "n_rebalances": len(rebal), "state_share": {k: float(st[k].mean()) for k in st},
                 "arms": {}, "paired": {}}

    def run(preds: pd.Series) -> pd.DataFrame:
        r = ExecutableNavEngine(preds.rename("score").reset_index(), prices,
                                NavConfig(horizon_td=hd, rebalance_step=step, top_n=300, universe_size=1500),
                                amihud=amihud).run()
        return r["nav_daily"].set_index("date")[["nav_net", "nav_bench"]]
    navR = run(ens)
    # S1：拥挤的调仓日去掉原研究池里流通市值最小的三分之一
    circ = wide_at(D / "lake/daily_basic.parquet", "circ_mv", dates, (dates[0] - pd.Timedelta(days=30)).strftime("%Y%m%d"))
    cs = circ.stack()
    cs.index.names = ["as_of", "ticker"]
    pool = meta["top1500"].reindex(ens.index).astype(bool)
    cv = cs.reindex(ens.index)
    small = pd.Series(False, index=ens.index)
    for d in [d for d in rebal if st.loc[d, "crowd"]]:
        m = (ens.index.get_level_values("as_of") == d) & pool.to_numpy()
        q = cv[m].rank(pct=True)
        small.loc[q.index[q <= 1 / 3]] = True
    navS = run(ens[~small.to_numpy()])
    bench = navR["nav_bench"]
    rR = navR["nav_net"].pct_change()
    rb = bench.pct_change()
    # T1 / T2：调仓日确定仓位，持有到下一个调仓日
    days = navR.index
    arms = {"R": rR, "S1": navS["nav_net"].pct_change().reindex(days)}
    for name, key in (("T1", "trend"), ("T2", "vol")):
        e = pd.Series(np.nan, index=days)
        for d in rebal:
            nxt = days[days > d]
            if len(nxt):
                e.loc[nxt[0]] = LOW_EXPOSURE if st.loc[d, key] else 1.0   # 决策日收盘后调整，次日起生效
        e = e.ffill().fillna(1.0)
        cost = e.diff().abs().fillna(0) * SWITCH_COST
        arms[name] = e * rR - cost
    j = pd.DataFrame({**arms, "b": rb}).dropna()
    for name in arms:
        w = (1 + j[name]).cumprod()
        res["arms"][name] = nav_metrics(pd.DataFrame({"nav_net": w, "nav_bench": (1 + j["b"]).cumprod()}))
    bi = boot_idx(len(j), blk, N_BOOT, SEED)
    rr, bb = j["R"].to_numpy(), j["b"].to_numpy()
    for name in ("T1", "T2", "S1"):
        ra = j[name].to_numpy()
        d = ra - rr
        bt = d[bi].mean(axis=1) * 252
        out = {"ann_mean_diff": float(d.mean() * 252),
               "ci95": [float(np.quantile(bt, 0.025)), float(np.quantile(bt, 0.975))],
               "mdd_diff": float(mdd(ra[None])[0] - mdd(rr[None])[0]),
               "mrwd_diff": float(mrwd(ra[None], bb[None])[0] - mrwd(rr[None], bb[None])[0])}
        bmdd = mdd(ra[bi]) - mdd(rr[bi])
        bmr = mrwd(ra[bi], bb[bi]) - mrwd(rr[bi], bb[bi])
        out["mdd_diff_ci95"] = [float(np.quantile(bmdd, 0.025)), float(np.quantile(bmdd, 0.975))]
        out["mrwd_diff_ci95"] = [float(np.quantile(bmr, 0.025)), float(np.quantile(bmr, 0.975))]
        key = {"T1": "trend", "T2": "vol", "S1": "crowd"}[name]
        out["trigger_share"] = float(st[key].mean())
        out["rarely_triggered"] = out["trigger_share"] < 0.05
        if name in ("T1", "T2"):
            out["risk_improved_suggested_1pp"] = bool(not out["rarely_triggered"] and out["mdd_diff_ci95"][1] < 0
                                                      and out["ci95"][0] > -0.01)
        else:
            out["noninferior_suggested_1pp_2pp"] = bool(not out["rarely_triggered"] and out["ci95"][0] > -0.01
                                                        and out["mrwd_diff_ci95"][1] < 0.02)
            out["risk_improved"] = bool(not out["rarely_triggered"] and out["mrwd_diff_ci95"][1] < 0)
        res["paired"][f"{name}-R"] = out
    (a.out / f"x12_{h}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"state_share": res["state_share"], "paired": res["paired"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
