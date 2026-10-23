"""T-042 X2：扩展研究域（原研究池 1500 名以外）的历史净值与容量（规则在 docs/work/tasks/T-042.md 登记，先于本脚本运行）.

三个臂，引擎与契约照搬 T-030 的 ENS 净值口径（eval_primary.py）：
- R：ENS 原样，PIT adv20 前 1500 里取前 300 等权；
- E：只在 top1500 = False 且按时点 adv20 ≥ 1,000 万元的股票里取 ENS 前 300；
- Q：原研究池 ENS 前 240 + 扩展域（同一 adv20 下限）ENS 前 60。
三个臂都和 R 的基准（PIT 前 1500 等权）比；配对差与回撤差用同一组移动块自助样本。
用法：PYTHONPATH=. python docs/work/evidence/T-042/x2_extended.py <data_root> <20d|63d> <out_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs/work/evidence/T-030"))

from eval_primary import ENS_MEMBERS, NAV, load_family, nav_metrics  # noqa: E402

from quantmind.execution.nav_engine import ExecutableNavEngine, NavConfig  # noqa: E402
from quantmind.line_a.f8_books import capacity_report  # noqa: E402
from quantmind.research.gpu.ensemble import rank_average_frames  # noqa: E402
from quantmind.research.gpu.folds import calendar_and_grid, make_folds  # noqa: E402

ADV_FLOOR_RMB = 1e7
TOP_N, Q_POOL, Q_EXT = 300, 240, 60
N_BOOT, SEED = 10_000, 0
T030_REF = {"20d": {"ann_net_excess": 0.0770786891731835, "max_relative_wealth_drawdown": 0.2913148590169523}}


def adv20_at(prices: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.Series:
    """按时点 adv20（元）：每只股票过去 20 个交易日成交额均值（至少 10 天），只用 ≤ as_of 的数据。amount 单位是千元。"""
    px = prices[["ts_code", "trade_date", "amount"]].copy()
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    wide = px.pivot_table(index="trade_date", columns="ts_code", values="amount").sort_index()
    adv = wide.rolling(20, min_periods=10).mean() * 1000
    adv = adv.reindex(pd.DatetimeIndex(sorted(set(dates))), method="ffill")
    s = adv.stack()
    s.index.names = ["as_of", "ticker"]
    return s


def boot_idx(n: int, blk: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / blk))
    starts = rng.integers(0, n - blk + 1, size=(n_boot, nb))
    return (starts[:, :, None] + np.arange(blk)).reshape(n_boot, -1)[:, :n]


def mrwd(r_arm: np.ndarray, r_bench: np.ndarray) -> np.ndarray:
    """相对财富最大回撤（与 nav_metrics 同一定义），r_* 形状 (样本, 天)。"""
    q = np.cumprod(1 + r_arm, axis=-1) / np.cumprod(1 + r_bench, axis=-1)
    return (1 - q / np.maximum.accumulate(q, axis=-1)).max(axis=-1)


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
    assert all(v is not None for v in fams.values()), "missing ENS member"
    ens = rank_average_frames({m: s.reindex(meta.index) for m, s in fams.items()}).reindex(meta.index).dropna()
    hd, step, blk = NAV[h]
    cols = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "adj_factor", "amount"]
    prices = pd.read_parquet(D / "raw/alpha_prices_panel_v6_20261009.parquet", columns=cols)
    amihud = pd.read_parquet(D / "panel/alpha_panel_weekly_v6_1.parquet", columns=["amihud_illiquidity"])
    adv = adv20_at(prices, list(ens.index.get_level_values("as_of").unique())).reindex(ens.index)
    df = pd.DataFrame({"score": ens, "pool": meta["top1500"].reindex(ens.index).astype(bool), "adv20": adv})
    ext_ok = (~df["pool"]) & (df["adv20"] >= ADV_FLOOR_RMB)

    def top_per_date(mask: pd.Series, k: int) -> pd.Series:
        sub = df.loc[mask, "score"]
        return sub.groupby(level="as_of", group_keys=False).apply(lambda s: s.sort_values(ascending=False).head(k))

    arms = {"R": (df["score"], 1500),
            "E": (df.loc[ext_ok, "score"], 10**6),
            "Q": (pd.concat([top_per_date(df["pool"], Q_POOL), top_per_date(ext_ok, Q_EXT)]).sort_index(), 10**6)}
    res: dict = {"horizon": h, "n_test_asof": len(test), "adv_floor_rmb": ADV_FLOOR_RMB,
                 "ext_eligible_per_date_median": float(ext_ok.groupby(level="as_of").sum().median()),
                 "pool_per_date_median": float(df["pool"].groupby(level="as_of").sum().median()), "arms": {}}
    navs, runs = {}, {}
    for name, (s, usize) in arms.items():
        preds = s.rename("score").reset_index()
        r = ExecutableNavEngine(preds, prices, NavConfig(horizon_td=hd, rebalance_step=step, top_n=TOP_N,
                                                          universe_size=usize), amihud=amihud).run()
        runs[name] = r
        navs[name] = r["nav_daily"].set_index("date")[["nav_net", "nav_bench"]]
        print(name, "done", r["summary"]["period"], flush=True)
    start = max(n.index[0] for n in navs.values())
    end = min(n.index[-1] for n in navs.values())
    bench = navs["R"]["nav_bench"]
    for name, nv in navs.items():
        w = nv[(nv.index >= start) & (nv.index <= end)]
        common = pd.DataFrame({"nav_net": w["nav_net"], "nav_bench": bench.reindex(w.index)})
        s = runs[name]["summary"]
        res["arms"][name] = {"own_benchmark": nav_metrics(w), "common_benchmark": nav_metrics(common),
                             "turnover": s.get("turnover"), "costs": s.get("costs"),
                             "n_distinct_rejected_orders": s.get("execution", {}).get("n_distinct_rejected_orders"),
                             "n_trades_filled": s.get("execution", {}).get("n_trades_filled")}
        tg = runs[name]["targets"]
        if len(tg):
            cap = pd.concat([capacity_report(tg, prices, x) for x in (1e5, 1e6, 1e7)], ignore_index=True)
            cap.to_parquet(a.out / f"capacity_{h}_{name}.parquet")
            cap["over_share"] = cap["n_over_participation"] / cap["n_targets"]
            res["arms"][name]["capacity"] = {str(int(x)): {k: float(g[k].median()) for k in
                                                           ("n_targets", "n_unbuyable_one_lot", "cash_residual_share",
                                                            "max_participation", "n_over_participation", "over_share",
                                                            "min_commission_drag_bp")}
                                             for x, g in cap.groupby("aum")}
        w.to_parquet(a.out / f"nav_{h}_{name}.parquet")
    ref = T030_REF.get(h)
    r_own = res["arms"]["R"]["own_benchmark"]
    res["reproduction"] = {"t030": ref, "this_run": {k: r_own[k] for k in ("ann_net_excess", "max_relative_wealth_drawdown")}}
    if ref:
        res["reproduction"]["ok"] = all(abs(r_own[k] - v) < 1e-6 for k, v in ref.items())
    rets = {k: v["nav_net"].pct_change().reindex(pd.date_range(start, end)).dropna() for k, v in navs.items()}
    rb = bench.pct_change()
    idx_common = rets["R"].index.intersection(rb.dropna().index)
    for k in rets:
        idx_common = idx_common.intersection(rets[k].index)
    rb = rb.reindex(idx_common).to_numpy()
    rr = rets["R"].reindex(idx_common).to_numpy()
    bi = boot_idx(len(idx_common), blk, N_BOOT, SEED)
    res["paired"] = {}
    for k in ("E", "Q"):
        ra = rets[k].reindex(idx_common).to_numpy()
        d = ra - rr
        bt_mean = d[bi].mean(axis=1) * 252
        dd_point = float(mrwd(ra[None], rb[None])[0] - mrwd(rr[None], rb[None])[0])
        bt_dd = mrwd(ra[bi], rb[bi]) - mrwd(rr[bi], rb[bi])
        out = {"n_days": int(len(d)), "ann_mean_diff": float(d.mean() * 252),
               "ci95": [float(np.quantile(bt_mean, 0.025)), float(np.quantile(bt_mean, 0.975))],
               "ci97_5": [float(np.quantile(bt_mean, 0.0125)), float(np.quantile(bt_mean, 0.9875))],
               "mrwd_diff": dd_point,
               "mrwd_diff_ci95": [float(np.quantile(bt_dd, 0.025)), float(np.quantile(bt_dd, 0.975))]}
        cap = res["arms"][k].get("capacity", {}).get("1000000", {})
        out["capacity_1m_ok"] = bool(cap and cap["n_over_participation"] <= 0.05 * cap["n_targets"])
        for tag, dr, dd in (("suggested_1pp_2pp", 0.01, 0.02),):
            out[f"noninferior_{tag}"] = bool(out["ci95"][0] > -dr and out["mrwd_diff_ci95"][1] < dd)
        out["superior"] = bool(out["ci95"][0] > 0)
        res["paired"][f"{k}-R"] = out
    (a.out / f"x2_{h}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"reproduction": res["reproduction"], "paired": res["paired"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
