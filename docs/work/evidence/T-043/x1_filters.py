"""T-043 X1：F1-C 类研究政策过滤的配对账本评估（规则在 docs/work/tasks/T-043.md 登记，先于本脚本运行）.

R 与 T-042 相同（ENS 原样，已确认逐位复现 T-030）；挑战臂在 R 的决策截面上先打分、后过滤，引擎照常递补。
C1 ST/*ST/退；C2 收盘价 < 2 元；C3 总市值 < 10 亿元；C4 复牌不满 5 个交易日；C5 任一条。
用法：PYTHONPATH=. python docs/work/evidence/T-043/x1_filters.py <data_root> <20d|63d> <out_dir>
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
sys.path.insert(0, str(REPO / "docs/work/evidence/T-042"))

from eval_primary import ENS_MEMBERS, NAV, load_family, nav_metrics  # noqa: E402
from x2_extended import N_BOOT, SEED, boot_idx, mrwd  # noqa: E402

from quantmind.execution.nav_engine import ExecutableNavEngine, NavConfig  # noqa: E402
from quantmind.research.gpu.ensemble import rank_average_frames  # noqa: E402
from quantmind.research.gpu.folds import calendar_and_grid, make_folds  # noqa: E402

PRICE_MIN, MV_MIN_WAN, RESUME_TD = 2.0, 100_000.0, 5


def wide_at(path: Path, col: str, dates: pd.DatetimeIndex, lo: str) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["ts_code", "trade_date", col])
    df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str).str.replace("-", "").str.slice(0, 8))
    df = df[df["trade_date"] >= pd.Timestamp(lo)]
    w = df.pivot_table(index="trade_date", columns="ts_code", values=col).sort_index()
    return w.reindex(w.index.union(dates)).ffill().reindex(dates)


def st_flags(path: Path, dates: pd.DatetimeIndex) -> set[tuple]:
    n = pd.read_parquet(path)
    n = n[n["name"].astype(str).str.contains("ST|退", regex=True)]
    out = set()
    for t, s, e in zip(n["ts_code"], n["start_date"], n["end_date"], strict=True):
        s = pd.Timestamp(str(s))
        e = pd.Timestamp(str(e)) if e is not None and str(e) not in ("None", "nan", "") else pd.Timestamp("2100-01-01")
        for d in dates[(dates >= s) & (dates <= e)]:
            out.add((d, t))
    return out


def resume_flags(prices: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """复牌不满 RESUME_TD 个交易日：当前连续有 bar 的天数 < RESUME_TD，且这段之前已经有过 bar（中间停过牌）。"""
    has = prices.assign(trade_date=pd.to_datetime(prices["trade_date"])).pivot_table(
        index="trade_date", columns="ts_code", values="close").sort_index().notna()
    h = has.to_numpy()
    run = np.zeros(h.shape[1])
    cum = np.zeros(h.shape[1])
    out = np.zeros_like(h, dtype=bool)
    for i in range(h.shape[0]):
        run = (run + 1) * h[i]
        cum = cum + h[i]
        out[i] = h[i] & (run < RESUME_TD) & (cum > run)
    return pd.DataFrame(out, index=has.index, columns=has.columns).reindex(dates).fillna(False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("horizon", choices=["20d", "63d"])
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    root, h = a.root, a.horizon
    D, L = root / "data", root / "data/lake"
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
    prices = pd.read_parquet(D / "raw/alpha_prices_panel_v6_20261009.parquet", columns=cols)
    amihud = pd.read_parquet(D / "panel/alpha_panel_weekly_v6_1.parquet", columns=["amihud_illiquidity"])
    dates = pd.DatetimeIndex(sorted(ens.index.get_level_values("as_of").unique()))
    lo = (dates[0] - pd.Timedelta(days=30)).strftime("%Y%m%d")
    close = wide_at(L / "daily_basic.parquet", "close", dates, lo)
    mv = wide_at(L / "daily_basic.parquet", "total_mv", dates, lo)
    st = st_flags(sorted((L / "events").glob("namechange_full_*.parquet"))[-1], dates)
    rs = resume_flags(prices, dates)
    idx = ens.index
    as_of, tick = idx.get_level_values("as_of"), idx.get_level_values("ticker")

    def look(w: pd.DataFrame) -> np.ndarray:
        s = w.stack()
        s.index.names = ["as_of", "ticker"]
        return s.reindex(idx).to_numpy()
    flags = pd.DataFrame({"C1": [(d, t) in st for d, t in zip(as_of, tick, strict=True)],
                          "C2": look(close) < PRICE_MIN, "C3": look(mv) < MV_MIN_WAN,
                          "C4": np.nan_to_num(look(rs.astype(float)), nan=0.0) > 0}, index=idx)
    flags["C5"] = flags[["C1", "C2", "C3", "C4"]].any(axis=1)
    pool = meta["top1500"].reindex(idx).astype(bool)
    res: dict = {"horizon": h, "rules": {"price_min": PRICE_MIN, "mv_min_wan": MV_MIN_WAN, "resume_td": RESUME_TD},
                 "flagged_rows_in_pool": {k: int((flags[k] & pool).sum()) for k in flags},
                 "pool_rows": int(pool.sum()), "arms": {}}
    navs, runs = {}, {}
    for name in ["R", "C1", "C2", "C3", "C4", "C5"]:
        s = ens if name == "R" else ens[~flags[name].to_numpy()]
        r = ExecutableNavEngine(s.rename("score").reset_index(), prices,
                                NavConfig(horizon_td=hd, rebalance_step=step, top_n=300, universe_size=1500),
                                amihud=amihud).run()
        runs[name], navs[name] = r, r["nav_daily"].set_index("date")[["nav_net", "nav_bench"]]
        print(name, "done", flush=True)
    bench = navs["R"]["nav_bench"]
    tgt_r = {pd.Timestamp(x["as_of"]): set(x["targets"]) for _, x in runs["R"]["targets"].iterrows()}
    for name, nv in navs.items():
        common = pd.DataFrame({"nav_net": nv["nav_net"], "nav_bench": bench.reindex(nv.index)}).dropna()
        tg = {pd.Timestamp(x["as_of"]): set(x["targets"]) for _, x in runs[name]["targets"].iterrows()}
        removed = [len(tgt_r[d] - tg.get(d, set())) for d in tgt_r]
        res["arms"][name] = {"common_benchmark": nav_metrics(common),
                             "targets_changed_vs_R": {"mean": float(np.mean(removed)), "max": int(np.max(removed)),
                                                      "n_rebalances_changed": int(sum(x > 0 for x in removed))}}
        nv.to_parquet(a.out / f"nav_{h}_{name}.parquet")
    rr_s = navs["R"]["nav_net"].pct_change()
    rb_s = bench.pct_change()
    res["paired"] = {}
    for k in ["C1", "C2", "C3", "C4", "C5"]:
        ra_s = navs[k]["nav_net"].pct_change()
        j = pd.DataFrame({"a": ra_s, "r": rr_s, "b": rb_s}).dropna()
        ra, rr, rb = j["a"].to_numpy(), j["r"].to_numpy(), j["b"].to_numpy()
        bi = boot_idx(len(j), blk, N_BOOT, SEED)
        d = ra - rr
        bt = d[bi].mean(axis=1) * 252
        bdd = mrwd(ra[bi], rb[bi]) - mrwd(rr[bi], rb[bi])
        ch = res["arms"][k]["targets_changed_vs_R"]
        out = {"n_days": int(len(d)), "ann_mean_diff": float(d.mean() * 252),
               "ci95": [float(np.quantile(bt, 0.025)), float(np.quantile(bt, 0.975))],
               "mrwd_diff": float(mrwd(ra[None], rb[None])[0] - mrwd(rr[None], rb[None])[0]),
               "mrwd_diff_ci95": [float(np.quantile(bdd, 0.025)), float(np.quantile(bdd, 0.975))],
               "maxdd_net_diff": res["arms"][k]["common_benchmark"]["max_drawdown_net"]
               - res["arms"]["R"]["common_benchmark"]["max_drawdown_net"],
               "no_effect": ch["n_rebalances_changed"] == 0}
        out["noninferior_suggested_1pp_2pp"] = bool(not out["no_effect"] and out["ci95"][0] > -0.01
                                                    and out["mrwd_diff_ci95"][1] < 0.02)
        res["paired"][f"{k}-R"] = out
    (a.out / f"x1_{h}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"flagged_rows_in_pool": res["flagged_rows_in_pool"], "paired": res["paired"]}, ensure_ascii=False,
                     indent=1))


if __name__ == "__main__":
    main()
