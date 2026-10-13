"""T-030 P6：主比较（预登记 §2）与次要指标（§6）.

输入：data/gpu_v1/preds/<族>_<h>/fold_*.parquet（各族折预测）、基线 Ridge C parquet、训练矩阵 meta（标签 y20/y63 与 top1500）。
主比较：周频测试 as_of 上、1500 池内、ENS 与 Ridge C 都有分数且标签非缺失的股票，逐 as_of Spearman RankIC 差；
移动块自助法块长 13 / 26，10,000 次，种子 0；双侧 95% 与 97.5%（Bonferroni k=2）。
次要：各族对 Ridge C 的配对差、全面板 RankIC、分年 / 分板块 / 分市值 IC、与 Ridge 的截面相关、可执行 NAV 与研究闸门 v2（资金闸门 NOT_APPROVED）。
用法：PYTHONPATH=. python eval_primary.py <data_root> <20d|63d> <out.json> [--families xgb,cat,tabm,seqfusion]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pds

from quantmind.backtest.wf_costs import board_of
from quantmind.execution.nav_engine import ExecutableNavEngine, NavConfig
from quantmind.research.gpu.ensemble import rank_average_frames
from quantmind.research.gpu.folds import calendar_and_grid, make_folds
from quantmind.research.gpu.metrics import moving_block_bootstrap, two_sided_ci

BASE = {"20d": "bakeoff/preds_v6_1_t028/ridge_20d_v6_1_BASE.parquet",
        "63d": "bakeoff/preds_v6_1/ridge_63d_v6_1_C_infnan.parquet"}
ENS_MEMBERS = ["xgb", "cat", "tabm", "seqfusion"]
NAV = {"20d": (20, 4, 20), "63d": (63, 13, 65)}          # 持有期、调仓步长（周频期）、NAV 配对块长（日）
N_BOOT = 10_000


def ic_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby(level="as_of")
    return g.apply(lambda x: x[col].rank().corr(x["y"].rank()) if len(x) >= 30 else np.nan).dropna()


def paired(a: pd.Series, b: pd.Series) -> dict:
    d = (a - b).dropna().sort_index()
    x = d.to_numpy()
    out = {"n_dates": int(len(x)), "ic_a": float(a.reindex(d.index).mean()), "ic_b": float(b.reindex(d.index).mean()),
           "mean_diff": float(x.mean()), "share_a_better": float((x > 0).mean())}
    for blk in (13, 26):
        bt = moving_block_bootstrap(x, blk, N_BOOT, 0)
        out[f"block{blk}"] = {"ci95": two_sided_ci(bt, 0.95), "ci97_5": two_sided_ci(bt, 0.975)}
    lo = [out[f"block{b}"]["ci97_5"][0] for b in (13, 26)]
    out["decision"] = ("检测到增益" if all(v > 0 for v in lo) else
                       ("未检测到增益（块长敏感）" if any(v > 0 for v in lo) else "未检测到增益"))
    return out


def load_family(root: Path, fam: str, h: str) -> pd.Series | None:
    d = root / "data/gpu_v1/preds" / f"{fam}_{h}"
    files = sorted(d.glob("fold_*.parquet"))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f, columns=["as_of", "ticker", "score"]) for f in files], ignore_index=True)
    df["as_of"] = pd.to_datetime(df["as_of"])
    return df.set_index(["as_of", "ticker"])["score"].sort_index()


def nav_metrics(nav: pd.DataFrame) -> dict:
    p, bm = nav["nav_net"], nav["nav_bench"]
    yrs = len(nav) / 252.0
    ann_p, ann_b = float((p.iloc[-1] / p.iloc[0]) ** (1 / yrs) - 1), float((bm.iloc[-1] / bm.iloc[0]) ** (1 / yrs) - 1)
    ex = (p.pct_change() - bm.pct_change()).dropna()
    q = (p / p.iloc[0]) / (bm / bm.iloc[0])
    yearly = {str(int(y)): float((g["nav_net"].iloc[-1] / g["nav_bench"].iloc[-1])
                                 / (g["nav_net"].iloc[0] / g["nav_bench"].iloc[0]) - 1) for y, g in nav.groupby(nav.index.year)}
    m = {"ann_net_excess": ann_p - ann_b, "information_ratio": float(ex.mean() / ex.std() * np.sqrt(252)),
         "max_relative_wealth_drawdown": float((1 - q / q.cummax()).max()),
         "max_drawdown_net": float((p / p.cummax() - 1).min()), "yearly_net_excess": yearly}
    m["research_gate_v2_pass"] = bool(m["ann_net_excess"] >= 0.05 and m["max_relative_wealth_drawdown"] <= 0.12
                                      and m["information_ratio"] >= 1.0 and all(v > 0 for v in yearly.values()))
    m["capital_gate_status"] = "NOT_APPROVED"
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("horizon", choices=["20d", "63d"])
    ap.add_argument("out", type=Path)
    ap.add_argument("--families", default="xgb,cat,tabm,seqfusion,lgbm,mlp,gru,alstm,master")
    ap.add_argument("--no-nav", action="store_true")
    a = ap.parse_args()
    root, h = a.root, a.horizon
    D = root / "data"
    cal, grid = calendar_and_grid(root)
    test = sorted({d for f in make_folds(cal, grid, h) for d in f.test_asof})
    ycol = "y20" if h == "20d" else "y63"
    meta = pd.concat([pd.read_parquet(p, columns=["as_of", "ticker", ycol, "top1500"])
                      for p in sorted((D / "gpu_v1/model/v1").glob("*.meta.parquet"))], ignore_index=True)
    meta["as_of"] = pd.to_datetime(meta["as_of"])
    meta = meta[meta["as_of"].isin(set(test))].set_index(["as_of", "ticker"]).rename(columns={ycol: "y"})
    ridge = pd.read_parquet(D / BASE[h])
    ridge["as_of"] = pd.to_datetime(ridge["as_of"])
    ridge = ridge.set_index(["as_of", "ticker"])["score"]

    fams = {f: s for f in a.families.split(",") if (s := load_family(root, f, h)) is not None}
    members = [m for m in ENS_MEMBERS if m in fams]
    res: dict = {"horizon": h, "baseline": BASE[h], "n_test_asof": len(test), "families_available": list(fams),
                 "ens_members": members, "ens_members_missing": [m for m in ENS_MEMBERS if m not in fams]}
    test_idx = meta.index
    scores = {f: s.reindex(test_idx) for f, s in fams.items()}
    if members:
        scores["ENS"] = rank_average_frames({m: scores[m] for m in members}).reindex(test_idx)
    for extra, mem in (("ENS+master", members + ["master"]), ("TREES", ["xgb", "cat"])):
        if all(m in scores for m in mem):
            scores[extra] = rank_average_frames({m: scores[m] for m in mem}).reindex(test_idx)
    scores["ridge"] = ridge.reindex(test_idx)

    pool = meta[meta["top1500"]]
    ic_pool, ic_full = {}, {}
    for name, s in scores.items():
        ic_pool[name] = ic_by_date(pool.assign(s=s.reindex(pool.index)).dropna(subset=["s", "y"]), "s")
        ic_full[name] = ic_by_date(meta.assign(s=s).dropna(subset=["s", "y"]), "s")
    if "ENS" in scores:                                    # 主比较：两者都有分数的行
        both = pool.assign(e=scores["ENS"].reindex(pool.index), r=scores["ridge"].reindex(pool.index)).dropna()
        res["primary"] = paired(ic_by_date(both, "e"), ic_by_date(both, "r"))
        res["primary"]["n_rows"] = int(len(both))
    res["exploratory_vs_ridge"] = {}
    for name in scores:
        if name in ("ridge", "ENS"):
            continue
        both = pool.assign(e=scores[name].reindex(pool.index), r=scores["ridge"].reindex(pool.index)).dropna()
        res["exploratory_vs_ridge"][name] = paired(ic_by_date(both, "e"), ic_by_date(both, "r"))
    res["ic_pool_mean"] = {k: float(v.mean()) for k, v in ic_pool.items()}
    res["ic_full_mean"] = {k: float(v.mean()) for k, v in ic_full.items()}
    res["ic_pool_by_year"] = {k: {str(y): float(g.mean()) for y, g in v.groupby(v.index.year)} for k, v in ic_pool.items()}
    boards = pd.Series([board_of(t) for t in pool.index.get_level_values("ticker")], index=pool.index)
    res["ic_pool_by_board"] = {}
    for b in ("MAIN", "CHINEXT", "STAR"):
        sub = pool[(boards == b).values]
        res["ic_pool_by_board"][b] = {k: float(ic_by_date(sub.assign(s=s.reindex(sub.index)).dropna(subset=["s", "y"]),
                                                           "s").mean()) for k, s in scores.items()}
    fnd = pds.dataset(sorted(str(p) for p in (D / "gpu_v1/panel/fnd").glob("*.parquet")), format="parquet").to_table(
        filter=pds.field("as_of").isin(pd.DatetimeIndex(test).as_unit("ns").to_numpy()),
        columns=["as_of", "ticker", "log_circ_market_cap"]).to_pandas()
    if "as_of" not in fnd.columns:
        fnd = fnd.reset_index()
    lcm = fnd.set_index(["as_of", "ticker"])["log_circ_market_cap"].reindex(pool.index)
    qs = lcm.groupby(level="as_of").transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False))
    res["ic_pool_by_size_quintile"] = {}
    for q in range(5):
        sub = pool[(qs == q).values]
        res["ic_pool_by_size_quintile"][str(q + 1)] = {
            k: float(ic_by_date(sub.assign(s=s.reindex(sub.index)).dropna(subset=["s", "y"]), "s").mean())
            for k, s in scores.items()}
    if "ENS" in scores:
        j = pd.DataFrame({"e": scores["ENS"], "r": scores["ridge"]}).reindex(pool.index).dropna()
        res["corr_ens_ridge_mean"] = float(j.groupby(level="as_of").apply(lambda x: x["e"].rank().corr(x["r"].rank())).mean())

    if not a.no_nav:
        hd, step, nblk = NAV[h]
        amihud = pd.read_parquet(D / "panel/alpha_panel_weekly_v6_1.parquet", columns=["amihud_illiquidity"])
        cols = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "adj_factor", "amount"]
        prices = pd.read_parquet(D / "raw/alpha_prices_panel_v6_20261009.parquet", columns=cols)
        navs = {}
        for name in ["ridge", "ENS", *fams]:
            if name not in scores:
                continue
            s = scores[name].dropna().rename("score").reset_index()
            r = ExecutableNavEngine(s, prices, NavConfig(horizon_td=hd, rebalance_step=step, top_n=300, universe_size=1500),
                                    amihud=amihud).run()
            navs[name] = r["nav_daily"].set_index("date")[["nav_net", "nav_bench"]]
        start = max(n.index[0] for n in navs.values())
        end = min(n.index[-1] for n in navs.values())
        res["nav_window"] = [str(start.date()), str(end.date())]
        res["nav"] = {k: nav_metrics(v[(v.index >= start) & (v.index <= end)]) for k, v in navs.items()}
        rb = navs["ridge"]["nav_net"].pct_change().dropna()
        res["nav_paired_vs_ridge"] = {}
        for k, v in navs.items():
            if k == "ridge":
                continue
            ra = v["nav_net"].pct_change().dropna()
            x, y = ra.align(rb, join="inner")
            dd = (x - y).to_numpy()
            bt = moving_block_bootstrap(dd, nblk, N_BOOT, 0) * 252
            res["nav_paired_vs_ridge"][k] = {"ann_mean_diff": float(dd.mean() * 252), "ci95": two_sided_ci(bt, 0.95)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("horizon", "ens_members", "primary") if k in res}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
