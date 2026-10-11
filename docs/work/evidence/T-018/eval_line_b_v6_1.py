"""T-018 评估（只读）：重训预测的样本外 RankIC（全体 / 分板块）、真实成交 NAV 与原闸门、与原预测对照、板块专用 vs 统一.

标签：v6_1 面板（完整交易日历，delist-aware）。NAV：ExecutableNavEngine + 完整价格表 alpha_prices_panel_v6_20261009，
amihud 取 v6_1 面板（原预测用原 v6 面板：两版 as_of 网格只有 110 个日期相同，取不到会退到最保守滑点档；2026-10-11 出结果前修正），Top 300 / 选股池 1500 / 原调仓步长与原闸门阈值（scripts/run_executable_nav.py 的 MODELS）。
用法：PYTHONPATH=. python eval_line_b_v6_1.py <data_root> <preds_dir> <out.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.run_executable_nav as ren
from quantmind.backtest.wf_costs import board_of
from quantmind.execution.nav_engine import ExecutableNavEngine, NavConfig
from quantmind.execution.nav_gate import evaluate_gate

root, preds_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
D = root / "data"
H = {"12d": 12, "63d": 63}
ORIG = {"12d": D / "bakeoff/preds/ridge_full_12d_quarterly_v6.parquet",
        "63d": D / "bakeoff/preds/ridge_full_fnd_63d_quarterly_v6.parquet"}
SPEC = {"12d": ren.MODELS["12d"], "63d": ren.MODELS["63d"]}
ORIG_END = pd.Timestamp("2026-04-20")   # 原预测的最后一个 as_of

panel = pd.read_parquet(D / "panel/alpha_panel_weekly_v6_1.parquet",
                        columns=["forward_return_12d", "forward_return_63d", "amihud_illiquidity"])
COLS = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "adj_factor", "amount"]
prices = pd.read_parquet(D / "raw/alpha_prices_panel_v6_20261009.parquet", columns=COLS)
amihud = panel[["amihud_illiquidity"]]
amihud_v6 = pd.read_parquet(D / "panel/alpha_panel_weekly_v6.parquet", columns=["amihud_illiquidity"])


def ic_table(preds: pd.DataFrame, h: int, lo=None, hi=None) -> pd.DataFrame:
    p = preds.copy()
    p["as_of"] = pd.to_datetime(p["as_of"])
    if lo is not None:
        p = p[p["as_of"] >= lo]
    if hi is not None:
        p = p[p["as_of"] <= hi]
    y = panel[f"forward_return_{h}d"]
    p = p.join(y.rename("y"), on=["as_of", "ticker"]).dropna(subset=["y", "score"])
    p["board"] = p["ticker"].map(board_of)
    rows = []
    for a, g in p.groupby("as_of"):
        for b, gb in [("ALL", g)] + [(k, v) for k, v in g.groupby("board") if k in ("MAIN", "CHINEXT", "STAR")]:
            if len(gb) >= 30:
                rows.append({"as_of": a, "board": b, "ic": gb["score"].rank().corr(gb["y"].rank())})
    return pd.DataFrame(rows)


def summarize_ic(t: pd.DataFrame) -> dict:
    out = {}
    for b, g in t.groupby("board"):
        out[b] = {"n_as_of": int(len(g)), "ic_mean": float(g.ic.mean()), "icir": float(g.ic.mean() / g.ic.std()),
                  "win": float((g.ic > 0).mean())}
    return out


def nav(preds: pd.DataFrame, seed: str, am: pd.DataFrame = None) -> dict:
    spec = SPEC[seed]
    p = preds.copy()
    p["as_of"] = pd.to_datetime(p["as_of"])
    cfg = NavConfig(horizon_td=spec["horizon_td"], rebalance_step=spec["rebalance_step"], top_n=300, universe_size=1500)
    s = ExecutableNavEngine(p, prices, cfg, amihud=amihud if am is None else am).run()["summary"]
    g = evaluate_gate(s, spec["thresholds"])
    n = s["nav"]
    return {"ann_net": n["ann_return_net"], "ann_bench": n["ann_return_bench"], "excess": n["ann_net_excess"],
            "ir": n["information_ratio"], "mdd": n["max_drawdown_net"], "yearly_excess": n["yearly_net_excess"],
            "period": s["period"], "gate_pass": g["gate_pass"]}


def paired_board_diff(tp: pd.DataFrame, tc: pd.DataFrame) -> dict:
    """P（板块专用）− C（统一）在同板块、同截面的 IC 配对差；95% 区间用独立自助法（10,000 次）."""
    m = tp.merge(tc, on=["as_of", "board"], suffixes=("_p", "_c"))
    rng = np.random.default_rng(0)
    res = {}
    for b, g in m[m.board != "ALL"].groupby("board"):
        d = (g.ic_p - g.ic_c).to_numpy()
        boots = d[rng.integers(0, len(d), size=(10_000, len(d)))].mean(axis=1)
        res[b] = {"n": int(len(d)), "mean_diff": float(d.mean()),
                  "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                  "share_p_better": float((d > 0).mean())}
    return res


res = {"inputs": {"panel": "alpha_panel_weekly_v6_1.parquet", "prices": "alpha_prices_panel_v6_20261009.parquet"}}
for seed in ("12d", "63d"):
    h = H[seed]
    orig = pd.read_parquet(ORIG[seed])
    r = {"original_preds": {"ic_same_window": summarize_ic(ic_table(orig, h, hi=ORIG_END))}}
    for proto in ("R", "X", "C", "P"):
        f = preds_dir / f"ridge_{seed}_v6_1_{proto}.parquet"
        if not f.exists():
            continue
        pr = pd.read_parquet(f)
        t_all = ic_table(pr, h)
        r[proto] = {"as_of_range": [str(pd.to_datetime(pr.as_of).min().date()), str(pd.to_datetime(pr.as_of).max().date())],
                    "n_as_of": int(pr.as_of.nunique()), "ic": summarize_ic(t_all),
                    "ic_same_window_as_original": summarize_ic(ic_table(pr, h, hi=ORIG_END)),
                    "nav": nav(pr, seed)}
        r[proto]["_t"] = t_all
        print(seed, proto, json.dumps({k: v for k, v in r[proto].items() if k not in ("_t",)}, default=str)[:600], flush=True)
    r["original_preds"]["nav_complete_prices"] = nav(orig, seed, am=amihud_v6)
    if "P" in r and "C" in r:
        r["board_P_minus_C"] = paired_board_diff(r["P"]["_t"], r["C"]["_t"])
        print(seed, "P-C", r["board_P_minus_C"], flush=True)
    for proto in ("R", "X", "C", "P"):
        if proto in r:
            r[proto].pop("_t")
    res[seed] = r
out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
