"""T-023 评估（只读）：12d / 20d / 40d / 63d 四个期限的方案 C 预测，同一套 IC 与真实成交 NAV.

NAV：ExecutableNavEngine，Top 300 / 选股池 1500 / 完整价格 / v6_1 amihud；调仓步长 = H / 5 个周截面（12d 沿用原 3）：
  12d→3、20d→4、40d→8、63d→13；闸门阈值同原 12d（max_drawdown 0.15）与 63d（0.12），新期限用 0.15。
比较（预登记，描述性）：各新期限与 63d C 的日净收益差，移动块自助法（块长 20 个交易日，10,000 次，种子 0）95% 区间。
判读：某新期限 IR 高于 63d C 且配对差区间下界 > 0 → 值得作为新种子继续；否则维持 63d 为主种子。12d / 63d 的数字应与 T-018 相同（一致性检查）。
用法：PYTHONPATH=. python eval_horizons.py <data_root> <preds_dir> <out_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quantmind.backtest.wf_costs import board_of
from quantmind.execution.nav_engine import ExecutableNavEngine, NavConfig
from quantmind.execution.nav_gate import GateThresholds, evaluate_gate

root, preds_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)
D = root / "data"
SPEC = {"12d": (12, 3, 0.15), "20d": (20, 4, 0.15), "40d": (40, 8, 0.15), "63d": (63, 13, 0.12)}
panel = pd.read_parquet(D / "panel/alpha_panel_weekly_v6_1.parquet",
                        columns=["forward_return_12d", "forward_return_63d", "amihud_illiquidity"])
extra = pd.read_parquet(D / "panel/labels_extra_v6_1.parquet").reindex(panel.index)
labels = {"12d": panel["forward_return_12d"], "20d": extra["forward_return_20d"],
          "40d": extra["forward_return_40d"], "63d": panel["forward_return_63d"]}
amihud = panel[["amihud_illiquidity"]]
COLS = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "adj_factor", "amount"]
prices = pd.read_parquet(D / "raw/alpha_prices_panel_v6_20261009.parquet", columns=COLS)


def ic_summary(preds: pd.DataFrame, y: pd.Series) -> dict:
    p = preds.copy()
    p["as_of"] = pd.to_datetime(p["as_of"])
    p = p.join(y.rename("y"), on=["as_of", "ticker"]).dropna(subset=["y", "score"])
    p["board"] = p["ticker"].map(board_of)
    out = {}
    for b, g0 in [("ALL", p)] + [(k, v) for k, v in p.groupby("board") if k in ("MAIN", "CHINEXT", "STAR")]:
        ic = g0.groupby("as_of").apply(lambda g: g["score"].rank().corr(g["y"].rank()) if len(g) >= 30 else np.nan).dropna()
        out[b] = {"n_as_of": int(len(ic)), "ic_mean": float(ic.mean()), "icir": float(ic.mean() / ic.std()),
                  "win": float((ic > 0).mean())}
    return out


def block_ci(d: np.ndarray, block: int = 20, n: int = 10_000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    T = len(d)
    k = int(np.ceil(T / block))
    idx = (rng.integers(0, T - block + 1, size=(n, k))[:, :, None] + np.arange(block)).reshape(n, -1)[:, :T]
    m = d[idx].mean(axis=1) * 252
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


res, daily = {}, {}
for seed, (h, step, mdd) in SPEC.items():
    f = preds_dir / f"ridge_{seed}_v6_1_C.parquet"
    if not f.exists():
        res[seed] = {"missing": str(f)}
        continue
    pr = pd.read_parquet(f)
    pr["as_of"] = pd.to_datetime(pr["as_of"])
    r = ExecutableNavEngine(pr, prices, NavConfig(horizon_td=h, rebalance_step=step, top_n=300, universe_size=1500),
                            amihud=amihud).run()
    s = r["summary"]
    g = evaluate_gate(s, GateThresholds(max_drawdown=mdd))
    nav = r["nav_daily"].set_index("date")
    nav[["nav_net", "nav_gross", "nav_bench", "cum_cost"]].to_parquet(out_dir / f"nav_{seed}_C.parquet")
    daily[seed] = nav["nav_net"].pct_change().dropna()
    ex = nav["nav_net"] / nav["nav_bench"]
    n = s["nav"]
    res[seed] = {"n_as_of": int(pr.as_of.nunique()), "as_of_range": [str(pr.as_of.min().date()), str(pr.as_of.max().date())],
                 "ic": ic_summary(pr, labels[seed]),
                 "nav": {"ann_net_excess": n["ann_net_excess"], "ann_gross_excess": n["ann_return_gross"] - n["ann_return_bench"],
                         "ir": n["information_ratio"], "abs_mdd": n["max_drawdown_net"],
                         "relative_mdd": float((ex / ex.cummax() - 1).min()), "yearly": n["yearly_net_excess"],
                         "cost_drag": s["costs"]["cost_drag_ann"], "turnover_twoway": s["turnover"]["annualized_twoway"],
                         "n_rebalances": s["period"]["n_rebalances"], "period": s["period"]},
                 "gate_pass": g["gate_pass"], "gate_failed": [c["criterion"] for c in g["checks"] if not c["passed"]]}
    print(seed, json.dumps({k: res[seed]["nav"][k] for k in ("ann_net_excess", "ir", "relative_mdd", "n_rebalances")}),
          "IC", round(res[seed]["ic"]["ALL"]["ic_mean"], 4), flush=True)
comp = {}
for seed in ("12d", "20d", "40d"):
    if seed in daily and "63d" in daily:
        a, b = daily[seed].align(daily["63d"], join="inner")
        d = (a - b).to_numpy()
        comp[f"{seed}-63d"] = {"n_days": int(len(d)), "ann_mean_diff": float(d.mean() * 252), "ci95": block_ci(d)}
res["paired_vs_63d"] = comp
verdict = {}
for seed in ("20d", "40d"):
    if seed in res and "nav" in res[seed] and "63d" in res and f"{seed}-63d" in comp:
        better_ir = res[seed]["nav"]["ir"] > res["63d"]["nav"]["ir"]
        lo = comp[f"{seed}-63d"]["ci95"][0]
        verdict[seed] = ("值得作为新种子继续" if better_ir and lo > 0 else "不优于 63d C（维持 63d 为主种子）")
res["verdict"] = verdict
(out_dir / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(json.dumps({"paired_vs_63d": comp, "verdict": verdict}, ensure_ascii=False, indent=1))
