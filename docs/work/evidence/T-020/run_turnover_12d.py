"""T-020：12d 种子的换手控制实验（只读数据；结果写到 <out_dir>）.

方案（运行前写定，见 docs/work/tasks/T-020.md）：
  B0 基线          ExecutableNavEngine 原配置（Top 300 / 选股池 1500 / rebalance_step 3 / 等权、每次调回等权）
  B1 缓冲带        已持有且排名仍在前 600 的票不卖
  B2 容忍带        目标内持仓偏离等权目标 ≤ 50% 时不调仓
  B3 = B1 + B2     ← 主比较（预先指定）
  B4 降频          rebalance_step 6（每 30 个交易日调仓），其余同 B0
主估计量：B3 与 B0 的日净收益差 d_t 的年化均值（×252）；95% 区间 = 移动块自助法（块长 15 个交易日 = 一个调仓间隔，10,000 次，种子 0）。
两者基准相同（同一选股池等权、无成本），所以 d_t 也是日超额收益之差。
滑点档用 amihud：v6_1 预测配 v6_1 面板（默认）；原 v6 预测必须配原 v6 面板（两版网格只有 110 个日期相同，取不到会退到最保守档）。
用法：PYTHONPATH=. python run_turnover_12d.py <data_root> <preds.parquet> <out_dir> [amihud_panel，默认 alpha_panel_weekly_v6_1.parquet]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quantmind.execution.nav_engine import NavConfig
from quantmind.execution.nav_gate import GateThresholds, evaluate_gate
from quantmind.execution.nav_turnover import TurnoverControlledNavEngine

VARIANTS = {
    "B0_baseline": {"step": 3, "exit_rank": None, "drift_band": None},
    "B1_buffer600": {"step": 3, "exit_rank": 600, "drift_band": None},
    "B2_band50": {"step": 3, "exit_rank": None, "drift_band": 0.5},
    "B3_buffer600_band50": {"step": 3, "exit_rank": 600, "drift_band": 0.5},
    "B4_step6": {"step": 6, "exit_rank": None, "drift_band": None},
}
PRIMARY, BASE = "B3_buffer600_band50", "B0_baseline"
BLOCK, N_BOOT, SEED = 15, 10_000, 0
THRESH = GateThresholds(max_drawdown=0.15)      # 与 scripts/run_executable_nav.py 的 12d 相同


def block_bootstrap_mean(d: np.ndarray, block: int = BLOCK, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    T = len(d)
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :T]
    return d[idx].mean(axis=1)


def main() -> None:
    root, preds_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    D = root / "data"
    preds = pd.read_parquet(preds_path)
    preds["as_of"] = pd.to_datetime(preds["as_of"])
    cols = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "adj_factor", "amount"]
    prices = pd.read_parquet(D / "raw/alpha_prices_panel_v6_20261009.parquet", columns=cols)
    am_file = sys.argv[4] if len(sys.argv) > 4 else "alpha_panel_weekly_v6_1.parquet"
    amihud = pd.read_parquet(D / "panel" / am_file, columns=["amihud_illiquidity"])
    am_dates = pd.DatetimeIndex(amihud.index.get_level_values("as_of").unique())
    cover = float(pd.DatetimeIndex(preds.as_of.unique()).isin(am_dates).mean())
    if cover < 0.99:
        raise SystemExit(f"amihud panel covers only {cover:.1%} of prediction dates; pass the matching panel")

    res = {"inputs": {"preds": str(preds_path), "n_as_of": int(preds.as_of.nunique()),
                      "as_of_range": [str(preds.as_of.min().date()), str(preds.as_of.max().date())],
                      "prices": "alpha_prices_panel_v6_20261009.parquet", "amihud": am_file,
                      "amihud_date_coverage": cover},
           "variants": {}}
    daily = {}
    for name, v in VARIANTS.items():
        t0 = time.time()
        cfg = NavConfig(horizon_td=12, rebalance_step=v["step"], top_n=300, universe_size=1500)
        eng = TurnoverControlledNavEngine(preds, prices, cfg, amihud=amihud,
                                          exit_rank=v["exit_rank"], drift_band=v["drift_band"])
        r = eng.run()
        s = r["summary"]
        g = evaluate_gate(s, THRESH)
        n = s["nav"]
        nav = r["nav_daily"].set_index("date")
        daily[name] = nav["nav_net"].pct_change().dropna()
        nav[["nav_net", "nav_gross", "nav_bench", "cum_cost", "n_positions"]].to_parquet(out_dir / f"nav_{name}.parquet")
        kept = pd.DataFrame(eng.kept_by_buffer)
        res["variants"][name] = {
            "params": v, "period": s["period"],
            "ann_net_excess": n["ann_net_excess"], "ann_gross_excess": n["ann_return_gross"] - n["ann_return_bench"],
            "information_ratio": n["information_ratio"], "max_drawdown_net": n["max_drawdown_net"],
            "yearly_net_excess": n["yearly_net_excess"], "cost_drag_ann": s["costs"]["cost_drag_ann"],
            "turnover_annualized_twoway": s["turnover"]["annualized_twoway"],
            "n_trades_filled": s["execution"]["n_trades_filled"],
            "rejections_distinct": s["execution"]["rejection_distinct_order_counts"],
            "mean_kept_outside_top300": float(kept["n_kept_outside_top_n"].mean()) if len(kept) else None,
            "gate_pass": g["gate_pass"], "gate_checks": [{k: c[k] for k in ("criterion", "passed")} for c in g["checks"]],
            "seconds": round(time.time() - t0, 1),
        }
        print(name, json.dumps({k: res["variants"][name][k] for k in
                                ("ann_net_excess", "ann_gross_excess", "information_ratio", "cost_drag_ann",
                                 "turnover_annualized_twoway", "gate_pass", "seconds")}), flush=True)

    comp = {}
    for name in VARIANTS:
        if name == BASE:
            continue
        a, b = daily[name].align(daily[BASE], join="inner")
        d = (a - b).to_numpy()
        boots = block_bootstrap_mean(d) * 252
        comp[f"{name}_minus_{BASE}"] = {
            "n_days": int(len(d)), "ann_mean_diff": float(d.mean() * 252),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "primary": name == PRIMARY}
    res["paired_vs_baseline"] = comp
    p = comp[f"{PRIMARY}_minus_{BASE}"]
    res["primary_verdict"] = {
        "effective": bool(p["ann_mean_diff"] > 0 and p["ci95"][0] > 0),
        "primary_gate_pass": res["variants"][PRIMARY]["gate_pass"],
        "rule": "effective = 年化均值差 > 0 且 95% 区间下界 > 0；12d 是否可用只看 B3 是否通过原闸门（阈值不变）",
    }
    (out_dir / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res["paired_vs_baseline"], indent=1), json.dumps(res["primary_verdict"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
