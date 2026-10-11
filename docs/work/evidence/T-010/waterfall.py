"""T-010 补充：主线 A 旧口径 → 真实成交口径的瀑布分解（同一批样本外分数）.

每一步只多打开一个约束，差值即该约束的代价：
  L0  旧口径 equal，零成本（run_nav_backtest：as_of 收盘建仓、持有到下一个 as_of）
  L1  旧口径 equal，单边 13bp
  E0  ExecutableNavEngine，零成本、不限选股池（T+1 开盘成交 + 涨跌停/停牌拒单 + v6 价格）
  E1  E0 + PIT adv20 前 1500 选股池
  E2  E1 + 真实成本（分层滑点 + 佣金 + 过户费 + 时变印花税）= T-010 主结果

用法（仓库根目录）：
  PYTHONPATH=. python docs/work/evidence/T-010/waterfall.py --data-root ~/projects/quantmind \
      --scores docs/work/evidence/T-010/run_v4_1/oos_scores.csv --out /tmp/t010_waterfall.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

import quantmind.execution.nav_engine as ne
from quantmind.backtest.wf_costs import SlippageTiers
from scripts.run_line_a_oos import PANEL_REL, executable_nav, legacy_nav


def run_exec(scores: pd.DataFrame, root: Path, universe_size: int, costs: bool) -> dict:
    orig_cfg, orig_stamp = ne.NavConfig, ne.stamp_duty_rate
    try:
        if not costs:
            zero = SlippageTiers(large_bp=0.0, mid_bp=0.0, small_bp=0.0)
            ne.NavConfig = lambda *a, **k: replace(orig_cfg(*a, **k), commission_bp=0.0,
                                                   transfer_bp=0.0, tiers=zero)
            ne.stamp_duty_rate = lambda *_: 0.0
        res = executable_nav(scores, 30, universe_size, root)
    finally:
        ne.NavConfig, ne.stamp_duty_rate = orig_cfg, orig_stamp
    n = res["summary"]["nav"]
    return {"ann_net": n["ann_return_net"], "ann_gross": n["ann_return_gross"],
            "ann_bench": n["ann_return_bench"], "excess": n["ann_net_excess"],
            "ir": n["information_ratio"], "mdd": n["max_drawdown_net"],
            "turnover_oneway": res["summary"]["turnover"]["annualized_oneway"],
            "cost_drag": res["summary"]["costs"]["cost_drag_ann"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    scores = pd.read_csv(a.scores, parse_dates=["as_of"])
    panel = pd.read_parquet(a.data_root / PANEL_REL)
    out = {}
    for name, bps in (("L0_legacy_gross", 0.0), ("L1_legacy_13bp", 13.0)):
        m = legacy_nav(panel, scores, 30, "equal", bps, a.data_root)["metrics"]
        out[name] = {"ann": m["ann_return"], "sharpe": m["sharpe_ratio"], "mdd": m["max_drawdown"]}
        print(name, out[name], flush=True)
    for name, uni, costs in (("E0_exec_nocost_nofilter", 1_000_000, False),
                             ("E1_exec_nocost_top1500", 1500, False),
                             ("E2_exec_full", 1500, True)):
        out[name] = run_exec(scores, a.data_root, uni, costs)
        print(name, {k: round(v, 4) for k, v in out[name].items()}, flush=True)
    a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
