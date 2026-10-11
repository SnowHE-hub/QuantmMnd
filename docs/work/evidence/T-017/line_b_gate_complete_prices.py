"""主线 B 两颗种子的真实成交 NAV 与闸门：原价格表（缺 18 个交易日）vs T-015 的完整价格表（只读，不写 reports/、不动 registry）.

复用 scripts/run_executable_nav.py 的 MODELS、NavConfig、evaluate_gate；只替换价格表：
  same_window  完整价格表截到 2026-05-11（与原评估同一区间：只看补齐 18 天的影响）
  extended     完整价格表到 2026-10-09（预测止于 2026-04-20，最后一批持仓的持有期得以走完）
用法（t/017 工作区根目录）：PYTHONPATH=. python docs/work/evidence/T-017/line_b_gate_complete_prices.py <data_root> <out.json>
"""
import json
import sys
from pathlib import Path

import pandas as pd

import scripts.run_executable_nav as ren
from quantmind.execution.nav_engine import ExecutableNavEngine, NavConfig
from quantmind.execution.nav_gate import evaluate_gate

root, out = Path(sys.argv[1]), Path(sys.argv[2])
COLS = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "adj_factor", "amount"]
px_full = pd.read_parquet(root / "data/raw/alpha_prices_panel_v6_20261009.parquet", columns=COLS)
amihud = pd.read_parquet(root / "data/panel/alpha_panel_weekly_v6.parquet", columns=["amihud_illiquidity"])
res = {}
for key, spec in ren.MODELS.items():
    preds = pd.read_parquet(root / spec["preds"].relative_to(ren.REPO))
    old = json.loads((root / f"reports/executable_nav/{spec['model_id']}/summary.json").read_text(encoding="utf-8"))
    res[key] = {"model_id": spec["model_id"], "original": {k: old["nav"][k] for k in (
        "ann_return_net", "ann_return_bench", "ann_net_excess", "information_ratio", "max_drawdown_net")} | {
        "period": old["period"]}}
    for name, end in (("same_window", "2026-05-11"), ("extended", None)):
        px = px_full if end is None else px_full[pd.to_datetime(px_full["trade_date"]) <= pd.Timestamp(end)]
        cfg = NavConfig(horizon_td=spec["horizon_td"], rebalance_step=spec["rebalance_step"], top_n=300,
                        universe_size=1500)
        s = ExecutableNavEngine(preds, px, cfg, amihud=amihud).run()["summary"]
        g = evaluate_gate(s, spec["thresholds"])
        res[key][name] = {k: s["nav"][k] for k in ("ann_return_net", "ann_return_bench", "ann_net_excess",
                                                    "information_ratio", "max_drawdown_net")} | {
            "period": s["period"], "gate_pass": g["gate_pass"]}
        r = res[key][name]
        print(f"{key} {name}: excess {r['ann_net_excess']:+.4f} IR {r['information_ratio']:+.3f} "
              f"MDD {r['max_drawdown_net']:+.3f} gate {r['gate_pass']} ({r['period']['start']}→{r['period']['end']})",
              flush=True)
    o = res[key]["original"]
    print(f"{key} original: excess {o['ann_net_excess']:+.4f} IR {o['information_ratio']:+.3f}", flush=True)
out.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
