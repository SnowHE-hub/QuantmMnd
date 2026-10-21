"""T-036 X0（账本）：参考契约导出的引擎配置与研究期 eval_primary.py 的配置逐项相同，并在研究期末段用冻结分数跑出相同的
目标持仓与净值（规则在运行前写定）.

1. 配置：对四个参考臂，f8_books.nav_config(契约) == eval_primary 的 NavConfig(horizon_td, rebalance_step, top_n=300,
   universe_size=1500)（数据类逐字段相等，含滑点档位与退市冲销天数）；
2. 运行：Ridge 20 日参考臂的冻结研究期分数（ridge_20d_v6_1_BASE，2025-07-01 以后的 as_of），同一份输入分别用两种配置
   跑引擎，目标持仓逐次相同、日净值最大差 = 0（检验的是"契约 → 引擎配置"这一步，不是模型）。
用法：python x0_books.py <data_root> <out.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


def main() -> None:
    from quantmind.execution.nav_engine import ExecutableNavEngine, NavConfig
    from quantmind.line_a.f8_books import nav_config
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "T-030"))
    import eval_primary as ep
    contracts = json.loads((Path(__file__).resolve().parents[4] / "configs/line_a_v2/reference_strategies.json")
                           .read_text(encoding="utf-8"))["strategies"]
    res: dict = {"config": {}}
    for arm, c in contracts.items():
        hd, step, _ = ep.NAV[c["horizon"]]
        a, b = nav_config(c), NavConfig(horizon_td=hd, rebalance_step=step, top_n=300, universe_size=1500)
        res["config"][arm] = {"equal": a == b, "contract": repr(a), "eval_primary": repr(b)}
    D = root / "data"
    ridge = pd.read_parquet(D / ep.BASE["20d"])
    ridge["as_of"] = pd.to_datetime(ridge["as_of"])
    s = ridge[ridge["as_of"] >= pd.Timestamp("2025-07-01")][["as_of", "ticker", "score"]]
    cols = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "adj_factor", "amount"]
    prices = pd.read_parquet(D / "raw/alpha_prices_panel_v6_20261009.parquet", columns=cols)
    amihud = pd.read_parquet(D / "panel/alpha_panel_weekly_v6_1.parquet", columns=["amihud_illiquidity"])
    runs = {}
    for name, cfg in (("contract", nav_config(contracts["ridge_native_20d"])),
                      ("eval_primary", NavConfig(horizon_td=20, rebalance_step=4, top_n=300, universe_size=1500))):
        r = ExecutableNavEngine(s, prices, cfg, amihud=amihud).run()
        runs[name] = r
    t1, t2 = runs["contract"]["targets"], runs["eval_primary"]["targets"]
    same_targets = len(t1) == len(t2) and all(list(x) == list(y) for x, y in zip(t1["targets"], t2["targets"], strict=True))
    n1 = runs["contract"]["nav_daily"].set_index("date")["nav_net"]
    n2 = runs["eval_primary"]["nav_daily"].set_index("date")["nav_net"]
    res["run_ridge_20d"] = {"rebalances": int(len(t1)), "targets_identical": bool(same_targets),
                            "nav_max_abs_diff": float(np.abs(n1 - n2.reindex(n1.index)).max())}
    res["pass"] = all(v["equal"] for v in res["config"].values()) and same_targets and \
        res["run_ridge_20d"]["nav_max_abs_diff"] == 0
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"pass": res["pass"], "config_equal": {k: v["equal"] for k, v in res["config"].items()},
                      "run": res["run_ridge_20d"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
