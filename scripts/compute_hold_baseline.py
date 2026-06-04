"""scripts/compute_hold_baseline.py — 用真实价格重算"死扛"基准.

为什么需要它
============
optimize_execution_params.py 里的 HOLD_BASELINE 原来是硬编码常量
（cum_return=+5.22%）。这个数字本身是真实的——等于
mean(realized_pnl.actual_return_63d)——但它和网格搜索用的**收益口径不一致**：

  - 旧基准：actual_return_63d（面板 63 日前向收益，无滑点，覆盖全部 80 笔）
  - 网格组合：HistoricalReplayEngine 回放（次日开盘成交 + 0.1% 卖出滑点，
              到期按收盘×滑点；无价格数据的票记 0）

两者口径不同 → "击败死扛"的比较不是 apples-to-apples。

本脚本把"死扛"也用**同一个 HistoricalReplayEngine** 跑一遍（关掉止损/止盈/
追踪止损，只按 holding_days 到期），从而与每个网格组合**完全同口径**。
输出 data/execution_research/hold_baseline_real.json，供 optimize 脚本读取。

用法：
  conda run -n quantmind python scripts/compute_hold_baseline.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantmind.execution import (  # noqa: E402
    HistoricalReplayEngine,
    ReplayParams,
    load_historical_recommendations,
    preload_price_history,
)

HOLD_DAYS = 63  # "死扛 63 天"，与 realized_pnl 的 63 日口径一致


def main() -> int:
    out_dir = ROOT / "data" / "execution_research"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] 加载历史推荐 + 真实价格（parquet）...")
    recs = load_historical_recommendations()           # → price_source.load_realized_pnl (parquet)
    prices = preload_price_history(None, recs, max_holding_days=200)  # → alpha_prices_panel.parquet
    covered = sum(1 for r in recs if r.ticker in prices and prices[r.ticker] is not None
                  and not prices[r.ticker].empty)
    print(f"  推荐数: {len(recs)}  有价格覆盖: {covered}  缺覆盖: {len(recs) - covered}")

    print(f"[2/3] 用同一回放引擎跑死扛（sl/tg/trl=None, hold={HOLD_DAYS}d）...")
    engine = HistoricalReplayEngine(recs, prices)
    res = engine.replay(ReplayParams(
        stop_loss=None, target_price=None, trailing_stop=None, holding_days=HOLD_DAYS,
    ))
    m = res["metrics"]

    baseline = {
        "cum_return":       round(float(m["cum_return"]), 6),
        "maxdd":            round(float(m["maxdd"]), 6),
        "sharpe":           (round(float(m["sharpe"]), 4) if m["sharpe"] is not None else None),
        "win_rate":         round(float(m["win_rate"]), 6),
        "avg_holding_days": round(float(m["avg_holding_days"]), 2),
        "n":                int(m["n"]),
        "exit_reasons":     m.get("exit_reasons", {}),
    }

    # 参照：旧口径 mean(actual_return_63d)（无滑点、全 80 笔）
    rp = pd.read_parquet(ROOT / "data" / "feedback" / "realized_pnl.parquet")
    legacy_mean = float(rp["actual_return_63d"].dropna().mean())

    payload = {
        "method": ("HistoricalReplayEngine.replay(sl=None,tg=None,trl=None,"
                   f"hold={HOLD_DAYS}d); next-open fill + 0.1% sell slippage; "
                   "equal-weight 1/N NAV; same convention as grid combos"),
        "price_source": "data/raw/alpha_prices_panel.parquet (via price_source.load_price_panel)",
        "computed_at": datetime.now().isoformat(timespec="seconds"),
        "baseline": baseline,
        "legacy_hardcoded": {"cum_return": 0.0522, "maxdd": -0.0085, "sharpe": 3.67,
                             "win_rate": 0.488, "avg_holding_days": 63,
                             "note": "= mean(actual_return_63d) over 80 trades, no slippage"},
        "legacy_mean_actual_return_63d": round(legacy_mean, 6),
    }

    out_path = out_dir / "hold_baseline_real.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[3/3] 结果")
    print(f"  真实同口径死扛: cum={baseline['cum_return']*100:+.2f}%  "
          f"maxdd={baseline['maxdd']*100:.2f}%  sharpe={baseline['sharpe']}  "
          f"win={baseline['win_rate']*100:.1f}%  n={baseline['n']}")
    print(f"  exit_reasons: {baseline['exit_reasons']}")
    print(f"  旧硬编码:       cum=+5.22% (= mean actual_return_63d={legacy_mean*100:+.2f}%, 无滑点/全80笔)")
    print(f"  → 写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
