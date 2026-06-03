"""quantmind/execution/optimizer.py — 执行参数网格搜索 + Pareto 前沿.

用法：
  from quantmind.execution.replay_engine import (
      HistoricalReplayEngine, load_historical_recommendations, preload_price_history,
  )
  from quantmind.execution.optimizer import ExecutionParamOptimizer
  from app.db.postgres import get_pg_engine

  eng = get_pg_engine()
  recs = load_historical_recommendations(eng)
  prices = preload_price_history(eng, recs)
  replay = HistoricalReplayEngine(recs, prices)

  opt = ExecutionParamOptimizer()
  results = opt.run_grid_search(replay)
  pareto = opt.find_pareto_optimal(results)
  best = opt.recommend_best_params(results, criteria="sharpe")
"""
from __future__ import annotations

import itertools
import logging
from typing import Any

import numpy as np
import pandas as pd

from quantmind.execution.replay_engine import (
    HistoricalReplayEngine, ReplayParams,
)

log = logging.getLogger(__name__)


class ExecutionParamOptimizer:
    """在历史推荐上扫参数网格，寻找 Pareto 最优。"""

    DEFAULT_GRID = {
        "stop_loss":     [-0.05, -0.08, -0.10, -0.12, -0.15, -0.20],
        "target_price":  [0.15, 0.20, 0.25, 0.30, 0.40, None],   # None = 不止盈
        "trailing_stop": [-0.10, -0.15, -0.20, None],            # None = 不追踪
        "holding_days":  [42, 63, 84, 126],                       # 2m/3m/4m/6m
    }

    def __init__(self, grid: dict | None = None) -> None:
        self.grid = grid or self.DEFAULT_GRID

    # ── 网格搜索 ─────────────────────────────────────────────────────────────

    def run_grid_search(
        self,
        replay_engine: HistoricalReplayEngine,
        progress_cb=None,
    ) -> pd.DataFrame:
        """对每个参数组合跑一次回放，返回 (n_combos, ...) DataFrame。"""
        keys = list(self.grid.keys())
        values = [self.grid[k] for k in keys]
        all_combos = list(itertools.product(*values))
        total = len(all_combos)

        rows: list[dict] = []
        for idx, combo in enumerate(all_combos):
            params_dict = dict(zip(keys, combo))
            params = ReplayParams(
                stop_loss=params_dict.get("stop_loss"),
                target_price=params_dict.get("target_price"),
                trailing_stop=params_dict.get("trailing_stop"),
                holding_days=int(params_dict.get("holding_days") or 63),
            )
            result = replay_engine.replay(params)
            metrics = result["metrics"]
            reasons = metrics.get("exit_reasons", {})

            row = {
                **params_dict,
                "n":               metrics.get("n"),
                "cum_return":      metrics.get("cum_return"),
                "maxdd":           metrics.get("maxdd"),
                "sharpe":          metrics.get("sharpe"),
                "win_rate":        metrics.get("win_rate"),
                "avg_return":      metrics.get("avg_return"),
                "avg_holding_days": metrics.get("avg_holding_days"),
                "n_stop_loss":     int(reasons.get("stop_loss", 0)),
                "n_target_hit":    int(reasons.get("target_hit", 0)),
                "n_trailing_stop": int(reasons.get("trailing_stop", 0)),
                "n_time_expired":  int(reasons.get("time_expired", 0)),
            }
            rows.append(row)

            if progress_cb is not None and (idx + 1) % 50 == 0:
                progress_cb(idx + 1, total)

        return pd.DataFrame(rows)

    # ── Pareto 前沿 ──────────────────────────────────────────────────────────

    @staticmethod
    def find_pareto_optimal(
        results_df: pd.DataFrame,
        objectives: tuple[str, str] = ("cum_return", "maxdd"),
    ) -> pd.DataFrame:
        """两目标 Pareto 前沿。

        默认 (cum_return, maxdd)：
          - cum_return 越大越好
          - maxdd 越大越好（即更接近 0 / 数值更"靠右"）
        返回不被任何其他点严格支配的子集。
        """
        df = results_df.dropna(subset=list(objectives)).copy()
        if df.empty:
            return df

        a, b = objectives
        # 用 numpy 加速
        arr = df[[a, b]].to_numpy()
        n = len(arr)
        is_pareto = np.ones(n, dtype=bool)
        for i in range(n):
            if not is_pareto[i]:
                continue
            # 被其他点 j "支配"：j 的两个目标都 >= i，且至少一个严格 >
            mask = ((arr[:, 0] >= arr[i, 0]) & (arr[:, 1] >= arr[i, 1]) &
                    ((arr[:, 0] > arr[i, 0]) | (arr[:, 1] > arr[i, 1])))
            if mask.any():
                is_pareto[i] = False

        pareto = df[is_pareto].copy()
        # 按 cum_return 排序便于看
        return pareto.sort_values(a, ascending=False).reset_index(drop=True)

    # ── 单目标最优 ───────────────────────────────────────────────────────────

    @staticmethod
    def recommend_best_params(
        results_df: pd.DataFrame,
        criteria: str = "sharpe",
        constraints: dict[str, float] | None = None,
    ) -> dict | None:
        """挑选单一目标最优的参数。

        criteria: 'sharpe' / 'return' / 'win_rate' / 'maxdd'
        constraints: 例如 {'maxdd_min': -0.05} 表示 MaxDD 不能比 -5% 更差
        """
        df = results_df.dropna(subset=[]).copy() if results_df is not None else None
        if df is None or df.empty:
            return None

        # 应用约束
        if constraints:
            for k, v in constraints.items():
                if k == "maxdd_min":
                    df = df[df["maxdd"] >= v]
                elif k == "min_n_trades":
                    df = df[df["n"] >= v]
                elif k == "min_win_rate":
                    df = df[df["win_rate"] >= v]
            if df.empty:
                return None

        column_map = {
            "sharpe":   "sharpe",
            "return":   "cum_return",
            "win_rate": "win_rate",
            "maxdd":    "maxdd",
        }
        col = column_map.get(criteria, "sharpe")
        df = df.dropna(subset=[col])
        if df.empty:
            return None
        best = df.loc[df[col].idxmax()]
        return best.to_dict()

    # ── 与基准对比 ───────────────────────────────────────────────────────────

    @staticmethod
    def beat_baseline(
        results_df: pd.DataFrame,
        baseline_cum_return: float,
        baseline_maxdd: float,
    ) -> dict[str, Any]:
        """统计有多少组合能击败给定基准。"""
        if results_df is None or results_df.empty:
            return {}
        df = results_df.dropna(subset=["cum_return", "maxdd"])
        beat_return = df["cum_return"] > baseline_cum_return
        beat_dd = df["maxdd"] > baseline_maxdd
        return {
            "total":              int(len(df)),
            "beat_return":        int(beat_return.sum()),
            "beat_maxdd":         int(beat_dd.sum()),
            "beat_both":          int((beat_return & beat_dd).sum()),
            "beat_return_pct":    float(beat_return.mean()),
            "beat_maxdd_pct":     float(beat_dd.mean()),
            "beat_both_pct":      float((beat_return & beat_dd).mean()),
        }
