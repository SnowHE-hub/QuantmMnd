"""
tests/test_execution_optimizer.py — E3.5 参数优化研究测试.

覆盖：
  - ReplayParams dataclass 行为
  - replay_single_order 在合成价格下的退出规则
  - HistoricalReplayEngine 多笔聚合 + NAV 计算
  - ExecutionParamOptimizer grid_search 形状 / Pareto / 单目标最优
  - 极端参数：stop_loss=None 不止损、target=None 不止盈
  - beat_baseline 统计正确
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── 合成价格数据 ─────────────────────────────────────────────────────────────

def _make_bars(
    ticker: str = "TEST.SZ",
    start: date = date(2026, 1, 1),
    closes: list[float] | None = None,
) -> pd.DataFrame:
    if closes is None:
        closes = [10.0, 10.1, 10.2, 10.0, 9.9, 10.3, 10.5]
    rows = []
    for i, c in enumerate(closes):
        d = start + timedelta(days=i)
        rows.append({
            "ts_code": ticker, "trade_date": d,
            "open": c, "high": c + 0.1, "low": c - 0.1, "close": c,
        })
    return pd.DataFrame(rows)


# ── Tests ────────────────────────────────────────────────────────────────────

class TestReplayParams:
    def test_dataclass_frozen(self):
        from quantmind.execution.replay_engine import ReplayParams
        p = ReplayParams(stop_loss=-0.1, target_price=0.2,
                          trailing_stop=-0.15, holding_days=63)
        with pytest.raises(Exception):
            p.stop_loss = -0.2  # frozen

    def test_to_dict(self):
        from quantmind.execution.replay_engine import ReplayParams
        p = ReplayParams(stop_loss=-0.1, target_price=None,
                          trailing_stop=None, holding_days=63)
        d = p.to_dict()
        assert d["stop_loss"] == -0.1
        assert d["target_price"] is None
        assert d["holding_days"] == 63


class TestReplaySingleOrder:
    def _make_rec(self, ticker="TEST.SZ", entry_price=10.0):
        from quantmind.execution.replay_engine import HistoricalRecommendation
        return HistoricalRecommendation(
            ticker=ticker, recommend_date=date(2026, 1, 1),
            entry_date=date(2026, 1, 1), entry_price=entry_price,
        )

    def test_stop_loss_triggers(self):
        from quantmind.execution.replay_engine import (
            replay_single_order, ReplayParams,
        )
        # 暴跌：index=2 这根 low=8.5-0.1=8.4 < 9（-10% 阈值）触发
        # next index=3 open=7.0 → close_price = 7.0 * 0.999 = 6.993
        closes = [10.0, 9.5, 8.5, 7.0]
        bars = _make_bars(closes=closes)
        params = ReplayParams(stop_loss=-0.10, target_price=None,
                              trailing_stop=None, holding_days=63)
        result = replay_single_order(self._make_rec(), bars, params)
        assert result["close_reason"] == "stop_loss"
        # 触发在 index=2（next index=3 的 open=7.0），加滑点 ×0.999
        assert result["close_price"] == pytest.approx(7.0 * 0.999, abs=1e-4)
        # 损失应深于 -10% 阈值（次日跳空体现 T+1 真实代价）
        assert result["pnl_pct"] < -0.10

    def test_target_hit(self):
        from quantmind.execution.replay_engine import (
            replay_single_order, ReplayParams,
        )
        # 大涨：第 2 根 high=12.1 > 12（+20% 阈值）
        closes = [10.0, 12.0, 13.0, 13.5]
        bars = _make_bars(closes=closes)
        params = ReplayParams(stop_loss=None, target_price=0.20,
                              trailing_stop=None, holding_days=63)
        result = replay_single_order(self._make_rec(), bars, params)
        assert result["close_reason"] == "target_hit"
        assert result["pnl_pct"] > 0

    def test_no_stop_loss_when_none(self):
        from quantmind.execution.replay_engine import (
            replay_single_order, ReplayParams,
        )
        # 暴跌但 stop_loss=None 应不触发，走到到期
        closes = [10.0] + [8.0] * 70
        bars = _make_bars(closes=closes)
        params = ReplayParams(stop_loss=None, target_price=None,
                              trailing_stop=None, holding_days=5)
        result = replay_single_order(self._make_rec(), bars, params)
        # 不触发止损 → time_expired
        assert result["close_reason"] == "time_expired"
        assert result["pnl_pct"] < 0  # 实际损失

    def test_time_expired(self):
        from quantmind.execution.replay_engine import (
            replay_single_order, ReplayParams,
        )
        # 平稳走势 → 持仓到期
        closes = [10.0] * 20
        bars = _make_bars(closes=closes)
        params = ReplayParams(stop_loss=-0.50, target_price=0.50,
                              trailing_stop=None, holding_days=5)
        result = replay_single_order(self._make_rec(), bars, params)
        assert result["close_reason"] == "time_expired"
        # 滑点：close=10 → close_price ≈ 9.99
        assert result["close_price"] == pytest.approx(10.0 * 0.999, abs=1e-4)

    def test_trailing_stop(self):
        from quantmind.execution.replay_engine import (
            replay_single_order, ReplayParams,
        )
        # 涨到 12 后回撤到 10.1（高点 12.1 → -16.5% 回撤）
        closes = [10.0, 11.0, 12.0, 10.1, 10.0]
        bars = _make_bars(closes=closes)
        params = ReplayParams(stop_loss=None, target_price=None,
                              trailing_stop=-0.15, holding_days=63)
        result = replay_single_order(self._make_rec(), bars, params)
        assert result["close_reason"] == "trailing_stop"

    def test_no_data(self):
        from quantmind.execution.replay_engine import (
            replay_single_order, ReplayParams,
        )
        result = replay_single_order(
            self._make_rec(),
            pd.DataFrame(),
            ReplayParams(stop_loss=-0.1, target_price=0.2,
                          trailing_stop=None, holding_days=63),
        )
        assert result["close_reason"] == "no_data"
        assert result["pnl_pct"] == 0.0

    def test_stop_loss_priority(self):
        """同一天 high>target 且 low<stop_loss，优先止损。"""
        from quantmind.execution.replay_engine import (
            replay_single_order, ReplayParams,
        )
        bars = pd.DataFrame([
            {"ts_code": "X", "trade_date": date(2026, 1, 1),
             "open": 10, "high": 10, "low": 10, "close": 10},
            {"ts_code": "X", "trade_date": date(2026, 1, 2),
             "open": 12, "high": 13, "low": 8, "close": 9},  # 日内 8 < 9, 13 > 12
            {"ts_code": "X", "trade_date": date(2026, 1, 3),
             "open": 9, "high": 9, "low": 9, "close": 9},
        ])
        params = ReplayParams(stop_loss=-0.10, target_price=0.20,
                              trailing_stop=None, holding_days=63)
        from quantmind.execution.replay_engine import HistoricalRecommendation
        rec = HistoricalRecommendation(
            ticker="X", recommend_date=date(2026, 1, 1),
            entry_date=date(2026, 1, 1), entry_price=10.0)
        result = replay_single_order(rec, bars, params)
        assert result["close_reason"] == "stop_loss"


class TestReplayEngineAggregate:
    def test_replay_multiple_orders(self):
        from quantmind.execution.replay_engine import (
            HistoricalReplayEngine, HistoricalRecommendation, ReplayParams,
        )
        recs = [
            HistoricalRecommendation("A", date(2026, 1, 1), date(2026, 1, 1), 10.0),
            HistoricalRecommendation("B", date(2026, 1, 1), date(2026, 1, 1), 20.0),
        ]
        prices = {
            "A": _make_bars("A", closes=[10.0] + [10.5] * 70),
            "B": _make_bars("B", closes=[20.0] + [20.5] * 70),
        }
        engine = HistoricalReplayEngine(recs, prices)
        result = engine.replay(ReplayParams(
            stop_loss=-0.5, target_price=0.5,
            trailing_stop=None, holding_days=5,
        ))
        assert len(result["orders"]) == 2
        assert "metrics" in result
        assert result["metrics"]["n"] == 2

    def test_equal_weight_nav_not_cumprod(self):
        """两笔 +20% 收益 → 等权 NAV = 1.20，不是 1.44。"""
        from quantmind.execution.replay_engine import (
            HistoricalReplayEngine, HistoricalRecommendation, ReplayParams,
        )
        # 构造两个一定 +20% 平仓的 ticker（next open >= target）
        recs = [
            HistoricalRecommendation("A", date(2026, 1, 1), date(2026, 1, 1), 10.0),
            HistoricalRecommendation("B", date(2026, 1, 1), date(2026, 1, 1), 20.0),
        ]
        prices = {
            "A": _make_bars("A", closes=[10.0, 12.5, 13.0]),
            "B": _make_bars("B", closes=[20.0, 25.0, 26.0]),
        }
        engine = HistoricalReplayEngine(recs, prices)
        result = engine.replay(ReplayParams(
            stop_loss=None, target_price=0.20,
            trailing_stop=None, holding_days=63,
        ))
        cum = result["metrics"]["cum_return"]
        # 两笔都触发 target_hit（≈+20%）, 等权平均 → cum_return ≈ +20% 左右
        assert 0.15 < cum < 0.30, f"等权 NAV cum={cum}, 不应是 cumprod 失真"


class TestOptimizerGridSearch:
    def _build_engine(self, n_recs=3):
        from quantmind.execution.replay_engine import (
            HistoricalReplayEngine, HistoricalRecommendation,
        )
        recs = [
            HistoricalRecommendation(f"T{i}", date(2026, 1, 1),
                                       date(2026, 1, 1), 10.0)
            for i in range(n_recs)
        ]
        # 不同走势：T0 暴跌，T1 平稳，T2 大涨
        prices = {
            "T0": _make_bars("T0", closes=[10.0] + [8.0] * 70),
            "T1": _make_bars("T1", closes=[10.0] * 71),
            "T2": _make_bars("T2", closes=[10.0] + [13.0] * 70),
        }
        return HistoricalReplayEngine(recs, prices)

    def test_grid_search_returns_correct_shape(self):
        from quantmind.execution.optimizer import ExecutionParamOptimizer
        # 极简网格：2×2×1×1 = 4 组合
        opt = ExecutionParamOptimizer(grid={
            "stop_loss":     [-0.05, -0.10],
            "target_price":  [0.10, 0.20],
            "trailing_stop": [None],
            "holding_days":  [21],
        })
        engine = self._build_engine()
        results = opt.run_grid_search(engine)
        assert len(results) == 4
        assert "cum_return" in results.columns
        assert "sharpe" in results.columns

    def test_pareto_at_most_n_points(self):
        from quantmind.execution.optimizer import ExecutionParamOptimizer
        opt = ExecutionParamOptimizer(grid={
            "stop_loss":     [-0.05, -0.10, -0.15],
            "target_price":  [0.10, 0.20],
            "trailing_stop": [None],
            "holding_days":  [21],
        })
        engine = self._build_engine()
        results = opt.run_grid_search(engine)
        pareto = opt.find_pareto_optimal(results)
        assert len(pareto) <= len(results)
        # Pareto 集中应有最大 cum_return
        if not pareto.empty:
            assert pareto["cum_return"].max() == results["cum_return"].max()

    def test_pareto_excludes_dominated_points(self):
        """构造明显被支配的点，确认被剔除。"""
        from quantmind.execution.optimizer import ExecutionParamOptimizer
        results = pd.DataFrame([
            {"cum_return": 0.10, "maxdd": -0.05},  # A: 最高收益 + 最小回撤 → 支配 B 和 C
            {"cum_return": 0.05, "maxdd": -0.10},  # B: 都被 A 支配
            {"cum_return": 0.08, "maxdd": -0.08},  # C: 都被 A 支配
        ])
        pareto = ExecutionParamOptimizer.find_pareto_optimal(results)
        # 应该只有 A 在前沿
        assert len(pareto) == 1
        assert pareto.iloc[0]["cum_return"] == pytest.approx(0.10)

    def test_pareto_keeps_tradeoff_points(self):
        """两个互不支配的点都应保留。"""
        from quantmind.execution.optimizer import ExecutionParamOptimizer
        results = pd.DataFrame([
            {"cum_return": 0.10, "maxdd": -0.20},  # 高收益 高回撤
            {"cum_return": 0.05, "maxdd": -0.05},  # 低收益 低回撤
        ])
        pareto = ExecutionParamOptimizer.find_pareto_optimal(results)
        assert len(pareto) == 2

    def test_recommend_best_sharpe(self):
        from quantmind.execution.optimizer import ExecutionParamOptimizer
        results = pd.DataFrame([
            {"sharpe": 1.0, "cum_return": 0.10, "maxdd": -0.05, "win_rate": 0.5},
            {"sharpe": 2.0, "cum_return": 0.08, "maxdd": -0.03, "win_rate": 0.6},
            {"sharpe": 1.5, "cum_return": 0.12, "maxdd": -0.10, "win_rate": 0.55},
        ])
        best = ExecutionParamOptimizer.recommend_best_params(results, criteria="sharpe")
        assert best["sharpe"] == pytest.approx(2.0)

    def test_recommend_with_maxdd_constraint(self):
        from quantmind.execution.optimizer import ExecutionParamOptimizer
        results = pd.DataFrame([
            {"sharpe": 5.0, "cum_return": 0.30, "maxdd": -0.20, "win_rate": 0.5},  # 高收益但 DD 超约束
            {"sharpe": 2.0, "cum_return": 0.10, "maxdd": -0.04, "win_rate": 0.5},  # DD 合规
        ])
        best = ExecutionParamOptimizer.recommend_best_params(
            results, criteria="return", constraints={"maxdd_min": -0.05},
        )
        # 第一个被约束排除
        assert best["cum_return"] == pytest.approx(0.10)

    def test_beat_baseline_counts(self):
        from quantmind.execution.optimizer import ExecutionParamOptimizer
        results = pd.DataFrame([
            {"cum_return": 0.10, "maxdd": -0.05},   # 都击败 (0.05, -0.10)
            {"cum_return": 0.03, "maxdd": -0.03},   # 仅 DD 击败
            {"cum_return": 0.08, "maxdd": -0.15},   # 仅收益击败
        ])
        stats = ExecutionParamOptimizer.beat_baseline(
            results, baseline_cum_return=0.05, baseline_maxdd=-0.10,
        )
        assert stats["total"] == 3
        assert stats["beat_return"] == 2
        assert stats["beat_maxdd"] == 2
        assert stats["beat_both"] == 1


@pytest.mark.integration
class TestEndToEnd:
    """端到端：80 笔真实数据 + 极简网格。"""

    def test_real_data_grid_runs(self):
        from app.db.postgres import get_pg_engine
        from quantmind.execution import (
            ExecutionParamOptimizer, HistoricalReplayEngine,
            load_historical_recommendations, preload_price_history,
        )
        eng = get_pg_engine()
        recs = load_historical_recommendations(eng)
        if len(recs) < 10:
            pytest.skip("realized_pnl 数据不足")
        prices = preload_price_history(eng, recs[:10])
        replay = HistoricalReplayEngine(recs[:10], prices)
        opt = ExecutionParamOptimizer(grid={
            "stop_loss": [-0.10, -0.20],
            "target_price": [0.20, None],
            "trailing_stop": [None],
            "holding_days": [63],
        })
        results = opt.run_grid_search(replay)
        assert len(results) == 4
        assert results["cum_return"].notna().all()
