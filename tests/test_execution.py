"""
tests/test_execution.py — E3 执行层测试套件

覆盖：
  StopLossEngine: 6 个规则测试 + 优先级
  ExecutionManager: 开仓/平仓/每日维护/has_open_position
  DataService: get_simulated_orders / get_execution_stats / vs_hold
  集成: 完整生命周期 3 个场景
  辅助: parse_holding_horizon / parse_position_size
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── 辅助：构造订单 dict ──────────────────────────────────────────────────────

def _make_order(**overrides) -> dict:
    base = {
        "order_id":        1,
        "ticker":          "000001.SZ",
        "open_date":       date(2026, 5, 1),
        "open_price":      10.0,
        "target_price":    12.0,
        "stop_loss_price": 9.0,
        "holding_period":  63,
        "status":          "OPEN",
        "high_price":      10.0,
        "low_price":       10.0,
        "industry":        "测试",
    }
    base.update(overrides)
    return base


# ═════════════════════════════════════════════════════════════════════════════
# 1. StopLossEngine 规则测试
# ═════════════════════════════════════════════════════════════════════════════

class TestStopLossRules:
    def setup_method(self):
        from quantmind.execution.stop_loss_engine import StopLossEngine
        self.engine = StopLossEngine(trailing_stop_pct=0.15)

    def test_stop_loss_triggered_when_price_below_threshold(self):
        order = _make_order(stop_loss_price=9.0)
        d = self.engine.check_single(order, current_price=8.5, as_of=date(2026, 5, 10))
        assert d is not None
        assert d.exit_reason == "stop_loss"
        assert d.exit_price == 8.5

    def test_target_hit_when_price_reaches_target(self):
        order = _make_order(target_price=12.0)
        d = self.engine.check_single(order, current_price=12.5, as_of=date(2026, 5, 10))
        assert d is not None
        assert d.exit_reason == "target_hit"

    def test_no_exit_when_within_bounds(self):
        order = _make_order(target_price=12.0, stop_loss_price=9.0, high_price=10.5)
        d = self.engine.check_single(order, current_price=10.5, as_of=date(2026, 5, 10))
        assert d is None

    def test_trailing_stop_from_peak(self):
        # 高点 12，回撤 20% 到 9.6（>15%阈值）
        order = _make_order(open_price=10.0, high_price=12.0, target_price=15.0,
                            stop_loss_price=8.0)
        d = self.engine.check_single(order, current_price=9.6, as_of=date(2026, 5, 10))
        assert d is not None
        assert d.exit_reason == "trailing_stop"

    def test_trailing_stop_not_triggered_below_threshold(self):
        # 高点 12，回撤 10% 到 10.8（<15%）
        order = _make_order(open_price=10.0, high_price=12.0, target_price=15.0,
                            stop_loss_price=8.0)
        d = self.engine.check_single(order, current_price=10.8, as_of=date(2026, 5, 10))
        assert d is None

    def test_time_expired_at_holding_period_end(self):
        order = _make_order(
            open_date=date(2026, 1, 1), holding_period=63,
            target_price=20, stop_loss_price=5, high_price=10.5,
        )
        # 64 天后
        d = self.engine.check_single(order, current_price=10.2,
                                      as_of=date(2026, 3, 5))
        assert d is not None
        assert d.exit_reason == "time_expired"

    def test_stop_loss_priority_over_time_expired(self):
        order = _make_order(
            open_date=date(2026, 1, 1), holding_period=63,
            stop_loss_price=9.0, target_price=20,
        )
        # 同时满足止损 + 到期，优先止损
        d = self.engine.check_single(order, current_price=8.0,
                                      as_of=date(2026, 5, 1))
        assert d.exit_reason == "stop_loss"

    def test_regime_change_disabled_by_default(self):
        order = _make_order(industry="证券", target_price=20, stop_loss_price=5)
        d = self.engine.check_single(order, current_price=10.5,
                                      as_of=date(2026, 5, 10),
                                      regime="bear", regime_prev="bull")
        # 默认 use_regime_exit=False
        assert d is None

    def test_regime_change_triggers_when_enabled(self):
        from quantmind.execution.stop_loss_engine import StopLossEngine
        eng = StopLossEngine(use_regime_exit=True,
                              regime_exit_industries={"证券"})
        order = _make_order(industry="证券", target_price=20, stop_loss_price=5,
                            high_price=10.5)
        d = eng.check_single(order, current_price=10.4,
                              as_of=date(2026, 5, 10),
                              regime="bear", regime_prev="bull")
        assert d is not None
        assert d.exit_reason == "regime_change"

    def test_evaluate_orders_batch(self):
        orders = [
            _make_order(order_id=1, ticker="A", stop_loss_price=9),
            _make_order(order_id=2, ticker="B", target_price=12),
            _make_order(order_id=3, ticker="C", target_price=20,
                        stop_loss_price=5, high_price=10.2),
        ]
        prices = {"A": 8.5, "B": 12.5, "C": 10.2}
        decisions = self.engine.evaluate_orders(orders, prices, date(2026, 5, 10))
        assert len(decisions) == 2  # A 和 B 触发，C 没事
        reasons = {d.ticker: d.exit_reason for d in decisions}
        assert reasons["A"] == "stop_loss"
        assert reasons["B"] == "target_hit"

    def test_skip_non_open_orders(self):
        order = _make_order(status="CLOSED", stop_loss_price=9)
        d = self.engine.check_single(order, 8, date(2026, 5, 10))
        assert d is None

    def test_invalid_price_returns_none(self):
        order = _make_order(stop_loss_price=9)
        assert self.engine.check_single(order, current_price=0, as_of=date.today()) is None
        assert self.engine.check_single(order, current_price=None, as_of=date.today()) is None


# ═════════════════════════════════════════════════════════════════════════════
# 2. parse_holding_horizon / parse_position_size
# ═════════════════════════════════════════════════════════════════════════════

class TestParsingHelpers:
    def test_parse_horizon_days(self):
        from quantmind.execution.manager import parse_holding_horizon
        assert parse_holding_horizon("21d") == 21
        assert parse_holding_horizon("63d") == 63

    def test_parse_horizon_months(self):
        from quantmind.execution.manager import parse_holding_horizon
        assert parse_holding_horizon("3m") == 63
        assert parse_holding_horizon("1m") == 21

    def test_parse_horizon_default(self):
        from quantmind.execution.manager import parse_holding_horizon, DEFAULT_HOLDING_DAYS
        assert parse_holding_horizon(None) == DEFAULT_HOLDING_DAYS
        assert parse_holding_horizon("") == DEFAULT_HOLDING_DAYS

    def test_parse_horizon_int(self):
        from quantmind.execution.manager import parse_holding_horizon
        assert parse_holding_horizon(63) == 63
        assert parse_holding_horizon(42.0) == 42

    def test_parse_position_size_pct_string(self):
        from quantmind.execution.manager import parse_position_size
        assert parse_position_size("5%") == 0.05
        assert parse_position_size("3-5%") == pytest.approx(0.04, abs=1e-3)

    def test_parse_position_size_chinese(self):
        from quantmind.execution.manager import parse_position_size
        assert parse_position_size("轻仓(1-3%)") == pytest.approx(0.02, abs=1e-3)
        assert parse_position_size("重仓") == 0.10

    def test_parse_position_size_float(self):
        from quantmind.execution.manager import parse_position_size
        assert parse_position_size(0.07) == 0.07


# ═════════════════════════════════════════════════════════════════════════════
# 3. ExecutionManager（集成测试，需要 PG）
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestExecutionManagerIntegration:
    """需要 PG simulated_orders 表已存在。"""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """每个测试前后清理测试用的 ticker。"""
        from sqlalchemy import text
        from app.db.postgres import get_pg_engine
        TEST_TICKERS = ("TEST.SZ", "TEST2.SZ", "TEST3.SZ")
        eng = get_pg_engine()
        with eng.begin() as conn:
            for t in TEST_TICKERS:
                conn.execute(text("DELETE FROM simulated_orders WHERE ticker=:t"),
                             {"t": t})
        yield
        with eng.begin() as conn:
            for t in TEST_TICKERS:
                conn.execute(text("DELETE FROM simulated_orders WHERE ticker=:t"),
                             {"t": t})

    def _mock_prices_panel(self) -> pd.DataFrame:
        """构造测试用价格面板。"""
        rows = []
        # TEST.SZ 平稳走势
        for i in range(70):
            d = date(2026, 1, 1) + timedelta(days=i)
            rows.append({"ts_code": "TEST.SZ", "trade_date": d,
                         "close": 10.0 + (i % 3) * 0.1,
                         "high": 10.2, "low": 9.9})
        # TEST2.SZ 暴跌触发止损
        for i in range(20):
            d = date(2026, 1, 1) + timedelta(days=i)
            close = 10.0 - i * 0.2
            rows.append({"ts_code": "TEST2.SZ", "trade_date": d,
                         "close": close, "high": close + 0.1, "low": close - 0.1})
        # TEST3.SZ 大涨触发止盈
        for i in range(20):
            d = date(2026, 1, 1) + timedelta(days=i)
            close = 10.0 + i * 0.3
            rows.append({"ts_code": "TEST3.SZ", "trade_date": d,
                         "close": close, "high": close + 0.2, "low": close - 0.1})
        return pd.DataFrame(rows)

    def test_open_position_inserts_db_row(self):
        from quantmind.execution.manager import ExecutionManager
        mgr = ExecutionManager(prices_panel=self._mock_prices_panel())
        rec = {"ticker": "TEST.SZ", "name": "测试1", "industry": "测试",
               "entry_price": 10.0, "rank": 1}
        order_id = mgr.open_position_from_recommendation(
            rec, agent_analysis=None, as_of=date(2026, 1, 2))
        assert order_id is not None
        # 验证 DB
        orders = mgr.get_open_orders()
        ours = [o for o in orders if o["ticker"] == "TEST.SZ"]
        assert len(ours) == 1
        assert ours[0]["open_price"] == 10.0
        assert ours[0]["target_price"] == pytest.approx(10.0 * 1.20)
        assert ours[0]["stop_loss_price"] == pytest.approx(10.0 * 0.90)

    def test_has_open_position_returns_correct_bool(self):
        from quantmind.execution.manager import ExecutionManager
        mgr = ExecutionManager(prices_panel=self._mock_prices_panel())
        assert mgr.has_open_position("TEST.SZ") is False
        mgr.open_position_from_recommendation(
            {"ticker": "TEST.SZ", "entry_price": 10.0},
            agent_analysis=None, as_of=date(2026, 1, 2))
        assert mgr.has_open_position("TEST.SZ") is True

    def test_open_position_skips_when_already_open(self):
        from quantmind.execution.manager import ExecutionManager
        mgr = ExecutionManager(prices_panel=self._mock_prices_panel())
        rec = {"ticker": "TEST.SZ", "entry_price": 10.0}
        id1 = mgr.open_position_from_recommendation(rec, None, date(2026, 1, 2))
        id2 = mgr.open_position_from_recommendation(rec, None, date(2026, 1, 3))
        assert id1 is not None
        assert id2 is None  # 已有 OPEN，跳过

    def test_close_position_calculates_pnl(self):
        from quantmind.execution.manager import ExecutionManager
        mgr = ExecutionManager(prices_panel=self._mock_prices_panel())
        order_id = mgr.open_position_from_recommendation(
            {"ticker": "TEST.SZ", "entry_price": 10.0},
            None, date(2026, 1, 2))
        result = mgr.close_position(order_id, close_price=11.0,
                                     close_reason="target_hit",
                                     as_of=date(2026, 2, 1))
        assert result["pnl_pct"] == pytest.approx(0.10, abs=1e-3)
        assert result["close_reason"] == "target_hit"
        # 验证 DB 状态
        assert mgr.has_open_position("TEST.SZ") is False

    def test_lifecycle_stop_loss(self):
        """完整生命周期：开仓 → 暴跌 → 止损触发。"""
        from quantmind.execution.manager import ExecutionManager
        mgr = ExecutionManager(prices_panel=self._mock_prices_panel())
        mgr.open_position_from_recommendation(
            {"ticker": "TEST2.SZ", "entry_price": 10.0},
            None, date(2026, 1, 1))

        # daily_update 在 day 6 时价格已经跌到 8.8（< 9.0 止损）
        summary = mgr.daily_update(as_of=date(2026, 1, 7))
        # 应该已经平仓
        closed = [c for c in summary["closes"] if c["ticker"] == "TEST2.SZ"]
        assert len(closed) == 1
        assert closed[0]["close_reason"] == "stop_loss"

    def test_lifecycle_target_hit(self):
        """完整生命周期：开仓 → 大涨 → 止盈触发。"""
        from quantmind.execution.manager import ExecutionManager
        mgr = ExecutionManager(prices_panel=self._mock_prices_panel())
        mgr.open_position_from_recommendation(
            {"ticker": "TEST3.SZ", "entry_price": 10.0},
            None, date(2026, 1, 1))

        # day 8: 价格 12.1，超过 12.0 目标
        summary = mgr.daily_update(as_of=date(2026, 1, 9))
        closed = [c for c in summary["closes"] if c["ticker"] == "TEST3.SZ"]
        assert len(closed) == 1
        assert closed[0]["close_reason"] == "target_hit"

    def test_lifecycle_time_expired(self):
        """完整生命周期：平稳走势 → 到期。"""
        from quantmind.execution.manager import ExecutionManager
        mgr = ExecutionManager(prices_panel=self._mock_prices_panel())
        mgr.open_position_from_recommendation(
            {"ticker": "TEST.SZ", "entry_price": 10.0},
            None, date(2026, 1, 1))

        # day 65 = 到期（默认 63 天）
        summary = mgr.daily_update(as_of=date(2026, 3, 10))
        closed = [c for c in summary["closes"] if c["ticker"] == "TEST.SZ"]
        assert len(closed) == 1
        assert closed[0]["close_reason"] == "time_expired"

    def test_daily_update_no_open_orders(self):
        from quantmind.execution.manager import ExecutionManager
        mgr = ExecutionManager(prices_panel=self._mock_prices_panel())
        summary = mgr.daily_update(as_of=date(2026, 6, 1))
        # 不要求没有，但至少应能正常返回
        assert "as_of" in summary
        assert "n_closed" in summary


# ═════════════════════════════════════════════════════════════════════════════
# 4. DataService 执行层方法
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestDataServiceExecution:
    """DataService 接口测试（依赖回填的真实 simulated_orders 数据）。"""

    def test_get_simulated_orders_all(self):
        from app.services.data_service import get_data_service
        svc = get_data_service()
        df = svc.get_simulated_orders(status="all")
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            for col in ("ticker", "open_date", "open_price", "status"):
                assert col in df.columns

    def test_get_simulated_orders_open_only(self):
        from app.services.data_service import get_data_service
        svc = get_data_service()
        df = svc.get_simulated_orders(status="OPEN")
        if not df.empty:
            assert (df["status"] == "OPEN").all()

    def test_get_execution_stats_returns_required_keys(self):
        from app.services.data_service import get_data_service
        svc = get_data_service()
        stats = svc.get_execution_stats(days=365)
        for key in ("total_orders", "open_orders", "closed_orders",
                    "win_rate", "avg_return", "exit_reasons",
                    "best_trade", "worst_trade"):
            assert key in stats, f"缺少 key: {key}"

    def test_get_execution_stats_consistency(self):
        from app.services.data_service import get_data_service
        svc = get_data_service()
        stats = svc.get_execution_stats(days=365)
        if stats.get("total_orders", 0) > 0:
            # open + closed 应该 <= total
            assert (stats["open_orders"] + stats["closed_orders"]) <= stats["total_orders"]
            # 胜率 in [0, 1]
            if stats.get("win_rate") is not None:
                assert 0 <= stats["win_rate"] <= 1

    def test_get_execution_vs_hold_comparison(self):
        from app.services.data_service import get_data_service
        svc = get_data_service()
        cmp = svc.get_execution_vs_hold_comparison()
        if "error" not in cmp:
            assert "execute" in cmp
            assert "hold_to_expiry" in cmp
            assert "curve" in cmp["execute"]
            assert "total_return" in cmp["execute"]
            assert "max_dd" in cmp["execute"]
