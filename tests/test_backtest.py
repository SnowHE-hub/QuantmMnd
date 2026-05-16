"""tests/test_backtest.py — Phase 6 回测引擎单元测试.

测试覆盖：
  1. test_t1_restriction       — T+1 限制（今日买入明日才能卖）
  2. test_price_limit_handling — 涨跌停不能成交
  3. test_metrics_correctness  — 用已知序列验证 Sharpe 计算
  4. test_dsr_detects_overfitting — 随机噪声策略 DSR 不显著
  5. test_walk_forward_no_lookahead — 验证每个测试窗口不用未来数据

全部使用 mock 数据，不依赖真实 API 或外部服务。
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── 被测模块 ──────────────────────────────────────────────────────────────────
from quantmind.backtest.portfolio import Portfolio, Position
from quantmind.backtest.execution import ExecutionSimulator, Order, Fill
from quantmind.backtest.metrics import PerformanceMetrics
from quantmind.backtest.statistical_tests import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from quantmind.backtest.walk_forward import WalkForwardConfig, WalkForwardValidator
from quantmind.backtest.engine import BacktestConfig


# ============================================================================
# 辅助函数
# ============================================================================

def _make_bar(
    ticker: str,
    open_: float = 10.0,
    close: float = 10.0,
    high: float = 10.5,
    low: float = 9.5,
    volume: float = 1_000_000.0,
    prev_close: float = 10.0,
) -> dict:
    """构造单支股票的 bar 字典."""
    return {
        "open": open_,
        "close": close,
        "high": high,
        "low": low,
        "volume": volume,
        "prev_close": prev_close,
        "vwap": (open_ + close) / 2.0,
    }


def _make_today_bar(tickers_bars: dict[str, dict]) -> pd.DataFrame:
    """从 {ticker: bar_dict} 构造 today_bar DataFrame."""
    return pd.DataFrame.from_dict(tickers_bars, orient="index")


def _make_portfolio(initial_capital: float = 10_000_000.0, t1: bool = True) -> Portfolio:
    """构造一个带初始资金的 Portfolio."""
    return Portfolio(initial_capital=initial_capital, t1_restriction=t1)


def _make_execution_sim(
    execution_mode: str = "close_price",
    commission_rate: float = 3e-4,
    stamp_duty: float = 1e-3,
    transfer_fee: float = 2e-5,
    slippage_bp: float = 0.0,   # 测试中关闭滑点，方便验证金额
    max_volume_pct: float = 1.0,
) -> ExecutionSimulator:
    cfg = BacktestConfig(
        execution_mode=execution_mode,
        commission_rate=commission_rate,
        stamp_duty=stamp_duty,
        transfer_fee=transfer_fee,
        slippage_bp=slippage_bp,
        max_volume_pct=max_volume_pct,
    )
    return ExecutionSimulator(cfg)


# ============================================================================
# Test 1: T+1 限制
# ============================================================================

class TestT1Restriction:
    """今日买入，明日才能卖（T+1）."""

    def test_buy_today_cannot_sell_same_day(self):
        """买入后当日 can_sell 应返回 False."""
        portfolio = _make_portfolio(t1=True)
        buy_date = date(2024, 1, 2)

        # 手动插入仓位（模拟今日买入）
        portfolio.positions["000001.SZ"] = Position(
            ticker="000001.SZ",
            shares=1000,
            avg_cost=10.0,
            buy_date=buy_date,
        )

        assert portfolio.can_sell("000001.SZ", buy_date) is False, \
            "当日买入不能当日卖出（T+1）"

    def test_buy_today_can_sell_next_day(self):
        """买入后次日 can_sell 应返回 True."""
        portfolio = _make_portfolio(t1=True)
        buy_date = date(2024, 1, 2)
        next_day = date(2024, 1, 3)

        portfolio.positions["000001.SZ"] = Position(
            ticker="000001.SZ",
            shares=1000,
            avg_cost=10.0,
            buy_date=buy_date,
        )

        assert portfolio.can_sell("000001.SZ", next_day) is True, \
            "次日应可卖出"

    def test_sellable_shares_zero_on_buy_day(self):
        """当日买入后 sellable_shares 应为 0."""
        portfolio = _make_portfolio(t1=True)
        buy_date = date(2024, 1, 2)

        portfolio.positions["600036.SH"] = Position(
            ticker="600036.SH",
            shares=500,
            avg_cost=50.0,
            buy_date=buy_date,
        )

        assert portfolio.sellable_shares("600036.SH", buy_date) == 0, \
            "当日买入后可卖股数应为 0"

    def test_t1_disabled_allows_same_day_sell(self):
        """关闭 T+1 后，当日买入可以当日卖出."""
        portfolio = _make_portfolio(t1=False)
        today = date(2024, 1, 2)

        portfolio.positions["000002.SZ"] = Position(
            ticker="000002.SZ",
            shares=200,
            avg_cost=20.0,
            buy_date=today,
        )

        assert portfolio.can_sell("000002.SZ", today) is True, \
            "关闭 T+1 后当日应可卖出"

    def test_execution_blocks_sell_on_t1_day(self):
        """通过 ExecutionSimulator.execute，T+1 当日卖出订单应被拒绝（0 成交）."""
        portfolio = _make_portfolio(t1=True)
        sim = _make_execution_sim()
        buy_date = date(2024, 1, 2)

        # 模拟今日买入持仓
        portfolio.positions["000001.SZ"] = Position(
            ticker="000001.SZ",
            shares=1000,
            avg_cost=10.0,
            buy_date=buy_date,
        )
        portfolio.cash = 5_000_000.0  # 剩余现金

        today_bar = _make_today_bar({
            "000001.SZ": _make_bar("000001.SZ", close=10.5),
        })

        # 当日发出卖出订单
        orders = [Order(ticker="000001.SZ", shares=500, direction="sell")]
        fills = sim.execute(orders, today_bar, portfolio, buy_date)

        sell_fills = [f for f in fills if f.direction == "sell"]
        assert len(sell_fills) == 0, "T+1 限制：当日买入不能当日卖出，应无成交"


# ============================================================================
# Test 2: 涨跌停不能成交
# ============================================================================

class TestPriceLimitHandling:
    """涨停不能买入，跌停不能卖出."""

    def _make_limit_up_bar(self, ticker: str, prev_close: float = 10.0) -> dict:
        """构造涨停 bar：close 比 prev_close 涨超 9.95%."""
        close = round(prev_close * 1.10, 2)  # 涨 10%
        return _make_bar(ticker, open_=close, close=close, prev_close=prev_close)

    def _make_limit_down_bar(self, ticker: str, prev_close: float = 10.0) -> dict:
        """构造跌停 bar：close 比 prev_close 跌超 9.95%."""
        close = round(prev_close * 0.90, 2)  # 跌 10%
        return _make_bar(ticker, open_=close, close=close, prev_close=prev_close)

    def test_buy_blocked_on_limit_up(self):
        """涨停时买入应被拒绝（0 成交）."""
        portfolio = _make_portfolio()
        sim = _make_execution_sim()

        today_bar = _make_today_bar({
            "300750.SZ": self._make_limit_up_bar("300750.SZ", prev_close=100.0),
        })
        orders = [Order(ticker="300750.SZ", shares=100, direction="buy")]
        fills = sim.execute(orders, today_bar, portfolio, date(2024, 3, 1))

        buy_fills = [f for f in fills if f.direction == "buy"]
        assert len(buy_fills) == 0, "涨停时买入不应成交"

    def test_sell_blocked_on_limit_down(self):
        """跌停时卖出应被拒绝（0 成交）."""
        portfolio = _make_portfolio()
        sim = _make_execution_sim()

        today = date(2024, 3, 1)
        # 持有股票（昨日买入，满足 T+1）
        portfolio.positions["000858.SZ"] = Position(
            ticker="000858.SZ",
            shares=500,
            avg_cost=50.0,
            buy_date=date(2024, 2, 28),
        )

        today_bar = _make_today_bar({
            "000858.SZ": self._make_limit_down_bar("000858.SZ", prev_close=50.0),
        })
        orders = [Order(ticker="000858.SZ", shares=200, direction="sell")]
        fills = sim.execute(orders, today_bar, portfolio, today)

        sell_fills = [f for f in fills if f.direction == "sell"]
        assert len(sell_fills) == 0, "跌停时卖出不应成交"

    def test_normal_price_allows_trade(self):
        """正常价格（无涨跌停）应可以正常成交."""
        portfolio = _make_portfolio()
        sim = _make_execution_sim()

        today_bar = _make_today_bar({
            "600519.SH": _make_bar("600519.SH", close=1800.0, prev_close=1800.0, volume=500_000),
        })
        orders = [Order(ticker="600519.SH", shares=100, direction="buy")]
        fills = sim.execute(orders, today_bar, portfolio, date(2024, 3, 1))

        buy_fills = [f for f in fills if f.direction == "buy"]
        assert len(buy_fills) == 1, "正常行情应可以成交"
        assert buy_fills[0].shares == 100

    def test_suspended_stock_not_traded(self):
        """停牌（volume=0）不能成交."""
        portfolio = _make_portfolio()
        sim = _make_execution_sim()

        today_bar = _make_today_bar({
            "000001.SZ": _make_bar("000001.SZ", volume=0.0),  # 停牌
        })
        orders = [Order(ticker="000001.SZ", shares=100, direction="buy")]
        fills = sim.execute(orders, today_bar, portfolio, date(2024, 3, 1))

        assert len(fills) == 0, "停牌股票不应成交"


# ============================================================================
# Test 3: 指标计算正确性
# ============================================================================

class TestMetricsCorrectness:
    """用已知序列验证 PerformanceMetrics 关键指标的数值正确性."""

    def _annual_sharpe(self, returns: np.ndarray, rf_annual: float = 0.025) -> float:
        """手动计算年化 Sharpe（与 PerformanceMetrics 相同逻辑）."""
        rf_daily = (1 + rf_annual) ** (1 / 252) - 1
        excess = returns - rf_daily
        return float(np.mean(excess) / (np.std(excess, ddof=1) + 1e-10) * np.sqrt(252))

    def test_zero_return_sharpe_negative(self):
        """全零收益序列的 Sharpe 应为负（因 rf > 0）."""
        returns = pd.Series(np.zeros(252))
        metrics = PerformanceMetrics().compute_all(returns)
        assert metrics["sharpe_ratio"] < 0, \
            "全零收益序列的 Sharpe 应为负（无风险利率 > 0）"

    def test_known_constant_return_metrics(self):
        """等额日收益 0.1% 的序列，可手动验证 Sharpe 值."""
        daily_ret = 0.001  # 每日 0.1%
        n = 252
        returns = pd.Series(np.full(n, daily_ret))

        metrics = PerformanceMetrics().compute_all(returns)

        # 手动计算期望 Sharpe
        expected_sharpe = self._annual_sharpe(np.full(n, daily_ret))

        assert abs(metrics["sharpe_ratio"] - expected_sharpe) < 0.01, \
            f"Sharpe 偏差过大：期望 {expected_sharpe:.4f}，实际 {metrics['sharpe_ratio']:.4f}"

    def test_max_drawdown_calculation(self):
        """验证最大回撤计算：先涨后跌的序列."""
        # 构造一个先涨 10% 再跌 15% 的序列
        # 净值：1 → 1.10 → 0.935；MDD 约 -15%
        np.random.seed(0)
        up = np.full(50, 0.002)      # 每日 +0.2%，上涨段
        down = np.full(50, -0.003)   # 每日 -0.3%，下跌段
        returns = pd.Series(np.concatenate([up, down]))

        metrics = PerformanceMetrics().compute_all(returns)

        # 最大回撤应为负数
        assert metrics["max_drawdown"] < 0, "最大回撤应为负数"
        # 下跌 50×0.3% = 15%，所以 MDD 约 -15%（累积净值下跌）
        assert metrics["max_drawdown"] < -0.10, \
            f"最大回撤应超过 -10%，实际 {metrics['max_drawdown']:.4f}"

    def test_total_return_calculation(self):
        """验证总收益计算：(1+r)^n - 1."""
        daily_ret = 0.001
        n = 100
        returns = pd.Series(np.full(n, daily_ret))

        metrics = PerformanceMetrics().compute_all(returns)

        expected_total = (1 + daily_ret) ** n - 1
        assert abs(metrics["total_return"] - expected_total) < 1e-4, \
            f"总收益偏差过大：期望 {expected_total:.6f}，实际 {metrics['total_return']:.6f}"

    def test_positive_sharpe_for_consistent_gains(self):
        """稳定正收益序列的 Sharpe 应显著大于 0."""
        daily_ret = 0.002  # 每日 0.2%
        returns = pd.Series(np.full(252, daily_ret))
        metrics = PerformanceMetrics().compute_all(returns)
        assert metrics["sharpe_ratio"] > 2.0, \
            f"稳定正收益的 Sharpe 应 > 2，实际 {metrics['sharpe_ratio']:.4f}"


# ============================================================================
# Test 4: DSR 检测过拟合
# ============================================================================

class TestDsrDetectsOverfitting:
    """对随机噪声策略，DSR 数据窥探修正后应不显著（p_value < 0.95）."""

    def _random_sharpe_list(self, n: int = 100, n_obs: int = 252, seed: int = 42) -> list[float]:
        """生成 n 个随机策略的年化 Sharpe（模拟研究员试验 n 个策略）.

        关键：SR 估计量的标准差 σ ≈ 1/√n_obs（正态假设下）。
        这才是 DSR 公式里 sigma_sr 的量级，max 才会落在 sharpe_star 附近。
        如果用 σ=0.3，max 会远大于 sharpe_star，导致 p_value→1。
        """
        rng = np.random.default_rng(seed)
        # 真实 SR=0，仅因有限样本估计产生抽样误差
        sigma = 1.0 / np.sqrt(n_obs)   # ≈ 0.063 for n_obs=252
        sharpes = rng.normal(loc=0.0, scale=sigma, size=n).tolist()
        return sharpes

    def test_dsr_not_significant_for_noise(self):
        """随机噪声策略：100 次试验后，最优 Sharpe 经 DSR 修正应不显著（p < 0.95）."""
        n_obs = 252
        sharpe_list = self._random_sharpe_list(n=100, n_obs=n_obs)
        best_sharpe = max(sharpe_list)

        result = deflated_sharpe_ratio(
            strategies_sharpe_list=sharpe_list,
            candidate_sharpe=best_sharpe,
            n_obs=n_obs,
        )

        # 真实 SR=0 的策略，多次试验后 DSR 修正值应不显著（best≈sharpe_star → p≈0.5）
        assert result["p_value"] < 0.95, (
            f"随机噪声策略不应通过 DSR 显著性检验，"
            f"p_value={result['p_value']:.4f}，"
            f"best_sharpe={best_sharpe:.4f}，sharpe_star={result['sharpe_star']:.4f}"
        )

    def test_dsr_significant_for_strong_alpha(self):
        """真实 Alpha 策略（高 Sharpe）：DSR 修正后应显著."""
        # 模拟一组差策略 + 一个真实强策略
        sharpe_list = self._random_sharpe_list(n=20, n_obs=252)
        strong_sharpe = 3.0  # 年化 Sharpe=3，非常强

        result = deflated_sharpe_ratio(
            strategies_sharpe_list=sharpe_list + [strong_sharpe],
            candidate_sharpe=strong_sharpe,
            n_obs=1000,  # 4年数据，更多观测
        )

        assert result["p_value"] >= 0.95, (
            f"高 Sharpe 策略应通过 DSR 显著性检验，"
            f"p_value={result['p_value']:.4f}"
        )

    def test_dsr_more_trials_harder_to_pass(self):
        """试验次数越多，通过 DSR 越难（修正后 sharpe_star 更高）."""
        base_sharpe = 1.0
        sharpe_list_small = [base_sharpe] * 5
        sharpe_list_large = [base_sharpe] * 200

        result_small = deflated_sharpe_ratio(sharpe_list_small, base_sharpe, n_obs=252)
        result_large = deflated_sharpe_ratio(sharpe_list_large, base_sharpe, n_obs=252)

        # 试验次数多时，sharpe_star（修正阈值）应更高
        assert result_large["sharpe_star"] >= result_small["sharpe_star"], \
            "更多试验次数应导致更高的 sharpe_star（更严格的修正）"

    def test_dsr_empty_list_returns_nan(self):
        """空策略列表应返回 NaN，不崩溃."""
        result = deflated_sharpe_ratio([], 1.0)
        assert math.isnan(result["dsr"]), "空策略列表应返回 NaN"

    def test_psr_significant_for_strong_strategy(self):
        """稳定正收益的 PSR 应显著（>= 0.95）."""
        np.random.seed(123)
        # 构造年化 Sharpe ≈ 2 的收益序列（日均 0.15%，波动 0.75%）
        daily_ret = 0.0015
        daily_vol = 0.0075
        returns = pd.Series(
            np.random.normal(daily_ret, daily_vol, 504)  # 2年
        )

        result = probabilistic_sharpe_ratio(returns, benchmark_sharpe=0.0)

        assert result["psr"] >= 0.95, (
            f"稳定正收益策略的 PSR 应显著，"
            f"psr={result['psr']:.4f}，sharpe={result['sharpe_hat']:.4f}"
        )


# ============================================================================
# Test 5: Walk-Forward 无前视偏差
# ============================================================================

class TestWalkForwardNoLookahead:
    """验证每个测试窗口严格不使用未来数据（无前视偏差）."""

    def _make_trading_dates(self, n_days: int = 1200) -> list[date]:
        """生成 n_days 个交易日（跳过周末）."""
        dates = []
        d = date(2020, 1, 2)
        while len(dates) < n_days:
            if d.weekday() < 5:  # 周一到周五
                dates.append(d)
            d += timedelta(days=1)
        return dates

    def test_window_dates_strictly_ordered(self):
        """所有窗口的日期应严格有序：train ≤ val ≤ test."""
        cfg = WalkForwardConfig(
            train_days=252,
            val_days=63,
            test_days=63,
            step_days=63,
        )
        validator = WalkForwardValidator(config=cfg)
        all_dates = self._make_trading_dates(1000)
        windows = validator.generate_windows(all_dates)

        assert len(windows) > 0, "应至少生成 1 个窗口"
        for w in windows:
            assert w.train_start <= w.train_end, \
                f"窗口{w.window_idx}: train_start > train_end"
            assert w.train_end < w.val_start, \
                f"窗口{w.window_idx}: train 与 val 重叠"
            assert w.val_end < w.test_start, \
                f"窗口{w.window_idx}: val 与 test 重叠"
            assert w.test_start <= w.test_end, \
                f"窗口{w.window_idx}: test_start > test_end"

    def test_test_windows_do_not_overlap(self):
        """相邻窗口的 test 区间不应重叠（step == test_days）."""
        cfg = WalkForwardConfig(
            train_days=252,
            val_days=63,
            test_days=63,
            step_days=63,
        )
        validator = WalkForwardValidator(config=cfg)
        all_dates = self._make_trading_dates(1000)
        windows = validator.generate_windows(all_dates)

        for i in range(1, len(windows)):
            prev = windows[i - 1]
            curr = windows[i]
            assert curr.test_start > prev.test_end, (
                f"窗口 {i-1} 和 {i} 的 test 区间重叠："
                f"{prev.test_start}~{prev.test_end} vs {curr.test_start}~{curr.test_end}"
            )

    def test_no_future_data_in_train(self):
        """训练集的 train_end 严格早于对应 test_start，不包含未来信息."""
        cfg = WalkForwardConfig(
            train_days=252,
            val_days=63,
            test_days=63,
            step_days=63,
        )
        validator = WalkForwardValidator(config=cfg)
        all_dates = self._make_trading_dates(1000)
        windows = validator.generate_windows(all_dates)

        for w in windows:
            # 训练集最晚日期应早于测试集开始日期
            assert w.train_end < w.test_start, (
                f"窗口 {w.window_idx}: 训练集 train_end={w.train_end} "
                f">= test_start={w.test_start}（前视偏差！）"
            )

    def test_feature_slice_excludes_test_period(self):
        """WalkForwardValidator._slice_features 应严格不包含 test 期数据."""
        # 构造 MultiIndex features DataFrame
        all_dates = self._make_trading_dates(600)
        tickers = ["000001.SZ", "000002.SZ"]
        idx = pd.MultiIndex.from_product(
            [pd.to_datetime(all_dates), tickers],
            names=["date", "ticker"],
        )
        features = pd.DataFrame(
            {"momentum": np.random.randn(len(idx))},
            index=idx,
        )

        # 取第一个窗口的参数
        cfg = WalkForwardConfig(train_days=252, val_days=63, test_days=63, step_days=63)
        validator = WalkForwardValidator(config=cfg)
        windows = validator.generate_windows(all_dates)
        assert windows, "应生成至少 1 个窗口"

        w = windows[0]
        train_features = WalkForwardValidator._slice_features(
            features, w.train_start, w.train_end
        )

        # 切片后的特征不应包含 >= val_start 的日期
        sliced_dates = train_features.index.get_level_values("date")
        assert all(d <= pd.Timestamp(w.train_end) for d in sliced_dates), \
            "训练特征不应包含训练窗口结束日之后的数据（前视偏差）"

    def test_window_count_consistent_with_step(self):
        """窗口数量应与滚动步长一致（n ≈ (total - train - val - test) // step + 1）."""
        cfg = WalkForwardConfig(
            train_days=252,
            val_days=63,
            test_days=63,
            step_days=63,
        )
        validator = WalkForwardValidator(config=cfg)
        n_total = 800
        all_dates = self._make_trading_dates(n_total)
        windows = validator.generate_windows(all_dates)

        # 理论窗口数
        min_required = cfg.train_days + cfg.val_days + cfg.test_days  # = 378
        expected = max(0, (n_total - min_required) // cfg.step_days + 1)

        assert len(windows) == expected, (
            f"窗口数量不符：期望 {expected}，实际 {len(windows)}"
        )


# ============================================================================
# Test 6: 换手率计算
# ============================================================================

class TestPortfolioTurnover:
    """通过 PerformanceMetrics 验证换手率计算逻辑."""

    def _make_fill(
        self,
        ticker: str,
        direction: str,
        shares: int,
        fill_price: float,
    ) -> Fill:
        """构造一条 Fill 记录."""
        gross = shares * fill_price
        commission = gross * 3e-4
        stamp_duty = gross * 1e-3 if direction == "sell" else 0.0
        transfer_fee = gross * 2e-5
        net = gross + commission + stamp_duty + transfer_fee if direction == "buy" \
            else gross - commission - stamp_duty - transfer_fee
        return Fill(
            ticker=ticker,
            direction=direction,
            shares=shares,
            fill_price=fill_price,
            gross_amount=gross,
            commission=commission,
            stamp_duty=stamp_duty,
            transfer_fee=transfer_fee,
            net_amount=net,
            fill_date=date(2024, 1, 2),
        )

    def test_zero_turnover_when_no_fills(self):
        """无成交时换手率应为 NaN（无法计算）."""
        returns = pd.Series(np.random.randn(20) * 0.01)
        metrics = PerformanceMetrics().compute_all(returns, fills=[])
        # 无成交时 turnover_rate 应为 nan
        assert np.isnan(metrics.get("turnover_rate", np.nan)), \
            "无成交时换手率应为 NaN"

    def test_turnover_rate_positive_with_buys(self):
        """有买入成交时换手率应为正数."""
        # 构造 50 天收益 + 1 笔买入
        returns = pd.Series(np.random.randn(50) * 0.005)
        fills = [
            self._make_fill("000001.SZ", "buy", 1000, 10.0),   # 买入 1万元
            self._make_fill("000001.SZ", "sell", 500, 10.5),   # 卖出（不计入换手）
        ]
        metrics = PerformanceMetrics().compute_all(returns, fills=fills)

        assert "turnover_rate" in metrics, "指标中应包含 turnover_rate"
        tr = metrics["turnover_rate"]
        assert not np.isnan(tr), "有买入成交时换手率不应为 NaN"
        assert tr > 0, f"换手率应为正数，实际 {tr:.6f}"

    def test_turnover_proportional_to_buy_amount(self):
        """换手率与买入金额成正比（同样的持有天数，买入越多换手越高）."""
        n_days = 50
        returns = pd.Series(np.zeros(n_days))

        # 小买入
        fills_small = [self._make_fill("000001.SZ", "buy", 100, 10.0)]   # 1000 元
        # 大买入
        fills_large = [self._make_fill("000001.SZ", "buy", 1000, 10.0)]  # 10000 元

        m_small = PerformanceMetrics().compute_all(returns, fills=fills_small)
        m_large = PerformanceMetrics().compute_all(returns, fills=fills_large)

        tr_small = m_small.get("turnover_rate", 0)
        tr_large = m_large.get("turnover_rate", 0)

        assert tr_large > tr_small, (
            f"买入金额更大时换手率应更高：small={tr_small:.4f}, large={tr_large:.4f}"
        )

    def test_portfolio_position_tracking(self):
        """验证 Portfolio 正确记录持仓股数和均价."""
        portfolio = Portfolio(initial_capital=1_000_000.0, t1_restriction=False)

        # 手动应用一笔买入 Fill
        fill = self._make_fill("600519.SH", "buy", 100, 1800.0)
        fill.fill_date = date(2024, 2, 1)

        portfolio._apply_fill(fill, date(2024, 2, 1))

        assert "600519.SH" in portfolio.positions, "应记录仓位"
        pos = portfolio.positions["600519.SH"]
        assert pos.shares == 100, f"股数应为 100，实际 {pos.shares}"
        assert abs(pos.avg_cost - 1800.0) < 0.01, \
            f"均价应为 1800.0，实际 {pos.avg_cost:.2f}"
        assert portfolio.cash < 1_000_000.0, "现金应减少"

    def test_portfolio_cash_decreases_on_buy(self):
        """买入后现金减少，现金减少量等于 net_amount."""
        portfolio = Portfolio(initial_capital=500_000.0, t1_restriction=False)
        initial_cash = portfolio.cash

        fill = self._make_fill("000001.SZ", "buy", 1000, 10.0)
        portfolio._apply_fill(fill, date(2024, 3, 1))

        cash_decrease = initial_cash - portfolio.cash
        assert abs(cash_decrease - fill.net_amount) < 0.01, (
            f"现金减少量 {cash_decrease:.2f} 应等于 net_amount {fill.net_amount:.2f}"
        )
