"""tests/test_risk.py — Phase 7 风险管理模块单元测试.

测试覆盖：
  1. test_equal_weight              — n 只股票各占 1/n
  2. test_risk_parity_equal_vol     — 等波动率时 risk_parity ≈ equal_weight
  3. test_hrp_sums_to_one           — HRP 权重之和 = 1
  4. test_drawdown_trigger          — 回撤 25% 时仓位降至 40%
  5. test_volatility_targeting      — 目标波动率计算正确
  6. test_inverse_vol_weights       — 逆波动率：低波动股权重更高
  7. test_minimum_variance_lower    — 最小方差权重使组合方差 <= 等权
  8. test_cppi_floor_protection     — CPPI NAV<=Floor 时返回 0
  9. test_kelly_sums_to_one         — Kelly 权重归一化
 10. test_drawdown_clearout         — 回撤 > 30% 时返回 0（清仓信号）

全部使用 mock 数据，不依赖真实 API 或外部服务。
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantmind.risk.position_sizing import PositionSizer
from quantmind.risk.drawdown import DrawdownController, DrawdownRule
from quantmind.risk.factor_risk import FactorRiskModel


# ============================================================================
# 辅助函数
# ============================================================================

def _make_returns(
    n_days: int = 252,
    n_stocks: int = 5,
    seed: int = 42,
    vols: list[float] | None = None,
) -> pd.DataFrame:
    """生成模拟日收益 DataFrame（可指定各股波动率）."""
    rng = np.random.default_rng(seed)
    tickers = [f"Stock{i+1}" for i in range(n_stocks)]
    if vols is None:
        vols = [0.01] * n_stocks
    data = {
        t: rng.normal(0.0, v, n_days)
        for t, v in zip(tickers, vols)
    }
    idx = pd.date_range("2022-01-01", periods=n_days, freq="B")
    return pd.DataFrame(data, index=idx)


def _make_cov(returns_df: pd.DataFrame) -> pd.DataFrame:
    """从收益序列计算协方差矩阵."""
    return returns_df.cov()


# ============================================================================
# Test 1: 等权组合
# ============================================================================

class TestEqualWeight:
    """n 只股票各占 1/n."""

    def test_5_stocks_equal(self):
        tickers = ["A", "B", "C", "D", "E"]
        w = PositionSizer.equal_weight(tickers)
        assert len(w) == 5
        for t, weight in w.items():
            assert abs(weight - 0.2) < 1e-9, f"{t} 权重应为 0.2，实际 {weight}"

    def test_weights_sum_to_one(self):
        tickers = [f"S{i}" for i in range(10)]
        w = PositionSizer.equal_weight(tickers)
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_single_stock(self):
        w = PositionSizer.equal_weight(["ONLY"])
        assert abs(w["ONLY"] - 1.0) < 1e-9

    def test_empty_returns_empty(self):
        w = PositionSizer.equal_weight([])
        assert w == {}


# ============================================================================
# Test 2: 风险平价（等波动率时 ≈ 等权）
# ============================================================================

class TestRiskParity:
    """等波动率时 risk_parity 结果应接近 equal_weight."""

    def test_equal_vol_gives_equal_weight(self):
        """所有股票波动率相同时，风险平价应退化为等权."""
        n = 5
        # 构造等波动率、零相关的协方差矩阵
        vol = 0.02  # 所有股票日波动率相同
        cov = pd.DataFrame(
            np.eye(n) * vol**2,
            index=[f"S{i}" for i in range(n)],
            columns=[f"S{i}" for i in range(n)],
        )
        w = PositionSizer.risk_parity(cov)
        expected = 1.0 / n
        for t, weight in w.items():
            assert abs(weight - expected) < 1e-4, (
                f"等波动率时风险平价应≈等权，{t}: {weight:.6f} vs {expected:.6f}"
            )

    def test_high_vol_gets_lower_weight(self):
        """高波动率股票应获得更低的权重."""
        # Stock0: 低波动，Stock1: 高波动（4倍）
        cov_data = np.array([
            [0.0001, 0.0],  # Stock0: vol=1%
            [0.0,  0.0016], # Stock1: vol=4%
        ])
        cov = pd.DataFrame(cov_data, index=["LowVol", "HighVol"], columns=["LowVol", "HighVol"])
        w = PositionSizer.risk_parity(cov)
        # 低波动应该权重更大
        assert w["LowVol"] > w["HighVol"], (
            f"低波动应权重更大：LowVol={w['LowVol']:.4f}, HighVol={w['HighVol']:.4f}"
        )

    def test_risk_contributions_equal(self):
        """验证风险平价时每只股票的风险贡献近似相等."""
        n = 4
        np.random.seed(7)
        vols = [0.01, 0.02, 0.015, 0.025]
        cov_data = np.diag([v**2 for v in vols])
        cov = pd.DataFrame(
            cov_data,
            index=[f"S{i}" for i in range(n)],
            columns=[f"S{i}" for i in range(n)],
        )
        w_dict = PositionSizer.risk_parity(cov)
        w = np.array(list(w_dict.values()))
        Σ = cov.values
        σ_p = np.sqrt(w @ Σ @ w)
        # 各股风险贡献 RC_i = w_i × (Σw)_i / σ_p
        Σw = Σ @ w
        RCs = w * Σw / σ_p
        # 检查 RC 是否均等
        assert np.std(RCs) / np.mean(RCs) < 0.01, (
            f"风险贡献应均等，变异系数 {np.std(RCs)/np.mean(RCs):.4f}"
        )


# ============================================================================
# Test 3: HRP 权重之和 = 1
# ============================================================================

class TestHRP:
    """HRP 权重应归一化到 1，且所有权重 >= 0."""

    def test_weights_sum_to_one(self):
        returns = _make_returns(n_days=120, n_stocks=6)
        w = PositionSizer.hierarchical_risk_parity(returns)
        total = sum(w.values())
        assert abs(total - 1.0) < 1e-6, f"HRP 权重之和应为 1，实际 {total:.8f}"

    def test_all_weights_nonnegative(self):
        returns = _make_returns(n_days=120, n_stocks=8)
        w = PositionSizer.hierarchical_risk_parity(returns)
        for t, weight in w.items():
            assert weight >= -1e-9, f"HRP 权重不应为负：{t}={weight:.6f}"

    def test_covers_all_tickers(self):
        returns = _make_returns(n_days=100, n_stocks=5)
        w = PositionSizer.hierarchical_risk_parity(returns)
        assert set(w.keys()) == set(returns.columns), "HRP 应包含所有股票"

    def test_single_stock(self):
        returns = _make_returns(n_days=50, n_stocks=1)
        w = PositionSizer.hierarchical_risk_parity(returns)
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_two_stocks(self):
        returns = _make_returns(n_days=100, n_stocks=2)
        w = PositionSizer.hierarchical_risk_parity(returns)
        assert abs(sum(w.values()) - 1.0) < 1e-6


# ============================================================================
# Test 4: 回撤触发规则
# ============================================================================

class TestDrawdownTrigger:
    """回撤阈值触发仓位调整规则."""

    def _controller(self) -> DrawdownController:
        return DrawdownController(verbose=False)

    def test_no_drawdown_full_position(self):
        """回撤 0% 时应满仓（100%）."""
        ctrl = self._controller()
        pos = ctrl.check_and_adjust(0.0)
        assert pos == 1.0

    def test_small_drawdown_full_position(self):
        """回撤 5% 未触发任何规则，仍满仓."""
        ctrl = self._controller()
        pos = ctrl.check_and_adjust(0.05)
        assert pos == 1.0

    def test_drawdown_10pct_reduces_to_70pct(self):
        """回撤 11% 触发第一条规则，仓位降至 70%."""
        ctrl = self._controller()
        pos = ctrl.check_and_adjust(0.11)
        assert pos == 0.70, f"回撤 11% 应降至 70%，实际 {pos:.2%}"

    def test_drawdown_25pct_reduces_to_40pct(self):
        """回撤 25% 触发第二条规则，仓位降至 40%."""
        ctrl = self._controller()
        pos = ctrl.check_and_adjust(0.25)
        assert pos == 0.40, f"回撤 25% 应降至 40%，实际 {pos:.2%}"

    def test_drawdown_30pct_clearout(self):
        """回撤 > 30% 触发清仓规则，仓位降至 0%."""
        ctrl = self._controller()
        pos = ctrl.check_and_adjust(0.35)
        assert pos == 0.0, f"回撤 35% 应清仓（0%），实际 {pos:.2%}"

    def test_negative_drawdown_treated_as_positive(self):
        """传入负数回撤（如 -0.25）应取绝对值处理."""
        ctrl = self._controller()
        pos = ctrl.check_and_adjust(-0.25)  # 等价于回撤 25%
        assert pos == 0.40, f"负数回撤应取绝对值，实际 {pos:.2%}"

    def test_custom_rules(self):
        """自定义规则覆盖默认规则."""
        custom_rules = [
            DrawdownRule(threshold=0.05, target_pct=0.50),
            DrawdownRule(threshold=0.15, target_pct=0.00),
        ]
        ctrl = DrawdownController(rules=custom_rules, verbose=False)
        assert ctrl.check_and_adjust(0.06) == 0.50
        assert ctrl.check_and_adjust(0.20) == 0.00


# ============================================================================
# Test 5: 目标波动率计算
# ============================================================================

class TestVolatilityTargeting:
    """目标波动率 → 杠杆系数验证."""

    def test_high_vol_reduces_leverage(self):
        """实现波动率 > 目标波动率时，杠杆 < 1."""
        np.random.seed(42)
        # 日波动率 2%，年化 ≈ 31.7%
        returns = pd.Series(np.random.normal(0, 0.02, 252))
        leverage = DrawdownController.volatility_targeting(returns, target_vol=0.10)
        assert leverage < 1.0, f"高波动时杠杆应 < 1，实际 {leverage:.4f}"

    def test_low_vol_increases_leverage(self):
        """实现波动率 < 目标波动率时，杠杆 > 1（但受上限约束）."""
        np.random.seed(42)
        # 日波动率 0.3%，年化 ≈ 4.8%
        returns = pd.Series(np.random.normal(0, 0.003, 252))
        leverage = DrawdownController.volatility_targeting(returns, target_vol=0.10)
        assert leverage > 1.0, f"低波动时杠杆应 > 1，实际 {leverage:.4f}"

    def test_leverage_within_bounds(self):
        """杠杆应在 [min_leverage, max_leverage] 范围内."""
        np.random.seed(0)
        returns = pd.Series(np.random.normal(0, 0.001, 252))  # 极低波动
        leverage = DrawdownController.volatility_targeting(
            returns, target_vol=0.10, min_leverage=0.0, max_leverage=2.0
        )
        assert 0.0 <= leverage <= 2.0, f"杠杆 {leverage:.4f} 超出 [0, 2] 范围"

    def test_exact_target_vol(self):
        """实现波动率等于目标波动率时，杠杆应接近 1.0."""
        target_vol = 0.15  # 15% 年化
        daily_vol = target_vol / np.sqrt(252)
        np.random.seed(123)
        returns = pd.Series(np.random.normal(0, daily_vol, 1000))
        leverage = DrawdownController.volatility_targeting(returns, target_vol=target_vol)
        # 由于随机性，允许 ±30% 误差
        assert abs(leverage - 1.0) < 0.30, (
            f"实现波动率≈目标时，杠杆应≈1.0，实际 {leverage:.4f}"
        )

    def test_insufficient_data_returns_one(self):
        """数据不足时应返回 1.0（不调整）."""
        leverage = DrawdownController.volatility_targeting(
            pd.Series([0.01, 0.02]), target_vol=0.10
        )
        assert leverage == 1.0


# ============================================================================
# Test 6: 逆波动率权重
# ============================================================================

class TestInverseVolatility:
    """低波动率的股票应获得更高权重."""

    def test_low_vol_higher_weight(self):
        vols = [0.005, 0.02, 0.01]  # Stock1 波动最低
        returns = _make_returns(n_days=200, n_stocks=3, vols=vols)
        w = PositionSizer.inverse_volatility(returns)
        assert w["Stock1"] > w["Stock2"] > w["Stock3"] or \
               w["Stock1"] > w["Stock3"], \
            f"最低波动率的股票应权重最大: {w}"

    def test_weights_sum_to_one(self):
        returns = _make_returns(n_days=100, n_stocks=4)
        w = PositionSizer.inverse_volatility(returns)
        assert abs(sum(w.values()) - 1.0) < 1e-9


# ============================================================================
# Test 7: 最小方差组合方差 <= 等权方差
# ============================================================================

class TestMinimumVariance:
    """最小方差组合的方差应 <= 等权组合的方差."""

    def test_lower_variance_than_equal_weight(self):
        returns = _make_returns(n_days=252, n_stocks=5, seed=99)
        cov = _make_cov(returns)

        w_min = PositionSizer.minimum_variance(cov)
        w_eq = PositionSizer.equal_weight(list(cov.columns))

        def portfolio_var(weights: dict) -> float:
            w = np.array(list(weights.values()))
            Σ = cov.values
            return float(w @ Σ @ w)

        var_min = portfolio_var(w_min)
        var_eq = portfolio_var(w_eq)
        assert var_min <= var_eq + 1e-8, (
            f"最小方差组合方差 {var_min:.8f} 应 <= 等权方差 {var_eq:.8f}"
        )

    def test_weights_sum_to_one(self):
        returns = _make_returns(n_days=200, n_stocks=4)
        cov = _make_cov(returns)
        w = PositionSizer.minimum_variance(cov)
        assert abs(sum(w.values()) - 1.0) < 1e-6


# ============================================================================
# Test 8: CPPI 安全垫
# ============================================================================

class TestCPPI:
    """CPPI 当 NAV <= Floor 时返回 0."""

    def test_below_floor_returns_zero(self):
        nav = 75.0
        floor = 80.0
        w = DrawdownController.cppi(nav, floor, multiplier=3)
        assert w == 0.0, f"NAV < Floor 时应返回 0，实际 {w}"

    def test_at_floor_returns_zero(self):
        w = DrawdownController.cppi(80.0, 80.0, multiplier=3)
        assert w == 0.0

    def test_above_floor_returns_positive(self):
        nav = 100.0
        floor = 80.0
        w = DrawdownController.cppi(nav, floor, multiplier=3)
        # Cushion=20，Exposure=60，Risky%=60%
        expected = min(3 * (100 - 80) / 100, 1.0)
        assert abs(w - expected) < 1e-6, f"CPPI 权重应为 {expected:.4f}，实际 {w:.4f}"

    def test_large_cushion_capped_at_one(self):
        w = DrawdownController.cppi(200.0, 50.0, multiplier=10)
        assert w == 1.0, "风险资产比例不应超过 100%"


# ============================================================================
# Test 9: Kelly 权重归一化
# ============================================================================

class TestKelly:
    """Kelly 仓位的权重应归一化为 1，且无负权重（半 Kelly 多头）."""

    def _make_kelly_inputs(self, n: int = 4) -> tuple:
        tickers = [f"S{i}" for i in range(n)]
        er = pd.Series([0.10, 0.08, 0.12, 0.06], index=tickers)
        vols = [0.15, 0.12, 0.18, 0.10]
        cov_data = np.diag([v**2 for v in vols])
        cov = pd.DataFrame(cov_data, index=tickers, columns=tickers)
        return er, cov

    def test_weights_sum_to_one(self):
        er, cov = self._make_kelly_inputs()
        w = PositionSizer.kelly_criterion(er, cov, fraction=0.5)
        assert abs(sum(w.values()) - 1.0) < 1e-6, (
            f"Kelly 权重之和应为 1，实际 {sum(w.values()):.8f}"
        )

    def test_no_negative_weights(self):
        er, cov = self._make_kelly_inputs()
        w = PositionSizer.kelly_criterion(er, cov, fraction=0.5)
        for t, weight in w.items():
            assert weight >= -1e-9, f"半 Kelly 不应有负权重：{t}={weight:.6f}"

    def test_higher_return_higher_weight(self):
        """预期收益更高的股票应获得更大的权重（μ/σ² 比值小，权重不触及上界）."""
        tickers = ["Low", "High"]
        # μ/σ² 小（μ=0.001/0.002，σ=10%），使 Kelly 权重自然落在 [0, 0.25] 内
        er = pd.Series({"Low": 0.001, "High": 0.002})
        cov = pd.DataFrame(
            [[0.01, 0.0], [0.0, 0.01]],   # σ=10%，σ²=0.01
            index=tickers, columns=tickers
        )
        # 原始 Kelly: [0.001/0.01, 0.002/0.01]=[0.1, 0.2]，半 Kelly:[0.05, 0.1]
        # 都在 [0, 0.25] 内，归一化后 High 权重应 = 2×Low 权重
        w = PositionSizer.kelly_criterion(er, cov, fraction=0.5)
        assert w["High"] > w["Low"], (
            f"高预期收益应有更高权重：High={w['High']:.4f}, Low={w['Low']:.4f}"
        )


# ============================================================================
# Test 10: 回撤 > 30% 清仓信号
# ============================================================================

class TestDrawdownClearout:
    """回撤超过 30% 时应返回 0（清仓信号）."""

    def test_extreme_drawdown_clearout(self):
        ctrl = DrawdownController(verbose=False)
        # 回撤 31%、40%、50% 均应触发清仓
        for dd in [0.31, 0.40, 0.50, 0.99]:
            pos = ctrl.check_and_adjust(dd)
            assert pos == 0.0, f"回撤 {dd:.0%} 应触发清仓，实际仓位 {pos:.2%}"

    def test_compute_max_drawdown(self):
        """DrawdownController.compute_max_drawdown 计算正确."""
        # 净值先涨至 120，再跌至 90 → 最大回撤 = (90-120)/120 = -25%
        nav = pd.Series([100.0, 110.0, 120.0, 105.0, 95.0, 90.0])
        mdd = DrawdownController.compute_max_drawdown(nav)
        expected = (90 - 120) / 120
        assert abs(mdd - expected) < 1e-6, f"最大回撤应为 {expected:.4f}，实际 {mdd:.4f}"

    def test_no_drawdown_zero_mdd(self):
        """单调上涨的净值序列，最大回撤应为 0."""
        nav = pd.Series([100.0, 105.0, 110.0, 115.0])
        mdd = DrawdownController.compute_max_drawdown(nav)
        assert mdd >= -1e-9, f"单调上涨的 MDD 应为 0，实际 {mdd:.6f}"
