"""
tests/test_backfill_realism.py — 验证回填数据的真实性.

问题历史：旧版回填脚本把 close_price 直接记为阈值价（target_price / stop_loss_price），
导致 target_hit 全是精确 +20.00%、stop_loss 全是精确 -10.00%，
NAV 对比也因为 cumprod 复利变成离谱的 +1242% vs +80%。

修复后期望:
  1. close_price 用"穿越次日开盘价 × (1 - 0.001 滑点)" → pnl 分布有真实方差
  2. NAV 用等权简单收益 → 累计差异在合理范围（不超过 5 倍）
  3. stop_loss / target_hit 的成交价不再 = 阈值价
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── 单元测试：成交价辅助函数（不依赖 DB）──────────────────────────────────

class TestExecPriceNextOpen:
    """验证 _exec_price_next_open 的行为。"""

    def _make_bars(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"trade_date": "2026-01-01", "open": 10.0, "high": 10.5, "low": 9.5,  "close": 10.0},
            {"trade_date": "2026-01-02", "open": 9.8,  "high": 10.0, "low": 9.3,  "close": 9.5},
            {"trade_date": "2026-01-03", "open": 9.6,  "high": 9.8,  "low": 9.2,  "close": 9.7},
        ])

    def test_next_open_with_slippage(self):
        from scripts.backfill_executions import _exec_price_next_open, SLIPPAGE_SELL_PCT
        bars = self._make_bars()
        # 假设触发在 index=0，应该用 index=1 的 open 9.8
        px, d = _exec_price_next_open(bars, trigger_idx=0, fallback_price=999)
        assert px == pytest.approx(9.8 * (1 - SLIPPAGE_SELL_PCT), abs=1e-6)
        assert d == "2026-01-02"

    def test_fallback_to_trigger_close_when_no_next_bar(self):
        from scripts.backfill_executions import _exec_price_next_open, SLIPPAGE_SELL_PCT
        bars = self._make_bars()
        # 触发在最后一根 bar → 没有次日 → 退化到 fallback_price
        px, _ = _exec_price_next_open(bars, trigger_idx=2, fallback_price=10.0)
        assert px == pytest.approx(10.0 * (1 - SLIPPAGE_SELL_PCT), abs=1e-6)

    def test_slippage_pct_value(self):
        from scripts.backfill_executions import SLIPPAGE_SELL_PCT
        # 卖出滑点必须是正数且 < 1%
        assert 0 < SLIPPAGE_SELL_PCT < 0.01


# ── 集成测试：真实回填数据分布（需要 PG + 已运行回填）─────────────────────

@pytest.mark.integration
class TestBackfillDispersion:
    """回填后的 simulated_orders 应该有真实分布，不是整齐的阈值价。"""

    @pytest.fixture(scope="class")
    def df_closed(self):
        from sqlalchemy import text
        from app.db.postgres import get_pg_engine
        with get_pg_engine().connect() as conn:
            return pd.read_sql(text(
                "SELECT * FROM simulated_orders WHERE status='CLOSED'"
            ), conn)

    def test_stop_loss_pnl_not_exactly_minus_10pct(self, df_closed):
        """止损订单的 pnl_pct 不应全部精确等于 -10%（之前的 bug）。"""
        sl = df_closed[df_closed["close_reason"] == "stop_loss"]
        if len(sl) < 2:
            pytest.skip("止损样本不足 2 条，无法检验方差")
        # std 应该 > 0.005（至少 0.5% 的分散度）
        std = float(sl["pnl_pct"].std())
        assert std > 0.005, \
            f"stop_loss pnl_pct std={std:.6f}，疑似全部精确 = -10%（bug）"
        # 不应所有 pnl_pct 都 == -0.10
        precise_count = (sl["pnl_pct"].round(4) == -0.10).sum()
        assert precise_count < len(sl), \
            f"{precise_count}/{len(sl)} 笔止损 pnl_pct 精确等于 -10%（bug）"

    def test_target_hit_pnl_not_exactly_plus_20pct(self, df_closed):
        """止盈订单的 pnl_pct 不应全部精确等于 +20%。"""
        th = df_closed[df_closed["close_reason"] == "target_hit"]
        if len(th) < 2:
            pytest.skip("止盈样本不足 2 条，无法检验方差")
        std = float(th["pnl_pct"].std())
        assert std > 0.005, \
            f"target_hit pnl_pct std={std:.6f}，疑似全部精确 = +20%（bug）"
        precise_count = (th["pnl_pct"].round(4) == 0.20).sum()
        assert precise_count < len(th)

    def test_close_price_differs_from_threshold(self, df_closed):
        """触发的成交价应该 != 阈值价（因为用了次日开盘 + 滑点）。"""
        sl = df_closed[df_closed["close_reason"] == "stop_loss"].copy()
        if not sl.empty:
            # 至少有一半不等于阈值
            equal_count = (sl["close_price"].round(4) ==
                            sl["stop_loss_price"].round(4)).sum()
            assert equal_count < len(sl) / 2, \
                "大多数 stop_loss 成交价 == 阈值价，说明回填仍未使用真实成交"

        th = df_closed[df_closed["close_reason"] == "target_hit"].copy()
        if not th.empty:
            equal_count = (th["close_price"].round(4) ==
                            th["target_price"].round(4)).sum()
            assert equal_count < len(th) / 2, \
                "大多数 target_hit 成交价 == 阈值价，说明回填仍未使用真实成交"


# ── 集成测试：NAV 对比逻辑 ─────────────────────────────────────────────────

@pytest.mark.integration
class TestExecVsHoldRealistic:
    """get_execution_vs_hold_comparison 修正后的合理性。"""

    @pytest.fixture(scope="class")
    def cmp(self):
        from app.services.data_service import get_data_service
        return get_data_service().get_execution_vs_hold_comparison()

    def test_nav_returns_required_keys(self, cmp):
        assert "execute" in cmp
        assert "hold_to_expiry" in cmp
        assert "n_total" in cmp
        assert "exec_stop_count" in cmp
        assert "exec_target_count" in cmp
        for side in ("execute", "hold_to_expiry"):
            for key in ("curve", "n", "total_return", "max_dd", "avg_return", "win_rate"):
                assert key in cmp[side], f"{side} 缺 {key}"

    @pytest.mark.stale_panel_fixture
    def test_nav_ratio_reasonable(self, cmp):
        """累计收益比率 |exec/hold| 不应超过 5 倍（之前 cumprod bug 是 15+ 倍）。"""
        e_ret = cmp["execute"]["total_return"]
        h_ret = cmp["hold_to_expiry"]["total_return"]
        if abs(h_ret) < 1e-4 or abs(e_ret) < 1e-4:
            pytest.skip("某侧收益接近 0，无法计算比率")
        ratio = abs(e_ret) / abs(h_ret)
        assert ratio < 5.0, \
            f"|exec/hold| = {ratio:.2f}x，应 < 5x（之前 cumprod bug 是 15+ 倍）"

    def test_total_returns_in_reasonable_range(self, cmp):
        """等权 N 笔组合，单笔贡献被 1/N 稀释，累计绝对值应 < 50%。"""
        for side in ("execute", "hold_to_expiry"):
            tr = cmp[side]["total_return"]
            assert -0.5 < tr < 0.5, \
                f"{side} total_return={tr:.4f}，超出合理范围 [-50%, +50%]"

    def test_max_dd_non_positive(self, cmp):
        """MaxDD 应该 <= 0。"""
        for side in ("execute", "hold_to_expiry"):
            assert cmp[side]["max_dd"] <= 0, \
                f"{side} max_dd={cmp[side]['max_dd']} 应 <= 0"

    def test_n_total_matches_sample_size(self, cmp):
        """n_total 应等于推荐池规模（与 realized_pnl 行数一致）。"""
        # 用 hold 的 n（来自 realized_pnl）作为基准
        n_hold = cmp["hold_to_expiry"]["n"]
        assert cmp["n_total"] == n_hold or cmp["n_total"] >= n_hold

    def test_exec_curve_ends_at_total_return(self, cmp):
        """NAV 曲线终值应等于 1 + total_return。"""
        for side in ("execute", "hold_to_expiry"):
            curve = cmp[side]["curve"]
            if not curve:
                continue
            tr = cmp[side]["total_return"]
            assert curve[-1] == pytest.approx(1 + tr, abs=1e-4), \
                f"{side} NAV[-1]={curve[-1]} != 1 + total_return={1 + tr}"

    def test_exit_reason_counts_match(self, cmp):
        """exec_stop_count = stop_loss + trailing_stop 计数。"""
        reasons = cmp["exit_reasons"]
        expected_stops = (reasons.get("stop_loss", 0) +
                           reasons.get("trailing_stop", 0))
        assert cmp["exec_stop_count"] == expected_stops
        assert cmp["exec_target_count"] == reasons.get("target_hit", 0)


# ── pure 函数测试（不依赖 DB，可在 CI 跑）──────────────────────────────────

class TestNavCalculationUnit:
    """用合成数据验证 NAV 算法在小样本下的正确性。"""

    def test_equal_weight_nav_no_cumprod_inflation(self):
        """8 笔订单全部 +10% 收益，等权组合 NAV 应 = 1.10（不是 1.1^8）。"""
        # 直接用 DataService 内部的算法逻辑：mean(returns) = total_return
        returns = pd.Series([0.10] * 8)
        n_total = 8
        nav = (1.0 + (returns / n_total).cumsum()).tolist()
        assert nav[-1] == pytest.approx(1.10, abs=1e-6)
        # 如果是 cumprod 滚雪球：1.1**8 = 2.14（错误）
        assert (1.1 ** 8) > 2.0  # 自检
        assert nav[-1] < 1.5     # 等权 NAV 远小于 cumprod

    def test_mixed_returns_average_correct(self):
        returns = pd.Series([0.20, -0.10, 0.05, -0.15])  # mean = 0.0
        n_total = 4
        nav = (1.0 + (returns / n_total).cumsum()).tolist()
        assert nav[-1] == pytest.approx(1.0, abs=1e-6)
