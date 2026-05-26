"""tests/test_position_nav.py — 持仓 NAV 曲线 + 止损模拟单元测试.

共 13 个测试，覆盖：
  - 固定止损触发 / 不触发
  - 追踪止损触发（比固定止损晚）
  - 最大回撤计算
  - 组合等权 NAV
  - 止损版 MaxDD ≤ 不止损版 MaxDD
  - 空数据 / 缺失股票 fallback
  - 止损冻结后 NAV 不再下降
  - 多股混合（部分触发、部分不触发）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_position_nav import build_nav_for_batch, _empty_result  # noqa: E402


# ── 辅助工厂 ─────────────────────────────────────────────────────────────────

def _make_prices(
    tickers: dict[str, list[float]],
    start: str = "2025-10-09",
) -> pd.DataFrame:
    """构造宽表价格 DataFrame（日期 × ticker）。

    Parameters
    ----------
    tickers : {ticker: [day0_price, day1_price, ...]}
    start   : 起始日期字符串
    """
    n = max(len(v) for v in tickers.values())
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {ticker: prices + [np.nan] * (n - len(prices))
         for ticker, prices in tickers.items()},
        index=dates,
    )


def _get_ticker(result: dict, ticker: str) -> dict:
    return result["stocks"][ticker]


# ── 测试 1：单股连续下跌 15% 应触发固定止损 ──────────────────────────────────

def test_fixed_stop_loss_triggered():
    """单股从 100 跌到 84（-16%）应触发 -15% 固定止损。"""
    # Day 0=100, Day 1=98, Day 2=94, Day 3=88, Day 4=84  → 第 4 日达 -16%
    prices = _make_prices({"AAA": [100, 98, 94, 88, 84, 83, 82]})
    result = build_nav_for_batch(
        entry_date  = "20251009",
        stock_list  = ["AAA"],
        prices_df   = prices,
        stop_loss   = -0.15,
        trailing    = False,
        holding_days= 10,
    )
    st = _get_ticker(result, "AAA")
    assert st["stop_loss_triggered"] is True, "应触发固定止损"
    assert st["stop_loss_date"] is not None


# ── 测试 2：不跌到阈值则不触发止损 ───────────────────────────────────────────

def test_fixed_stop_loss_not_triggered():
    """单股最多下跌 10%，未达 -15% 阈值，不应触发止损。"""
    prices = _make_prices({"BBB": [100, 98, 95, 93, 92, 91, 90]})
    result = build_nav_for_batch(
        entry_date  = "20251009",
        stock_list  = ["BBB"],
        prices_df   = prices,
        stop_loss   = -0.15,
        trailing    = False,
        holding_days= 10,
    )
    st = _get_ticker(result, "BBB")
    assert st["stop_loss_triggered"] is False, "未达阈值不应触发"
    assert st["stop_loss_date"] is None


# ── 测试 3：不触发时 holding_days_actual == 持有全程 ─────────────────────────

def test_holding_days_actual_full_when_no_stop():
    """不触发止损时，实际持仓天数 = 价格数据天数 - 1。"""
    n = 10
    prices = _make_prices({"CCC": [100] * (n + 1)})
    result = build_nav_for_batch(
        entry_date  = "20251009",
        stock_list  = ["CCC"],
        prices_df   = prices,
        stop_loss   = -0.15,
        trailing    = False,
        holding_days= n,
    )
    st = _get_ticker(result, "CCC")
    assert st["stop_loss_triggered"] is False
    assert st["holding_days_actual"] == n


# ── 测试 4：追踪止损从峰值回撤触发 ───────────────────────────────────────────

def test_trailing_stop_triggers_from_peak():
    """追踪止损：先涨后跌，从最高点回撤 >15% 应触发。

    Day 0=100 → Day 3=130（峰值）→ Day 6=109（相对峰值 -16%）
    固定止损不触发（绝对涨 +9%），追踪止损触发。
    """
    prices = _make_prices({
        "DDD": [100, 110, 120, 130, 120, 115, 109, 108]
    })
    # 追踪止损
    result_trailing = build_nav_for_batch(
        entry_date  = "20251009",
        stock_list  = ["DDD"],
        prices_df   = prices,
        stop_loss   = -0.15,
        trailing    = True,
        holding_days= 20,
    )
    # 固定止损
    result_fixed = build_nav_for_batch(
        entry_date  = "20251009",
        stock_list  = ["DDD"],
        prices_df   = prices,
        stop_loss   = -0.15,
        trailing    = False,
        holding_days= 20,
    )
    trailing_st = _get_ticker(result_trailing, "DDD")
    fixed_st    = _get_ticker(result_fixed,    "DDD")

    assert trailing_st["stop_loss_triggered"] is True,  "追踪止损应触发"
    assert fixed_st["stop_loss_triggered"]    is False, "固定止损不应触发（绝对收益为正）"


# ── 测试 5：先涨后跌场景下追踪止损先于固定止损触发 ────────────────────────

def test_trailing_stop_triggers_earlier_after_peak():
    """先涨后跌：追踪止损从峰值触发（更保守），比固定止损更早。

    价格: 100 → 120(峰值) → 102(从峰跌 15%) → 84(从入场跌 16%)
    - 追踪止损: 120 * 0.85 = 102 → 在价格降到 102 时触发（较早）
    - 固定止损: 100 * 0.85 = 85  → 需要跌到 85 才触发（较晚）
    """
    prices = _make_prices({
        "EEE": [100, 105, 110, 120, 115, 108, 102, 98, 90, 84, 80]
    })
    r_fixed    = build_nav_for_batch("20251009", ["EEE"], prices, -0.15, False, 20)
    r_trailing = build_nav_for_batch("20251009", ["EEE"], prices, -0.15, True,  20)

    fixed_st    = _get_ticker(r_fixed,    "EEE")
    trailing_st = _get_ticker(r_trailing, "EEE")

    # 两者都触发
    assert trailing_st["stop_loss_triggered"] is True,  "追踪止损应触发"
    assert fixed_st["stop_loss_triggered"]    is True,  "固定止损应触发"

    # 追踪止损触发日 <= 固定止损触发日（从峰值触发更早或同日）
    fixed_date    = fixed_st["stop_loss_date"]
    trailing_date = trailing_st["stop_loss_date"]
    assert trailing_date <= fixed_date, (
        f"追踪止损({trailing_date}) 应 <= 固定止损({fixed_date})（从峰值触发更早）"
    )


# ── 测试 6：组合 NAV = 各股等权均值 ──────────────────────────────────────────

def test_portfolio_nav_equals_equal_weight_mean():
    """两股等权组合第 2 日净值应等于各自净值的均值。"""
    prices = _make_prices({
        "F1": [100, 110, 120],   # 第1日 +10%, 第2日 +9.09%
        "F2": [100,  90,  80],   # 第1日 -10%, 第2日 -11.11%
    })
    result = build_nav_for_batch("20251009", ["F1", "F2"], prices, -0.50, False, 10)

    port_nav = pd.DataFrame(result["portfolio_nav"], columns=["date", "nav"])
    # Day 0: both=1.0 → mean=1.0
    # Day 1: F1=1.10, F2=0.90 → mean=1.00
    # Day 2: F1=1.20, F2=0.80 → mean=1.00
    day0_nav = port_nav.iloc[0]["nav"]
    assert abs(day0_nav - 1.0) < 1e-6, f"Day0 组合净值应为 1.0，得 {day0_nav}"

    day1_nav = port_nav.iloc[1]["nav"]
    expected = (1.10 + 0.90) / 2
    assert abs(day1_nav - expected) < 1e-4, (
        f"Day1 组合净值应为 {expected:.4f}，得 {day1_nav:.4f}"
    )


# ── 测试 7：止损版 MaxDD ≤ 不止损版 MaxDD ────────────────────────────────────

def test_stop_loss_limits_max_drawdown():
    """加入止损后，组合最大回撤应 ≤ 不止损版本。"""
    # 构造下跌行情使止损生效
    prices = _make_prices({
        "G1": [100, 95, 88, 80, 74, 70, 68],
        "G2": [100, 92, 85, 78, 72, 68, 65],
    })
    result = build_nav_for_batch("20251009", ["G1", "G2"], prices, -0.15, False, 20)

    mdd_no_sl = abs(result["portfolio_max_drawdown"])
    mdd_sl    = abs(result["with_stop_loss"]["portfolio_max_drawdown"])

    assert mdd_sl <= mdd_no_sl + 1e-6, (
        f"止损版 MaxDD({mdd_sl:.4f}) 应 ≤ 不止损版({mdd_no_sl:.4f})"
    )


# ── 测试 8：止损触发后 NAV 冻结（不再下降） ──────────────────────────────────

def test_stop_loss_freezes_nav_after_trigger():
    """止损触发后，止损版组合 NAV 不应继续下降。"""
    # H1 迅速跌破止损，之后继续暴跌
    prices = _make_prices({
        "H1": [100, 90, 80, 70, 60, 50, 40, 30]
    })
    result = build_nav_for_batch("20251009", ["H1"], prices, -0.15, False, 20)

    sl_nav_list = result["with_stop_loss"]["portfolio_nav"]
    assert len(sl_nav_list) > 0

    sl_navs = [v for _, v in sl_nav_list]
    # 找到止损触发点后的 NAV，应全部相同（冻结）
    triggered_idx = None
    for i, v in enumerate(sl_navs):
        if i > 0 and abs(v - sl_navs[i - 1]) < 1e-9:
            triggered_idx = i
            break

    if triggered_idx is not None:
        frozen_val = sl_navs[triggered_idx]
        for v in sl_navs[triggered_idx:]:
            assert abs(v - frozen_val) < 1e-6, (
                f"止损冻结后 NAV 不应变化：期望 {frozen_val:.6f}，得 {v:.6f}"
            )


# ── 测试 9：缺失股票被跳过，不影响其他股票结果 ───────────────────────────────

def test_missing_ticker_skipped_gracefully():
    """stock_list 中包含不在 prices_df 的股票，应被跳过，不引发异常。"""
    prices = _make_prices({"I1": [100, 105, 110, 108]})
    result = build_nav_for_batch(
        entry_date = "20251009",
        stock_list = ["I1", "MISSING_TICKER"],
        prices_df  = prices,
        stop_loss  = -0.15,
        trailing   = False,
        holding_days = 10,
    )
    # I1 有结果
    assert "I1" in result["stocks"]
    assert result["stocks"]["I1"].get("cumulative_nav")
    # 组合有数据
    assert len(result["portfolio_nav"]) > 0
    # MISSING_TICKER 在 stocks 里但是空的（或不存在）
    missing_data = result["stocks"].get("MISSING_TICKER", {})
    assert not missing_data.get("cumulative_nav"), "缺失股票不应有净值数据"


# ── 测试 10：空 prices_df 返回 empty result ───────────────────────────────────

def test_empty_prices_returns_empty_result():
    """prices_df 为空 DataFrame 时，应返回 fallback 空结果。"""
    empty_prices = pd.DataFrame()
    result = build_nav_for_batch(
        entry_date = "20251009",
        stock_list = ["J1", "J2"],
        prices_df  = empty_prices,
        stop_loss  = -0.15,
        trailing   = False,
    )
    assert result["n_stocks"] == 0
    assert result["portfolio_nav"] == []


# ── 测试 11：最终收益计算正确 ────────────────────────────────────────────────

def test_final_return_correct():
    """单股持有到期，最终收益 = (final_price/entry_price) - 1。"""
    prices = _make_prices({"K1": [100, 102, 105, 110, 115]})
    result = build_nav_for_batch("20251009", ["K1"], prices, -0.50, False, 10)
    st = _get_ticker(result, "K1")
    expected_return = 115 / 100 - 1   # +15%
    assert abs(st["final_return"] - expected_return) < 1e-4, (
        f"期望最终收益 {expected_return:.4f}，得 {st['final_return']:.4f}"
    )


# ── 测试 12：多股混合，部分触发止损 ──────────────────────────────────────────

def test_mixed_stop_loss_partial_trigger():
    """3 只股票：L1 暴跌触发止损，L2 L3 正常上涨，不应触发。"""
    prices = _make_prices({
        "L1": [100, 92, 84, 78, 73, 70],   # -30% → 触发
        "L2": [100, 105, 110, 112, 115, 118],
        "L3": [100, 101, 103, 104, 106, 108],
    })
    result = build_nav_for_batch("20251009", ["L1", "L2", "L3"], prices, -0.15, False, 20)

    assert _get_ticker(result, "L1")["stop_loss_triggered"] is True
    assert _get_ticker(result, "L2")["stop_loss_triggered"] is False
    assert _get_ticker(result, "L3")["stop_loss_triggered"] is False

    stopped = result["with_stop_loss"]["stopped_stocks"]
    assert "L1" in stopped
    assert "L2" not in stopped
    assert "L3" not in stopped


# ── 测试 13：cumulative_nav 第一条净值为 1.0 ─────────────────────────────────

def test_cumulative_nav_starts_at_one():
    """所有股票 cumulative_nav 第一个值应为 1.0（归一化）。"""
    prices = _make_prices({
        "M1": [50, 52, 55, 58],
        "M2": [200, 195, 190, 198],
    })
    result = build_nav_for_batch("20251009", ["M1", "M2"], prices, -0.50, False, 10)

    for ticker in ["M1", "M2"]:
        nav_list = _get_ticker(result, ticker).get("cumulative_nav", [])
        assert len(nav_list) > 0
        first_nav = nav_list[0][1]
        assert abs(first_nav - 1.0) < 1e-6, (
            f"{ticker} 起始净值应为 1.0，得 {first_nav}"
        )
