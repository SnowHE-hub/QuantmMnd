"""tests/test_wf_costs.py — 成本三处扩展 + T+1 时点（合成/小样本）.

H-B 滑点流动性分层、时变印花税切换点、H-E 板块×时变涨跌停、T+1 入场。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantmind.backtest import wf_costs as wc


# ===========================================================================
# 时变印花税（2023-08-28 减半 0.1% → 0.05%）
# ===========================================================================
def test_stamp_duty_halving_switch():
    assert wc.stamp_duty_rate("2023-08-27") == pytest.approx(0.001)   # 减半前一日
    assert wc.stamp_duty_rate("2023-08-28") == pytest.approx(0.0005)  # 生效当日
    assert wc.stamp_duty_rate("2025-06-30") == pytest.approx(0.0005)  # OOS 全程减半后
    assert wc.stamp_duty_rate("2019-01-02") == pytest.approx(0.001)


# ===========================================================================
# H-E 板块 × 时变涨跌停
# ===========================================================================
def test_board_classification():
    assert wc.board_of("688981.SH") == "STAR"
    assert wc.board_of("300750.SZ") == "CHINEXT"
    assert wc.board_of("600519.SH") == "MAIN"
    assert wc.board_of("000001.SZ") == "MAIN"


def test_price_limit_sector_and_time():
    d_late = "2025-06-30"
    # 科创板 20%
    assert wc.price_limit_pct("688981.SH", d_late) == pytest.approx(0.20)
    # 主板 10%
    assert wc.price_limit_pct("600519.SH", d_late) == pytest.approx(0.10)
    assert wc.price_limit_pct("000001.SZ", d_late) == pytest.approx(0.10)
    # 创业板跨 2020-08-24：之前 10%，当日及之后 20%
    assert wc.price_limit_pct("300750.SZ", "2020-08-21") == pytest.approx(0.10)
    assert wc.price_limit_pct("300750.SZ", "2020-08-24") == pytest.approx(0.20)
    assert wc.price_limit_pct("300750.SZ", "2025-06-30") == pytest.approx(0.20)
    # ST 5%
    assert wc.price_limit_pct("600519.SH", d_late, is_st=True) == pytest.approx(0.05)


# ===========================================================================
# H-B 滑点流动性分层（本版必做）
# ===========================================================================
def test_slippage_tiers_by_quantile():
    t = wc.SlippageTiers()
    assert t.bp_for_quantile(0.10) == 5.0    # 大盘（最流动）
    assert t.bp_for_quantile(0.50) == 15.0   # 中盘
    assert t.bp_for_quantile(0.90) == 30.0   # 小盘（最不流动）
    # 中小盘滑点 > 大盘（H-B 核心：防低估成本）
    assert t.bp_for_quantile(0.90) > t.bp_for_quantile(0.10)


def test_slippage_nan_is_most_conservative():
    # 缺流动性信息 → 按最贵档（绝不低估成本 → 不造假阳性 alpha）
    t = wc.SlippageTiers()
    assert t.bp_for_quantile(float("nan")) == t.small_bp


def test_slippage_cross_section_amihud_to_bp():
    # amihud 越大=越不流动 → 分位越高 → bp 越大
    amihud = pd.Series({"A": 0.1, "B": 1.0, "C": 5.0, "D": 50.0, "E": 100.0, "F": 500.0})
    bp = wc.slippage_bp_series(amihud)
    assert bp["A"] == 5.0          # 最流动 → 大盘档
    assert bp["F"] == 30.0         # 最不流动 → 小盘档
    assert bp["A"] <= bp["C"] <= bp["F"]  # 单调


# ===========================================================================
# T+1 入场 / 出场时点
# ===========================================================================
def test_t_plus_1_entry_exit_index():
    # 信号 as_of 收盘出 → T+1 次日开盘入场
    assert wc.entry_fill_index(100) == 101
    # 持有 12 交易日 → as_of+1+12 出场
    assert wc.exit_fill_index(100, holding_td=12) == 113


def test_wf_cost_model_bundle():
    m = wc.WFCostModel()
    assert m.stamp_duty("2025-01-02") == pytest.approx(0.0005)
    assert m.limit_pct("688981.SH", "2025-01-02") == pytest.approx(0.20)
    amihud = pd.Series({"A": 0.1, "Z": 999.0})
    bp = m.slippage_bp(amihud)
    assert bp["Z"] > bp["A"]
