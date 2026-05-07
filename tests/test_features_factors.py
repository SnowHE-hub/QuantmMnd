"""tests/test_features_factors.py — 因子计算正确性测试.

用合成数据验证因子的数学正确性（不依赖真实 snapshot）。
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantmind.features import (
    compute_all_fundamental_factors,
    compute_all_sentiment_factors,
    compute_all_technical_factors,
)
from quantmind.features.fundamental import (
    book_to_market,
    earnings_yield,
    log_market_cap,
    pb,
    pe_ttm,
    roe_ttm,
)
from quantmind.features.technical import (
    momentum_1m,
    momentum_3m,
    rsi_14,
    volatility_3m,
)

# ============================================================================
# 合成 snapshot
# ============================================================================


@pytest.fixture
def synthetic_snapshot() -> dict:
    """构造一个 3 ticker × 280 天的最小 snapshot."""
    tickers = ["A", "B", "C"]
    n_days = 280
    dates = pd.bdate_range(end="2024-06-28", periods=n_days)

    # universe
    universe = pd.DataFrame(
        {"ticker": tickers, "weight": [40.0, 35.0, 25.0], "trade_date": dates[-1]}
    )

    # prices: A 单调上涨，B 单调下跌，C 平稳
    rng = np.random.default_rng(42)
    prices_rows = []
    for t in tickers:
        if t == "A":
            ret = rng.normal(0.001, 0.02, n_days)  # +0.1%/day
        elif t == "B":
            ret = rng.normal(-0.001, 0.02, n_days)  # -0.1%/day
        else:
            ret = rng.normal(0.0, 0.005, n_days)  # 平稳低波动
        close = 100 * np.exp(np.cumsum(ret))
        prices_rows.extend(
            [
                {
                    "trade_date": d,
                    "ticker": t,
                    "open": c,
                    "high": c * 1.01,
                    "low": c * 0.99,
                    "close": c,
                    "volume": 1e6,
                    "amount": c * 1e6,
                    "pct_change": r * 100,
                    "turnover_rate": 1.0,
                    "total_mv": c * 1e8,
                    "pe_ttm": 20.0,
                    "pb": 2.0,
                }
                for d, c, r in zip(dates, close, ret, strict=True)
            ]
        )
    prices = pd.DataFrame(prices_rows)

    # daily_basic（取 last day 的快照）
    last = prices[prices["trade_date"] == dates[-1]].copy()
    daily_basic = last[["ticker", "trade_date", "close", "pe_ttm", "pb", "total_mv"]].copy()
    daily_basic["pe"] = daily_basic["pe_ttm"]
    daily_basic["ps"] = 3.0
    daily_basic["ps_ttm"] = 3.0
    daily_basic["dv_ttm"] = 1.5
    daily_basic["circ_mv"] = daily_basic["total_mv"] * 0.7
    daily_basic["turnover_rate"] = 1.0
    daily_basic["turnover_rate_f"] = 1.5
    daily_basic["volume_ratio"] = 1.2
    daily_basic["dv_ratio"] = 1.5
    daily_basic["total_share"] = daily_basic["total_mv"] / 100
    daily_basic["float_share"] = daily_basic["total_share"] * 0.8
    daily_basic["free_share"] = daily_basic["total_share"] * 0.5

    # financial_indicators (8 个季度 × 3 票)
    periods = pd.to_datetime(
        [f"{y}-{m:02d}-{d:02d}" for y, m, d in [
            (2022, 12, 31), (2023, 3, 31), (2023, 6, 30), (2023, 9, 30),
            (2023, 12, 31), (2024, 3, 31),
        ]]
    )
    ind_rows = []
    for i, t in enumerate(tickers):
        for p in periods:
            ind_rows.append({
                "ticker": t,
                "report_date": p,
                "ann_date": p + pd.Timedelta(days=30),
                "roe": 10.0 + i * 5,
                "roe_yearly": 12.0 + i * 5,
                "roa": 5.0 + i * 2,
                "roa_yearly": 6.0 + i * 2,
                "gross_margin": 30.0 + i * 10,
                "grossprofit_margin": 30.0 + i * 10,
                "netprofit_margin": 10.0 + i * 5,
                "debt_to_assets": 50.0 - i * 10,
                "current_ratio": 2.0 + i * 0.5,
                "assets_turn": 0.5 + i * 0.1,
                "assets_to_eqt": 2.0,
                "tr_yoy": 5.0 + i * 3,
                "or_yoy": 5.0,
                "op_yoy": 6.0,
                "netprofit_yoy": 7.0,
                "q_sales_yoy": 8.0,
            })
    ind = pd.DataFrame(ind_rows)

    # financial statements (minimal — only required for accruals/ocf_to_revenue)
    fin_inc = pd.DataFrame([
        {"ticker": t, "report_date": p, "f_ann_date": p + pd.Timedelta(days=30),
         "ann_date": p + pd.Timedelta(days=30),
         "n_income_attr_p": 1e9, "n_income": 1e9, "total_revenue": 1e10}
        for t in tickers for p in periods
    ])
    fin_bs = pd.DataFrame([
        {"ticker": t, "report_date": p, "f_ann_date": p + pd.Timedelta(days=30),
         "total_assets": 1e11}
        for t in tickers for p in periods
    ])
    fin_cf = pd.DataFrame([
        {"ticker": t, "report_date": p, "f_ann_date": p + pd.Timedelta(days=30),
         "n_cashflow_act": 9e8}
        for t in tickers for p in periods
    ])

    # north_bound: 30 trading days, constant
    nb_dates = pd.bdate_range(end=dates[-1], periods=30)
    nb = pd.DataFrame({
        "trade_date": nb_dates,
        "north_money": [100.0] * 30,
    })

    return {
        "universe": universe,
        "prices": prices,
        "daily_basic": daily_basic,
        "financial_indicators": ind,
        "financials_income": fin_inc,
        "financials_balance_sheet": fin_bs,
        "financials_cashflow": fin_cf,
        "north_bound": nb,
    }


# ============================================================================
# 基本面因子（直接从 daily_basic / fina_indicator 取）
# ============================================================================


class TestFundamentalFactors:
    def test_pe_ttm_pb_ps(self, synthetic_snapshot: dict) -> None:
        s = pe_ttm(synthetic_snapshot, date(2024, 6, 30))
        assert (s == 20.0).all()
        s = pb(synthetic_snapshot, date(2024, 6, 30))
        assert (s == 2.0).all()

    def test_book_to_market_inverse_of_pb(self, synthetic_snapshot: dict) -> None:
        s = book_to_market(synthetic_snapshot, date(2024, 6, 30))
        assert np.allclose(s.values, 0.5)

    def test_earnings_yield_inverse_of_pe(self, synthetic_snapshot: dict) -> None:
        s = earnings_yield(synthetic_snapshot, date(2024, 6, 30))
        assert np.allclose(s.values, 0.05)

    def test_log_market_cap_monotonic_in_total_mv(self, synthetic_snapshot: dict) -> None:
        s = log_market_cap(synthetic_snapshot, date(2024, 6, 30))
        # A 价格涨最多 → total_mv 最大 → log_market_cap 最大
        assert s.notna().all()
        assert s["A"] > s["C"] > s["B"]

    def test_roe_ttm_monotonic_with_synthetic_data(self, synthetic_snapshot: dict) -> None:
        # synthetic: ticker A=10, B=15, C=20  for roe_yearly
        s = roe_ttm(synthetic_snapshot, date(2024, 6, 30))
        assert s["A"] == 12.0
        assert s["B"] == 17.0
        assert s["C"] == 22.0

    def test_compute_all_returns_expected_columns(self, synthetic_snapshot: dict) -> None:
        df = compute_all_fundamental_factors(synthetic_snapshot, date(2024, 6, 30))
        assert df.shape[0] == 3  # 3 tickers
        assert df.shape[1] >= 20  # at least 20 fundamental factors


# ============================================================================
# 量价因子
# ============================================================================


class TestTechnicalFactors:
    def test_momentum_1m_positive_for_uptrend(self, synthetic_snapshot: dict) -> None:
        s = momentum_1m(synthetic_snapshot, date(2024, 6, 30))
        # A 长期上涨，1月动量应为正；B 长期下跌应为负
        assert s["A"] > 0
        assert s["B"] < 0

    def test_momentum_3m_orders(self, synthetic_snapshot: dict) -> None:
        s = momentum_3m(synthetic_snapshot, date(2024, 6, 30))
        assert s["A"] > s["B"]

    def test_volatility_higher_for_high_vol_stock(self, synthetic_snapshot: dict) -> None:
        s = volatility_3m(synthetic_snapshot, date(2024, 6, 30))
        # A、B 都 sigma=0.02；C sigma=0.005 → C 应低
        assert s["C"] < s["A"]
        assert s["C"] < s["B"]

    def test_rsi_in_valid_range(self, synthetic_snapshot: dict) -> None:
        s = rsi_14(synthetic_snapshot, date(2024, 6, 30))
        assert ((s >= 0) & (s <= 100)).all()
        # 上涨股 A 的 RSI 应高于下跌股 B
        assert s["A"] > s["B"]

    def test_compute_all_technical(self, synthetic_snapshot: dict) -> None:
        df = compute_all_technical_factors(synthetic_snapshot, date(2024, 6, 30))
        assert df.shape[0] == 3
        assert df.shape[1] >= 14


# ============================================================================
# 情绪因子
# ============================================================================


class TestSentimentFactors:
    def test_north_bound_constant_across_tickers(self, synthetic_snapshot: dict) -> None:
        df = compute_all_sentiment_factors(synthetic_snapshot, date(2024, 6, 30))
        col = df["north_bound_30d_net_inflow"]
        assert col.nunique() == 1  # market-level signal -> 同值

    def test_free_float_ratio_in_zero_one(self, synthetic_snapshot: dict) -> None:
        df = compute_all_sentiment_factors(synthetic_snapshot, date(2024, 6, 30))
        col = df["free_float_ratio"].dropna()
        assert ((col > 0) & (col <= 1)).all()
