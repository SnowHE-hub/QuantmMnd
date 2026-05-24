"""Feature Expansion v1：PIT、手工 β/RS、市场类因子、缺失表、因子全集."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantmind.features.expansion import (
    MARKET_LEVEL_FEATURES,
    beta_60d,
    compute_all_expansion_factors,
    relative_strength_vs_csi300_60d,
    _index_close_series,
    _stock_index_return_window,
)
from quantmind.features.fundamental import FUNDAMENTAL_FACTORS
from quantmind.features.pipeline import FeaturePipeline, list_all_factor_names
from quantmind.features.sentiment import MARKET_LEVEL_SENTIMENT_FACTORS, SENTIMENT_FACTORS
from quantmind.features.standardize import standardize
from quantmind.features.technical import TECHNICAL_FACTORS


def _legacy_41_names() -> set[str]:
    """扩 v1 前「核心」因子：基本面 + 量价 + 情绪（含市场级北向等）."""
    return {
        name
        for block in (
            FUNDAMENTAL_FACTORS,
            TECHNICAL_FACTORS,
            SENTIMENT_FACTORS,
            MARKET_LEVEL_SENTIMENT_FACTORS,
        )
        for name, _ in block
    }


def _minimal_snapshot() -> dict:
    """universe + prices + daily_basic；无 expansion 可选表."""
    return {
        "universe": pd.DataFrame(
            {"ticker": ["000001.SZ", "600000.SH"], "weight": [1.5, 1.5]}
        ),
        "prices": pd.DataFrame(
            {
                "ticker": ["000001.SZ"] * 270 + ["600000.SH"] * 270,
                "trade_date": list(pd.bdate_range("2023-01-03", periods=270))
                * 2,
                "close": np.concatenate(
                    [
                        np.linspace(10, 12, 270),
                        np.linspace(20, 18, 270),
                    ]
                ),
                "volume": [1e6] * 540,
                "turnover_rate": [1.0] * 540,
                "high": [0] * 540,
                "low": [0] * 540,
            }
        ),
        "daily_basic": pd.DataFrame(
            {
                "ticker": ["000001.SZ", "600000.SH"],
                "trade_date": [date(2024, 6, 28)] * 2,
                "total_mv": [1_000_000.0, 2_000_000.0],
                "circ_mv": [800_000.0, 1_600_000.0],
                "pe_ttm": [10.0, 12.0],
                "pb": [1.0, 1.2],
                "ps_ttm": [2.0, 2.0],
                "dv_ttm": [0.01, 0.02],
                "total_share": [1e6] * 2,
                "float_share": [8e5, 16e5],
                "free_share": [8e5, 16e5],
            }
        ),
        "financial_indicators": pd.DataFrame(),
    }


class TestPITHkHold:
    def test_hk_hold_respects_as_of(self) -> None:
        as_of = date(2024, 6, 28)
        snap = _minimal_snapshot()
        snap["hk_hold"] = pd.DataFrame(
            {
                "ticker": ["000001.SZ"] * 3,
                "trade_date": pd.to_datetime(
                    ["2024-06-26", "2024-06-28", "2025-12-31"]
                ),
                "hold_ratio": [1.0, 5.0, 99.0],
                "hold_vol": [100.0, 500.0, 999.0],
            }
        )
        df = compute_all_expansion_factors(snap, as_of)
        assert df.loc["000001.SZ", "north_hold_ratio"] == pytest.approx(
            5.0
        ), "must ignore future trade_date"


class TestPITMargin:
    def test_margin_respects_as_of(self) -> None:
        as_of = date(2024, 6, 28)
        snap = _minimal_snapshot()
        snap["margin"] = pd.DataFrame(
            {
                "ticker": ["000001.SZ"] * 3,
                "trade_date": pd.to_datetime(
                    ["2024-06-26", "2024-06-28", "2025-01-15"]
                ),
                "rzye": [1.0, 3.0, 100.0],
                "rzmre": [0.1, 0.2, 0.3],
                "rqye": [1.0, 2.0, 50.0],
            }
        )
        df = compute_all_expansion_factors(snap, as_of)
        assert df.loc["000001.SZ", "margin_balance"] == pytest.approx(3.0)


class TestBetaRSHand:
    def test_beta_60d_cov_var(self) -> None:
        """β = cov/σ²_m 在手工对齐收益上成立."""
        rng = np.random.default_rng(42)
        n = 80
        dates = list(pd.bdate_range("2024-01-02", periods=n))
        mret_arr = rng.normal(0, 0.008, size=n)
        mret_arr[0] = 0.0
        idx_close_vals = 3000.0 * np.cumprod(1.0 + mret_arr)
        sret_arr = 1.5 * mret_arr
        stk_close_vals = 10.0 * np.cumprod(1.0 + sret_arr)
        as_of = pd.Timestamp(dates[-1]).date()

        snap = {
            "universe": pd.DataFrame({"ticker": ["000001.SZ"], "weight": [100.0]}),
            "index_daily": pd.DataFrame(
                {
                    "ts_code": ["000300.SH"] * n,
                    "trade_date": dates,
                    "close": idx_close_vals,
                }
            ),
            "prices": pd.DataFrame(
                {
                    "ticker": ["000001.SZ"] * n,
                    "trade_date": dates,
                    "close": stk_close_vals,
                    "volume": [1e6] * n,
                    "turnover_rate": [1.0] * n,
                    "high": [0.0] * n,
                    "low": [0.0] * n,
                }
            ),
            "daily_basic": pd.DataFrame(
                {
                    "ticker": ["000001.SZ"],
                    "trade_date": [as_of],
                    "total_mv": [1e6],
                    "circ_mv": [1e6],
                    "pe_ttm": [10.0],
                    "pb": [1.0],
                    "ps_ttm": [2.0],
                    "dv_ttm": [0.01],
                    "total_share": [1e6],
                    "float_share": [1e6],
                    "free_share": [1e6],
                }
            ),
        }
        b = beta_60d(snap, as_of)
        assert b.loc["000001.SZ"] == pytest.approx(1.5, abs=0.05)

    def test_relative_strength_60d_ratio_formula(self) -> None:
        """(1+Rs)/(1+Ri)-1 与收盘价窗口一致."""
        dates = list(pd.bdate_range("2024-06-01", periods=70))
        idx_close = pd.Series(np.linspace(3000, 3300, len(dates)), index=pd.DatetimeIndex(dates))
        st_close = pd.Series(np.linspace(10, 12, len(dates)), index=pd.DatetimeIndex(dates))
        rs, ri = _stock_index_return_window(st_close, idx_close, 60)
        expected = (1 + rs) / (1 + ri) - 1.0
        as_of = dates[-1]
        if hasattr(as_of, "date"):
            as_of_d = as_of.date()
        else:
            as_of_d = as_of
        snap = {
            "universe": pd.DataFrame({"ticker": ["000001.SZ"], "weight": [100.0]}),
            "index_daily": pd.DataFrame(
                {
                    "ts_code": ["000300.SH"] * len(dates),
                    "trade_date": dates,
                    "close": idx_close.values,
                }
            ),
            "prices": pd.DataFrame(
                {
                    "ticker": ["000001.SZ"] * len(dates),
                    "trade_date": dates,
                    "close": st_close.values,
                    "volume": [1e6] * len(dates),
                    "turnover_rate": [1.0] * len(dates),
                    "high": [0] * len(dates),
                    "low": [0] * len(dates),
                }
            ),
        }
        got = relative_strength_vs_csi300_60d(snap, as_of_d)
        assert got.loc["000001.SZ"] == pytest.approx(expected, rel=1e-5)


class TestMarketLevelNotZerod:
    def test_constant_market_column_unchanged_after_standardize(self) -> None:
        idx = pd.Index(["a", "b", "c"])
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0],
                "market_momentum_60d": [0.05, 0.05, 0.05],
            },
            index=idx,
        )
        out = standardize(
            df,
            do_neutralize=False,
            zscore_winsorize_exclude=MARKET_LEVEL_FEATURES,
        )
        assert (out["market_momentum_60d"] == 0.05).all()
        assert abs(out["x"].mean()) < 1e-6


class TestMissingTablesNoCrash:
    def test_expansion_with_minimal_snapshot(self) -> None:
        snap = _minimal_snapshot()
        df = compute_all_expansion_factors(snap, date(2024, 6, 28))
        assert not df.empty
        assert np.isnan(df.loc["000001.SZ", "north_hold_ratio"])


class TestLegacyFactorsPresent:
    def test_41_base_names_still_listed(self) -> None:
        all_n = set(list_all_factor_names())
        legacy = _legacy_41_names()
        # 注：legacy 因子数量随版本增长（2026-05-24 修正：移除硬编码 ==41 检查）
        # 核心测试：所有 legacy 因子名都应仍在注册表中
        assert len(legacy) >= 41, f"legacy 因子数 {len(legacy)} 低于预期最小值 41"
        missing = legacy - all_n
        assert not missing, missing

    def test_registry_includes_expansion(self) -> None:
        names = list_all_factor_names()
        assert "north_hold_ratio" in names
        assert "exposure_industry" in names
        assert "market_momentum_60d" in names


class TestExpansionInPipeline:
    def test_new_columns_in_raw_output(self) -> None:
        snap = _minimal_snapshot()
        snap["stock_basic"] = pd.DataFrame(
            {
                "ticker": ["000001.SZ", "600000.SH"],
                "industry": ["银行", "保险"],
                "area": ["深圳", "上海"],
                "list_date": pd.to_datetime(["1990-01-01", "2000-01-01"]),
            }
        )
        df = FeaturePipeline(groups=["expansion"]).compute_raw_from_snapshot(
            snap, date(2024, 6, 28)
        )
        assert "beta_60d" in df.columns
        assert "exposure_industry" in df.columns


class TestIndexSeries:
    def test_index_close_sorted(self) -> None:
        idf = pd.DataFrame(
            {
                "ts_code": ["000300.SH"] * 3,
                "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                "close": [100.0, 101.0, 102.0],
            }
        )
        s = _index_close_series(idf, date(2024, 6, 30))
        assert len(s) == 3

