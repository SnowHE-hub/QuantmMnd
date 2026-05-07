"""tests/test_features_utils.py — features.utils 单元测试.

覆盖：
    - safe_divide / safe_log
    - latest_report_per_ticker
    - lookup_period
    - ytd_to_ttm（核心：YTD → TTM 转换）
    - cagr
    - pivot_prices
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantmind.features.utils import (
    cagr,
    latest_report_per_ticker,
    lookup_period,
    pivot_prices,
    safe_divide,
    safe_log,
    ytd_to_ttm,
)

# ============================================================================
# 安全运算
# ============================================================================


class TestSafeOps:
    def test_safe_divide_handles_zero_denom(self) -> None:
        out = safe_divide(pd.Series([1.0, 2.0]), pd.Series([0.0, 4.0]))
        assert pd.isna(out.iloc[0])
        assert out.iloc[1] == 0.5

    def test_safe_divide_strips_inf(self) -> None:
        # 人工塞个 inf，确保替换为 NaN
        num = pd.Series([1.0, np.inf, -np.inf])
        den = pd.Series([2.0, 1.0, 1.0])
        out = safe_divide(num, den)
        assert out.iloc[0] == 0.5
        assert pd.isna(out.iloc[1])
        assert pd.isna(out.iloc[2])

    def test_safe_log_handles_negative(self) -> None:
        out = safe_log(pd.Series([1.0, -1.0, 0.0, 100.0]))
        assert out.iloc[0] == 0.0
        assert pd.isna(out.iloc[1])
        assert pd.isna(out.iloc[2])
        assert out.iloc[3] == pytest.approx(np.log(100))


# ============================================================================
# latest_report_per_ticker
# ============================================================================


class TestLatestReport:
    def _df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ticker": ["A", "A", "A", "B", "B"],
                "report_date": pd.to_datetime(
                    ["2023-12-31", "2024-03-31", "2023-09-30", "2023-12-31", "2024-03-31"]
                ),
                "f_ann_date": pd.to_datetime(
                    ["2024-04-01", "2024-04-25", "2023-10-30", "2024-03-15", "2024-04-20"]
                ),
                "value": [10, 11, 9, 20, 22],
            }
        )

    def test_picks_latest_period_per_ticker(self) -> None:
        out = latest_report_per_ticker(self._df())
        assert "value" in out.columns
        assert out.loc["A", "value"] == 11
        assert out.loc["B", "value"] == 22

    def test_handles_empty(self) -> None:
        out = latest_report_per_ticker(pd.DataFrame())
        assert out.empty


# ============================================================================
# lookup_period
# ============================================================================


class TestLookupPeriod:
    def test_returns_specific_period(self) -> None:
        df = pd.DataFrame(
            {
                "ticker": ["A", "A", "B"],
                "report_date": pd.to_datetime(["2023-12-31", "2024-03-31", "2023-12-31"]),
                "value": [10, 11, 20],
            }
        )
        out = lookup_period(df, pd.Timestamp("2023-12-31"))
        assert len(out) == 2
        assert out.loc["A", "value"] == 10
        assert out.loc["B", "value"] == 20


# ============================================================================
# ytd_to_ttm
# ============================================================================


class TestYTDToTTM:
    def _build(self) -> pd.DataFrame:
        """构造一个含 5 个季度的财报样本."""
        # 2022 全年 = 100, 2023Q1=20, 2023Q2=50(YTD), 2023Q3=80, 2023Q4=120, 2024Q1=30
        rows = [
            ("A", "2022-12-31", 100.0),
            ("A", "2023-03-31", 20.0),
            ("A", "2023-06-30", 50.0),
            ("A", "2023-09-30", 80.0),
            ("A", "2023-12-31", 120.0),
            ("A", "2024-03-31", 30.0),
        ]
        return pd.DataFrame(
            [
                {
                    "ticker": t,
                    "report_date": pd.Timestamp(d),
                    "f_ann_date": pd.Timestamp(d) + pd.Timedelta(days=30),
                    "rev": v,
                }
                for t, d, v in rows
            ]
        )

    def test_ttm_at_q1_2024(self) -> None:
        """TTM Q1 2024 = annual_2023 + Q1_2024 - Q1_2023 = 120 + 30 - 20 = 130."""
        df = self._build()
        out = ytd_to_ttm(df, "rev")
        assert out["A"] == pytest.approx(130.0)

    def test_ttm_at_annual_returns_annual(self) -> None:
        """末期是 12-31（年报）→ TTM = 当期 YTD."""
        df = self._build().iloc[:5]  # 只到 2023 年报
        out = ytd_to_ttm(df, "rev")
        assert out["A"] == pytest.approx(120.0)

    def test_missing_prior_year_returns_nan(self) -> None:
        df = self._build().iloc[-1:]  # 仅 2024-03-31
        out = ytd_to_ttm(df, "rev")
        assert pd.isna(out["A"])

    def test_pit_filter_excludes_future_announcements(self) -> None:
        df = self._build()
        # 强制 f_ann_date 是 2026-01-01（未来），应被 PIT 过滤掉
        df.loc[df["report_date"] == pd.Timestamp("2024-03-31"), "f_ann_date"] = pd.Timestamp(
            "2099-12-31"
        )
        out = ytd_to_ttm(df, "rev", as_of=date(2024, 6, 30))
        # Q1 2024 行被剔除，最新就是 2023 年报，TTM = 120
        assert out["A"] == pytest.approx(120.0)


# ============================================================================
# CAGR
# ============================================================================


class TestCAGR:
    def test_basic(self) -> None:
        # 100 → 161.05 in 5 years = 10% CAGR
        out = cagr(pd.Series([100.0]), pd.Series([161.051]), 5.0)
        assert out.iloc[0] == pytest.approx(0.10, rel=1e-3)

    def test_zero_or_negative_start_returns_nan(self) -> None:
        out = cagr(pd.Series([0.0, -10.0]), pd.Series([100.0, 100.0]), 3.0)
        assert pd.isna(out.iloc[0])
        assert pd.isna(out.iloc[1])


# ============================================================================
# pivot_prices
# ============================================================================


class TestPivotPrices:
    def test_pivot(self) -> None:
        prices = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]
                ),
                "ticker": ["A", "A", "B", "B"],
                "close": [10, 11, 20, 22],
            }
        )
        wide = pivot_prices(prices)
        assert wide.shape == (2, 2)
        assert wide.columns.tolist() == ["A", "B"]
        assert wide.iloc[0, 0] == 10
        assert wide.iloc[1, 1] == 22
