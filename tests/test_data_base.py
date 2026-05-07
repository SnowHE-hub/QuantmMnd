"""tests/test_data_base.py — DataProvider 抽象层纯单元测试（无网络）.

覆盖：
    - ticker 标准化：``normalize_ticker``、``to_akshare_*``、``to_tushare_code``
    - PIT 工具：``_assert_pit``、``_filter_pit``
    - 列名标准化：``_rename``
    - 元数据 stamping：``_stamp``
    - 日期格式化：``_to_yyyymmdd``、``_to_yyyy_dash``
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quantmind.data.base import (
    DataProvider,
    PITViolationError,
    normalize_ticker,
    to_akshare_em_symbol,
    to_akshare_symbol,
    to_tushare_code,
)

# ============================================================================
# Ticker 标准化
# ============================================================================


class TestNormalizeTicker:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("600519", "600519.SH"),
            ("600519.SH", "600519.SH"),
            ("600519.sh", "600519.SH"),
            ("sh600519", "600519.SH"),
            ("SH.600519", "600519.SH"),
            ("300750", "300750.SZ"),
            ("000001", "000001.SZ"),
            ("002594", "002594.SZ"),
            ("688981", "688981.SH"),  # 科创板
            ("832145", "832145.BJ"),  # 北交所
            (" 600519 ", "600519.SH"),
            ("600519/sh", "600519.SH"),
        ],
    )
    def test_various_inputs(self, raw: str, expected: str) -> None:
        assert normalize_ticker(raw) == expected

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_ticker("ABC")
        with pytest.raises(ValueError):
            normalize_ticker("60051")  # 5 位

    def test_to_akshare_symbol(self) -> None:
        assert to_akshare_symbol("600519.SH") == "600519"
        assert to_akshare_symbol("300750") == "300750"

    def test_to_akshare_em_symbol(self) -> None:
        assert to_akshare_em_symbol("600519.SH") == "SH600519"
        assert to_akshare_em_symbol("300750") == "SZ300750"

    def test_to_tushare_code(self) -> None:
        assert to_tushare_code("600519") == "600519.SH"
        assert to_tushare_code("sz300750") == "300750.SZ"


# ============================================================================
# PIT 工具
# ============================================================================


class TestPITHelpers:
    def test_assert_pit_no_violation(self) -> None:
        df = pd.DataFrame({"f_ann_date": ["2024-01-15", "2024-03-30"]})
        DataProvider._assert_pit(df, date(2024, 4, 1), "f_ann_date")

    def test_assert_pit_with_violation_raises(self) -> None:
        df = pd.DataFrame({"f_ann_date": ["2024-01-15", "2025-01-01"]})
        with pytest.raises(PITViolationError, match="PIT violation"):
            DataProvider._assert_pit(df, date(2024, 6, 30), "f_ann_date")

    def test_assert_pit_strict_false_only_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        df = pd.DataFrame({"f_ann_date": ["2025-01-01"]})
        DataProvider._assert_pit(df, date(2024, 6, 30), "f_ann_date", strict=False)

    def test_assert_pit_skips_when_as_of_none(self) -> None:
        df = pd.DataFrame({"f_ann_date": ["2025-01-01"]})
        DataProvider._assert_pit(df, None, "f_ann_date")

    def test_assert_pit_skips_when_empty(self) -> None:
        DataProvider._assert_pit(pd.DataFrame(), date(2024, 6, 30), "f_ann_date")

    def test_assert_pit_handles_multiple_columns(self) -> None:
        # 任一列违规即抛出（按列表顺序检查）
        df = pd.DataFrame(
            {
                "ann_date": ["2024-03-30", "2024-04-15"],
                "f_ann_date": ["2024-04-01", "2024-04-20"],
            }
        )
        with pytest.raises(PITViolationError, match="ann_date"):
            DataProvider._assert_pit(df, date(2024, 4, 10), ["ann_date", "f_ann_date"])

        # 同一日期但只一列违规：应能定位到那一列
        df2 = pd.DataFrame(
            {"ann_date": ["2024-03-30"], "f_ann_date": ["2024-04-20"]}
        )
        with pytest.raises(PITViolationError, match="f_ann_date"):
            DataProvider._assert_pit(df2, date(2024, 4, 10), ["ann_date", "f_ann_date"])

    def test_assert_pit_skips_missing_columns(self) -> None:
        df = pd.DataFrame({"f_ann_date": ["2024-01-15"]})
        DataProvider._assert_pit(df, date(2024, 6, 30), ["f_ann_date", "nonexistent_col"])

    def test_filter_pit_drops_future_rows(self) -> None:
        df = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2024-01-01", "2024-06-30", "2024-09-15"]),
                "close": [100, 110, 120],
            }
        )
        out = DataProvider._filter_pit(df, date(2024, 7, 1), "trade_date")
        assert len(out) == 2
        assert out["close"].tolist() == [100, 110]

    def test_filter_pit_idempotent_when_as_of_none(self) -> None:
        df = pd.DataFrame(
            {"trade_date": pd.to_datetime(["2024-01-01", "2025-01-01"]), "close": [100, 200]}
        )
        out = DataProvider._filter_pit(df, None, "trade_date")
        assert len(out) == 2

    def test_filter_pit_handles_nat(self) -> None:
        df = pd.DataFrame({"trade_date": [pd.NaT, pd.Timestamp("2024-01-01")], "x": [1, 2]})
        out = DataProvider._filter_pit(df, date(2024, 6, 30), "trade_date")
        # NaT 行应被剔除
        assert len(out) == 1
        assert out["x"].iloc[0] == 2


# ============================================================================
# 列名标准化 + stamping
# ============================================================================


class _DummyProvider(DataProvider):
    name = "dummy"

    def get_price(self, *a, **kw):  # type: ignore[override]
        return pd.DataFrame()

    def get_financials(self, *a, **kw):  # type: ignore[override]
        return pd.DataFrame()

    def get_index_constituents(self, *a, **kw):  # type: ignore[override]
        return []

    def is_tradable(self, *a, **kw):  # type: ignore[override]
        return True


class TestColumnStandardization:
    def test_rename_only_existing_keys(self) -> None:
        df = pd.DataFrame({"日期": [1], "无关列": [2]})
        mapping = {"日期": "trade_date", "缺失列": "close"}
        out = DataProvider._rename(df, mapping)
        assert list(out.columns) == ["trade_date", "无关列"]


class TestStamp:
    def test_stamp_writes_attrs(self) -> None:
        p = _DummyProvider()
        df = pd.DataFrame({"a": [1]})
        out = p._stamp(df, date(2024, 6, 30), ticker="300750.SZ", extra={"k": "v"})
        assert out.attrs["data_source"] == "dummy"
        assert out.attrs["as_of"] == "2024-06-30"
        assert out.attrs["ticker"] == "300750.SZ"
        assert out.attrs["k"] == "v"
        assert "fetched_at" in out.attrs


class TestDateFormatters:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("2024-06-30", "20240630"),
            ("2024/06/30", "20240630"),
            ("20240630", "20240630"),
            (date(2024, 6, 30), "20240630"),
        ],
    )
    def test_to_yyyymmdd(self, raw, expected) -> None:
        assert DataProvider._to_yyyymmdd(raw) == expected

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("20240630", "2024-06-30"),
            ("2024-06-30", "2024-06-30"),
            (date(2024, 6, 30), "2024-06-30"),
        ],
    )
    def test_to_yyyy_dash(self, raw, expected) -> None:
        assert DataProvider._to_yyyy_dash(raw) == expected
