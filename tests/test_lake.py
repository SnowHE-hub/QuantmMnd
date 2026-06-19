"""tests/test_lake.py — 数据湖契约测试.

覆盖：
- read_lake_window 严格 PIT（不含未来 trade_date）
- lookback >= 252 交易日可用（beta_252d 前提）
- 原生列保留
- write_lake 幂等（重复写不增行；同 key keep='last' 覆盖）
- north_bound 按 trade_date 去重
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quantmind.data import lake


# ---------------------------------------------------------------------------
# 依赖已建好的真实湖表（scripts/build_data_lake.py 的产物）
# ---------------------------------------------------------------------------
_HAS_INDEX = lake.lake_path("index_daily").exists()
_HAS_PANEL = lake.PRICE_PANEL_PATH.exists()
needs_lake = pytest.mark.skipif(
    not (_HAS_INDEX and _HAS_PANEL),
    reason="需要 data/lake/index_daily.parquet 与 alpha_prices_panel.parquet",
)


@needs_lake
def test_read_window_is_strictly_pit():
    """返回窗口不得含 trade_date > as_of（PIT 不泄漏未来）。"""
    as_of = date(2023, 6, 30)
    df = lake.read_lake_window("index_daily", as_of, lookback_trading_days=252)
    assert not df.empty
    assert df["trade_date"].max() <= pd.Timestamp(as_of)


@needs_lake
def test_read_window_non_trading_asof_excludes_future():
    """as_of 取非交易日（周日），仍严格 <= as_of。"""
    as_of = date(2023, 7, 2)  # 周日
    df = lake.read_lake_window("index_daily", as_of, lookback_trading_days=60)
    assert not df.empty
    assert df["trade_date"].max() <= pd.Timestamp(as_of)


@needs_lake
def test_lookback_supports_252_trading_days():
    """lookback=252 时，单标的应取到约 252 个交易日（beta_252d 前提）。"""
    as_of = date(2024, 12, 31)
    df = lake.read_lake_window("index_daily", as_of, lookback_trading_days=252)
    # index_daily 含多个指数码，按单一 code 验证交易日数
    one = df[df["ts_code"] == "000300.SH"]
    n_days = one["trade_date"].nunique()
    assert 250 <= n_days <= 252, f"got {n_days} trading days"


@needs_lake
def test_read_window_keeps_native_columns():
    """保留原生列（index_daily 的 close/open/high/low 等）。"""
    df = lake.read_lake_window("index_daily", date(2024, 1, 31), 20)
    for col in ("ts_code", "trade_date", "close", "open", "high", "low"):
        assert col in df.columns


@needs_lake
def test_read_window_missing_series_returns_empty():
    df = lake.read_lake_window("does_not_exist_xyz", date(2024, 1, 1), 10)
    assert df.empty


def test_write_lake_idempotent_and_keep_last(monkeypatch, tmp_path):
    """重复写不增行；同 key 后写覆盖（keep='last'）。"""
    monkeypatch.setattr(lake, "DATA_LAKE_DIR", tmp_path)

    df1 = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"],
            "trade_date": ["20200102", "20200102"],
            "val": [1.0, 2.0],
        }
    )
    n1 = lake.write_lake("daily_basic", df1)
    assert n1 == 2

    # 同批重写 → 行数不变（幂等）
    n2 = lake.write_lake("daily_basic", df1)
    assert n2 == 2

    # 同 key 改值 → 行数不变，值被覆盖为最新
    df1b = df1.copy()
    df1b["val"] = [9.0, 9.0]
    lake.write_lake("daily_basic", df1b)
    got = pd.read_parquet(lake.lake_path("daily_basic"))
    assert len(got) == 2
    assert set(got["val"]) == {9.0}

    # 追加新 key → 行数增加
    df2 = pd.DataFrame(
        {"ts_code": ["000001.SZ"], "trade_date": ["20200103"], "val": [3.0]}
    )
    n3 = lake.write_lake("daily_basic", df2)
    assert n3 == 3


def test_write_lake_north_bound_dedup_by_trade_date(monkeypatch, tmp_path):
    """north_bound 仅按 trade_date 去重（无 ts_code 维度）。"""
    monkeypatch.setattr(lake, "DATA_LAKE_DIR", tmp_path)
    df = pd.DataFrame(
        {"trade_date": ["20200102", "20200102", "20200103"], "north_money": [10, 99, 20]}
    )
    n = lake.write_lake("north_bound", df)
    assert n == 2  # 20200102 去重保留最后一条
    got = pd.read_parquet(lake.lake_path("north_bound"))
    v = got.loc[got["trade_date"] == pd.Timestamp("2020-01-02"), "north_money"].iloc[0]
    assert v == 99


def test_write_lake_missing_key_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(lake, "DATA_LAKE_DIR", tmp_path)
    bad = pd.DataFrame({"trade_date": ["20200102"], "x": [1]})  # 缺 ts_code
    with pytest.raises(ValueError):
        lake.write_lake("margin", bad)


@needs_lake
def test_calendar_monotonic_and_covers_range():
    cal = lake.load_trading_calendar()
    assert cal.is_monotonic_increasing
    assert cal.min() <= pd.Timestamp("2019-01-02")
    assert cal.max() >= pd.Timestamp("2026-05-11")
