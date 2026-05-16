"""Data Expansion v1：PIT、manifest、include 开关（不发起全量网络请求）."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from quantmind.data import snapshot as snapshot_mod
from quantmind.data.tushare_provider import TushareProvider
from quantmind.data.snapshot import module_manifest_for_dataframe


# ---------------------------------------------------------------------------
# Provider PIT（Tushare 返回含未来 trade_date 时应截断）
# ---------------------------------------------------------------------------


def test_hk_hold_pit_trade_date_le_as_of(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = pd.DataFrame(
        {
            "ts_code": ["600519.SH", "600519.SH"],
            "trade_date": ["20240102", "20990101"],
            "vol": [100.0, 200.0],
        }
    )
    monkeypatch.setattr(
        "quantmind.data.tushare_provider._raw_hk_hold",
        lambda ts_code, start, end: raw,
    )
    p = TushareProvider()
    out = p.get_hk_hold("600519.SH", "20240101", "20241231", as_of=date(2024, 6, 30))
    assert not out.empty
    assert (out["trade_date"] <= pd.Timestamp("2024-06-30")).all()


def test_margin_detail_pit_trade_date_le_as_of(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = pd.DataFrame(
        {
            "ts_code": ["600519.SH", "600519.SH"],
            "trade_date": ["20240102", "20990101"],
            "rzye": [1.0, 2.0],
            "rqyl": [0.0, 0.0],
        }
    )
    monkeypatch.setattr(
        "quantmind.data.tushare_provider._raw_margin_detail",
        lambda ts_code, start, end: raw,
    )
    p = TushareProvider()
    out = p.get_margin_detail("600519.SH", "20240101", "20241231", as_of=date(2024, 6, 30))
    assert not out.empty
    assert (out["trade_date"] <= pd.Timestamp("2024-06-30")).all()


def test_index_daily_pit_trade_date_le_as_of(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = pd.DataFrame(
        {
            "ts_code": ["000300.SH", "000300.SH"],
            "trade_date": ["20240102", "20990101"],
            "close": [3000.0, 4000.0],
        }
    )
    monkeypatch.setattr(
        "quantmind.data.tushare_provider._raw_index_daily",
        lambda ts_code, start, end: raw,
    )
    p = TushareProvider()
    out = p.get_index_daily("000300.SH", "20240101", "20241231", as_of=date(2024, 6, 30))
    assert not out.empty
    assert (out["trade_date"] <= pd.Timestamp("2024-06-30")).all()


def test_stock_basic_has_industry(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = pd.DataFrame(
        {
            "ts_code": ["600519.SH"],
            "name": ["贵州茅台"],
            "industry": ["白酒"],
            "area": ["贵州"],
            "market": ["主板"],
            "exchange": ["SSE"],
            "list_date": ["20010827"],
        }
    )
    monkeypatch.setattr(
        "quantmind.data.tushare_provider._raw_stock_basic",
        lambda: raw,
    )
    p = TushareProvider()
    df = p.get_stock_basic(include_delisted=False)
    assert "industry" in df.columns
    assert df["industry"].iloc[0] == "白酒"


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def test_module_manifest_trade_date_bounds() -> None:
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-03-01"]),
            "x": [1, 2],
        }
    )
    m = module_manifest_for_dataframe("test", df, pit_date_column="trade_date")
    assert m["row_count"] == 2
    assert "trade_date" in m["pit_date_column"] or m["pit_date_column"] == "trade_date"
    assert m["date_min"] is not None and m["date_max"] is not None


def test_module_manifest_stock_basic_list_date() -> None:
    df = pd.DataFrame(
        {
            "list_date": pd.to_datetime(["20100101", "20150202"]),
            "ticker": ["600519.SH", "000001.SZ"],
        }
    )
    m = module_manifest_for_dataframe(
        "stock_basic",
        df,
        pit_date_column=None,
        date_bounds_from_list_date=True,
    )
    assert m["pit_date_column"] is None
    assert m["date_min"] is not None


# ---------------------------------------------------------------------------
# build_snapshot：include 开关 + strict=False 不阻塞
# ---------------------------------------------------------------------------


class _StubTushare:
    name = "tushare"

    def __init__(self) -> None:
        self.hk_calls = 0
        self.margin_calls = 0

    def get_price(
        self,
        ticker: str,
        start,
        end,
        as_of=None,
        freq="D",
        adjust="qfq",
        with_basic=True,
    ):
        return pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2024-01-02"]),
                "ticker": [ticker],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "pre_close": [1.0],
                "volume": [1.0],
                "amount": [1.0],
                "pct_change": [0.0],
            }
        )

    def get_financials(self, ticker, statement_type, as_of=None, start=None, end=None):
        return pd.DataFrame()

    def get_financial_indicators(self, ticker, as_of=None, start=None, end=None):
        return pd.DataFrame()

    def get_daily_basic_market(self, trade_date, as_of=None):
        return pd.DataFrame(
            {
                "ticker": ["600519.SH"],
                "trade_date": [pd.Timestamp(trade_date)],
                "total_mv": [1e6],
                "close": [10.0],
            }
        )

    def get_north_bound_flow(
        self, ticker=None, start=None, end=None, as_of=None
    ):
        return pd.DataFrame(
            {
                "trade_date": pd.to_datetime([str(end)]),
                "north_money": [1.0],
            }
        )

    def get_stock_basic(self, include_delisted=True):
        return pd.DataFrame(
            {
                "ts_code": ["600519.SH"],
                "ticker": ["600519.SH"],
                "name": ["M"],
                "industry": ["T"],
                "area": ["Z"],
                "list_date": pd.to_datetime(["20010827"]),
                "exchange": ["SSE"],
                "market": ["主板"],
            }
        )

    def get_hk_hold(self, ticker, start, end, as_of=None):
        self.hk_calls += 1
        return pd.DataFrame()

    def get_margin_detail(self, ticker, start, end, as_of=None):
        self.margin_calls += 1
        raise RuntimeError("simulated margin_detail API failure")

    def get_index_daily(self, index_code, start, end, as_of=None):
        return pd.DataFrame(
            {
                "ts_code": [index_code],
                "trade_date": pd.to_datetime(["2024-01-02"]),
                "close": [3000.0],
            }
        )


@pytest.fixture
def snap_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """将 snapshot 写入 tmp_path/data/snapshots."""
    root = tmp_path / "data"
    root.mkdir(parents=True)

    data_ns = SimpleNamespace(
        dir=str(root),
        snapshot_strict=False,
        include_stock_basic=True,
        include_hk_hold=True,
        include_margin=True,
        include_index_daily=True,
        index_daily_codes=["000300.SH", "000905.SH"],
        hk_hold_lookback_calendar_days=120,
        margin_lookback_calendar_days=120,
        index_daily_lookback_calendar_days=400,
    )

    def _gs():
        return SimpleNamespace(data=data_ns)

    monkeypatch.setattr(snapshot_mod, "get_settings", _gs)

    def _univ(universe_name, as_of):
        return pd.DataFrame({"ticker": ["600519.SH"], "weight": [100.0]})

    monkeypatch.setattr(snapshot_mod, "get_universe_with_weights", _univ)
    return root


def test_include_hk_hold_false_skips_hk_fetch(snap_tmp, monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubTushare()
    monkeypatch.setattr(snapshot_mod, "TushareProvider", lambda *a, **k: stub)
    meta = snapshot_mod.build_snapshot(
        date(2024, 6, 28),
        universe_name="csi300",
        include_financials=False,
        include_indicators=False,
        include_hk_hold=False,
        strict=False,
        overwrite=True,
    )
    assert stub.hk_calls == 0
    assert meta.get("include_flags", {}).get("hk_hold") is False


def test_include_true_margin_fails_strict_false_completes(snap_tmp, monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubTushare()
    monkeypatch.setattr(snapshot_mod, "TushareProvider", lambda *a, **k: stub)
    meta = snapshot_mod.build_snapshot(
        date(2024, 6, 28),
        universe_name="csi300",
        include_financials=False,
        include_indicators=False,
        strict=False,
        include_stock_basic=True,
        include_hk_hold=False,
        include_margin=True,
        include_index_daily=True,
        overwrite=True,
    )
    assert stub.margin_calls >= 1
    assert "margin" not in meta.get("rows_per_table", {})


def test_snapshot_manifest_has_modules_when_rows_written(
    snap_tmp, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _StubTushare()
    monkeypatch.setattr(snapshot_mod, "TushareProvider", lambda *a, **k: stub)
    meta = snapshot_mod.build_snapshot(
        date(2024, 6, 28),
        universe_name="csi300",
        include_financials=False,
        include_indicators=False,
        include_hk_hold=False,
        include_margin=False,
        strict=False,
        overwrite=True,
    )
    assert meta.get("data_expansion_version") == "v1"
    assert "stock_basic" in meta.get("modules", {})
    assert "index_daily" in meta.get("modules", {})


def test_strict_true_margin_empty_raises(snap_tmp, monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubTushare()
    monkeypatch.setattr(snapshot_mod, "TushareProvider", lambda *a, **k: stub)
    with pytest.raises(RuntimeError, match="margin"):
        snapshot_mod.build_snapshot(
            date(2024, 6, 28),
            universe_name="csi300",
            include_financials=False,
            include_indicators=False,
            include_hk_hold=False,
            include_margin=True,
            include_index_daily=False,
            strict=True,
            overwrite=True,
        )
