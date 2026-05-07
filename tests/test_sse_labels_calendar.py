"""SSE 月线 mock + forward-return 标签（离线逻辑）."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quantmind.data.sse_calendar import monthly_last_trade_days
from quantmind.features.labels import attach_forward_returns, forward_return_n_bars


def test_forward_return_n_bars_simple_path() -> None:
    px = pd.DataFrame(
        {
            "ticker": ["A"] * 4,
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
            "close": [10.0, 10.5, 11.0, 12.0],
        }
    )
    r = forward_return_n_bars(px, ticker="A", anchor_trade_date=date(2024, 1, 3), n_bars=2)
    assert r == pytest.approx(12.0 / 10.5 - 1.0)


def test_forward_return_n_bars_insufficient_future() -> None:
    px = pd.DataFrame(
        {
            "ticker": ["B"] * 2,
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "close": [1.0, 2.0],
        }
    )
    r = forward_return_n_bars(px, ticker="B", anchor_trade_date=date(2024, 1, 2), n_bars=5)
    assert pd.isna(r)


def test_monthly_last_trade_days_monkeypatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_trade_days(s: date, e: date) -> list[date]:  # noqa: ARG001
        return [date(2024, 1, 15), date(2024, 1, 31), date(2024, 2, 29)]

    monkeypatch.setattr(
        "quantmind.data.sse_calendar.list_sse_trade_dates",
        fake_trade_days,
    )

    got = monthly_last_trade_days(date(2024, 1, 1), date(2024, 2, 29))
    assert got == [date(2024, 1, 31), date(2024, 2, 29)]


def test_attach_forward_returns_shapes() -> None:
    feats = pd.DataFrame(
        {
            "fx": [1.0, 2.0],
        },
        index=pd.MultiIndex.from_tuples(
            [(date(2024, 1, 2), "A"), (date(2024, 1, 2), "B")],
            names=["as_of", "ticker"],
        ),
    )
    px = pd.DataFrame(
        {
            "ticker": ["A", "A", "B", "B"],
            "trade_date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"]
            ),
            "close": [10.0, 11.0, 50.0, 55.0],
        }
    )
    out = attach_forward_returns(feats, px, horizons=[1])
    assert "fwd_ret_1" in out.columns
    assert out.loc[(date(2024, 1, 2), "A"), "fwd_ret_1"] == pytest.approx(0.1)
    assert out.loc[(date(2024, 1, 2), "B"), "fwd_ret_1"] == pytest.approx(0.1)
