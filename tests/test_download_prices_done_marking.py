"""T-012：by_date 下载时，交易日返回空表视为失败，不得记入 price_date_done.txt."""
from __future__ import annotations

import pandas as pd

import scripts.data_pipeline.step2_download_prices as dl

DATES = ["20240102", "20240103", "20240104"]


class FakePro:
    """trade_cal 给出 3 个交易日；20240103 的 daily 返回空表（模拟限频 / 代理失败）."""

    def __init__(self, empty_dates=("20240103",)):
        self.empty_dates = set(empty_dates)
        self.daily_calls: list[str] = []

    def trade_cal(self, **_):
        return pd.DataFrame({"cal_date": DATES})

    def daily(self, trade_date):
        self.daily_calls.append(trade_date)
        if trade_date in self.empty_dates:
            return pd.DataFrame()
        return pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": [trade_date], "close": [10.0]})

    def adj_factor(self, **_):
        return pd.DataFrame()


def _run(tmp_path, monkeypatch, pro):
    monkeypatch.setattr(dl, "AUX_DIR", tmp_path)
    monkeypatch.setattr(dl, "OUT_DIR", tmp_path)
    monkeypatch.setattr(dl.time, "sleep", lambda *_: None)
    dl.run_by_date(pro, {"000001.SZ"}, sleep=0.0)
    return set((tmp_path / "price_date_done.txt").read_text().splitlines())


def test_empty_trading_day_is_not_marked_done(tmp_path, monkeypatch):
    done = _run(tmp_path, monkeypatch, FakePro())
    assert done == {"20240102", "20240104"}


def test_failed_day_is_retried_on_next_run(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, FakePro())
    pro = FakePro(empty_dates=())
    done = _run(tmp_path, monkeypatch, pro)
    assert pro.daily_calls == ["20240103"]
    assert done == set(DATES)


def test_downloaded_rows_written(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, FakePro())
    panel = pd.read_parquet(tmp_path / "alpha_prices_panel.parquet")
    assert sorted(panel["trade_date"].dt.strftime("%Y%m%d")) == ["20240102", "20240104"]
