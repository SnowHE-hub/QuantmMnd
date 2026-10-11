"""T-017：v6 面板在完整交易日历上重建——日历校验、标签 horizon、输出不覆盖、lake 窗口用同一日历."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import quantmind.data.lake as lake
import scripts.survivorship.p2_build_v6 as p2

DAYS = pd.bdate_range("2024-01-02", periods=40)


@pytest.fixture()
def restore_globals():
    names = ["DATA_ROOT", "V5_PRICES", "V5_PANEL", "V6_PRICES", "V6_PANEL", "SB_FULL", "SH_OUT", "PANEL_IN",
             "CALENDAR", "REFUSE_OVERWRITE"]
    saved = {n: getattr(p2, n) for n in names}
    lake_saved = {n: getattr(lake, n) for n in ("DATA_LAKE_DIR", "CHECKPOINT_DIR", "PRICE_PANEL_PATH",
                                                "load_trading_calendar")}
    yield
    for n, v in saved.items():
        setattr(p2, n, v)
    for n, v in lake_saved.items():
        setattr(lake, n, v)


def _cal_file(tmp_path, days=DAYS):
    p = tmp_path / "trade_cal.parquet"
    pd.DataFrame({"cal_date": days}).to_parquet(p)
    return p


def test_calendar_must_match_price_dates_exactly(tmp_path):
    cal = _cal_file(tmp_path)
    assert list(p2.validated_calendar(cal, DAYS[5:30])) == list(DAYS[5:30])      # 截到价格区间
    with pytest.raises(ValueError, match="trading days without prices"):
        p2.validated_calendar(cal, DAYS.delete(10))                                # 价格表缺日（旧 v6 的问题）
    with pytest.raises(ValueError, match="non-trading days"):
        p2.validated_calendar(cal, DAYS.append(pd.DatetimeIndex(["2024-01-06"])))   # 周六


def test_missing_price_day_shifts_the_horizon():
    """旧 v6：价格表缺 DAYS[8]，horizon 按价格表日期计数 → 目标日后移一天；完整日历上目标日正确."""
    full = pd.DataFrame({"A": 10.0 * 1.01 ** np.arange(40)}, index=DAYS)
    gap = full.drop(DAYS[8])
    args = (pd.Series({"A": DAYS[-1]}), pd.Series({"A": False}))
    good = p2.compute_forward_returns_delist_aware(full, full.ffill(), DAYS[5], (12,), *args)
    bad = p2.compute_forward_returns_delist_aware(gap, gap.ffill(), DAYS[5], (12,), *args)
    assert good.loc["A", "forward_return_12d"] == pytest.approx(1.01 ** 12 - 1)
    assert bad.loc["A", "forward_return_12d"] == pytest.approx(1.01 ** 13 - 1)


def test_new_outputs_never_overwrite(tmp_path, restore_globals):
    prices = tmp_path / "data/raw/px.parquet"
    prices.parent.mkdir(parents=True)
    pd.DataFrame({"trade_date": DAYS}).to_parquet(prices)
    out = tmp_path / "data/panel/new.parquet"
    out.parent.mkdir(parents=True)
    out.write_bytes(b"keep")
    p2.configure(data_root=tmp_path, prices_file=prices, panel_out=out)
    assert out == p2.V6_PANEL and p2.REFUSE_OVERWRITE
    with pytest.raises(RuntimeError, match="refuse to overwrite existing"):
        p2._write(pd.DataFrame({"x": [1]}), out)
    assert out.read_bytes() == b"keep"
    with pytest.raises(RuntimeError, match="v5"):
        p2._write(pd.DataFrame({"x": [1]}), p2.V5_PANEL)
    fresh = tmp_path / "data/panel/fresh.parquet"
    p2._write(pd.DataFrame({"x": [1]}), fresh)
    assert fresh.exists()


def test_default_configuration_keeps_old_behaviour(restore_globals):
    before = (p2.V6_PRICES, p2.V6_PANEL, p2.SB_FULL, p2.REFUSE_OVERWRITE, p2.CALENDAR)
    p2.configure()
    assert before == (p2.V6_PRICES, p2.V6_PANEL, p2.SB_FULL, p2.REFUSE_OVERWRITE, p2.CALENDAR)
    assert p2.trading_calendar is not None and p2.CALENDAR is None


def test_calendar_file_also_drives_lake_windows(tmp_path, restore_globals):
    """lake 的回看窗口（read_lake_window）必须用同一份完整日历，而不是由旧价格表反推的日历."""
    lake_dir = tmp_path / "data/lake"
    lake_dir.mkdir(parents=True)
    pd.DataFrame({"trade_date": DAYS, "x": np.arange(40)}).to_parquet(lake_dir / "north_bound.parquet")
    prices = tmp_path / "px.parquet"
    pd.DataFrame({"trade_date": DAYS}).to_parquet(prices)
    p2.configure(data_root=tmp_path, prices_file=prices, calendar_file=_cal_file(tmp_path))
    assert list(p2.trading_calendar()) == list(DAYS)
    w = lake.read_lake_window("north_bound", DAYS[20], 5)
    assert list(w["trade_date"]) == list(DAYS[16:21])


def test_fundamental_factor_builder_paths(tmp_path):
    import scripts.survivorship.p63_2_build_factors as p63

    saved = (p63.LAKE, p63.V6_PANEL, p63.OUT, p63.REFUSE_OVERWRITE)
    try:
        p63.configure()
        assert saved == (p63.LAKE, p63.V6_PANEL, p63.OUT, p63.REFUSE_OVERWRITE)   # 不传参数 = 原行为
        p63.configure(tmp_path, panel=tmp_path / "p.parquet", out=tmp_path / "f.parquet")
        assert tmp_path.resolve() / "data/lake" == p63.LAKE and tmp_path / "f.parquet" == p63.OUT
        assert p63.REFUSE_OVERWRITE
    finally:
        p63.LAKE, p63.V6_PANEL, p63.OUT, p63.REFUSE_OVERWRITE = saved
