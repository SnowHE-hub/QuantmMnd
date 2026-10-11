"""T-020 TurnoverControlledNavEngine：不设控制时与父类逐笔相同；缓冲带 / 容忍带按定义生效（合成数据）。"""
from __future__ import annotations

import pandas as pd
import pytest

from quantmind.execution.nav_engine import ExecutableNavEngine, NavConfig
from quantmind.execution.nav_turnover import TurnoverControlledNavEngine

DATES = pd.bdate_range("2024-01-01", periods=120)


def synth_prices(n: int = 8, drift_step: float = 0.0005) -> pd.DataFrame:
    rows = []
    for i in range(n):
        t, base = f"T{i:03d}.SZ", 10.0 + i
        prev = base
        for k, d in enumerate(DATES):
            # 每只票不同漂移，制造组合内权重漂移
            close = base * (1 + drift_step * (i - n / 2)) ** k
            rows.append({"ts_code": t, "trade_date": d, "open": close, "high": close * 1.015,
                         "low": close * 0.985, "close": close, "pre_close": prev,
                         "adj_factor": 1.0, "amount": 100.0 - i})
            prev = close
    return pd.DataFrame(rows)


def rotating_preds(as_of_list, n: int = 8) -> pd.DataFrame:
    """每个截面把每只票的名次后移一位（末位回到第 1）：T(i) 在第 k 期的名次 = (i + k) mod n。"""
    rows = []
    for k, a in enumerate(as_of_list):
        for i in range(n):
            rank = (i + k) % n
            rows.append({"as_of": a, "ticker": f"T{i:03d}.SZ", "score": float(n - rank)})
    return pd.DataFrame(rows)


def cfg(**kw) -> NavConfig:
    d = {"horizon_td": 12, "rebalance_step": 1, "top_n": 3, "universe_size": 8, "min_recent_bar_td": 5}
    d.update(kw)
    return NavConfig(**d)


AS_OF = list(DATES[30:110:5])


def test_no_control_identical_to_parent():
    px, preds = synth_prices(), rotating_preds(AS_OF)
    a = ExecutableNavEngine(preds, px, cfg()).run()
    b = TurnoverControlledNavEngine(preds, px, cfg()).run()
    pd.testing.assert_frame_equal(a["nav_daily"], b["nav_daily"])
    pd.testing.assert_frame_equal(a["trades"], b["trades"])
    assert a["summary"] == b["summary"]


def test_buffer_keeps_held_names_within_exit_rank():
    px, preds = synth_prices(), rotating_preds(AS_OF)
    base = ExecutableNavEngine(preds, px, cfg()).run()
    buf = TurnoverControlledNavEngine(preds, px, cfg(), exit_rank=6).run()
    tb, tbase = buf["targets"], base["targets"]
    # 初始持有 T0–T2；之后每期名次后移一位。前 4 期它们的名次 ≤ 5（在前 6 内）→ 缓冲带下不换票；父类每期都换
    first = set(tb["targets"].iloc[0])
    assert all(set(x) == first for x in tb["targets"].iloc[:4])
    assert set(tbase["targets"].iloc[1]) != set(tbase["targets"].iloc[0])
    assert max(r["n_kept_outside_top_n"] for r in buf_engine_log(preds, px)) > 0
    assert buf["summary"]["turnover"]["annualized_oneway"] < base["summary"]["turnover"]["annualized_oneway"]
    for x in tb["targets"]:
        assert len(x) == 3


def buf_engine_log(preds, px):
    e = TurnoverControlledNavEngine(preds, px, cfg(), exit_rank=6)
    e.run()
    return e.kept_by_buffer


def test_buffer_exit_rank_equal_top_n_is_no_buffer():
    px, preds = synth_prices(), rotating_preds(AS_OF)
    a = ExecutableNavEngine(preds, px, cfg()).run()
    b = TurnoverControlledNavEngine(preds, px, cfg(), exit_rank=3).run()
    pd.testing.assert_frame_equal(a["trades"], b["trades"])


def test_drift_band_skips_small_reweights():
    px = synth_prices(drift_step=0.002)
    const = pd.DataFrame([{"as_of": a, "ticker": f"T{i:03d}.SZ", "score": float(8 - i)}
                          for a in AS_OF for i in range(8)])            # 目标恒定，换手只来自调回等权
    base = ExecutableNavEngine(const, px, cfg()).run()
    band = TurnoverControlledNavEngine(const, px, cfg(), drift_band=10.0).run()   # 极宽：建仓后永不调
    n_base = len(base["trades"])
    n_band = len(band["trades"])
    assert n_base > 3 and n_band == 3          # 只有首次建仓 3 笔
    assert band["summary"]["costs"]["total_cost_nav_units"] < base["summary"]["costs"]["total_cost_nav_units"]


def test_invalid_parameters():
    px, preds = synth_prices(), rotating_preds(AS_OF)
    with pytest.raises(ValueError):
        TurnoverControlledNavEngine(preds, px, cfg(), exit_rank=2)
    with pytest.raises(ValueError):
        TurnoverControlledNavEngine(preds, px, cfg(), drift_band=-0.1)
