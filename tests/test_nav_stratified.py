"""T-027 StratifiedNavEngine：不分层时与父类逐笔相同；分层名额与组内取票按定义生效（合成数据）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantmind.execution.nav_engine import ExecutableNavEngine, NavConfig
from quantmind.execution.nav_stratified import StratifiedNavEngine, stratum_labels, stratum_quotas

DATES = pd.bdate_range("2024-01-01", periods=120)
TICKERS = [f"T{i:03d}.SZ" for i in range(8)]
AS_OF = list(DATES[30:110:5])


def synth_prices(n: int = 8) -> pd.DataFrame:
    rows = []
    for i in range(n):
        t, base = TICKERS[i], 10.0 + i
        prev = base
        for k, d in enumerate(DATES):
            close = base * (1 + 0.0005 * (i - n / 2)) ** k
            rows.append({"ts_code": t, "trade_date": d, "open": close, "high": close * 1.015,
                         "low": close * 0.985, "close": close, "pre_close": prev,
                         "adj_factor": 1.0, "amount": 100.0 - i})
            prev = close
    return pd.DataFrame(rows)


def const_preds() -> pd.DataFrame:
    """分数 T000 最高、T007 最低；前 4 名全在行业 A。"""
    return pd.DataFrame([{"as_of": a, "ticker": t, "score": float(8 - i)} for a in AS_OF for i, t in enumerate(TICKERS)])


def exposures() -> pd.DataFrame:
    rows = [{"as_of": a, "ticker": t, "sw_l1": "A" if i < 4 else "B", "size": float(i % 4), "beta": np.nan if i == 7 else float(i)}
            for a in AS_OF for i, t in enumerate(TICKERS)]
    return pd.DataFrame(rows).set_index(["as_of", "ticker"])


def cfg(**kw) -> NavConfig:
    d = {"horizon_td": 12, "rebalance_step": 1, "top_n": 4, "universe_size": 8, "min_recent_bar_td": 5}
    d.update(kw)
    return NavConfig(**d)


def test_no_dims_identical_to_parent():
    px, preds = synth_prices(), const_preds()
    a = ExecutableNavEngine(preds, px, cfg()).run()
    b = StratifiedNavEngine(preds, px, cfg(), exposures=exposures(), dims=()).run()
    pd.testing.assert_frame_equal(a["nav_daily"], b["nav_daily"])
    pd.testing.assert_frame_equal(a["trades"], b["trades"])
    assert a["summary"] == b["summary"]


def test_industry_strata_take_top_of_each_group():
    px, preds = synth_prices(), const_preds()
    plain = ExecutableNavEngine(preds, px, cfg()).run()["targets"]["targets"].iloc[0]
    eng = StratifiedNavEngine(preds, px, cfg(), exposures=exposures(), dims=("sw_l1",))
    strat = eng.run()["targets"]["targets"].iloc[0]
    assert set(plain) == {"T000.SZ", "T001.SZ", "T002.SZ", "T003.SZ"}
    assert strat == ["T000.SZ", "T001.SZ", "T004.SZ", "T005.SZ"]      # 每个行业 2 个名额，组内按分数，输出保持全局名次顺序
    assert eng.strata_log[0]["n_swapped_vs_plain_top"] == 2


def test_quotas_sum_and_largest_remainder():
    lab = pd.Series(["a"] * 5 + ["b"] * 3 + ["c"] * 2)
    q = stratum_quotas(lab, 4)          # 原始 2.0 / 1.2 / 0.8 → 2 / 1 / 1
    assert q.to_dict() == {"a": 2, "b": 1, "c": 1}
    assert int(stratum_quotas(lab, 7).sum()) == 7
    tie = pd.Series(["x", "y"])         # 余数相同 → 按组名
    assert stratum_quotas(tie, 1).to_dict() == {"x": 1, "y": 0}


def test_numeric_quantiles_and_missing_group():
    x = pd.DataFrame({"beta": [5.0, np.nan, 1.0, 3.0, 2.0, 4.0]}, index=list("abcdef"))
    lab = stratum_labels(x, ("beta",), 2)
    assert lab["b"] == "NA"
    assert set(lab[["c", "e"]]) == {"q0"} and set(lab[["a", "f"]]) == {"q1"}
    assert (lab != "NA").sum() == 5
    combo = stratum_labels(pd.DataFrame({"ind": ["A", None], "beta": [1.0, 2.0]}, index=["u", "v"]), ("ind", "beta"), 2)
    assert combo.tolist() == ["A|q0", "NA|q1"]


def test_invalid_parameters():
    px, preds = synth_prices(), const_preds()
    with pytest.raises(ValueError):
        StratifiedNavEngine(preds, px, cfg(), exposures=None, dims=("sw_l1",))
    with pytest.raises(ValueError):
        StratifiedNavEngine(preds, px, cfg(), exposures=exposures(), dims=("nope",))
    with pytest.raises(ValueError):
        StratifiedNavEngine(preds, px, cfg(), exposures=exposures(), dims=("size",), n_quantiles=1)
