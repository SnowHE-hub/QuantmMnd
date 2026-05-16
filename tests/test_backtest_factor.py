"""scripts/run_backtest_factor.py — 截面分层回测单元测试."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_backtest_script():
    path = ROOT / "scripts" / "run_backtest_factor.py"
    name = "quantmind_scripts_run_backtest_factor"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec.loader.exec_module(mod)
    return mod


def _synthetic_panel(
    *,
    dates: list[pd.Timestamp],
    n_stocks: int,
    label_col: str,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.Series]:
    """fwd 截面 iid；score ≈ fwd 使 Q5 层系统性优于 Q1。"""
    idx = pd.MultiIndex.from_product(
        [dates, [f"T{i:03d}" for i in range(n_stocks)]],
        names=["as_of", "ticker"],
    )
    n = len(idx)
    fwd = rng.normal(0.01, 0.06, size=n).astype(np.float64)
    df = pd.DataFrame({label_col: fwd}, index=idx)
    scores = pd.Series(fwd + rng.normal(0, 0.001, size=n), index=idx, name="score")
    return df, scores


@pytest.fixture(scope="module")
def bt():
    return _load_backtest_script()


def test_quintile_layers_count_and_spread(bt) -> None:
    rng = np.random.default_rng(42)
    label = "forward_return_21d"
    dates = pd.to_datetime(["2023-03-31", "2023-06-30", "2023-09-29", "2023-12-29"])
    panel, scores = _synthetic_panel(dates=dates, n_stocks=40, label_col=label, rng=rng)
    n_bins = 5
    tbl = bt.per_period_quintile_returns(panel, scores, label, n_bins)
    assert not tbl.empty
    assert set(tbl.columns) >= {"as_of", "quantile", "mean_return"}
    assert len(bt.layer_summary(tbl)) == n_bins

    summ = bt.layer_summary(tbl).set_index("quantile")
    assert summ.loc["Q5", "mean_monthly"] >= summ.loc["Q1", "mean_monthly"]

    ls = bt.long_short_returns(tbl, n_bins)
    sh = bt.sharpe_monthly(ls.values.astype(float), rf_annual=0.03)
    assert sh == sh
    assert np.isfinite(sh)

    q5 = bt.q5_returns_series(tbl, n_bins).sort_index()
    nav = bt.nav_from_returns_indexed(q5.dropna())
    mdd = bt.DRAWDOWN_TOOL.compute_max_drawdown(nav)
    assert -1.0 <= mdd <= 0.0

def test_output_csv_schema_exists(tmp_path: Path, bt) -> None:
    rng = np.random.default_rng(7)
    label = "forward_return_21d"
    dates = [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-04-30")]
    panel, scores = _synthetic_panel(dates=dates, n_stocks=30, label_col=label, rng=rng)
    tbl = bt.per_period_quintile_returns(panel, scores, label, 5)
    p = tmp_path / "q.csv"
    tbl.to_csv(p, index=False)
    got = pd.read_csv(p)
    assert {"as_of", "quantile", "mean_return"}.issubset(got.columns)
