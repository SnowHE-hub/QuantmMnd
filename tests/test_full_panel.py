"""tests for full panel build and walk-forward split."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantmind.core.config import PROJECT_ROOT
from quantmind.features.expansion import EXPANSION_FACTORS
from quantmind.features.panel import compute_forward_returns

PANEL_PATH = PROJECT_ROOT / "data" / "features" / "csi300_full_panel.parquet"
SPLIT_META = PROJECT_ROOT / "data" / "panel" / "split_meta.json"
SNAP_ROOT = PROJECT_ROOT / "data" / "snapshots"


@pytest.mark.integration
@pytest.mark.skipif(not PANEL_PATH.exists(), reason="csi300_full_panel.parquet not built")
@pytest.mark.stale_panel_fixture
def test_full_panel_rowcount_vs_snapshots() -> None:
    panel = pd.read_parquet(PANEL_PATH)
    as_of_level = panel.index.get_level_values(0).unique()
    expected = 0
    for ts in as_of_level:
        d = pd.Timestamp(ts).date()
        u_path = SNAP_ROOT / d.isoformat() / "universe.parquet"
        assert u_path.exists(), f"missing universe for {d}"
        u = pd.read_parquet(u_path)
        expected += len(u)
    ratio = abs(len(panel) - expected) / max(expected, 1)
    assert ratio <= 0.05, f"row count {len(panel)} vs expected {expected}, ratio={ratio}"


@pytest.mark.integration
@pytest.mark.skipif(not PANEL_PATH.exists(), reason="panel missing")
@pytest.mark.stale_panel_fixture
def test_expansion_nonempty_after_2021() -> None:
    panel = pd.read_parquet(PANEL_PATH)
    exp_names = [n for n, _ in EXPANSION_FACTORS]
    cut = pd.Timestamp(date(2021, 1, 1))
    sub = panel.loc[panel.index.get_level_values(0) >= cut, exp_names]
    rate = sub.notna().to_numpy().mean()
    assert rate > 0.8, f"expansion non-NaN rate {rate}"


@pytest.mark.integration
@pytest.mark.skipif(not PANEL_PATH.exists(), reason="panel missing")
@pytest.mark.stale_panel_fixture
def test_expansion_all_nan_2019_2020() -> None:
    panel = pd.read_parquet(PANEL_PATH)
    exp_names = [n for n, _ in EXPANSION_FACTORS]
    end = pd.Timestamp(date(2020, 12, 31))
    sub = panel.loc[panel.index.get_level_values(0) <= end, exp_names]
    assert sub.isna().to_numpy().all(), "2019-2020 expansion should be all NA/NaN"


@pytest.mark.skipif(not PANEL_PATH.exists(), reason="panel missing")
def test_labels_nonnull_except_last_two_periods() -> None:
    panel = pd.read_parquet(PANEL_PATH)
    as_ofs = sorted(panel.index.get_level_values(0).unique())
    assert len(as_ofs) >= 3
    early = as_ofs[:-2]
    for lab in ("forward_return_21d", "forward_return_63d"):
        for ts in early:
            col = panel.loc[panel.index.get_level_values(0) == ts, lab]
            assert col.notna().mean() > 0.9, f"{lab} @ {ts} too sparse"


@pytest.mark.integration
@pytest.mark.skipif(not PANEL_PATH.exists(), reason="panel missing")
@pytest.mark.stale_panel_fixture
def test_expansion_many_nonempty_columns_modern() -> None:
    panel = pd.read_parquet(PANEL_PATH)
    cut = pd.Timestamp(date(2021, 1, 1))
    sub = panel.loc[panel.index.get_level_values(0) >= cut]
    exp_names = [n for n, _ in EXPANSION_FACTORS]
    non_full_nan = sum(
        1 for c in exp_names if not sub[c].isna().all()
    )
    assert non_full_nan >= 20, f"only {non_full_nan} expansion cols have values"


def test_forward_returns_no_lookahead() -> None:
    idx = pd.bdate_range("2024-01-01", periods=40)
    raw = np.linspace(100.0, 140.0, len(idx))
    pivot = pd.DataFrame({"AAA.SZ": raw}, index=idx)
    as_of = idx[10].date()
    fr = compute_forward_returns(pivot, as_of, (5,))
    base_close = pivot.loc[idx[10], "AAA.SZ"]
    future_close = pivot.loc[idx[15], "AAA.SZ"]
    exp = future_close / base_close - 1.0
    assert abs(float(fr.loc["AAA.SZ", "forward_return_5d"]) - exp) < 1e-9
    # distort a price strictly before base_idx — should not affect label
    pivot2 = pivot.copy()
    pivot2.loc[idx[5], "AAA.SZ"] = 1.0
    fr2 = compute_forward_returns(pivot2, as_of, (5,))
    assert abs(float(fr2.loc["AAA.SZ", "forward_return_5d"]) - exp) < 1e-9


@pytest.mark.integration
@pytest.mark.skipif(not SPLIT_META.exists(), reason="split not built")
@pytest.mark.stale_panel_fixture
def test_split_no_date_overlap() -> None:
    meta = json.loads(SPLIT_META.read_text(encoding="utf-8"))

    def _parse(dr: dict) -> tuple[date | None, date | None]:
        a, b = dr.get("min"), dr.get("max")
        if a is None or b is None:
            return None, None
        return date.fromisoformat(a), date.fromisoformat(b)

    tr = _parse(meta["train"]["date_range"])
    te = _parse(meta["test"]["date_range"])
    va = _parse(meta["val"]["date_range"])
    assert tr[1] is not None and te[0] is not None
    assert tr[1] < te[0], "train should end before test starts"
    assert te[1] is not None and va[0] is not None
    assert te[1] < va[0], "test should end before val starts"


@pytest.mark.skipif(not SPLIT_META.exists(), reason="split not built")
def test_split_meta_structure() -> None:
    meta = json.loads(SPLIT_META.read_text(encoding="utf-8"))
    for k in ("train", "val", "test", "predict"):
        assert k in meta
        assert "date_range" in meta[k]
        assert "n_rows" in meta[k]
        assert "as_of_dates" in meta[k]
    assert "feature_columns" in meta
    assert isinstance(meta["feature_columns"], list)
    assert len(meta["feature_columns"]) > 40


@pytest.mark.skipif(not PANEL_PATH.exists(), reason="panel missing")
def test_panel_acceptance_shape() -> None:
    panel = pd.read_parquet(PANEL_PATH)
    n_dates = panel.index.get_level_values(0).nunique()
    # 约 24 个季末 × ~300 票；成分股略少于 300 时总行数可能 < 7000
    assert panel.shape[0] >= 6800
    assert panel.shape[1] >= 67
    assert n_dates >= 22
