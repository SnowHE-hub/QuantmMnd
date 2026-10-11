"""T-013：已实现收益用复权价；meta-learner 训练样本按 (join_date, ticker) 去重."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import track_realized_pnl as trp  # noqa: E402
import train_meta_learner as tml  # noqa: E402

DAYS = pd.bdate_range("2024-01-02", periods=80)
SPLIT = DAYS[30]


def _prices(tmp_path: Path, cols: list[str]) -> Path:
    """AAA 在 SPLIT 日 10 送 10：原始 close 从 10 跳到 5，adj_factor 从 1 变 2，经济上没有涨跌."""
    raw = np.where(DAYS >= SPLIT, 5.0, 10.0)
    af = np.where(DAYS >= SPLIT, 2.0, 1.0)
    df = pd.DataFrame({"ts_code": "AAA", "trade_date": DAYS, "close": raw, "adj_factor": af,
                       "adj_close": raw * af})
    p = tmp_path / "prices.parquet"
    df[["ts_code", "trade_date", *cols]].to_parquet(p)
    return p


def test_split_is_not_a_loss(tmp_path):
    wide = trp.load_price_pivot(_prices(tmp_path, ["close", "adj_factor"]))
    rec = {"as_of": str(DAYS[5].date()), "top10": [{"ticker": "AAA", "lgbm_score": 1.0, "rank": 1}]}
    rows = trp.compute_realized_pnl_for_rec(rec, wide, holding_days=40)
    assert rows[0]["actual_return_63d"] == pytest.approx(0.0)


def test_adj_close_is_used_when_no_adj_factor(tmp_path):
    wide = trp.load_price_pivot(_prices(tmp_path, ["adj_close"]))
    assert wide["AAA"].nunique() == 1


def test_raw_close_only_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="复权"):
        trp.load_price_pivot(_prices(tmp_path, ["close"]))


def test_meta_learner_dedups_same_recommendation(tmp_path, monkeypatch):
    """2025-06-28 与 2025-06-30 两批相同推荐：规范化后每个持仓只保留一行."""
    tickers = ["A", "B", "C"]
    pnl = pd.DataFrame({
        "as_of_date": ["2025-06-28"] * 3 + ["2025-06-30"] * 3 + ["2025-09-30"] * 3,
        "ticker": tickers * 3,
        "actual_return_63d": [0.1, 0.2, 0.3] * 3,
        "hit": [True, False, True] * 3,
    })
    idx = pd.MultiIndex.from_product([pd.to_datetime(["2025-06-30", "2025-09-30"]), tickers],
                                     names=["as_of", "ticker"])
    panel = pd.DataFrame({"pe_ttm": np.arange(6.0)}, index=idx)
    pnl.to_parquet(tmp_path / "pnl.parquet")
    panel.to_parquet(tmp_path / "panel.parquet")

    import quantmind.models.meta_learner as ml
    monkeypatch.setattr(ml, "compute_agent_proxies",
                        lambda xs: pd.DataFrame({c: 0.0 for c in tml.AGENT_COLS}, index=xs.index))
    out = tml.build_training_data(tmp_path / "pnl.parquet", tmp_path / "panel.parquet")
    assert len(out) == 6
    assert not out.duplicated(["join_date", "ticker"]).any()
