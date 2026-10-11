"""T-014：模拟盘的持有期收益用复权价."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import paper_trading_sim as pts  # noqa: E402

DAYS = pd.bdate_range("2024-01-02", periods=40)
SPLIT = DAYS[20]


def _prices(tmp_path: Path, cols: list[str]) -> Path:
    raw = np.where(DAYS >= SPLIT, 5.0, 10.0)        # 10 送 10：原始价减半，经济上没有涨跌
    af = np.where(DAYS >= SPLIT, 2.0, 1.0)
    df = pd.DataFrame({"ts_code": "AAA", "trade_date": DAYS, "close": raw, "adj_factor": af,
                       "adj_close": raw * af})
    p = tmp_path / "p.parquet"
    df[["ts_code", "trade_date", *cols]].to_parquet(p)
    return p


def test_split_is_not_a_loss(tmp_path):
    wide = pts.load_price_pivot(_prices(tmp_path, ["close", "adj_factor"]))
    assert pts.calc_return(wide, "AAA", DAYS[5], 21) == pytest.approx(0.0)


def test_raw_close_only_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="复权"):
        pts.load_price_pivot(_prices(tmp_path, ["close"]))
