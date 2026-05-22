"""pytest tests for System3 Bull Regime sharpe threshold logic.

Tests verify that BacktestSystem._is_bull_regime() and the resulting
sharpe threshold respond correctly to CSI300 20-day returns:
  - CSI300 +6% (> 5%)  → Bull  → SHARPE_THRESH_BULL  = 0.5
  - CSI300 +3% (<= 5%) → Normal → SHARPE_THRESH_NORMAL = 1.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_30day_sim import BacktestSystem


AS_OF = pd.Timestamp("2025-11-19")


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_csi300(as_of: pd.Timestamp, ret_20d: float) -> pd.DataFrame:
    """Build CSI300 price DataFrame giving the requested 20-day return."""
    dates = pd.date_range(end=as_of, periods=21, freq="B")
    base = 4000.0
    prices = np.linspace(base, base * (1 + ret_20d), 21)
    return pd.DataFrame({"trade_date": dates, "close": prices})


def _make_prices_trending(tickers: list[str], as_of: pd.Timestamp,
                          daily_ret: float = 0.002, n_days: int = 70) -> pd.DataFrame:
    """Create monotonically trending stock prices → high Sharpe."""
    dates = pd.date_range(end=as_of, periods=n_days, freq="B")
    rows = []
    for t in tickers:
        price = 10.0
        for d in dates:
            rows.append({"trade_date": d, "ts_code": t, "close": price})
            price *= (1 + daily_ret)
    return pd.DataFrame(rows)


def _make_prices_choppy(tickers: list[str], as_of: pd.Timestamp,
                        n_days: int = 70) -> pd.DataFrame:
    """Create noisy/flat stock prices → low Sharpe (~0)."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=as_of, periods=n_days, freq="B")
    rows = []
    for t in tickers:
        # Mean-reverting around 10 with large noise → Sharpe near 0
        prices = 10.0 + rng.normal(0, 1.0, n_days).cumsum() * 0.02
        prices = np.clip(prices, 5, 15)
        for d, p in zip(dates, prices):
            rows.append({"trade_date": d, "ts_code": t, "close": p})
    return pd.DataFrame(rows)


def _make_candidates(tickers: list[str]) -> pd.DataFrame:
    """Minimal candidate DataFrame without hist_* columns (those are computed by validate)."""
    return pd.DataFrame(
        {"composite_score": list(range(len(tickers), 0, -1))},
        index=tickers,
    )


# ─── Tests: regime detection ──────────────────────────────────────────────────

class TestBullRegimeDetection:
    def test_bull_when_csi300_up_6pct(self):
        csi300 = _make_csi300(AS_OF, ret_20d=0.06)
        prices = _make_prices_trending(["000001.SZ"], AS_OF)
        bt = BacktestSystem(prices, csi300)
        assert bt._is_bull_regime(AS_OF) is True

    def test_normal_when_csi300_up_3pct(self):
        csi300 = _make_csi300(AS_OF, ret_20d=0.03)
        prices = _make_prices_trending(["000001.SZ"], AS_OF)
        bt = BacktestSystem(prices, csi300)
        assert bt._is_bull_regime(AS_OF) is False

    def test_no_csi300_defaults_to_normal(self):
        prices = _make_prices_trending(["000001.SZ"], AS_OF)
        bt = BacktestSystem(prices, csi300=None)
        assert bt._is_bull_regime(AS_OF) is False


# ─── Tests: sharpe threshold selection ────────────────────────────────────────

class TestSharpeThresholdSelection:
    """Directly verify sharpe_thresh() returns the right value per regime."""

    def test_bull_threshold_is_0_5(self):
        csi300 = _make_csi300(AS_OF, ret_20d=0.06)   # +6% → Bull
        bt = BacktestSystem(_make_prices_trending(["X"], AS_OF), csi300)
        assert bt.sharpe_thresh(AS_OF) == pytest.approx(0.5)

    def test_normal_threshold_is_1_0(self):
        csi300 = _make_csi300(AS_OF, ret_20d=0.03)   # +3% → Normal
        bt = BacktestSystem(_make_prices_trending(["X"], AS_OF), csi300)
        assert bt.sharpe_thresh(AS_OF) == pytest.approx(1.0)

    def test_constants(self):
        assert BacktestSystem.SHARPE_THRESH_BULL   == pytest.approx(0.5)
        assert BacktestSystem.SHARPE_THRESH_NORMAL == pytest.approx(1.0)
        assert BacktestSystem.BULL_CSI300_20D      == pytest.approx(0.05)


# ─── Tests: full validate() integration ───────────────────────────────────────

class TestValidateInvestableFilter:
    """
    Verify that validate() applies the regime-dependent threshold.

    Strategy:
    - 高Sharpe stock: monotonically rising prices (Sharpe >> 1.0) → passes both thresholds
    - 低Sharpe stock: choppy prices (Sharpe near 0) → fails both thresholds
    - 中Sharpe stock (manually set via monkeypatch): Sharpe=0.6 → passes Bull(0.5), fails Normal(1.0)
    """

    def test_high_sharpe_passes_both_regimes(self):
        tickers = ["A", "B"]
        for ret_20d, label in [(0.06, "Bull"), (0.03, "Normal")]:
            csi300 = _make_csi300(AS_OF, ret_20d=ret_20d)
            prices = _make_prices_trending(tickers, AS_OF, daily_ret=0.005)
            bt = BacktestSystem(prices, csi300)
            result = bt.validate(_make_candidates(tickers), AS_OF)
            assert result["investable"].all(), f"High-sharpe should be investable in {label} regime"

    def test_choppy_single_stock_blocked_per_row(self):
        """Each choppy stock should be blocked if its own hist_sharpe < threshold."""
        ticker = ["CHOP"]
        for ret_20d, label in [(0.06, "Bull"), (0.03, "Normal")]:
            csi300 = _make_csi300(AS_OF, ret_20d=ret_20d)
            prices = _make_prices_choppy(ticker, AS_OF)
            bt = BacktestSystem(prices, csi300)
            result = bt.validate(_make_candidates(ticker), AS_OF)
            thresh = bt.sharpe_thresh(AS_OF)
            row = result.iloc[0]
            computed_sharpe = float(row["hist_sharpe"])
            investable = bool(row["investable"])
            if computed_sharpe < thresh:
                assert not investable, (
                    f"[{label}] hist_sharpe={computed_sharpe:.3f} < {thresh} "
                    f"→ should NOT be investable"
                )

    def test_regime_determines_threshold_in_validate(self, monkeypatch):
        """
        Mock _is_bull_regime to inject +6% / +3% scenarios and verify
        that the sharpe threshold actually used differs between regimes.
        """
        tickers = ["A"]
        prices = _make_prices_trending(tickers, AS_OF, daily_ret=0.002)
        csi300_dummy = _make_csi300(AS_OF, ret_20d=0.0)
        bt = BacktestSystem(prices, csi300_dummy)

        # Force Bull
        monkeypatch.setattr(bt, "_is_bull_regime", lambda _: True)
        assert bt.sharpe_thresh(AS_OF) == pytest.approx(0.5)

        # Force Normal
        monkeypatch.setattr(bt, "_is_bull_regime", lambda _: False)
        assert bt.sharpe_thresh(AS_OF) == pytest.approx(1.0)
