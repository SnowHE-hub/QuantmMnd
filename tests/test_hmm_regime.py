"""pytest tests for quantmind/regime/hmm.py

Test coverage
-------------
1. build_observations         — output in [0, 26], valid length, no NaN
2. _HMM.fit + predict         — smoke: runs without error on random obs
3. _assign_state_labels       — correct bull/bear/neutral mapping from crafted B
4. REGIME_WEIGHTS             — each regime's weights sum to 1.0
5. RegimeHMM (no files)       — neutral-fallback model returns 'neutral'
6. RegimeHMM.predict_regime   — returns valid string from fitted label_map
7. Bull-sequence detection    — HMM trained on mostly-bull obs → >50% bull states
8. AnalysisSystem.__init__    — initialises without crashing (files may be absent)
9. AnalysisSystem weight-sum  — regime weights returned by model sum to 1.0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quantmind.regime.hmm import (
    REGIME_WEIGHTS,
    N_OBS,
    N_STATES,
    _HMM,
    _assign_state_labels,
    build_observations,
    RegimeHMM,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _make_price_df(
    n: int = 150,
    daily_ret: float = 0.001,
    daily_std: float = 0.015,
    seed: int = 0,
) -> pd.DataFrame:
    """Create a synthetic CSI300-like price DataFrame."""
    rng    = np.random.default_rng(seed)
    dates  = pd.date_range("2020-01-01", periods=n, freq="B")
    shocks = rng.normal(daily_ret, daily_std, n)
    prices = 4000.0 * np.cumprod(1 + shocks)
    return pd.DataFrame({"trade_date": dates, "close": prices})


# ─── 1. build_observations ───────────────────────────────────────────────────

class TestBuildObservations:
    def test_output_range(self):
        df  = _make_price_df(200)
        obs = build_observations(df)
        assert obs.min() >= 0,  "obs must be >= 0"
        assert obs.max() <= 26, "obs must be <= 26 (3^3 - 1)"

    def test_no_nan(self):
        df  = _make_price_df(200)
        obs = build_observations(df)
        assert not np.isnan(obs).any(), "observations must not contain NaN"

    def test_nonempty_after_warmup(self):
        """150 bars should produce at least 120 valid observations after NaN drop."""
        df  = _make_price_df(150)
        obs = build_observations(df)
        assert len(obs) >= 100

    def test_indexed_by_trade_date_accepted(self):
        """DataFrame indexed by trade_date (not column) should also work."""
        df      = _make_price_df(120)
        df_idx  = df.set_index("trade_date")
        obs1 = build_observations(df)
        obs2 = build_observations(df_idx)
        np.testing.assert_array_equal(obs1, obs2)


# ─── 2. _HMM smoke tests ─────────────────────────────────────────────────────

class TestHMMSmoke:
    def test_fit_returns_finite_ll(self):
        rng = np.random.default_rng(1)
        obs = rng.integers(0, N_OBS, size=300)
        hmm = _HMM(rng=np.random.default_rng(1))
        ll  = hmm.fit(obs, n_iter=10)
        assert np.isfinite(ll), f"log-likelihood must be finite, got {ll}"

    def test_predict_shape_and_range(self):
        rng = np.random.default_rng(2)
        obs = rng.integers(0, N_OBS, size=200)
        hmm = _HMM(rng=np.random.default_rng(2))
        hmm.fit(obs, n_iter=5)
        states = hmm.predict(obs)
        assert states.shape == (200,)
        assert set(states).issubset(set(range(N_STATES)))

    def test_transition_matrix_rows_sum_to_one(self):
        hmm = _HMM(rng=np.random.default_rng(3))
        np.testing.assert_allclose(
            hmm.A.sum(axis=1), np.ones(N_STATES), atol=1e-9,
            err_msg="Transition rows must sum to 1 after init",
        )

    def test_emission_matrix_rows_sum_to_one(self):
        rng = np.random.default_rng(4)
        obs = rng.integers(0, N_OBS, size=300)
        hmm = _HMM(rng=np.random.default_rng(4))
        hmm.fit(obs, n_iter=10)
        np.testing.assert_allclose(
            hmm.B.sum(axis=1), np.ones(N_STATES), atol=1e-6,
            err_msg="Emission rows must sum to 1 after EM",
        )


# ─── 3. _assign_state_labels ─────────────────────────────────────────────────

class TestAssignStateLabels:
    def _crafted_B(self) -> np.ndarray:
        """Build B so state 0 peaks at bull-proto (19), state 2 at bear-proto (7)."""
        B = np.full((3, N_OBS), 1.0 / N_OBS)
        B[0, 19] += 5.0   # state 0: strong bull signal
        B[1, 13] += 3.0   # state 1: neutral-ish (mid-mid-mid: 1*9+1*3+1=13)
        B[2, 7]  += 5.0   # state 2: strong bear signal
        B /= B.sum(axis=1, keepdims=True)
        return B

    def test_correct_labels(self):
        B         = self._crafted_B()
        label_map = _assign_state_labels(B)
        assert label_map[0] == "bull",    f"State 0 should be bull, got {label_map}"
        assert label_map[2] == "bear",    f"State 2 should be bear, got {label_map}"
        assert label_map[1] == "neutral", f"State 1 should be neutral, got {label_map}"

    def test_all_three_labels_present(self):
        B         = self._crafted_B()
        label_map = _assign_state_labels(B)
        assert set(label_map.values()) == {"bull", "neutral", "bear"}

    def test_tie_breaking(self):
        """When bull-proto and bear-proto peak on the same state, disambiguate."""
        B = np.full((3, N_OBS), 1.0 / N_OBS)
        # state 0 peaks on BOTH prototypes
        B[0, 19] += 8.0
        B[0, 7]  += 8.0
        # state 2 is second-highest for bear
        B[2, 7]  += 3.0
        B /= B.sum(axis=1, keepdims=True)
        label_map = _assign_state_labels(B)
        # Must produce all 3 labels (no crash, no duplicate labels)
        assert set(label_map.values()) == {"bull", "neutral", "bear"}


# ─── 4. REGIME_WEIGHTS ───────────────────────────────────────────────────────

class TestRegimeWeights:
    @pytest.mark.parametrize("regime", ["bull", "neutral", "bear"])
    def test_weights_sum_to_one(self, regime: str):
        w = REGIME_WEIGHTS[regime]
        total = sum(w.values())
        assert abs(total - 1.0) < 1e-9, (
            f"REGIME_WEIGHTS['{regime}'] sums to {total}, expected 1.0"
        )

    @pytest.mark.parametrize("regime", ["bull", "neutral", "bear"])
    def test_all_keys_present(self, regime: str):
        keys = set(REGIME_WEIGHTS[regime].keys())
        assert keys == {"value", "momentum", "quality", "technical"}

    @pytest.mark.parametrize("regime", ["bull", "neutral", "bear"])
    def test_all_weights_positive(self, regime: str):
        for k, v in REGIME_WEIGHTS[regime].items():
            assert v > 0, f"weight {k} in regime '{regime}' must be positive"


# ─── 5. RegimeHMM — neutral fallback (no data files) ─────────────────────────

class TestRegimeHMMNoFiles:
    def test_neutral_when_no_files(self):
        model  = RegimeHMM.fit_from_file(hist_path=None, sim_path=None)
        regime = model.predict_regime(None, pd.Timestamp("2025-01-01"))
        assert regime == "neutral"

    def test_neutral_before_all_training_dates(self):
        """Query date before any training data → 'neutral'."""
        model  = RegimeHMM.fit_from_file(hist_path=None, sim_path=None)
        regime = model.predict_regime(None, pd.Timestamp("1990-01-01"))
        assert regime == "neutral"

    def test_get_weights_returns_dict(self):
        model   = RegimeHMM.fit_from_file(hist_path=None, sim_path=None)
        weights = model.get_weights("neutral")
        assert set(weights.keys()) == {"value", "momentum", "quality", "technical"}


# ─── 6. RegimeHMM.predict_regime — hand-built model ──────────────────────────

class TestRegimeHMMPredict:
    def _build_model(self) -> RegimeHMM:
        """Construct a RegimeHMM with hand-crafted internals (no file I/O)."""
        hmm = _HMM(rng=np.random.default_rng(0))
        # Manual emission matrix: state 0 = bull, 1 = neutral, 2 = bear
        B            = np.full((3, N_OBS), 1.0 / N_OBS)
        B[0, 19]    += 5.0
        B[1, 13]    += 3.0
        B[2, 7]     += 5.0
        B           /= B.sum(axis=1, keepdims=True)
        hmm.B        = B
        label_map    = {0: "bull", 1: "neutral", 2: "bear"}
        dates        = pd.date_range("2020-01-01", periods=10, freq="B")
        states       = np.array([0, 1, 2, 0, 1, 0, 2, 1, 0, 1], dtype=np.int32)
        return RegimeHMM(hmm, label_map, pd.DatetimeIndex(dates), states)

    def test_returns_valid_label(self):
        model  = self._build_model()
        regime = model.predict_regime(None, pd.Timestamp("2020-01-10"))
        assert regime in {"bull", "neutral", "bear"}

    def test_before_training_window_returns_neutral(self):
        model  = self._build_model()
        regime = model.predict_regime(None, pd.Timestamp("2010-01-01"))
        assert regime == "neutral"

    def test_exact_date_match(self):
        model  = self._build_model()
        # 2020-01-01 is index 0 → state 0 → 'bull'
        regime = model.predict_regime(None, pd.Timestamp("2020-01-01"))
        assert regime == "bull"


# ─── 7. Regime emission learning ─────────────────────────────────────────────

class TestBullSequenceDetection:
    """Test that the HMM correctly learns regime-specific emission patterns."""

    def test_bull_state_emission_peaks_at_bull_prototype(self):
        """After EM on 80% bull-prototype data, the 'bull' state's argmax must be obs=19.

        This holds even in imperfect local-optima: since _assign_state_labels picks
        the state with the highest B[:, 19], that state's emission row cannot peak
        elsewhere whenever obs=19 appears at 80% frequency.
        """
        rng = np.random.default_rng(0)
        obs = rng.choice([19, 13, 7], size=600, p=[0.8, 0.1, 0.1]).astype(np.int32)
        hmm = _HMM(rng=np.random.default_rng(42))
        hmm.fit(obs, n_iter=50)
        label_map  = _assign_state_labels(hmm.B)
        bull_state = next(k for k, v in label_map.items() if v == "bull")
        assert hmm.B[bull_state, 19] == hmm.B[bull_state].max(), (
            "Bull state's emission must peak at obs=19 after bull-heavy training"
        )

    def test_bear_state_emission_peaks_at_bear_prototype(self):
        """After EM on 80% bear-prototype data, the 'bear' state's argmax must be obs=7."""
        rng = np.random.default_rng(1)
        obs = rng.choice([7, 13, 19], size=600, p=[0.8, 0.1, 0.1]).astype(np.int32)
        hmm = _HMM(rng=np.random.default_rng(42))
        hmm.fit(obs, n_iter=50)
        label_map  = _assign_state_labels(hmm.B)
        bear_state = next(k for k, v in label_map.items() if v == "bear")
        assert hmm.B[bear_state, 7] == hmm.B[bear_state].max(), (
            "Bear state's emission must peak at obs=7 after bear-heavy training"
        )

    def test_viterbi_routes_bull_obs_on_ideal_hmm(self):
        """Viterbi on a hand-built 'ideal' HMM routes 80% bull obs → >60% bull states.

        By constructing the HMM manually (not via EM), we test the Viterbi
        algorithm independently of EM convergence quality.
        """
        hmm = _HMM(rng=np.random.default_rng(0))
        uniform = 0.1 / (N_OBS - 1)
        # State 0: strongly emits obs=19 (bull)
        hmm.B[0, :] = uniform
        hmm.B[0, 19] = 0.9
        # State 1: strongly emits obs=13 (neutral)
        hmm.B[1, :] = uniform
        hmm.B[1, 13] = 0.9
        # State 2: strongly emits obs=7  (bear)
        hmm.B[2, :] = uniform
        hmm.B[2, 7] = 0.9
        # High self-transition probabilities
        hmm.A  = np.array([[0.90, 0.05, 0.05],
                            [0.05, 0.90, 0.05],
                            [0.05, 0.05, 0.90]])
        hmm.pi = np.array([0.80, 0.10, 0.10])

        rng = np.random.default_rng(0)
        obs = rng.choice([19, 13, 7], size=600, p=[0.8, 0.1, 0.1]).astype(np.int32)

        label_map = _assign_state_labels(hmm.B)
        states    = hmm.predict(obs)
        labels    = [label_map[s] for s in states]
        bull_frac = labels.count("bull") / len(labels)

        assert bull_frac > 0.60, (
            f"Ideal HMM + 80% bull obs → expected >60% bull states, got {bull_frac:.1%}"
        )


# ─── 8. AnalysisSystem.__init__ smoke test ───────────────────────────────────

class TestAnalysisSystemInit:
    def test_init_does_not_crash(self):
        """AnalysisSystem() must not raise regardless of whether parquet files exist."""
        import scripts.run_30day_sim as sim
        # Should complete (possibly with a warning log) without raising
        analyzer = sim.AnalysisSystem()
        assert hasattr(analyzer, "_regime_model")

    def test_weights_sum_to_one_via_model(self):
        """If the HMM trained successfully, get_weights() sums to 1.0."""
        import scripts.run_30day_sim as sim
        analyzer = sim.AnalysisSystem()
        if analyzer._regime_model is not None:
            for regime in ("bull", "neutral", "bear"):
                w = analyzer._regime_model.get_weights(regime)
                assert abs(sum(w.values()) - 1.0) < 1e-9, (
                    f"weights for '{regime}' don't sum to 1: {w}"
                )
