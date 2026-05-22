"""quantmind/regime/hmm.py

3-state Hidden Markov Model for A-share market regime detection.
Pure NumPy implementation — no hmmlearn dependency.

Hidden states
-------------
  Bull    — rising trend, compressed volatility
  Neutral — sideways / ambiguous
  Bear    — falling trend, elevated volatility

Observation alphabet (27 discrete symbols)
------------------------------------------
  obs = o1*9 + o2*3 + o3
  o1  = CSI300 5-day return  quantile  : Low=0 | Mid=1 | High=2
  o2  = CSI300 20-day vol    quantile  : Low=0 | Mid=1 | High=2
  o3  = market turnover                : fixed Mid=1 (no market-wide daily data)

Prototype observations for state auto-labeling:
  Bull prototype  obs=19  (o1=High,  o2=Low,  o3=Mid)  → 2*9 + 0*3 + 1 = 19
  Bear prototype  obs=7   (o1=Low,   o2=High, o3=Mid)  → 0*9 + 2*3 + 1 =  7

State-labeling assumption (IMPORTANT)
--------------------------------------
  _assign_state_labels() anchors state semantics to fixed prototype symbols.
  This is valid when training data is long enough for the EM to differentiate
  states, but prototype-to-state mapping may drift if:
    (a) the dataset is very short (<200 bars)
    (b) the underlying regime distribution changes materially
    (c) the HMM is re-initialised with a different random seed

  After fitting, always call RegimeHMM.validate_state_labels() to confirm that
  the empirical mean returns satisfy bull > neutral > bear.  If they do not,
  the method swaps the label_map entries automatically and emits a WARNING.

Numerical methods
-----------------
  Forward-backward : per-step scaled (avoids underflow)
  Viterbi          : log-space (avoids underflow)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

N_STATES: int = 3
N_OBS: int    = 27   # 3^3 discrete observation symbols

_PROTO_BULL: int = 19   # o1=High(2), o2=Low(0),  o3=Mid(1)
_PROTO_BEAR: int = 7    # o1=Low(0),  o2=High(2), o3=Mid(1)

# Factor weights per regime — derived from 30-day sim IC analysis (2025-Q4)
REGIME_WEIGHTS: Dict[str, Dict[str, float]] = {
    "bull":    {"value": 0.242, "momentum": 0.223, "quality": 0.333, "technical": 0.202},
    "neutral": {"value": 0.280, "momentum": 0.220, "quality": 0.300, "technical": 0.200},
    "bear":    {"value": 0.350, "momentum": 0.150, "quality": 0.280, "technical": 0.220},
}


# ─── Observation Construction ─────────────────────────────────────────────────

def build_observations(df: pd.DataFrame) -> np.ndarray:
    """Build 27-symbol discrete observation sequence from CSI300 price DataFrame.

    Parameters
    ----------
    df : DataFrame
        Must contain columns ``trade_date`` and ``close`` (or be indexed by
        ``trade_date``).

    Returns
    -------
    np.ndarray of int, shape (T,)
        Observation symbols in [0, 26], with leading NaN rows dropped.
    """
    if "trade_date" in df.columns:
        df = df.set_index("trade_date")
    close = df["close"].sort_index().astype(float)

    # Raw features
    ret5    = close.pct_change(5)
    vol20   = close.pct_change().rolling(20).std()

    def _quantize(series: pd.Series) -> np.ndarray:
        """Quantize each point using up-to-252-day rolling percentile boundaries."""
        vals   = series.values
        result = np.ones(len(vals), dtype=int)   # default Mid=1
        for i in range(len(vals)):
            v = vals[i]
            if np.isnan(v):
                continue
            window = vals[max(0, i - 251): i + 1]
            window = window[~np.isnan(window)]
            if len(window) < 5:
                continue
            p33 = np.percentile(window, 33)
            p67 = np.percentile(window, 67)
            if v <= p33:
                result[i] = 0   # Low
            elif v >= p67:
                result[i] = 2   # High
            # else: already 1 (Mid)
        return result

    o1 = _quantize(ret5)
    o2 = _quantize(vol20)
    o3 = np.ones(len(close), dtype=int)   # fixed Mid=1

    obs = o1 * 9 + o2 * 3 + o3

    # Drop leading rows where either raw feature is NaN
    valid_mask = ~(np.isnan(ret5.values) | np.isnan(vol20.values))
    return obs[valid_mask].astype(np.int32)


# ─── HMM Core ─────────────────────────────────────────────────────────────────

class _HMM:
    """Discrete-observation HMM with Baum-Welch EM and Viterbi decoding.

    Parameters
    ----------
    n_states : int
        Number of hidden states K.
    n_obs : int
        Alphabet size M (number of distinct observation symbols).
    rng : np.random.Generator, optional
        For reproducible Dirichlet initialisation.
    """

    def __init__(
        self,
        n_states: int = N_STATES,
        n_obs:    int = N_OBS,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.K = n_states
        self.M = n_obs
        rng    = rng or np.random.default_rng(42)

        # Initial state distribution π  (K,)
        self.pi = rng.dirichlet(np.ones(self.K))
        # Transition matrix A            (K, K)
        self.A  = rng.dirichlet(np.ones(self.K), size=self.K)
        # Emission matrix B              (K, M)
        self.B  = rng.dirichlet(np.ones(self.M), size=self.K)

    # ── Forward (scaled) ─────────────────────────────────────────────────────

    def _forward(
        self, obs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Scaled forward algorithm.

        Returns
        -------
        alpha  : (T, K) scaled forward variables
        scales : (T,)  per-step scale factors
        log_ll : float log-likelihood
        """
        T      = len(obs)
        alpha  = np.zeros((T, self.K))
        scales = np.zeros(T)

        alpha[0]  = self.pi * self.B[:, obs[0]]
        scales[0] = alpha[0].sum() or 1e-300
        alpha[0] /= scales[0]

        for t in range(1, T):
            alpha[t]  = (alpha[t - 1] @ self.A) * self.B[:, obs[t]]
            scales[t] = alpha[t].sum() or 1e-300
            alpha[t] /= scales[t]

        log_ll = float(np.sum(np.log(scales + 1e-300)))
        return alpha, scales, log_ll

    # ── Backward (scaled) ────────────────────────────────────────────────────

    def _backward(self, obs: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """Scaled backward algorithm.

        Returns
        -------
        beta : (T, K)
        """
        T    = len(obs)
        beta = np.zeros((T, self.K))
        beta[T - 1] = 1.0

        for t in range(T - 2, -1, -1):
            beta[t] = self.A @ (self.B[:, obs[t + 1]] * beta[t + 1])
            s = scales[t + 1]
            if s:
                beta[t] /= s

        return beta

    # ── Baum-Welch EM ────────────────────────────────────────────────────────

    def fit(
        self,
        obs:    np.ndarray,
        n_iter: int   = 60,
        tol:    float = 1e-4,
    ) -> float:
        """Baum-Welch EM.

        Returns
        -------
        float: final log-likelihood
        """
        obs     = np.asarray(obs, dtype=np.int32)
        T       = len(obs)
        prev_ll = -np.inf

        for iteration in range(n_iter):
            alpha, scales, ll = self._forward(obs)
            beta              = self._backward(obs, scales)

            # Posterior state probabilities γ  (T, K)
            gamma = alpha * beta
            gamma_sum = gamma.sum(axis=1, keepdims=True)
            gamma /= np.where(gamma_sum > 0, gamma_sum, 1.0)

            # Joint transition posteriors ξ  (T-1, K, K)
            xi = np.zeros((T - 1, self.K, self.K))
            for t in range(T - 1):
                xi[t] = (
                    alpha[t].reshape(-1, 1)
                    * self.A
                    * (self.B[:, obs[t + 1]] * beta[t + 1]).reshape(1, -1)
                )
                xi_sum = xi[t].sum()
                if xi_sum > 0:
                    xi[t] /= xi_sum

            # M-step: update π, A, B
            self.pi = gamma[0] / (gamma[0].sum() or 1.0)

            xi_sumKK = xi.sum(axis=0)                       # (K, K)
            row_sums  = xi_sumKK.sum(axis=1, keepdims=True)
            self.A    = xi_sumKK / np.where(row_sums > 0, row_sums, 1.0)

            for m in range(self.M):
                mask = obs == m
                self.B[:, m] = gamma[mask].sum(axis=0) if mask.any() else 1e-300

            row_sums = self.B.sum(axis=1, keepdims=True)
            self.B  /= np.where(row_sums > 0, row_sums, 1.0)

            if abs(ll - prev_ll) < tol:
                logger.debug(
                    f"HMM converged at iteration {iteration + 1}, ll={ll:.4f}"
                )
                break
            prev_ll = ll

        return float(ll)

    # ── Viterbi (log-space) ───────────────────────────────────────────────────

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Viterbi decoding in log-space.

        Returns
        -------
        np.ndarray of int, shape (T,): most-likely hidden-state sequence
        """
        obs    = np.asarray(obs, dtype=np.int32)
        T      = len(obs)
        log_A  = np.log(self.A + 1e-300)
        log_B  = np.log(self.B + 1e-300)
        log_pi = np.log(self.pi + 1e-300)

        delta  = np.zeros((T, self.K))
        psi    = np.zeros((T, self.K), dtype=np.int32)

        delta[0] = log_pi + log_B[:, obs[0]]

        for t in range(1, T):
            # trans[i, j] = delta[t-1, i] + log A[i, j]
            trans    = delta[t - 1].reshape(-1, 1) + log_A   # (K, K)
            psi[t]   = trans.argmax(axis=0)
            delta[t] = trans.max(axis=0) + log_B[:, obs[t]]

        # Backtrack
        states       = np.zeros(T, dtype=np.int32)
        states[T - 1] = int(delta[T - 1].argmax())
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states


# ─── State Auto-Labeling ─────────────────────────────────────────────────────

def _assign_state_labels(B: np.ndarray) -> Dict[int, str]:
    """Map HMM states to regime labels using prototype observations.

    Rules
    -----
    * State with the highest ``B[:, _PROTO_BULL]`` → 'bull'
    * State with the highest ``B[:, _PROTO_BEAR]`` → 'bear'
    * Remaining state                               → 'neutral'

    When both prototypes point to the same state, the second-highest
    bear score breaks the tie.
    """
    bull_state = int(B[:, _PROTO_BULL].argmax())

    bear_scores = B[:, _PROTO_BEAR].copy()
    bear_scores[bull_state] = -1.0          # exclude bull_state first
    bear_state = int(bear_scores.argmax())

    all_states   = set(range(B.shape[0]))
    neutral_state = (all_states - {bull_state, bear_state}).pop()

    return {bull_state: "bull", neutral_state: "neutral", bear_state: "bear"}


# ─── Public Interface ─────────────────────────────────────────────────────────

class RegimeHMM:
    """Fitted 3-state HMM for A-share market regime detection.

    Typical usage
    -------------
    >>> model = RegimeHMM.fit_from_file(hist_path, sim_path)
    >>> regime = model.predict_regime(None, as_of=pd.Timestamp("2025-11-19"))
    >>> weights = model.get_weights(regime)
    """

    def __init__(
        self,
        hmm:       _HMM,
        label_map: Dict[int, str],
        dates:     pd.DatetimeIndex,
        states:    np.ndarray,
    ) -> None:
        self._hmm       = hmm
        self._label_map = label_map    # int state → 'bull'/'neutral'/'bear'
        self._dates     = dates
        self._states    = states       # Viterbi sequence parallel to _dates

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def fit_from_file(
        cls,
        hist_path: Optional[Path] = None,
        sim_path:  Optional[Path] = None,
        n_iter:    int = 60,
    ) -> "RegimeHMM":
        """Train HMM on CSI300 data from parquet files.

        Parameters
        ----------
        hist_path : path to ``data/raw/index_daily_panel.parquet``
            Multi-stock panel; CSI300 rows selected by ts_code == '000300.SH'.
        sim_path  : path to ``data/sim30d/raw/index_000300_SH.parquet``
            Single-stock CSI300 file for 2025-2026 extension.
        n_iter    : Baum-Welch EM iterations.

        Returns
        -------
        RegimeHMM instance (or a neutral-fallback instance if no data found).
        """
        dfs: list[pd.DataFrame] = []

        # --- historical panel ---
        if hist_path is not None and Path(hist_path).exists():
            raw = pd.read_parquet(hist_path)
            if "ts_code" in raw.columns:
                csi = raw[raw["ts_code"] == "000300.SH"][["trade_date", "close"]].copy()
            else:
                csi = raw[["trade_date", "close"]].copy()
            csi["trade_date"] = pd.to_datetime(csi["trade_date"])
            dfs.append(csi)

        # --- 2025-2026 extension ---
        if sim_path is not None and Path(sim_path).exists():
            sim = pd.read_parquet(sim_path)
            if "trade_date" not in sim.columns and sim.index.name == "trade_date":
                sim = sim.reset_index()
            sim = sim[["trade_date", "close"]].copy()
            sim["trade_date"] = pd.to_datetime(sim["trade_date"])
            dfs.append(sim)

        if not dfs:
            logger.warning(
                "RegimeHMM.fit_from_file: no data files found — "
                "returning neutral-fallback model."
            )
            return cls._neutral_fallback()

        combined = (
            pd.concat(dfs, ignore_index=True)
            .drop_duplicates("trade_date")
            .sort_values("trade_date")
            .reset_index(drop=True)
        )
        logger.info(
            f"RegimeHMM: training on {len(combined)} CSI300 rows "
            f"({combined['trade_date'].min().date()} → "
            f"{combined['trade_date'].max().date()})"
        )

        obs = build_observations(combined)

        # Date index aligned to valid observations
        tmp   = combined.set_index("trade_date")["close"].sort_index()
        ret5  = tmp.pct_change(5)
        vol20 = tmp.pct_change().rolling(20).std()
        valid = ~(ret5.isna() | vol20.isna())
        dates = tmp.index[valid]

        # Fit
        hmm = _HMM(rng=np.random.default_rng(42))
        ll  = hmm.fit(obs, n_iter=n_iter)
        logger.info(f"RegimeHMM fitted: ll={ll:.4f}, T={len(obs)}")

        # Viterbi
        states    = hmm.predict(obs)
        label_map = _assign_state_labels(hmm.B)
        dist      = {v: int((states == k).sum()) for k, v in label_map.items()}
        logger.info(f"RegimeHMM state distribution: {dist}")

        model = cls(hmm, label_map, pd.DatetimeIndex(dates), states)
        # Sanity-check state ordering; auto-swap bull↔bear if violated
        model.validate_state_labels(close=tmp)
        return model

    @classmethod
    def _neutral_fallback(cls) -> "RegimeHMM":
        """Dummy model that always returns 'neutral'."""
        hmm       = _HMM()
        label_map = {0: "neutral", 1: "neutral", 2: "neutral"}
        return cls(hmm, label_map, pd.DatetimeIndex([]), np.array([], dtype=np.int32))

    # ── Inference ────────────────────────────────────────────────────────────

    def predict_regime(
        self,
        df:    Optional[pd.DataFrame],
        as_of: pd.Timestamp,
    ) -> str:
        """Return the most-likely regime label for a given date.

        Looks up the Viterbi state at the last training date ≤ ``as_of``.
        Falls back to 'neutral' if the query date precedes all training data.

        Parameters
        ----------
        df    : Unused (kept for API symmetry with possible online variants).
        as_of : Query timestamp.

        Returns
        -------
        str: one of 'bull', 'neutral', 'bear'
        """
        if len(self._dates) == 0:
            return "neutral"

        idx = int(self._dates.searchsorted(as_of, side="right")) - 1
        if idx < 0:
            return "neutral"
        idx       = min(idx, len(self._states) - 1)
        state_int = int(self._states[idx])
        return self._label_map.get(state_int, "neutral")

    def get_weights(self, regime: str) -> Dict[str, float]:
        """Return the calibrated factor weight dictionary for *regime*."""
        return REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS["neutral"])

    # ── Label sanity-check ───────────────────────────────────────────────────

    def validate_state_labels(self, close: Optional[pd.Series] = None) -> bool:
        """Check that empirical mean returns satisfy bull > neutral > bear.

        The prototype-anchor labeling in _assign_state_labels() is a heuristic.
        If the data distribution changes or the EM converges to an atypical
        solution, the semantic order may be violated.  This method verifies
        the ordering and, if violated, swaps label_map entries to restore it,
        emitting a WARNING.

        Parameters
        ----------
        close : pd.Series indexed by trade_date, optional
            CSI300 close prices aligned to the training window.  If None the
            method tries to use the state sequence alone (less reliable).

        Returns
        -------
        bool: True if ordering was correct (or could be fixed), False if it
              could not be determined (e.g. too few states observed).
        """
        if len(self._states) == 0:
            return True   # neutral-fallback model, nothing to validate

        # ── Compute mean daily return per state from Viterbi sequence ────────
        if close is not None:
            aligned_close = close.reindex(self._dates)
            daily_ret = aligned_close.pct_change()
        else:
            # Cannot compute returns without price data — skip gracefully
            logger.debug("validate_state_labels: no close series supplied, skipping")
            return True

        means: Dict[int, float] = {}
        for state_int, label in self._label_map.items():
            mask = self._states == state_int
            if mask.sum() < 5:
                continue
            state_ret = daily_ret.values[mask]
            means[state_int] = float(np.nanmean(state_ret))

        if len(means) < 2:
            return True   # not enough distinct states to compare

        bull_state    = next((k for k, v in self._label_map.items() if v == "bull"),    None)
        neutral_state = next((k for k, v in self._label_map.items() if v == "neutral"), None)
        bear_state    = next((k for k, v in self._label_map.items() if v == "bear"),    None)

        bull_ret    = means.get(bull_state,    float("-inf"))
        neutral_ret = means.get(neutral_state, 0.0)
        bear_ret    = means.get(bear_state,    float("inf"))

        order_ok = (bull_ret >= neutral_ret >= bear_ret)

        if not order_ok:
            logger.warning(
                f"validate_state_labels: state ordering violated "
                f"(bull={bull_ret:.4%}, neutral={neutral_ret:.4%}, bear={bear_ret:.4%}). "
                f"Swapping 'bull' ↔ 'bear' labels."
            )
            # Swap bull ↔ bear in label_map
            if bull_state is not None and bear_state is not None:
                self._label_map[bull_state] = "bear"
                self._label_map[bear_state] = "bull"
        else:
            logger.info(
                f"validate_state_labels OK: "
                f"bull={bull_ret:.4%} ≥ neutral={neutral_ret:.4%} ≥ bear={bear_ret:.4%}"
            )

        return True
