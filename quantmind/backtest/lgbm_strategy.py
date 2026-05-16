"""LGBM + panel-driven quarterly/monthly rebalance strategy for BacktestEngine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from quantmind.backtest.engine import BacktestConfig, Strategy
from quantmind.backtest.execution import Order
from quantmind.models.factor_model import FactorModel
from quantmind.risk.position_sizing import PositionSizer
from quantmind.utils.score_order import order_preserving_pct_rank


class LGBMStrategy(Strategy):
    """Use FactorModel scores from latest PIT panel cross-section; fallback momentum."""

    def __init__(
        self,
        model_path: str | Path,
        panel_dir: str | Path = "data/panel",
        top_n: int = 50,
        long_n: int = 10,
        rebalance: str = "quarterly",
        price_path: str | Path = "data/prices/csi300_daily_ohlcv.parquet",
        adj_close_path: str | Path | None = None,
        snapshots_root: str | Path = "data/snapshots",
        max_industry_stocks: int | None = None,
        reversal_pct_cutoff: float | None = None,
        weighting: str = "equal",
        config: BacktestConfig | None = None,
        risk_manager: Any | None = None,
    ) -> None:
        super().__init__(config=config, risk_manager=risk_manager)
        self.model_path = Path(model_path)
        self.panel_dir = Path(panel_dir)
        self.top_n = int(top_n)
        self.long_n = int(long_n)
        self.rebalance = rebalance.lower().strip()
        self.price_path = Path(price_path)
        self.adj_close_path = Path(adj_close_path) if adj_close_path else None
        self.snapshots_root = Path(snapshots_root)
        self.max_industry_stocks = max_industry_stocks
        self.reversal_pct_cutoff = reversal_pct_cutoff
        self.weighting = weighting.lower().strip()
        self._model: FactorModel | None = None
        self._panel: pd.DataFrame | None = None
        self._feat_cols: list[str] | None = None
        self._reb_map: dict[tuple[int, int], pd.Timestamp] = {}
        self._industry_by_ticker: dict[str, str] = {}
        self._adj_close_wide: pd.DataFrame | None = None

    def on_start(self, start_date: Any, end_date: Any, universe: list[str]) -> None:
        del start_date, end_date, universe
        try:
            if self.model_path.is_file():
                self._model = FactorModel.load(self.model_path)
                logger.info("[LGBMStrategy] loaded model {}", self.model_path)
            else:
                logger.warning("[LGBMStrategy] model missing {}; momentum fallback", self.model_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("[LGBMStrategy] model load failed: {}", str(e)[:120])
            self._model = None

        self._panel = self._load_panel_union()
        meta_path = self.panel_dir / "split_meta.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                self._feat_cols = meta.get("feature_columns")
                alt = Path(meta.get("panel_file", ""))
                if alt.is_file() and (self._panel is None or self._panel.empty):
                    self._panel = pd.read_parquet(alt)
                    logger.info("[LGBMStrategy] panel from split_meta {}", alt)
            except Exception as e:  # noqa: BLE001
                logger.warning("[LGBMStrategy] split_meta read failed: {}", str(e)[:120])

        monthly_meta = self.panel_dir / "monthly_split_meta.json"
        if monthly_meta.is_file() and (self._panel is None or self._panel.empty):
            try:
                meta_m = json.loads(monthly_meta.read_text(encoding="utf-8"))
                self._feat_cols = meta_m.get("feature_columns") or self._feat_cols
                mp = Path(meta_m.get("panel_file", ""))
                if mp.is_file():
                    self._panel = pd.read_parquet(mp)
                    logger.info("[LGBMStrategy] monthly panel {}", mp)
            except Exception as e:  # noqa: BLE001
                logger.warning("[LGBMStrategy] monthly_split_meta failed: {}", str(e)[:120])

        if self._model is not None and self._model.feature_names:
            self._feat_cols = list(self._model.feature_names)

        if self._panel is not None:
            if not isinstance(self._panel.index, pd.MultiIndex):
                if "as_of" in self._panel.columns and "ticker" in self._panel.columns:
                    self._panel = self._panel.set_index(["as_of", "ticker"]).sort_index()
                elif "trade_date" in self._panel.columns:
                    self._panel = self._panel.rename(columns={"trade_date": "as_of"})
                    if "ticker" not in self._panel.columns and "ts_code" in self._panel.columns:
                        self._panel = self._panel.rename(columns={"ts_code": "ticker"})
                    self._panel = self._panel.set_index(["as_of", "ticker"]).sort_index()
            if isinstance(self._panel.index, pd.MultiIndex):
                self._panel.index.set_names(["as_of", "ticker"], inplace=True)
                self._panel = self._panel.sort_index()
                if self._panel.index.duplicated().any():
                    self._panel = self._panel[~self._panel.index.duplicated(keep="last")]

        cal = self._calendar_days()
        self._reb_map = self._first_trading_day_per_month(cal)
        logger.info("[LGBMStrategy] rebalance={} calendar_days={}", self.rebalance, len(cal))

        self._industry_by_ticker = self._load_industry_map()
        self._adj_close_wide = self._load_adj_close_wide()

    def _load_adj_close_wide(self) -> pd.DataFrame | None:
        path = self.adj_close_path
        if path is None or not path.is_file():
            return None
        try:
            px = pd.read_parquet(path)
            px.index = pd.to_datetime(px.index)
            return px.sort_index()
        except Exception as e:  # noqa: BLE001
            logger.warning("[LGBMStrategy] adj_close load failed: {}", str(e)[:120])
            return None

    def _load_industry_map(self) -> dict[str, str]:
        root = self.snapshots_root
        if not root.is_dir():
            return {}
        days = sorted(
            [p for p in root.iterdir() if p.is_dir() and (p / "stock_basic.parquet").is_file()],
            key=lambda p: p.name,
        )
        if not days:
            return {}
        p = days[-1] / "stock_basic.parquet"
        try:
            sb = pd.read_parquet(p)
            col = "ticker" if "ticker" in sb.columns else "ts_code"
            ind_col = "industry" if "industry" in sb.columns else None
            if ind_col is None:
                return {}
            out = dict(zip(sb[col].astype(str), sb[ind_col].astype(str)))
            logger.info("[LGBMStrategy] industry map from {} (n={})", p.name, len(out))
            return out
        except Exception as e:  # noqa: BLE001
            logger.warning("[LGBMStrategy] industry map failed: {}", str(e)[:120])
            return {}

    def _calendar_days(self) -> pd.DatetimeIndex:
        if not self.price_path.is_file():
            return pd.DatetimeIndex([])
        mi = pd.read_parquet(self.price_path)
        if isinstance(mi.index, pd.MultiIndex):
            lv0 = mi.index.get_level_values(0)
        else:
            lv0 = mi.index
        return pd.DatetimeIndex(pd.unique(lv0)).sort_values()

    def _load_panel_union(self) -> pd.DataFrame | None:
        frames: list[pd.DataFrame] = []
        for name in (
            "train.parquet",
            "val.parquet",
            "test.parquet",
            "predict.parquet",
            "monthly_train.parquet",
            "monthly_val.parquet",
            "monthly_test.parquet",
            "monthly_panel.parquet",
        ):
            p = self.panel_dir / name
            if p.is_file():
                frames.append(pd.read_parquet(p))
        if not frames:
            return None
        out = pd.concat(frames, axis=0).sort_index()
        if isinstance(out.index, pd.MultiIndex):
            out.index.set_names(["as_of", "ticker"], inplace=True)
            if out.index.duplicated().any():
                out = out[~out.index.duplicated(keep="last")]
        return out

    def _first_trading_day_per_month(self, cal: pd.DatetimeIndex) -> dict[tuple[int, int], pd.Timestamp]:
        m: dict[tuple[int, int], pd.Timestamp] = {}
        for d in cal:
            ts = pd.Timestamp(d).normalize()
            key = (ts.year, ts.month)
            if key not in m:
                m[key] = ts
        return m

    def _is_rebalance_day(self, current_date: Any) -> bool:
        ts = pd.Timestamp(current_date).normalize()
        key = (ts.year, ts.month)
        first = self._reb_map.get(key)
        if first is None or ts != first:
            return False
        if self.rebalance == "monthly":
            return True
        if self.rebalance == "quarterly":
            return ts.month in (3, 6, 9, 12)
        return False

    def _five_day_return(self, pit_prices: pd.DataFrame, ticker: str, ts: pd.Timestamp) -> float:
        try:
            px = pit_prices.xs(ticker, level="ticker")["close"].sort_index().loc[:ts]
            if len(px) < 6:
                return float("nan")
            return float(px.iloc[-1] / px.iloc[-6] - 1.0)
        except (KeyError, IndexError, ValueError):
            return float("nan")

    def _five_day_adj_return(self, ticker: str, ts: pd.Timestamp) -> float:
        w = self._adj_close_wide
        if w is None or ticker not in w.columns:
            return float("nan")
        try:
            s = w.loc[:ts, ticker].dropna()
            if len(s) < 6:
                return float("nan")
            return float(s.iloc[-1] / s.iloc[-6] - 1.0)
        except Exception:
            return float("nan")

    def _momentum_scores(
        self,
        pit_prices: pd.DataFrame,
        universe: list[str],
        current_date: Any,
    ) -> pd.Series:
        ts = pd.Timestamp(current_date).normalize()
        scores: dict[str, float] = {}
        for tkr in universe:
            try:
                px = pit_prices.xs(tkr, level="ticker")["close"].sort_index().loc[:ts]
                if len(px) < 22:
                    continue
                scores[tkr] = float(px.iloc[-1] / px.iloc[-22] - 1.0)
            except (KeyError, IndexError, ValueError):
                continue
        s = pd.Series(scores, dtype=float)
        if s.empty:
            return s
        return order_preserving_pct_rank(s)

    def _get_latest_panel_scores(self, as_of_date: Any, universe: list[str]) -> pd.Series:
        if self._panel is None or self._panel.empty or self._model is None:
            return pd.Series(dtype=float)
        dt = pd.Timestamp(as_of_date).normalize()
        dates = sorted(self._panel.index.get_level_values(0).unique())
        past = [d for d in dates if pd.Timestamp(d).normalize() <= dt]
        if not past:
            return pd.Series(dtype=float)
        latest = past[-1]
        try:
            slab = self._panel.xs(latest, level=0)
        except KeyError:
            return pd.Series(dtype=float)

        feats = self._feat_cols or getattr(self._model, "feature_names", None)
        if not feats:
            return pd.Series(dtype=float)

        use = slab.reindex(universe).dropna(how="all")
        if use.empty:
            use = slab

        X = use.reindex(columns=feats).fillna(0.0).astype(np.float32)
        tickers = list(use.index)
        try:
            pred = self._model.predict(X.values)
        except Exception as e:  # noqa: BLE001
            logger.debug("[LGBMStrategy] predict failed: {}", str(e)[:120])
            return pd.Series(dtype=float)
        raw_s = pd.Series(pred, index=tickers, dtype=float)
        s = order_preserving_pct_rank(raw_s)
        return s.reindex(universe)

    def _combined_scores(
        self,
        pit_prices: pd.DataFrame,
        universe: list[str],
        current_date: Any,
    ) -> pd.Series:
        mom = self._momentum_scores(pit_prices, universe, current_date)
        pan = self._get_latest_panel_scores(current_date, universe)
        if pan.empty or pan.dropna().shape[0] < max(self.long_n, 10):
            return mom.reindex(universe)
        out = pan.reindex(universe).fillna(mom)
        return out.fillna(0.0)

    def _filter_reversal(self, scores: pd.Series, pit_prices: pd.DataFrame, ts: pd.Timestamp) -> pd.Series:
        if self.reversal_pct_cutoff is None:
            return scores
        keep = []
        for tkr in scores.index:
            r = self._five_day_adj_return(tkr, ts)
            if r == r and r > float(self.reversal_pct_cutoff):
                continue
            if r != r:
                r2 = self._five_day_return(pit_prices, tkr, ts)
                if r2 == r2 and r2 > float(self.reversal_pct_cutoff):
                    continue
            keep.append(tkr)
        return scores.reindex(keep).dropna()

    def _apply_industry_cap(self, ranked: pd.Series) -> list[str]:
        lim = self.max_industry_stocks
        if lim is None:
            return list(ranked.sort_values(ascending=False).head(self.long_n).index)
        ranked = ranked.sort_values(ascending=False)
        ind_cnt: dict[str, int] = {}
        out: list[str] = []
        for tkr, _sc in ranked.items():
            ind = self._industry_by_ticker.get(str(tkr), "UNKNOWN")
            if ind_cnt.get(ind, 0) >= int(lim):
                continue
            ind_cnt[ind] = ind_cnt.get(ind, 0) + 1
            out.append(str(tkr))
            if len(out) >= self.long_n:
                break
        return out

    def _inverse_vol_weights(
        self,
        picks: list[str],
        pit_prices: pd.DataFrame,
        ts: pd.Timestamp,
    ) -> dict[str, float]:
        frames: list[pd.Series] = []
        for tkr in picks:
            try:
                colname = "close"
                s = pit_prices.xs(tkr, level="ticker")[colname].sort_index().loc[:ts].tail(22)
                r = s.pct_change().dropna()
                if len(r) >= 5:
                    frames.append(r.rename(tkr))
            except Exception:
                continue
        if not frames:
            return PositionSizer.equal_weight(picks)
        df = pd.concat(frames, axis=1)
        return PositionSizer.inverse_volatility(df, min_periods=5)

    def on_market_open(
        self,
        current_date: Any,
        prices: pd.DataFrame,
        universe: list[str],
    ) -> list[Order]:
        if self.portfolio is None:
            return []

        is_reb = self._is_rebalance_day(current_date)
        logger.debug("[LGBM] {}: rebalance={}", current_date, is_reb)
        if not is_reb:
            return []

        ts = pd.Timestamp(current_date).normalize()
        try:
            bar = prices.xs(ts, level="date")
        except KeyError:
            return []

        scores = self._combined_scores(prices, universe, current_date)
        scores = scores.dropna().sort_values(ascending=False)
        scores = self._filter_reversal(scores, prices, ts)
        if self.top_n:
            scores = scores.head(self.top_n)
        picks = self._apply_industry_cap(scores)
        picks = [p for p in picks if p in bar.index]
        logger.debug(
            "[LGBM] {}: Top{} preview={}…",
            current_date,
            self.long_n,
            picks[:3],
        )
        if not picks:
            logger.debug("[LGBMStrategy] {} no picks", current_date)
            return []

        if self.weighting == "inverse_vol":
            weights_raw = self._inverse_vol_weights(picks, prices, ts)
            sw = sum(weights_raw.get(t, 0.0) for t in picks)
            weights = {t: float(weights_raw.get(t, 0.0) / sw) if sw > 0 else 1.0 / len(picks) for t in picks}
        else:
            w = 1.0 / len(picks)
            weights = {t: w for t in picks}

        logger.debug(
            "[LGBM] {} target_weights n={} sum_w={:.6f} preview={}",
            current_date,
            len(weights),
            sum(weights.values()),
            {k: round(weights[k], 6) for k in list(weights.keys())[: min(4, len(weights))]},
        )

        price_open = {}
        tickers_for_px = set(weights) | set(self.portfolio.positions.keys())
        for tkr in tickers_for_px:
            if tkr not in bar.index:
                continue
            row = bar.loc[tkr]
            op = float(row.get("open", row.get("close", np.nan)))
            if np.isfinite(op) and op > 0:
                price_open[tkr] = op

        orders = self.portfolio.rebalance_to_weights(weights, price_open, pd.Timestamp(current_date).date())
        logger.info("[LGBMStrategy] rebalance {} picks={}", current_date, picks[: min(5, len(picks))])
        return orders


__all__ = ["LGBMStrategy"]
