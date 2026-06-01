"""quantmind/watchlist/daily_scorer.py — 自选股每日因子评分与 Agent 分析.

每日为自选股计算因子得分和 6-Agent 辩论结论。
结果缓存到 data/watchlist/scores/{date}.json，避免重复计算。
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]

# System2 四维权重（与漏斗页一致）
_DIM_WEIGHTS = {
    "value":     0.242,
    "momentum":  0.223,
    "quality":   0.333,
    "technical": 0.202,
}

# 因子 → 维度映射（每个维度取平均后标准化到 0-100）
_VALUE_FACTORS    = ["pe_ttm", "pb", "earnings_yield", "book_to_market", "dividend_yield_ttm"]
_MOMENTUM_FACTORS = ["momentum_1m", "momentum_3m", "momentum_6m", "reversal_1w"]
_QUALITY_FACTORS  = ["roe_ttm", "roa_ttm", "gross_margin", "net_margin",
                     "revenue_yoy", "net_profit_yoy", "ocf_to_revenue_ttm"]
_TECHNICAL_FACTORS = ["rsi_14", "bollinger_position", "volume_spike_5_30",
                      "distance_to_52w_high", "amihud_illiquidity"]


# ── StockScore 数据类 ─────────────────────────────────────────────────────────

@dataclass
class StockScore:
    """单只股票的完整评分快照."""

    ticker:  str
    name:    str
    as_of:   str

    # 价格相关
    current_price:        float = 0.0
    price_change_1d:      float = 0.0   # 日涨跌幅（小数，0.02=2%）
    price_change_5d:      float = 0.0
    ma20:                 float = 0.0
    ma60:                 float = 0.0
    drawdown_from_high:   float = 0.0   # 距52周高点回撤（负数）
    volume_ratio:         float = 1.0   # 量比

    # System2 四维得分（0-100）
    value_score:      float = 50.0
    momentum_score:   float = 50.0
    quality_score:    float = 50.0
    technical_score:  float = 50.0
    composite_score:  float = 50.0

    # LGBM 百分位（在 alpha 宇宙里的分位数，0-100）
    lgbm_percentile: Optional[float] = None

    # 6-Agent 辩论结论
    agent_signal:          float = 0.0
    agent_confidence:      float = 0.5
    agent_recommendation:  str   = "持有"
    agent_bull_reasons:    List[str] = field(default_factory=list)
    agent_bear_reasons:    List[str] = field(default_factory=list)
    agent_key_debate:      str   = ""
    agent_debate_summary:  str   = ""

    # 元信息
    in_alpha_universe: bool = False
    data_freshness:    str  = ""   # 最近一期因子日期

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StockScore":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


# ── WatchlistDailyScorer ─────────────────────────────────────────────────────

class WatchlistDailyScorer:
    """每日为自选股计算因子得分和 6-Agent 辩论结论.

    Parameters
    ----------
    alpha_panel_path : Path | None
        alpha_panel_v4.parquet 路径，默认 data/panel/alpha_panel_v4.parquet。
    prices_path : Path | None
        daily_prices_panel.parquet 路径，默认 data/raw/daily_prices_panel.parquet。
    cache_dir : Path | None
        分数缓存目录，默认 data/watchlist/scores/。
    """

    ALPHA_PATH  = _ROOT / "data" / "panel" / "alpha_panel_v4.parquet"
    PRICES_PATH = _ROOT / "data" / "raw" / "daily_prices_panel.parquet"
    CACHE_DIR   = _ROOT / "data" / "watchlist" / "scores"

    def __init__(
        self,
        alpha_panel_path: Optional[Path] = None,
        prices_path:      Optional[Path] = None,
        cache_dir:        Optional[Path] = None,
    ) -> None:
        self._alpha_path  = Path(alpha_panel_path) if alpha_panel_path else self.ALPHA_PATH
        self._prices_path = Path(prices_path)      if prices_path      else self.PRICES_PATH
        self._cache_dir   = Path(cache_dir)        if cache_dir        else self.CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # 懒加载缓存
        self._alpha_panel: Optional[pd.DataFrame] = None
        self._prices_df:   Optional[pd.DataFrame] = None
        self._alpha_tickers: Optional[set] = None

    # ── 公开入口 ──────────────────────────────────────────────────────────────

    def score_all(
        self,
        tickers:      List[str],
        as_of:        str,
        mode:         str  = "fast",
        use_cache:    bool = True,
        name_map:     Optional[Dict[str, str]] = None,
    ) -> Dict[str, StockScore]:
        """对每只股票计算完整评分.

        Parameters
        ----------
        tickers   : 股票代码列表
        as_of     : 分析日期 YYYY-MM-DD
        mode      : 'fast'=规则模式，'full'=Ollama LLM
        use_cache : 若缓存已有当日数据则跳过重新计算
        name_map  : ticker→name 映射，避免重复 IO
        """
        cache_file = self._cache_dir / f"{as_of}.json"
        cached: Dict[str, Any] = {}
        if use_cache and cache_file.exists():
            try:
                with cache_file.open(encoding="utf-8") as fh:
                    cached = json.load(fh)
            except Exception:
                cached = {}

        results: Dict[str, StockScore] = {}
        to_compute = []
        for ticker in tickers:
            if ticker in cached:
                results[ticker] = StockScore.from_dict(cached[ticker])
            else:
                to_compute.append(ticker)

        for ticker in to_compute:
            name = (name_map or {}).get(ticker, ticker)
            try:
                score = self._score_one(ticker, name, as_of, mode)
                results[ticker] = score
                cached[ticker]  = score.to_dict()
            except Exception as e:
                log.warning("[Scorer] %s 评分失败: %s", ticker, e)
                results[ticker] = StockScore(ticker=ticker, name=name, as_of=as_of)

        # 写缓存
        if to_compute:
            try:
                with cache_file.open("w", encoding="utf-8") as fh:
                    json.dump(cached, fh, ensure_ascii=False, indent=2)
            except Exception as e:
                log.debug("[Scorer] 写缓存失败: %s", e)

        return results

    def invalidate_cache(self, as_of: str) -> None:
        """清除指定日期的缓存."""
        cache_file = self._cache_dir / f"{as_of}.json"
        if cache_file.exists():
            cache_file.unlink()

    # ── 单只股票评分 ──────────────────────────────────────────────────────────

    def _score_one(self, ticker: str, name: str, as_of: str, mode: str) -> StockScore:
        score = StockScore(ticker=ticker, name=name, as_of=as_of)

        # 1. 价格信息
        price_info = self._get_price_info(ticker)
        score.current_price      = price_info.get("current_price", 0.0)
        score.price_change_1d    = price_info.get("price_change_1d", 0.0)
        score.price_change_5d    = price_info.get("price_change_5d", 0.0)
        score.ma20               = price_info.get("ma20", 0.0)
        score.ma60               = price_info.get("ma60", 0.0)
        score.drawdown_from_high = price_info.get("drawdown_from_high_52w", 0.0)
        score.volume_ratio       = price_info.get("volume_ratio", 1.0)

        # 2. 因子得分
        factor_info = self._get_factor_scores(ticker, as_of)
        score.in_alpha_universe = factor_info.get("in_alpha_universe", False)
        score.data_freshness    = factor_info.get("data_freshness", "")
        score.value_score       = factor_info.get("value_score",     50.0)
        score.momentum_score    = factor_info.get("momentum_score",  50.0)
        score.quality_score     = factor_info.get("quality_score",   50.0)
        score.technical_score   = factor_info.get("technical_score", 50.0)
        score.composite_score   = (
            score.value_score    * _DIM_WEIGHTS["value"]    +
            score.momentum_score * _DIM_WEIGHTS["momentum"] +
            score.quality_score  * _DIM_WEIGHTS["quality"]  +
            score.technical_score * _DIM_WEIGHTS["technical"]
        )
        score.lgbm_percentile = factor_info.get("lgbm_percentile")

        # 3. 6-Agent 辩论
        debate = self._run_debate(ticker, as_of, factor_info.get("raw_factors", {}), mode)
        score.agent_signal         = debate.get("signal",         0.0)
        score.agent_confidence     = debate.get("confidence",     0.5)
        score.agent_recommendation = debate.get("recommendation", "持有")
        score.agent_bull_reasons   = debate.get("bull_reasons",   [])
        score.agent_bear_reasons   = debate.get("bear_reasons",   [])
        score.agent_key_debate     = debate.get("key_debate",     "")
        score.agent_debate_summary = debate.get("debate_summary", "")

        return score

    # ── 因子得分 ──────────────────────────────────────────────────────────────

    def _get_factor_scores(self, ticker: str, as_of: str) -> Dict[str, Any]:
        """从 alpha_panel_v4 读取最近一期因子值，计算四维得分."""
        panel = self._load_alpha_panel()
        result: Dict[str, Any] = {
            "in_alpha_universe": False,
            "data_freshness": "",
            "raw_factors": {},
            "value_score": 50.0,
            "momentum_score": 50.0,
            "quality_score": 50.0,
            "technical_score": 50.0,
            "lgbm_percentile": None,
        }

        if panel is None or panel.empty:
            return result

        # 找该 ticker 在 panel 中的最近一期记录
        try:
            ticker_rows = panel.xs(ticker, level="ticker") if ticker in panel.index.get_level_values("ticker") else None
        except Exception:
            ticker_rows = None

        # 同时获取全量最近一期（用于百分位计算）
        all_dates = sorted(panel.index.get_level_values("as_of").unique())
        latest_date = all_dates[-1] if all_dates else None

        if ticker_rows is None or ticker_rows.empty:
            result["in_alpha_universe"] = False
            # 仍尝试用 price-based 信息估算 technical score
            if latest_date is not None:
                result["data_freshness"] = str(latest_date)[:10]
            return result

        result["in_alpha_universe"] = True

        # 取最近一期（优先 as_of 当天，否则最近历史）
        as_of_dt = pd.Timestamp(as_of)
        available_dates = sorted(ticker_rows.index)
        best_date = max((d for d in available_dates if pd.Timestamp(d) <= as_of_dt), default=available_dates[-1]) \
            if available_dates else available_dates[-1]

        row = ticker_rows.loc[best_date]
        result["data_freshness"] = str(best_date)[:10]
        raw: Dict[str, Any] = row.to_dict()
        result["raw_factors"] = {k: v for k, v in raw.items() if pd.notna(v)}

        # 截面分位数（用最近一期全量）
        if latest_date is not None:
            cross_section = panel.xs(latest_date, level="as_of") if latest_date in panel.index.get_level_values("as_of") else pd.DataFrame()
        else:
            cross_section = pd.DataFrame()

        def _dim_score(factors: list[str]) -> float:
            """将因子列表转为截面百分位均值（0-100）."""
            scores = []
            for f in factors:
                val = raw.get(f)
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    continue
                try:
                    val = float(val)
                    if not cross_section.empty and f in cross_section.columns:
                        col_vals = cross_section[f].dropna().values.astype(float)
                        if len(col_vals) > 0:
                            # 估值因子：PE/PB 越低越好，取倒序百分位
                            if f in ("pe_ttm", "pb", "ps_ttm", "amihud_illiquidity",
                                     "debt_to_assets", "volatility_3m", "volatility_1y",
                                     "downside_volatility_3m", "max_drawdown_3m"):
                                pct = float(np.sum(col_vals > val) / len(col_vals) * 100)
                            else:
                                pct = float(np.sum(col_vals < val) / len(col_vals) * 100)
                            scores.append(pct)
                        else:
                            scores.append(50.0)
                    else:
                        scores.append(50.0)
                except Exception:
                    continue
            return float(np.mean(scores)) if scores else 50.0

        result["value_score"]     = _dim_score(_VALUE_FACTORS)
        result["momentum_score"]  = _dim_score(_MOMENTUM_FACTORS)
        result["quality_score"]   = _dim_score(_QUALITY_FACTORS)
        result["technical_score"] = _dim_score(_TECHNICAL_FACTORS)

        # LGBM 百分位（若 lgbm_v6_alpha 可用）
        try:
            result["lgbm_percentile"] = self._compute_lgbm_percentile(
                ticker, row, cross_section
            )
        except Exception:
            pass

        return result

    def _compute_lgbm_percentile(
        self, ticker: str, row: pd.Series, cross_section: pd.DataFrame
    ) -> Optional[float]:
        """在 alpha 宇宙截面上计算该股票 LGBM 得分的百分位."""
        from quantmind.models.factor_model import FactorModel
        model_path = _ROOT / "models" / "lgbm_v6_alpha.pkl"
        if not model_path.exists():
            return None
        model = FactorModel.load(str(model_path))
        feat_names = getattr(model, "_feature_names", None) or []
        if not feat_names or cross_section.empty:
            return None

        # 该股票得分
        missing = [f for f in feat_names if f not in row.index]
        ticker_vec = np.array(
            [float(row.get(f, 0.0) or 0.0) for f in feat_names], dtype=np.float32
        ).reshape(1, -1)
        ticker_score = float(model.predict(ticker_vec)[0])

        # 全截面得分
        avail = [f for f in feat_names if f in cross_section.columns]
        X_all = cross_section[avail].fillna(0.0).reindex(columns=feat_names, fill_value=0.0).to_numpy(dtype=np.float32)
        all_scores = model.predict(X_all)

        pct = float(np.sum(all_scores < ticker_score) / len(all_scores) * 100)
        return round(pct, 1)

    # ── 价格信息 ──────────────────────────────────────────────────────────────

    def _get_price_info(self, ticker: str) -> Dict[str, Any]:
        """从 daily_prices_panel.parquet 读取价格相关指标."""
        result = {
            "current_price":        0.0,
            "price_change_1d":      0.0,
            "price_change_5d":      0.0,
            "ma20":                 0.0,
            "ma60":                 0.0,
            "drawdown_from_high_52w": 0.0,
            "volume_ratio":         1.0,
        }
        if not self._prices_path.exists():
            return result

        try:
            import pyarrow.parquet as pq
            schema_names = pq.ParquetFile(str(self._prices_path)).schema_arrow.names
            # 用 close（与市场行情一致）；adj_close 仅在 close 不存在时降级
            price_col = "close" if "close" in schema_names else "adj_close"
            date_col  = "trade_date" if "trade_date" in schema_names else "date"
            extra = [c for c in ["vol", "pct_chg"] if c in schema_names]

            df = pd.read_parquet(
                self._prices_path,
                filters=[("ts_code", "==", ticker)],
                columns=[date_col, price_col] + extra,
            )
            if df.empty:
                return result

            df = df.rename(columns={date_col: "trade_date", price_col: "close"})
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.sort_values("trade_date").tail(300)   # 最近300个交易日

            if len(df) < 2:
                return result

            close  = df["close"].astype(float)
            latest = float(close.iloc[-1])
            result["current_price"] = latest

            # 日涨跌幅
            if "pct_chg" in df.columns and pd.notna(df["pct_chg"].iloc[-1]):
                result["price_change_1d"] = float(df["pct_chg"].iloc[-1]) / 100.0
            elif len(close) >= 2:
                result["price_change_1d"] = (close.iloc[-1] / close.iloc[-2] - 1.0)

            # 5日涨跌幅
            if len(close) >= 6:
                result["price_change_5d"] = float(close.iloc[-1] / close.iloc[-6] - 1.0)

            # 均线
            if len(close) >= 20:
                result["ma20"] = float(close.tail(20).mean())
            if len(close) >= 60:
                result["ma60"] = float(close.tail(60).mean())

            # 52周高点回撤
            n252 = min(252, len(close))
            high_52w = float(close.tail(n252).max())
            if high_52w > 0:
                result["drawdown_from_high_52w"] = float(latest / high_52w - 1.0)

            # 量比（今日成交 / 20日均量）
            if "vol" in df.columns:
                vol = df["vol"].astype(float)
                if len(vol) >= 21 and float(vol.tail(20).mean()) > 0:
                    result["volume_ratio"] = float(vol.iloc[-1] / vol.tail(20).mean())

        except Exception as e:
            log.debug("[Scorer] 价格读取失败(%s): %s", ticker, e)

        return result

    def get_price_history(self, ticker: str, n: int = 90) -> pd.DataFrame:
        """返回最近 n 个交易日的收盘价序列，供 Streamlit 图表使用."""
        if not self._prices_path.exists():
            return pd.DataFrame()
        try:
            import pyarrow.parquet as pq
            schema_names = pq.ParquetFile(str(self._prices_path)).schema_arrow.names
            price_col = "close" if "close" in schema_names else "adj_close"
            date_col  = "trade_date" if "trade_date" in schema_names else "date"

            df = pd.read_parquet(
                self._prices_path,
                filters=[("ts_code", "==", ticker)],
                columns=[date_col, price_col],
            )
            if df.empty:
                return pd.DataFrame()
            df = df.rename(columns={date_col: "trade_date", price_col: "close"})
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.sort_values("trade_date").tail(n).reset_index(drop=True)
            return df
        except Exception as e:
            log.debug("[Scorer] 价格历史读取失败(%s): %s", ticker, e)
            return pd.DataFrame()

    # ── 6-Agent 辩论 ──────────────────────────────────────────────────────────

    def _run_debate(
        self, ticker: str, as_of: str, raw_factors: Dict[str, Any], mode: str
    ) -> Dict[str, Any]:
        """调用 DebateOrchestrator 运行 6-Agent 辩论."""
        result = {
            "signal": 0.0,
            "confidence": 0.5,
            "recommendation": "持有",
            "bull_reasons": [],
            "bear_reasons": [],
            "key_debate": "",
            "debate_summary": "",
        }
        try:
            from quantmind.agents.debate_orchestrator import DebateOrchestrator

            # 构建 context（与 daily_update.py 一致）
            context: Dict[str, Any] = {
                "ticker":          ticker,
                "as_of":           as_of,
                "news":            [],
                "reports":         [],
                "snapshot":        raw_factors,
                "lgbm_score":      0.0,
                "composite_score": 0.0,
                "regime":          "neutral",
            }
            for k, v in raw_factors.items():
                context[f"snapshot_{k}"] = v

            orch = DebateOrchestrator(
                ticker     = ticker,
                as_of      = as_of,
                context    = context,
                regime     = "neutral",
                agent_mode = mode,
                timeout    = 60.0 if mode == "full" else 30.0,
            )
            dr = orch.run_debate()

            # 解析 DebateResult
            avg_signal = getattr(dr, "avg_signal", 0.0)
            result["signal"]         = float(avg_signal)
            result["confidence"]     = float(dr.final_confidence)
            result["recommendation"] = dr.recommendation
            result["bull_reasons"]   = [
                s.argument for s in getattr(dr, "stances", []) if s.stance == "bull"
            ]
            result["bear_reasons"]   = [
                s.argument for s in getattr(dr, "stances", []) if s.stance == "bear"
            ]
            result["key_debate"]      = dr.key_debate_point
            result["debate_summary"]  = dr.debate_summary

        except Exception as e:
            log.warning("[Scorer] %s Agent 辩论失败: %s", ticker, e)

        return result

    # ── 懒加载 ────────────────────────────────────────────────────────────────

    def _load_alpha_panel(self) -> Optional[pd.DataFrame]:
        if self._alpha_panel is None and self._alpha_path.exists():
            try:
                self._alpha_panel = pd.read_parquet(self._alpha_path)
                self._alpha_tickers = set(
                    self._alpha_panel.index.get_level_values("ticker").unique()
                )
                log.info("[Scorer] alpha_panel 已加载: %s", self._alpha_panel.shape)
            except Exception as e:
                log.warning("[Scorer] alpha_panel 加载失败: %s", e)
        return self._alpha_panel
