"""quantmind.selection.funnel_selector — 全市场漏斗选股系统（LazyDataEngine版）.

6 层过滤，从 ~5500 只精选至 10-20 只。
每层独立，可单独测试，支持跳过某层（skip_layers）。

核心特性：
  - 使用 LazyDataEngine 懒加载数据（逐层按需拉取）
  - Layer N 只为 Layer(N-1) 的幸存者拉取数据
  - Layer 3 复用 Layer 2 的价格缓存（零额外请求）
  - 返回 FunnelResult，含每层统计和缓存命中率

层次结构：
  Layer 1: 基础质量过滤（ST、上市天数、市值）— 数据来源：get_spot_data()
  Layer 2: 流动性过滤（成交额、自由流通市值）  — 数据来源：get_price_window(20日)
  Layer 3: 趋势过滤（MA20/60、回撤、RSI）      — 数据来源：复用Layer2价格缓存
  Layer 4: 基本面过滤（ROE、PE、营收增速）       — 数据来源：get_fundamentals_light()
  Layer 5: LightGBM 打分（Top 50）              — 数据来源：复用价格+基本面缓存
  Layer 6: LLM Reranker 精排（Top top_n）       — 数据来源：get_kb_context()
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from quantmind.selection.lazy_data_engine import LazyDataEngine

_ROOT = Path(__file__).resolve().parent.parent.parent
_CSI300_PRICE_FILE = _ROOT / "data" / "prices" / "csi300_daily_adj_close.parquet"
_FEATURE_DIR = _ROOT / "data" / "features"
_DEFAULT_MODEL = _ROOT / "models" / "lgbm_v1_final.pkl"


@dataclass
class LayerStat:
    """单层进出统计."""
    layer: int
    n_in: int
    n_out: int
    skipped: bool = False


@dataclass
class FunnelStats:
    """漏斗各层进出统计（兼容旧接口）."""
    layer1_in: int = 0
    layer1_out: int = 0
    layer2_in: int = 0
    layer2_out: int = 0
    layer3_in: int = 0
    layer3_out: int = 0
    layer4_in: int = 0
    layer4_out: int = 0
    layer5_in: int = 0
    layer5_out: int = 0
    layer6_in: int = 0
    layer6_out: int = 0
    skipped_layers: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "skipped_layers"}


@dataclass
class FunnelResult:
    """漏斗选股结果."""
    candidates: pd.DataFrame
    layer_stats: list[LayerStat]
    cache_stats: dict
    as_of: str
    universe: str

    def to_json(self) -> dict:
        stats_dict = {}
        for ls in self.layer_stats:
            stats_dict[f"layer{ls.layer}_in"] = ls.n_in
            stats_dict[f"layer{ls.layer}_out"] = ls.n_out

        cand_list = []
        for _, row in self.candidates.iterrows():
            c: dict[str, Any] = {"ticker": row.get("ticker", "")}
            if "name" in row and pd.notna(row.get("name")):
                c["name"] = str(row["name"])
            for col in ["layer5_rank", "lgbm_score", "pe_ttm", "pb", "roe_ttm"]:
                if col in row and pd.notna(row.get(col)):
                    val = row[col]
                    c[col] = round(float(val), 4) if isinstance(val, (float, np.floating)) else val
            if "lgbm_score" in c:
                c["score"] = c["lgbm_score"]
            cand_list.append(c)

        return {
            "as_of": self.as_of,
            "universe": self.universe,
            "funnel_stats": stats_dict,
            "cache_stats": self.cache_stats,
            "candidates": cand_list,
        }

    def summary(self) -> str:
        lines = [f"[FunnelResult] as_of={self.as_of}, universe={self.universe}"]
        for ls in self.layer_stats:
            flag = " (skipped)" if ls.skipped else ""
            lines.append(f"  Layer{ls.layer}: {ls.n_in} → {ls.n_out}{flag}")
        cr = self.cache_stats.get("hit_rate", 0)
        lines.append(f"  缓存命中率: {cr:.1%} (mem={self.cache_stats.get('mem_hits',0)}, disk={self.cache_stats.get('disk_hits',0)})")
        lines.append(f"  最终候选: {len(self.candidates)} 只")
        return "\n".join(lines)


class FunnelSelector:
    """全A股漏斗选股，6层过滤，从~5500只精选至10-20只.

    Args:
        as_of: 截止日期（YYYY-MM-DD）
        data_engine: LazyDataEngine 实例（None 时自动创建）
        lgbm_model_path: LightGBM 模型路径
        provider: LLM provider（none/dashscope/openai）
        model_name: LLM model 名称
        universe: 'full_a' | 'csi300' | 'csi1000'
    """

    def __init__(
        self,
        as_of: str,
        data_engine: LazyDataEngine | None = None,
        lgbm_model_path: str | Path | None = None,
        provider: str = "none",
        model_name: str = "qwen-plus",
        universe: str = "csi300",
        # 向后兼容旧参数
        price_panel_path: str | Path | None = None,
        fundamentals_path: str | Path | None = None,
        model_path: str | Path | None = None,
        custom_universe_file: str | Path | None = None,
    ) -> None:
        self.as_of = as_of
        self.as_of_ts = pd.Timestamp(as_of)
        self.data_engine = data_engine or LazyDataEngine(
            as_of,
            universe=universe,
            custom_universe_file=custom_universe_file,
        )
        if data_engine is not None:
            self.data_engine.universe = universe
        self.model_path = Path(lgbm_model_path or model_path or _DEFAULT_MODEL)
        self.provider = provider
        self.model_name = model_name
        self.universe = universe
        # 向后兼容
        self.price_panel_path = Path(price_panel_path) if price_panel_path else _CSI300_PRICE_FILE
        self.fundamentals_path = Path(fundamentals_path) if fundamentals_path else None
        self.stats = FunnelStats()

    # ── 主入口 ─────────────────────────────────────────────────────────────────

    def run(
        self,
        skip_layers: list[int] | None = None,
        top_n: int = 15,
        lgbm_top: int = 50,
        universe: str | None = None,
        top_n_final: int | None = None,
        max_tickers: int | None = None,
    ) -> "FunnelResult | tuple[pd.DataFrame, FunnelStats]":
        """运行完整漏斗.

        Args:
            skip_layers: 要跳过的层编号列表，例如 [4]
            top_n: 最终输出数量（Layer 6 输出）
            lgbm_top: Layer 5 LGBM 保留数量
            universe: 覆盖实例化时的 universe
            top_n_final: top_n 的别名（兼容 run_investment_pipeline.py）
            max_tickers: 测试模式：Layer1 入口随机抽样至多 N 只（random_state=42）

        Returns:
            FunnelResult（新接口）
        """
        if top_n_final is not None:
            top_n = top_n_final
        if universe is not None:
            self.universe = universe
            self.data_engine.universe = universe

        skip = set(skip_layers or [])
        self.stats.skipped_layers = list(skip)
        layer_stats: list[LayerStat] = []

        # ── Layer 1: 基础质量过滤（全量拉取一次） ────────────────────────────
        spot_df = self.data_engine.get_spot_data()
        if max_tickers is not None and len(spot_df) > max_tickers:
            spot_df = spot_df.sample(n=min(max_tickers, len(spot_df)), random_state=42).reset_index(
                drop=True
            )
            logger.info(
                "[Funnel] 测试模式: 随机抽取 {} 只作为 Layer1 入口（universe={}）",
                len(spot_df),
                self.universe,
            )
        survivors = self._build_initial_universe(spot_df)
        n_in = len(survivors)
        self.stats.layer1_in = n_in
        if 1 not in skip:
            survivors = self._filter_basic_quality(survivors)
        self.stats.layer1_out = len(survivors)
        layer_stats.append(LayerStat(1, n_in, len(survivors), 1 in skip))
        logger.info(f"[Funnel] Layer1 基础质量: {n_in} → {len(survivors)}")

        # ── Layer 2: 流动性过滤（只为 Layer1 存活者拉价格）──────────────────
        n_in = len(survivors)
        self.stats.layer2_in = n_in
        prices_20d: pd.DataFrame = pd.DataFrame()
        if 2 not in skip and n_in > 0:
            prices_20d = self.data_engine.get_price_window(survivors["ticker"].tolist(), 20)
            survivors = self._filter_liquidity(survivors, prices_20d)
        self.stats.layer2_out = len(survivors)
        layer_stats.append(LayerStat(2, n_in, len(survivors), 2 in skip))
        logger.info(f"[Funnel] Layer2 流动性: {n_in} → {len(survivors)}")

        # ── Layer 3: 趋势过滤（扩展到63日，复用Layer2价格缓存）──────────────
        n_in = len(survivors)
        self.stats.layer3_in = n_in
        prices_63d: pd.DataFrame = pd.DataFrame()
        if 3 not in skip and n_in > 0:
            # get_price_window 会命中内存缓存（不同 window_days → 不同 key，但同 ticker）
            prices_63d = self.data_engine.get_price_window(survivors["ticker"].tolist(), 63)
            survivors = self._filter_trend(survivors, prices_63d)
        self.stats.layer3_out = len(survivors)
        layer_stats.append(LayerStat(3, n_in, len(survivors), 3 in skip))
        logger.info(f"[Funnel] Layer3 趋势: {n_in} → {len(survivors)}")

        # ── Layer 4: 基本面过滤（只为 Layer3 存活者拉基本面）────────────────
        n_in = len(survivors)
        self.stats.layer4_in = n_in
        fundamentals: pd.DataFrame = pd.DataFrame()
        if 4 not in skip and n_in > 0:
            fundamentals = self.data_engine.get_fundamentals_light(survivors["ticker"].tolist())
            survivors = self._filter_fundamentals(survivors, fundamentals)
        self.stats.layer4_out = len(survivors)
        layer_stats.append(LayerStat(4, n_in, len(survivors), 4 in skip))
        logger.info(f"[Funnel] Layer4 基本面: {n_in} → {len(survivors)}")

        # ── Layer 5: LightGBM 打分（复用已缓存的价格+基本面）────────────────
        n_in = len(survivors)
        self.stats.layer5_in = n_in
        if 5 not in skip and n_in > 0:
            survivors = self._score_lgbm(survivors, prices_63d, fundamentals, top_k=lgbm_top)
        self.stats.layer5_out = len(survivors)
        layer_stats.append(LayerStat(5, n_in, len(survivors), 5 in skip))
        logger.info(f"[Funnel] Layer5 LGBM: {n_in} → {len(survivors)}")

        # ── Layer 6: LLM Reranker（只拉 Top50 的 KB）─────────────────────────
        n_in = len(survivors)
        self.stats.layer6_in = n_in
        dash_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        use_llm_rerank = (
            6 not in skip
            and n_in > 0
            and self.provider != "none"
            and (self.provider != "dashscope" or bool(dash_key))
        )
        if use_llm_rerank:
            kb_contexts = self.data_engine.get_kb_context(survivors["ticker"].tolist())
            survivors = self._rerank_llm(survivors, kb_contexts, top_n=top_n)
        else:
            if 6 not in skip and n_in > 0:
                logger.info(f"[Funnel] Layer6 LLM 不可用，直接取 Top{top_n}")
            survivors = survivors.head(top_n)
        self.stats.layer6_out = len(survivors)
        layer_stats.append(LayerStat(6, n_in, len(survivors), 6 in skip))
        logger.info(f"[Funnel] Layer6 LLM精排: {n_in} → {len(survivors)}")

        candidates = survivors.reset_index(drop=True)
        result = FunnelResult(
            candidates=candidates,
            layer_stats=layer_stats,
            cache_stats=self.data_engine.cache_stats(),
            as_of=self.as_of,
            universe=self.universe,
        )
        logger.info(result.summary())
        return result

    # ── Layer 实现 ─────────────────────────────────────────────────────────────

    def _build_initial_universe(self, spot_df: pd.DataFrame) -> pd.DataFrame:
        """从 spot_data 构建初始 universe.

        csi300: 直接使用本地价格面板构建，避免 AkShare ticker 格式不一致的问题。
        full_a/csi1000: 优先使用 AkShare 全量行情，无则使用本地回退。
        """
        if self.universe == "csi300":
            # CSI300 直接从本地价格面板读取，稳定可靠
            fallback = self._fallback_universe()
            if not fallback.empty:
                return fallback
            # 若本地文件也不存在，尝试从 spot_df 过滤
            if not spot_df.empty and "ticker" in spot_df.columns:
                csi300_tickers = self._get_csi300_tickers()
                if csi300_tickers:
                    filtered = spot_df[spot_df["ticker"].isin(csi300_tickers)].copy()
                    if not filtered.empty:
                        return filtered
            return fallback

        # full_a / csi1000：使用 AkShare 全量行情
        if spot_df.empty:
            logger.info("[Funnel] AkShare spot_data 不可用，使用本地回退")
            return self._fallback_universe()
        return spot_df

    def _fallback_universe(self) -> pd.DataFrame:
        """当 spot_data 不可用时，从本地价格面板构建 universe."""
        try:
            price_file = (
                _CSI300_PRICE_FILE
                if self.universe == "csi300"
                else _ROOT / "data" / "prices" / "full_market_daily_adj_close.parquet"
            )
            import pyarrow.parquet as pq
            schema = pq.read_schema(price_file)
            tickers = [f for f in schema.names if f.endswith((".SH", ".SZ", ".BJ"))]
            return pd.DataFrame({
                "ticker": tickers,
                "name": tickers,
                "list_date": "2015-01-01",
                "total_mv": 500000,
            })
        except Exception as e:
            logger.warning(f"[Funnel] fallback universe 加载失败: {e}")
            return pd.DataFrame(columns=["ticker", "name", "list_date", "total_mv"])

    def _get_csi300_tickers(self) -> list[str]:
        """从本地价格面板获取 CSI300 成分股列表."""
        try:
            import pyarrow.parquet as pq
            schema = pq.read_schema(_CSI300_PRICE_FILE)
            return [f for f in schema.names if f.endswith((".SH", ".SZ", ".BJ"))]
        except Exception:
            return []

    def _filter_basic_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Layer 1: 基础质量过滤（ST、上市天数、市值）."""
        mask = pd.Series(True, index=df.index)

        # 过滤 ST / *ST
        if "name" in df.columns:
            st_mask = df["name"].astype(str).str.contains(r"ST|\*", na=False)
            mask &= ~st_mask

        if "is_st" in df.columns:
            mask &= ~df["is_st"].fillna(False).astype(bool)

        # 上市天数 > 90 天（缺失上市日期的行保留，避免 Tushare fallback 被误杀）
        if "list_date" in df.columns:
            list_dates = pd.to_datetime(df["list_date"], errors="coerce")
            days_listed = (self.as_of_ts - list_dates).dt.days
            listing_ok = days_listed.isna() | (days_listed > 90)
            mask &= listing_ok

        # 市值过滤：仅在「有足够有效市值样本」时才启用，避免 total_mv=0 时全灭
        mv_col = "total_mv" if "total_mv" in df.columns else None
        if mv_col is None and "circ_mv" in df.columns:
            mv_col = "circ_mv"
        if mv_col:
            mv = pd.to_numeric(df[mv_col], errors="coerce")
            has_mv = mv.notna() & (mv > 0)
            if has_mv.sum() > 50:
                med = float(mv.loc[has_mv].median())
                threshold = 2e9 if med > 1e8 else 200000  # 元 vs 万元（与原逻辑一致）
                mask &= (~has_mv) | (mv >= threshold)

        result = df[mask].copy()
        logger.debug(f"  Layer1: {len(df)} → {len(result)}")
        return result

    def _filter_liquidity(
        self, df: pd.DataFrame, prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Layer 2: 流动性过滤（近20日有效价格天数）."""
        if prices is None or prices.empty:
            logger.warning("[Funnel] Layer2: 价格数据不可用，跳过流动性过滤")
            return df

        tickers_in_price = [t for t in df["ticker"].tolist() if t in prices.columns]
        if not tickers_in_price:
            logger.warning("[Funnel] Layer2: 无 ticker 在价格面板中，跳过")
            return df

        hist = prices[tickers_in_price][prices.index <= self.as_of_ts].tail(20)
        if len(hist) < 5:
            logger.warning("[Funnel] Layer2: 价格历史不足，跳过流动性过滤")
            return df

        valid_count = (hist > 0).sum()
        liquid_tickers = set(valid_count[valid_count >= 15].index.tolist())

        # 不在价格面板中的保留（无法判断）
        not_in_price = set(df["ticker"].tolist()) - set(tickers_in_price)
        keep = liquid_tickers | not_in_price
        result = df[df["ticker"].isin(keep)].copy()
        logger.debug(f"  Layer2: {len(df)} → {len(result)}")
        return result

    def _filter_trend(
        self, df: pd.DataFrame, prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Layer 3: 趋势过滤（MA20/MA60、回撤、RSI）."""
        if prices is None or prices.empty:
            logger.warning("[Funnel] Layer3: 价格数据不可用，跳过趋势过滤")
            return df

        tickers_in_price = [t for t in df["ticker"].tolist() if t in prices.columns]
        if not tickers_in_price:
            return df

        hist = prices[tickers_in_price][prices.index <= self.as_of_ts].tail(70)
        if len(hist) < 25:
            logger.warning("[Funnel] Layer3: 价格历史不足，跳过趋势过滤")
            return df

        passing = set()
        for ticker in tickers_in_price:
            p = hist[ticker].dropna()
            if len(p) < 25:
                continue
            try:
                ma20 = p.rolling(20).mean().iloc[-1]
                ma60 = p.rolling(60).mean().iloc[-1] if len(p) >= 60 else p.mean()
                close = p.iloc[-1]
                if close <= ma20:
                    continue
                if ma20 <= ma60:
                    continue
                recent = p.tail(20)
                roll_max = recent.expanding().max()
                dd = abs(((recent - roll_max) / roll_max).min())
                if dd > 0.25:
                    continue
                rsi = _calc_rsi(p.tail(20))
                if rsi is not None and not (20 <= rsi <= 80):
                    continue
                passing.add(ticker)
            except Exception:
                pass

        not_in_price = set(df["ticker"].tolist()) - set(tickers_in_price)
        keep = passing | not_in_price
        result = df[df["ticker"].isin(keep)].copy()
        logger.debug(f"  Layer3: {len(df)} → {len(result)}")
        return result

    def _filter_fundamentals(
        self, df: pd.DataFrame, fundamentals: pd.DataFrame
    ) -> pd.DataFrame:
        """Layer 4: 基本面过滤（ROE、PE、营收增速）."""
        if fundamentals is None or fundamentals.empty:
            logger.warning("[Funnel] Layer4: 基本面数据不可用，跳过")
            return df

        fund = fundamentals.copy()
        if "ticker" not in fund.columns:
            logger.warning("[Funnel] Layer4: 基本面缺少 ticker 列，跳过")
            return df

        # 仅保留 df 中没有的列，避免重复
        existing_cols = [c for c in df.columns if c != "ticker"]
        fund_extra_cols = [c for c in ["roe_ttm", "pe_ttm", "pb", "revenue_yoy"]
                           if c in fund.columns and c not in existing_cols]
        if not fund_extra_cols:
            # 所有基本面列已在 df 中，直接使用
            merged = df
        else:
            merged = df.merge(fund[["ticker"] + fund_extra_cols], on="ticker", how="left")

        mask = pd.Series(True, index=merged.index)

        roe_col = "roe_ttm"
        if roe_col in merged.columns:
            roe = pd.to_numeric(merged[roe_col], errors="coerce")
            roe_pct = roe.where(roe.abs() > 1, roe * 100)
            mask &= roe_pct.isna() | (roe_pct > 8)

        pe_col = "pe_ttm"
        if pe_col in merged.columns:
            pe = pd.to_numeric(merged[pe_col], errors="coerce")
            mask &= pe.isna() | ((pe > 0) & (pe < 100))

        if "revenue_yoy" in merged.columns:
            rev = pd.to_numeric(merged["revenue_yoy"], errors="coerce")
            rev_pct = rev.where(rev.abs() > 1, rev * 100)
            mask &= rev_pct.isna() | (rev_pct > -30)

        passing = set(merged.loc[mask, "ticker"].tolist())
        result = df[df["ticker"].isin(passing)].copy()
        logger.debug(f"  Layer4: {len(df)} → {len(result)}")
        return result

    def _score_mv_fallback(self, df: pd.DataFrame, top_k: int) -> pd.DataFrame:
        """按总市值降序取 Top-K；分数做保序分位唯一化。"""
        from quantmind.utils.score_order import order_preserving_pct_rank

        d = df.copy().reset_index(drop=True)
        mv_col = "total_mv" if "total_mv" in d.columns else None
        if mv_col is None and "circ_mv" in d.columns:
            mv_col = "circ_mv"
        if mv_col:
            mv = pd.to_numeric(d[mv_col], errors="coerce").fillna(0.0)
        else:
            mv = pd.Series(range(len(d), 0, -1), dtype=float)
        d["lgbm_score"] = mv.astype(float).values
        d = d.sort_values("lgbm_score", ascending=False).head(top_k).reset_index(drop=True)
        d["lgbm_score"] = order_preserving_pct_rank(d["lgbm_score"], higher_is_better=True).fillna(0.5).values
        d["layer5_rank"] = range(1, len(d) + 1)
        return d

    def _score_lgbm(
        self,
        df: pd.DataFrame,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame,
        top_k: int = 50,
    ) -> pd.DataFrame:
        """Layer 5: LightGBM 打分并选 Top-K（复用已缓存的价格+基本面）."""
        if not self.model_path.exists():
            logger.warning(f"[Funnel] Layer5: 模型文件不存在 {self.model_path}，总市值降级")
            return self._score_mv_fallback(df, top_k)

        try:
            from quantmind.models.factor_model import FactorModel

            model = FactorModel.load(self.model_path)
            feat_names = model.feature_names
            if not feat_names:
                logger.warning("[Funnel] Layer5: 模型无特征列表，总市值降级")
                return self._score_mv_fallback(df, top_k)

            feat_df = self._load_features_for_tickers(df["ticker"].tolist())
            if feat_df is None:
                return self._score_mv_fallback(df, top_k)

            feat_df = feat_df.reindex(index=df["ticker"].values)
            available = [f for f in feat_names if f in feat_df.columns]
            if len(available) < 5:
                logger.warning(
                    f"[Funnel] Layer5: 可用特征仅 {len(available)} 个（<5），按总市值降级排序"
                )
                return self._score_mv_fallback(df, top_k)

            for f in feat_names:
                if f not in feat_df.columns:
                    feat_df[f] = np.nan

            X = feat_df[list(feat_names)].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            scores = model.predict(X)

            df = df.copy().reset_index(drop=True)
            # 使用保序百分位归一化，消除 LambdaRank 分数退化（多只股票同分）
            from quantmind.utils.score_order import order_preserving_pct_rank
            df["lgbm_score"] = order_preserving_pct_rank(scores)
            df = df.sort_values("lgbm_score", ascending=False).head(top_k).reset_index(drop=True)
            df["layer5_rank"] = range(1, len(df) + 1)
            return df

        except Exception as e:
            logger.warning(f"[Funnel] Layer5: LGBM打分失败({e})，总市值降级")
            return self._score_mv_fallback(df, top_k)

    def _score_rule_based(
        self, df: pd.DataFrame, fundamentals: pd.DataFrame, top_k: int
    ) -> pd.DataFrame:
        """规则打分降级方案."""
        df = df.copy().reset_index(drop=True)
        score = pd.Series(0.5, index=df.index)
        roe_col = "roe_ttm"
        if roe_col in df.columns:
            score += pd.to_numeric(df[roe_col], errors="coerce").fillna(0) * 0.01
        elif fundamentals is not None and not fundamentals.empty and roe_col in fundamentals.columns:
            roe_map = fundamentals.set_index("ticker")[roe_col].to_dict()
            score += df["ticker"].map(roe_map).fillna(0) * 0.01
        df["lgbm_score"] = score.values
        df = df.sort_values("lgbm_score", ascending=False).head(top_k).reset_index(drop=True)
        df["layer5_rank"] = range(1, len(df) + 1)
        return df

    def _rerank_llm(
        self,
        candidates: pd.DataFrame,
        kb_contexts: dict[str, dict],
        top_n: int = 15,
    ) -> pd.DataFrame:
        """Layer 6: LLM Reranker 精排（使用已获取的 KB 上下文）."""
        if self.provider == "none" or len(candidates) == 0:
            return candidates.head(top_n)

        try:
            from quantmind.models.llm_reranker import LLMListwiseReranker, RerankCandidate

            reranker_candidates = []
            for _, row in candidates.iterrows():
                kf = {}
                for col in ["pe_ttm", "pb", "roe_ttm", "momentum_6m"]:
                    if col in row and pd.notna(row.get(col)):
                        kf[col] = float(row[col])
                c = RerankCandidate(
                    ticker=row["ticker"],
                    lgbm_score=float(row.get("lgbm_score", 0.5)),
                    lgbm_rank=int(row.get("layer5_rank", 1)),
                    key_factors=kf,
                    shap_contributions={},
                )
                reranker_candidates.append(c)

            reranker = LLMListwiseReranker(
                provider=self.provider,
                model=self.model_name,
                top_n=top_n,
            )
            result = reranker.rerank(
                candidates=reranker_candidates,
                as_of=self.as_of,
                universe=self.universe,
            )

            if result and result.rankings:
                ranked_tickers = [r.ticker for r in result.rankings[:top_n]]
                reranked = candidates[candidates["ticker"].isin(ranked_tickers)].copy()
                order_map = {t: i for i, t in enumerate(ranked_tickers)}
                reranked["llm_rank"] = reranked["ticker"].map(order_map)
                reranked = reranked.sort_values("llm_rank").drop(columns=["llm_rank"])
                return reranked.reset_index(drop=True)

        except Exception as e:
            logger.warning(f"[Funnel] Layer6: LLM精排失败({e})，保留LGBM顺序")

        return candidates.head(top_n)

    # ── 辅助方法 ───────────────────────────────────────────────────────────────

    def _load_features_for_tickers(self, tickers: list[str]) -> pd.DataFrame | None:
        """为指定 ticker 列表加载特征数据."""
        feat_file = _FEATURE_DIR / f"{self.universe}_{self.as_of}.parquet"
        if feat_file.exists():
            df = pd.read_parquet(feat_file)
            return df.set_index("ticker") if "ticker" in df.columns else df

        feat_files = sorted(_FEATURE_DIR.glob("*.parquet"))
        if not feat_files:
            return None
        try:
            df = pd.read_parquet(feat_files[-1])
            if df.index.nlevels > 1:
                dates = df.index.get_level_values(0).unique()
                closest = min(dates, key=lambda d: abs((pd.Timestamp(d) - self.as_of_ts).days))
                df = df.xs(closest, level=0)
            if df.index.name != "ticker":
                df.index.name = "ticker"
            return df
        except Exception as e:
            logger.warning(f"[Funnel] 特征文件加载失败: {e}")
            return None

    def to_output_json(
        self, candidates: pd.DataFrame, stats: "FunnelStats | FunnelResult"
    ) -> dict:
        """生成标准输出 JSON（兼容旧接口）."""
        if isinstance(stats, FunnelResult):
            return stats.to_json()

        stats_dict = {}
        for i in range(1, 7):
            stats_dict[f"layer{i}_in"] = getattr(stats, f"layer{i}_in", 0)
            stats_dict[f"layer{i}_out"] = getattr(stats, f"layer{i}_out", 0)

        cand_list = []
        for _, row in candidates.iterrows():
            c: dict[str, Any] = {"ticker": row["ticker"]}
            if "name" in row and pd.notna(row.get("name")):
                c["name"] = str(row["name"])
            for col in ["layer5_rank", "lgbm_score", "pe_ttm", "pb", "roe_ttm"]:
                if col in row and pd.notna(row.get(col)):
                    val = row[col]
                    c[col] = round(float(val), 4) if isinstance(val, (float, np.floating)) else val
            if "lgbm_score" in c:
                c["score"] = c["lgbm_score"]
            cand_list.append(c)

        return {
            "as_of": self.as_of,
            "universe": self.universe,
            "funnel_stats": stats_dict,
            "skipped_layers": getattr(stats, "skipped_layers", []),
            "candidates": cand_list,
        }


# ── 模块级工具函数 ─────────────────────────────────────────────────────────────

def _calc_rsi(series: pd.Series, period: int = 5) -> float | None:
    if len(series) < period + 1:
        return None
    delta = series.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean().iloc[-1]
    avg_loss = loss.rolling(period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))
