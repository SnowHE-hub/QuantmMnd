"""quantmind.data.research_data — 投资研究系统专用深度数据提供者.

为 6 Agent 投资分析系统提供：快照上下文、详细财务、分析师评级、公告。
依赖 ChromaDB（可选）和 Tushare（需 Token）。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

_ROOT = Path(__file__).resolve().parent.parent.parent
_RESEARCH_DIR = _ROOT / "data" / "research"
_PRICES_DIR = _ROOT / "data" / "prices"


class ResearchDataProvider:
    """投资研究系统深度数据提供者，为 6 Agent 提供详细数据.

    注意：
    - TUSHARE_TOKEN 从环境变量读取，不打印
    - ChromaDB 不可用时自动降级到空 context
    """

    def __init__(
        self,
        chroma_dir: str = ".cache/chromadb",
        collection_name: str = "default",
        use_cache: bool = True,
    ) -> None:
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.use_cache = use_cache
        _RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

    # ── KB 上下文检索 ──────────────────────────────────────────────────────────

    def get_snapshot_context(self, ticker: str, as_of: str) -> dict:
        """获取股票 KB 快照上下文（复用 retrieve_stock_context 逻辑）.

        Returns:
            标准 context dict（含 news_context, report_context, snapshot_* 等）
        """
        try:
            from quantmind.kb.retriever import HybridRetriever
            from scripts.retrieve_stock_context import (
                SNAPSHOT_DOC_TYPES,
                retrieve_market_context,
                retrieve_news,
                retrieve_reports,
                retrieve_snapshot_by_doc_type,
            )
            from datetime import date as _date

            retriever = HybridRetriever(
                collection_name=self.collection_name,
                chroma_dir=self.chroma_dir,
                use_reranker=False,
            )
            if retriever.count() == 0:
                logger.warning(f"[ResearchData] ChromaDB 为空，{ticker} 使用空 context")
                return self._empty_context(ticker, as_of)

            as_of_date = _date.fromisoformat(as_of) if as_of else None
            news_items = retrieve_news(retriever, ticker, as_of_date, 5)
            report_items = retrieve_reports(retriever, ticker, as_of_date, 5)
            snapshot_results = {
                dt: retrieve_snapshot_by_doc_type(retriever, ticker, as_of_date, dt, 3)
                for dt in SNAPSHOT_DOC_TYPES
            }
            market_items = retrieve_market_context(retriever, as_of_date)

            ctx: dict[str, Any] = {
                "ticker": ticker,
                "as_of": as_of,
                "news_context": news_items,
                "news_count": len(news_items),
                "report_context": report_items,
                "report_count": len(report_items),
                "market_context": market_items,
            }
            for dt in SNAPSHOT_DOC_TYPES:
                ctx[f"snapshot_{dt}"] = snapshot_results[dt]
            return ctx

        except Exception as e:
            logger.warning(f"[ResearchData] KB 检索失败({e})，返回空 context")
            return self._empty_context(ticker, as_of)

    # ── 详细财务数据 ───────────────────────────────────────────────────────────

    def get_detailed_financials(self, ticker: str, periods: int = 8) -> pd.DataFrame:
        """获取多期财务数据（损益表 + 资产负债表 + 现金流）.

        数据来源（优先级）：
          1. data/research/{ticker}/financials.parquet
          2. Tushare API（需 TUSHARE_TOKEN）

        Args:
            ticker: 股票代码，如 600519.SH
            periods: 获取历史期数（最近 N 个季度）

        Returns:
            DataFrame，每行为一个报告期
        """
        cache_path = _RESEARCH_DIR / ticker / "financials.parquet"
        if self.use_cache and cache_path.exists():
            df = pd.read_parquet(cache_path)
            logger.info(f"[ResearchData] 财务数据缓存命中: {ticker} ({len(df)}期)")
            return df

        df = self._fetch_financials_tushare(ticker, periods)
        if df is not None and not df.empty:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)
        return df if df is not None else pd.DataFrame()

    def _fetch_financials_tushare(self, ticker: str, periods: int) -> pd.DataFrame | None:
        """从 Tushare 获取多期财务数据."""
        token = os.environ.get("TUSHARE_TOKEN", "")
        if not token:
            logger.warning("[ResearchData] TUSHARE_TOKEN 未设置，跳过财务数据获取")
            return None

        try:
            import tushare as ts
            pro = ts.pro_api(token)

            ts_code = _to_tushare_code(ticker)
            frames = []

            # 损益表
            try:
                income = pro.income(ts_code=ts_code, fields="ts_code,ann_date,f_ann_date,end_date,revenue,n_income,total_profit")
                if not income.empty:
                    income = income.head(periods)
                    frames.append(income)
                time.sleep(0.5)
            except Exception as e:
                logger.debug(f"[ResearchData] 损益表获取失败: {e}")

            # 资产负债表
            try:
                balance = pro.balancesheet(ts_code=ts_code, fields="ts_code,ann_date,end_date,total_assets,total_liab")
                if not balance.empty:
                    balance = balance.head(periods)
                    frames.append(balance)
                time.sleep(0.5)
            except Exception as e:
                logger.debug(f"[ResearchData] 资产负债表获取失败: {e}")

            if not frames:
                return None

            # 合并
            result = frames[0]
            for f in frames[1:]:
                merge_cols = ["ts_code", "end_date"]
                merge_cols = [c for c in merge_cols if c in result.columns and c in f.columns]
                if merge_cols:
                    result = result.merge(f, on=merge_cols, how="outer")

            return result

        except ImportError:
            logger.warning("[ResearchData] tushare 未安装")
            return None
        except Exception as e:
            logger.warning(f"[ResearchData] Tushare 财务数据获取失败: {e}")
            return None

    # ── 分析师评级 ─────────────────────────────────────────────────────────────

    def get_analyst_ratings(self, ticker: str, start_date: str = "2023-01-01") -> pd.DataFrame:
        """获取分析师评级历史.

        Args:
            ticker: 股票代码
            start_date: 开始日期

        Returns:
            DataFrame(date, analyst, rating, target_price, firm)
        """
        cache_path = _RESEARCH_DIR / ticker / "analyst_ratings.parquet"
        if self.use_cache and cache_path.exists():
            df = pd.read_parquet(cache_path)
            return df[df["date"] >= start_date] if "date" in df.columns else df

        token = os.environ.get("TUSHARE_TOKEN", "")
        if not token:
            return pd.DataFrame(columns=["date", "analyst", "rating", "target_price", "firm"])

        try:
            import tushare as ts
            pro = ts.pro_api(token)
            ts_code = _to_tushare_code(ticker)
            df = pro.report_rc(
                ts_code=ts_code,
                start_date=start_date.replace("-", ""),
                fields="ts_code,report_date,rating,target_price,analyst,org_name",
            )
            if df is not None and not df.empty:
                df = df.rename(columns={
                    "report_date": "date",
                    "org_name": "firm",
                })
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cache_path, index=False)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            logger.warning(f"[ResearchData] 分析师评级获取失败: {e}")
            return pd.DataFrame(columns=["date", "analyst", "rating", "target_price", "firm"])

    # ── 重大公告 ────────────────────────────────────────────────────────────────

    def get_announcements(self, ticker: str, start_date: str = "2023-01-01") -> list[dict]:
        """获取重大公告列表.

        Args:
            ticker: 股票代码（如 600519.SH）
            start_date: 开始日期

        Returns:
            list of dict(date, title, type)
        """
        try:
            import akshare as ak
            code = ticker.split(".")[0]
            df = ak.stock_notice_report(symbol=code)
            if df is None or df.empty:
                return []
            # 标准化列名
            col_map = {
                "公告日期": "date",
                "公告标题": "title",
                "公告类型": "type",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            if "date" in df.columns:
                df = df[pd.to_datetime(df["date"], errors="coerce") >= pd.Timestamp(start_date)]
            result = df[["date", "title", "type"]].head(20).to_dict("records")
            time.sleep(0.5)
            return result
        except Exception as e:
            logger.debug(f"[ResearchData] 公告获取失败({ticker}): {e}")
            return []

    # ── 辅助方法 ───────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_context(ticker: str, as_of: str) -> dict:
        """空 context 结构（KB 不可用时的降级）."""
        return {
            "ticker": ticker,
            "as_of": as_of,
            "news_context": [],
            "news_count": 0,
            "report_context": [],
            "report_count": 0,
            "market_context": [],
            "snapshot_company_profile": [],
            "snapshot_latest_market_metrics": [],
            "snapshot_financial_indicator_summary": [],
            "snapshot_northbound_summary": [],
            "snapshot_margin_summary": [],
        }


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def _to_tushare_code(ticker: str) -> str:
    """将标准 ticker 转换为 Tushare 格式（600519.SH → 600519.SH）."""
    return ticker  # Tushare 格式与标准格式一致
