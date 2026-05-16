"""quantmind.agents.tools.kb_tools — 知识库检索工具（Phase 5 实现）.

使用 HybridRetriever（ChromaDB Dense + BM25 Sparse + BGE-Reranker）。
知识库未初始化时优雅降级（返回空列表 + warning）。
"""

from __future__ import annotations

import threading
from datetime import date
from typing import Any

from loguru import logger

__all__ = [
    "search_research_reports",
    "search_news",
    "search_company_filings",
    "get_kb_retriever",
]

_RETRIEVER_LOCK = threading.Lock()
_RETRIEVER: Any | None = None  # HybridRetriever | None

# 控制是否使用实际知识库（False = 降级为 stub）
_KB_ENABLED = True


def get_kb_retriever(
    collection_name: str = "default",
    chroma_dir: str = ".cache/chromadb",
    use_reranker: bool = True,
) -> Any:
    """获取全局 HybridRetriever 实例（懒初始化，线程安全）."""
    global _RETRIEVER
    with _RETRIEVER_LOCK:
        if _RETRIEVER is None:
            try:
                from quantmind.kb.retriever import HybridRetriever
                _RETRIEVER = HybridRetriever(
                    collection_name=collection_name,
                    chroma_dir=chroma_dir,
                    use_reranker=use_reranker,
                )
                logger.debug(
                    f"[KBTools] HybridRetriever initialized: "
                    f"collection={collection_name}, count={_RETRIEVER.count()}"
                )
            except Exception as e:
                logger.warning(f"[KBTools] HybridRetriever 初始化失败: {e}，降级为 stub")
                _RETRIEVER = None
        return _RETRIEVER


def search_research_reports(
    query: str,
    date_range: tuple[date, date] | None = None,
    top_k: int = 5,
    ticker: str | None = None,
) -> list[dict[str, Any]]:
    """检索研究报告.

    Args:
        query:      检索关键词
        date_range: (start, end) 日期范围（end 作为 as_of PIT 过滤）
        top_k:      返回最多条数
        ticker:     可选：只返回指定股票代码

    Returns:
        list of dicts: [{text, source, published_date, title, score}]
    """
    if not _KB_ENABLED:
        logger.debug(f"search_research_reports('{query}') — KB disabled")
        return []

    retriever = get_kb_retriever()
    if retriever is None:
        return []

    try:
        as_of = date_range[1] if date_range else None
        results = retriever.retrieve(
            query=query,
            top_k=top_k,
            as_of=as_of,
            ticker_filter=ticker,
            source_type_filter="report",
        )
        return [
            {
                "text": r.text,
                "source": r.source,
                "published_date": r.published_date.isoformat() if r.published_date else "",
                "title": r.metadata.get("title", ""),
                "ticker": r.ticker,
                "score": r.score,
            }
            for r in results
        ]
    except Exception as e:
        logger.warning(f"[KBTools] search_research_reports 失败: {e}")
        return []


def search_news(
    query: str,
    date_range: tuple[date, date] | None = None,
    top_k: int = 10,
    ticker: str | None = None,
) -> list[dict[str, Any]]:
    """检索新闻.

    Args:
        query:      检索关键词或股票代码
        date_range: (start, end) 日期范围（end 作为 as_of PIT 过滤）
        top_k:      返回最多条数
        ticker:     可选：只返回指定股票代码

    Returns:
        list of dicts: [{text, source, published_date, title, ticker, score}]
    """
    if not _KB_ENABLED:
        logger.debug(f"search_news('{query}') — KB disabled")
        return []

    retriever = get_kb_retriever()
    if retriever is None:
        return []

    try:
        as_of = date_range[1] if date_range else None
        results = retriever.retrieve(
            query=query,
            top_k=top_k,
            as_of=as_of,
            ticker_filter=ticker,
            source_type_filter="news",
        )
        return [
            {
                "text": r.text,
                "source": r.source,
                "published_date": r.published_date.isoformat() if r.published_date else "",
                "title": r.metadata.get("title", ""),
                "ticker": r.ticker,
                "score": r.score,
            }
            for r in results
        ]
    except Exception as e:
        logger.warning(f"[KBTools] search_news 失败: {e}")
        return []


def search_company_filings(
    ticker: str,
    filing_type: str = "annual_report",
    as_of: date | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """检索公司公告/年报.

    Args:
        ticker:      股票代码
        filing_type: "annual_report" | "quarterly_report" | "announcement" | "filing"
        as_of:       截止日期（PIT）
        top_k:       返回最多条数

    Returns:
        list of dicts: [{text, source, published_date, title, score}]
    """
    if not _KB_ENABLED:
        logger.debug(f"search_company_filings('{ticker}', '{filing_type}') — KB disabled")
        return []

    retriever = get_kb_retriever()
    if retriever is None:
        return []

    try:
        query = f"{ticker} {filing_type} 财务报告 公告"
        results = retriever.retrieve(
            query=query,
            top_k=top_k,
            as_of=as_of,
            ticker_filter=ticker,
            source_type_filter="filing",
        )
        return [
            {
                "text": r.text,
                "source": r.source,
                "published_date": r.published_date.isoformat() if r.published_date else "",
                "title": r.metadata.get("title", ""),
                "ticker": r.ticker,
                "filing_type": r.metadata.get("filing_type", filing_type),
                "score": r.score,
            }
            for r in results
        ]
    except Exception as e:
        logger.warning(f"[KBTools] search_company_filings 失败: {e}")
        return []
