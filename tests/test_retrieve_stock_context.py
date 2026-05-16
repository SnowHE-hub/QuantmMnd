"""Tests for scripts/retrieve_stock_context.py and HybridRetriever.fetch_by_metadata.

覆盖：
1. 能按 ticker 过滤
2. 能按 source_type 分类
3. 能按 doc_type 分类 snapshot
4. 某类为空时不崩溃
5. JSON 输出结构稳定
6. Markdown 输出包含各 section
7. 不加载 reranker
8. 不读取密钥
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── 内联脚本中的纯函数（无 I/O 依赖）──────────────────────────────────────────


from scripts.retrieve_stock_context import (
    SNAPSHOT_DOC_TYPES,
    _post_filter,
    _preview,
    _result_dict,
    _ticker_variants,
    build_markdown,
)
from quantmind.kb.retriever import HybridRetriever, RetrievalResult


# ── 辅助工厂 ───────────────────────────────────────────────────────────────────


def _make_result(
    ticker: str = "600519.SH",
    source_type: str = "snapshot",
    doc_type: str = "company_profile",
    published_date: str = "2024-12-31",
    score: float = 0.9,
    text: str = "some text",
) -> RetrievalResult:
    return RetrievalResult(
        text=text,
        metadata={
            "ticker": ticker,
            "source_type": source_type,
            "doc_type": doc_type,
            "published_date": published_date,
            "source": f"snapshot:{ticker}:{doc_type}",
            "as_of": published_date,
            "title": f"{ticker} {doc_type}",
        },
        score=score,
    )


# ── 1. _post_filter: 按 ticker 过滤 ────────────────────────────────────────────


def test_post_filter_ticker() -> None:
    results = [
        _make_result(ticker="600519.SH"),
        _make_result(ticker="300750.SZ"),
        _make_result(ticker="000001.SZ"),
    ]
    filtered = _post_filter(results, ticker="600519.SH")
    assert len(filtered) == 1
    assert filtered[0].metadata["ticker"] == "600519.SH"


# ── 2. _post_filter: 按 source_type 过滤 ──────────────────────────────────────


def test_post_filter_source_type() -> None:
    results = [
        _make_result(source_type="news", doc_type=""),
        _make_result(source_type="snapshot", doc_type="company_profile"),
        _make_result(source_type="report", doc_type=""),
    ]
    snap = _post_filter(results, source_type="snapshot")
    assert all(r.metadata["source_type"] == "snapshot" for r in snap)
    assert len(snap) == 1


# ── 3. _post_filter: 按 doc_type 过滤 snapshot ──────────────────────────────────


def test_post_filter_doc_type() -> None:
    results = [
        _make_result(doc_type="company_profile"),
        _make_result(doc_type="financial_indicator_summary"),
        _make_result(doc_type="northbound_summary"),
    ]
    cp = _post_filter(results, doc_type="company_profile")
    assert len(cp) == 1
    assert cp[0].metadata["doc_type"] == "company_profile"


def test_post_filter_doc_type_missing_passes_not() -> None:
    """doc_type 为空的文档在 doc_type filter 时被排除."""
    results = [_make_result(doc_type="")]
    filtered = _post_filter(results, doc_type="company_profile")
    assert filtered == []


# ── 4. 某类为空时不崩溃 ────────────────────────────────────────────────────────


def test_empty_category_no_crash() -> None:
    ctx: dict[str, Any] = {
        "ticker": "999999.SH",
        "as_of": "2024-12-31",
        "collection_count": 1000,
        "news_context": [],
        "news_count": 0,
        "report_context": [],
        "report_count": 0,
        "market_context": [],
        "market_count": 0,
    }
    for dt in SNAPSHOT_DOC_TYPES:
        ctx[f"snapshot_{dt}"] = []
        ctx[f"snapshot_{dt}_count"] = 0

    md = build_markdown("999999.SH", date(2024, 12, 31), ctx)
    assert "999999.SH" in md
    assert "（无结果）" in md


# ── 5. JSON 输出结构稳定 ──────────────────────────────────────────────────────


def test_result_dict_stable_keys() -> None:
    r = _make_result()
    d = _result_dict(1, r)
    required_keys = {
        "rank", "score", "ticker", "source_type", "doc_type",
        "as_of", "published_date", "source", "title",
        "text_preview", "text",
    }
    assert required_keys.issubset(d.keys())
    assert d["rank"] == 1
    assert isinstance(d["score"], float)


def test_json_serializable() -> None:
    ctx: dict[str, Any] = {
        "ticker": "600519.SH",
        "as_of": "2024-12-31",
        "collection_count": 20966,
        "news_context": [_result_dict(1, _make_result(source_type="news", doc_type=""))],
        "news_count": 1,
        "report_context": [],
        "report_count": 0,
        "market_context": [],
        "market_count": 0,
    }
    for dt in SNAPSHOT_DOC_TYPES:
        ctx[f"snapshot_{dt}"] = []
        ctx[f"snapshot_{dt}_count"] = 0

    j = json.dumps(ctx, ensure_ascii=False)
    parsed = json.loads(j)
    assert parsed["ticker"] == "600519.SH"
    assert parsed["news_count"] == 1


# ── 6. Markdown 输出包含各 section ─────────────────────────────────────────────


def test_markdown_contains_all_sections() -> None:
    ctx: dict[str, Any] = {
        "ticker": "600519.SH",
        "as_of": "2024-12-31",
        "collection_count": 20966,
        "news_context": [],
        "news_count": 0,
        "report_context": [],
        "report_count": 0,
        "market_context": [],
        "market_count": 0,
    }
    for dt in SNAPSHOT_DOC_TYPES:
        ctx[f"snapshot_{dt}"] = []
        ctx[f"snapshot_{dt}_count"] = 0

    md = build_markdown("600519.SH", date(2024, 12, 31), ctx)
    assert "News Context" in md
    assert "Report Context" in md
    assert "Snapshot Context" in md
    assert "Market Index Context" in md
    for dt in SNAPSHOT_DOC_TYPES:
        assert dt in md


# ── 7. 不加载 reranker ─────────────────────────────────────────────────────────


def test_retriever_use_reranker_false(tmp_path: Path) -> None:
    r = HybridRetriever(
        collection_name="test_no_reranker",
        chroma_dir=str(tmp_path / "chroma"),
        use_reranker=False,
    )
    assert r.use_reranker is False
    assert r._reranker is None


# ── 8. 不读取密钥 ─────────────────────────────────────────────────────────────


def test_no_secret_access(tmp_path: Path) -> None:
    """retrieve_stock_context 不读取 .env 或任何密钥文件."""
    import scripts.retrieve_stock_context as mod
    import inspect
    src = inspect.getsource(mod)
    forbidden = [".env", "OPENAI_API_KEY", "api_key", "secret_key", "password"]
    for kw in forbidden:
        assert kw not in src, f"Script must not reference {kw!r}"


# ── fetch_by_metadata 单元测试 ─────────────────────────────────────────────────


def test_fetch_by_metadata_empty_collection(tmp_path: Path) -> None:
    """空 collection 上 fetch_by_metadata 返回空列表而不崩溃."""
    r = HybridRetriever(
        collection_name="test_fetch_meta",
        chroma_dir=str(tmp_path / "chroma"),
        use_reranker=False,
    )
    # 空 collection，where 过滤应返回 []
    results = r.fetch_by_metadata(where={"ticker": {"$eq": "600519.SH"}}, as_of=None, limit=10)
    assert results == []


def test_fetch_by_metadata_with_data(tmp_path: Path) -> None:
    """插入一条快照文档后，fetch_by_metadata 能精确检索。"""
    from quantmind.kb.chunker import Chunk

    r = HybridRetriever(
        collection_name="test_fetch_meta_data",
        chroma_dir=str(tmp_path / "chroma"),
        use_reranker=False,
    )

    chunk = Chunk(
        text="公司概览：贵州茅台 行业：白酒",
        metadata={
            "ticker": "600519.SH",
            "source_type": "snapshot",
            "doc_type": "company_profile",
            "published_date": "2024-12-31",
            "as_of": "2024-12-31",
            "source": "snapshot:600519.SH:company_profile",
            "title": "贵州茅台 公司概览",
        },
        chunk_idx=0,
    )
    r.add_chunks([chunk])

    # fetch by ticker + doc_type
    results = r.fetch_by_metadata(
        where={
            "$and": [
                {"ticker": {"$eq": "600519.SH"}},
                {"doc_type": {"$eq": "company_profile"}},
            ]
        },
        as_of=date(2024, 12, 31),
        limit=5,
    )
    assert len(results) == 1
    assert results[0].metadata["ticker"] == "600519.SH"
    assert results[0].metadata["doc_type"] == "company_profile"


def test_fetch_by_metadata_pit_filter(tmp_path: Path) -> None:
    """as_of 过滤：published_date 超过 as_of 的文档被排除。"""
    from quantmind.kb.chunker import Chunk

    r = HybridRetriever(
        collection_name="test_pit",
        chroma_dir=str(tmp_path / "chroma"),
        use_reranker=False,
    )

    for pub_date, label in [("2024-06-30", "past"), ("2025-06-30", "future")]:
        r.add_chunks([Chunk(
            text=f"财务数据 {label}",
            metadata={
                "ticker": "600519.SH",
                "source_type": "snapshot",
                "doc_type": "financial_indicator_summary",
                "published_date": pub_date,
                "as_of": pub_date,
                "source": f"snapshot:600519.SH:{label}",
                "title": f"test {label}",
            },
            chunk_idx=0,
        )])

    results = r.fetch_by_metadata(
        where={
            "$and": [
                {"ticker": {"$eq": "600519.SH"}},
                {"doc_type": {"$eq": "financial_indicator_summary"}},
            ]
        },
        as_of=date(2024, 12, 31),
        limit=10,
    )
    # Only past doc should be returned
    assert len(results) == 1
    assert results[0].metadata["published_date"] == "2024-06-30"


# ── _ticker_variants ──────────────────────────────────────────────────────────


def test_ticker_variants() -> None:
    assert "600519.SH" in _ticker_variants("600519.SH")
    assert "600519" in _ticker_variants("600519.SH")
    assert "300750.SZ" in _ticker_variants("300750.SZ")
    assert "300750" in _ticker_variants("300750.SZ")


# ── _preview 截断 ─────────────────────────────────────────────────────────────


def test_preview_truncation() -> None:
    long_text = "a" * 200
    p = _preview(long_text, max_len=50)
    assert len(p) == 50
    assert p.endswith("...")


def test_preview_short_text() -> None:
    short = "hello world"
    assert _preview(short) == "hello world"
