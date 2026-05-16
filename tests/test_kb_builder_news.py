"""KBBuilder news pipeline: field compatibility, chunking, dedup (no AkShare / BGE)."""

from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

from loguru import logger

from quantmind.kb.builder import (
    KBBuilder,
    analyze_raw_news_records,
)
from quantmind.kb.chunker import SmartChunker


def _chunker() -> SmartChunker:
    return SmartChunker(chunk_size=200, chunk_overlap=40)


def test_chinese_column_news_produces_chunks():
    news_list = [
        {
            "标题": "外媒称宁德时代",
            "内容": "这是一条测试正文，用于验证中文列名。",
            "发布时间": "2024-05-01 10:00:00",
            "链接": "https://example.com/1",
        }
    ]
    as_of = date(2024, 6, 1)
    stats = analyze_raw_news_records(
        news_list,
        "300750.SZ",
        as_of - timedelta(days=120),
        as_of,
        _chunker(),
        seen_hashes=None,
        enable_dedup=True,
    )
    assert stats["empty_text_docs"] == 0
    assert stats["len_chunks"] >= 1


def test_em_style_columns_news_title_content():
    """东方财富 / AkShare 常见列名：新闻标题、新闻内容."""
    news_list = [
        {
            "新闻标题": " company report",
            "新闻内容": "正文段落用于测试。",
            "发布时间": "2024-05-01 10:00:00",
            "新闻链接": "https://example.com/x",
        }
    ]
    as_of = date(2024, 6, 1)
    stats = analyze_raw_news_records(
        news_list,
        "300750.SZ",
        as_of - timedelta(days=120),
        as_of,
        _chunker(),
        seen_hashes=None,
        enable_dedup=True,
    )
    assert stats["empty_text_docs"] == 0
    assert stats["len_chunks"] >= 1


def test_title_only_produces_chunk():
    news_list = [
        {"title": "仅标题新闻", "publish_time": "2024-05-01"},
    ]
    as_of = date(2024, 6, 1)
    stats = analyze_raw_news_records(
        news_list,
        "600519.SH",
        as_of - timedelta(days=120),
        as_of,
        _chunker(),
        seen_hashes=None,
        enable_dedup=True,
    )
    assert stats["empty_text_docs"] == 0
    assert stats["len_chunks"] == 1


def test_empty_title_and_body_zero_chunks():
    news_list = [
        {"title": "", "content": "", "publish_time": "2024-05-01"},
    ]
    as_of = date(2024, 6, 1)
    stats = analyze_raw_news_records(
        news_list,
        "600519.SH",
        as_of - timedelta(days=120),
        as_of,
        _chunker(),
        seen_hashes=None,
        enable_dedup=True,
    )
    assert stats["empty_text_docs"] == 1
    assert stats["len_chunks"] == 0


def test_duplicate_news_deduped():
    news_list = [
        {"title": "重复", "content": "同样正文", "publish_time": "2024-05-01"},
        {"title": "重复", "content": "同样正文", "publish_time": "2024-05-02"},
    ]
    as_of = date(2024, 6, 1)
    stats = analyze_raw_news_records(
        news_list,
        "600519.SH",
        as_of - timedelta(days=120),
        as_of,
        _chunker(),
        seen_hashes=None,
        enable_dedup=True,
    )
    assert stats["docs_after_date_filter"] == 2
    assert stats["docs_after_dedup"] == 1
    assert stats["dedup_dropped"] == 1


def test_debug_reports_path_never_loads_embedding(tmp_path, monkeypatch):
    from quantmind.kb import embedding_service

    def boom(*_a, **_k):
        raise RuntimeError("embedding must not load in debug reports")

    monkeypatch.setattr(embedding_service.EmbeddingService, "_get_model", boom)
    p = tmp_path / "note.txt"
    p.write_text("段落 " * 50, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        "debug_kb_sources",
        Path(__file__).resolve().parents[1] / "scripts" / "debug_kb_sources.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.run_reports(tmp_path, date(2099, 1, 1))


def test_build_from_news_warns_when_add_chunks_returns_zero(monkeypatch):
    builder = KBBuilder(chunker=_chunker())
    monkeypatch.setattr(
        builder,
        "_fetch_news",
        lambda *_a, **_k: [
            {
                "title": "T",
                "content": "C",
                "publish_time": "2024-05-01",
                "url": "u",
            }
        ],
    )
    ret = MagicMock()
    ret.add_chunks.return_value = 0
    builder._retriever = ret

    sinked: list[str] = []

    def sink(message):
        sinked.append(message)

    hid_wrapped = logger.add(sink, level="WARNING")

    try:
        builder.build_from_news(["600519.SH"], as_of=date(2024, 6, 1), days_back=200)
    finally:
        logger.remove(hid_wrapped)

    ret.add_chunks.assert_called_once()
    assert any("add_chunks 返回 0" in m for m in sinked)
