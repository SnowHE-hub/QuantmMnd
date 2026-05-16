"""Tests for scripts/batch_generate_reports.py — no API keys, no Chroma."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from scripts import batch_generate_reports as bgr


def _fake_context(ticker: str = "TEST.SZ") -> dict:
    return {
        "ticker": ticker,
        "as_of": "2024-12-31",
        "collection_count": 1,
        "news_context": [{"rank": 1}],
        "news_count": 1,
        "report_context": [],
        "report_count": 0,
        "market_context": [],
        "market_count": 0,
        **{f"snapshot_{dt}": [{"rank": 1}] for dt in bgr.SNAPSHOT_DOC_TYPES},
        **{f"snapshot_{dt}_count": 1 for dt in bgr.SNAPSHOT_DOC_TYPES},
    }


def _empty_context(ticker: str = "EMPTY.SZ") -> dict:
    d = {
        "ticker": ticker,
        "as_of": "2024-12-31",
        "collection_count": 1,
        "news_context": [],
        "news_count": 0,
        "report_context": [],
        "report_count": 0,
        "market_context": [],
        "market_count": 0,
    }
    for dt in bgr.SNAPSHOT_DOC_TYPES:
        d[f"snapshot_{dt}"] = []
        d[f"snapshot_{dt}_count"] = 0
    return d


def test_parse_ticker_file_comments_and_blanks(tmp_path: Path) -> None:
    p = tmp_path / "t.txt"
    p.write_text(
        "\n  600519.SH  \n\n# skip\n000001.SZ\n# x\n",
        encoding="utf-8",
    )
    assert bgr.parse_ticker_file(p) == ["600519.SH", "000001.SZ"]


def test_single_ticker_writes_report_and_context(tmp_path: Path) -> None:
    out = tmp_path / "out"
    ctx_dir = tmp_path / "ctx"
    sj = tmp_path / "sum.json"
    sm = tmp_path / "sum.md"
    br = tmp_path / "batch.md"

    def fake_retrieve(ns: SimpleNamespace) -> dict:
        assert ns.ticker == "AAA.SH"
        return _fake_context("AAA.SH")

    agent = MagicMock()

    def factory():
        return agent

    agent.generate_report.return_value = {
        "ticker": "AAA.SH",
        "as_of": "2024-12-31",
        "report_markdown": "# AAA.SH 报告\n",
        "data_coverage": {},
        "warnings": [],
        "used_sources": [],
        "llm_used": False,
        "provider": "none",
        "model": "template",
    }

    ns = SimpleNamespace(
        tickers=["AAA.SH"],
        ticker_file=None,
        as_of="2024-12-31",
        provider="none",
        model="qwen-plus",
        output_dir=out,
        context_dir=ctx_dir,
        language="zh",
        top_k_news=5,
        top_k_reports=5,
        top_k_per_doc_type=3,
        workers=1,
        skip_existing=False,
        context_only=False,
        collection_name="default",
        chroma_dir=".cache/chromadb",
        summary_json=sj,
        summary_md=sm,
        batch_report_md=br,
    )

    summ = bgr.run_batch(ns, retrieve_fn=fake_retrieve, agent_factory=factory)
    safe = "AAA_SH"
    assert (out / f"{safe}_report.md").read_text(encoding="utf-8").startswith("# AAA.SH")
    ctx = json.loads((ctx_dir / f"{safe}_context.json").read_text(encoding="utf-8"))
    assert ctx["news_count"] == 1
    assert summ["counts"]["success"] == 1
    assert summ["counts"]["failed"] == 0
    data = json.loads(sj.read_text(encoding="utf-8"))
    assert "per_ticker" in data and "counts" in data
    assert len(data["per_ticker"]) == 1
    assert data["per_ticker"][0]["ticker"] == "AAA.SH"
    agent.generate_report.assert_called_once()


def test_empty_context_does_not_crash_recorded_success(tmp_path: Path) -> None:
    out = tmp_path / "out"
    ctx_dir = tmp_path / "ctx"

    agent = MagicMock()

    def factory():
        return agent

    agent.generate_report.return_value = {
        "ticker": "EMPTY.SZ",
        "as_of": "2024-12-31",
        "report_markdown": "# 空\n",
        "data_coverage": {},
        "warnings": ["snapshot 无数据"],
        "used_sources": [],
        "llm_used": False,
        "provider": "none",
        "model": "template",
    }

    ns = SimpleNamespace(
        tickers=["EMPTY.SZ"],
        ticker_file=None,
        as_of="2024-12-31",
        provider="none",
        model="qwen-plus",
        output_dir=out,
        context_dir=ctx_dir,
        language="zh",
        top_k_news=5,
        top_k_reports=5,
        top_k_per_doc_type=3,
        workers=1,
        skip_existing=False,
        context_only=False,
        collection_name="default",
        chroma_dir=".cache/chromadb",
        summary_json=tmp_path / "s.json",
        summary_md=tmp_path / "s.md",
        batch_report_md=tmp_path / "b.md",
    )

    summ = bgr.run_batch(ns, retrieve_fn=lambda _ns: _empty_context(), agent_factory=factory)
    assert summ["counts"]["success"] == 1
    row = summ["per_ticker"][0]
    assert row["snapshot_five_doc_types_complete"] is False


def test_skip_existing_skips_when_report_present(tmp_path: Path) -> None:
    out = tmp_path / "out"
    ctx_dir = tmp_path / "ctx"
    out.mkdir(parents=True)
    (out / "SKIP_SH_report.md").write_text("old", encoding="utf-8")

    ns = SimpleNamespace(
        tickers=["SKIP.SH"],
        ticker_file=None,
        as_of="2024-12-31",
        provider="none",
        model="qwen-plus",
        output_dir=out,
        context_dir=ctx_dir,
        language="zh",
        top_k_news=5,
        top_k_reports=5,
        top_k_per_doc_type=3,
        workers=1,
        skip_existing=True,
        context_only=False,
        collection_name="default",
        chroma_dir=".cache/chromadb",
        summary_json=tmp_path / "s.json",
        summary_md=tmp_path / "s.md",
        batch_report_md=tmp_path / "b.md",
    )

    summ = bgr.run_batch(ns, retrieve_fn=lambda _ns: pytest.fail("should not retrieve"), )
    assert summ["counts"]["skipped"] == 1
    assert (out / "SKIP_SH_report.md").read_text() == "old"


def test_context_only_writes_context_preserves_report(tmp_path: Path) -> None:
    out = tmp_path / "out"
    ctx_dir = tmp_path / "ctx"
    out.mkdir(parents=True)
    (out / "BOO_SH_report.md").write_text("# keep me\n", encoding="utf-8")

    retrieved: list[str] = []

    def fake_retrieve(ns: SimpleNamespace) -> dict:
        retrieved.append(ns.ticker)
        return _fake_context("BOO.SH")

    ns = SimpleNamespace(
        tickers=["BOO.SH"],
        ticker_file=None,
        as_of="2024-12-31",
        provider="none",
        model="qwen-plus",
        output_dir=out,
        context_dir=ctx_dir,
        language="zh",
        top_k_news=5,
        top_k_reports=5,
        top_k_per_doc_type=3,
        workers=1,
        skip_existing=False,
        context_only=True,
        collection_name="default",
        chroma_dir=".cache/chromadb",
        summary_json=tmp_path / "s.json",
        summary_md=tmp_path / "s.md",
        batch_report_md=tmp_path / "b.md",
    )

    summ = bgr.run_batch(ns, retrieve_fn=fake_retrieve, agent_factory=lambda: pytest.fail("no agent"))
    assert summ["counts"]["success"] == 1
    assert (out / "BOO_SH_report.md").read_text() == "# keep me\n"
    ctx = json.loads((ctx_dir / "BOO_SH_context.json").read_text(encoding="utf-8"))
    assert ctx["news_count"] == 1
    assert retrieved == ["BOO.SH"]


def test_collect_tickers_mutex_error() -> None:
    ns = SimpleNamespace(
        tickers=["A"],
        ticker_file=Path("x"),
        skip_existing=False,
        context_only=False,
    )
    with pytest.raises(SystemExit):
        bgr.collect_tickers(ns)


def test_retrieve_failure_recorded(tmp_path: Path) -> None:
    out = tmp_path / "out"
    ctx_dir = tmp_path / "ctx"

    ns = SimpleNamespace(
        tickers=["BAD.SH"],
        ticker_file=None,
        as_of="2024-12-31",
        provider="none",
        model="qwen-plus",
        output_dir=out,
        context_dir=ctx_dir,
        language="zh",
        top_k_news=5,
        top_k_reports=5,
        top_k_per_doc_type=3,
        workers=1,
        skip_existing=False,
        context_only=False,
        collection_name="default",
        chroma_dir=".cache/chromadb",
        summary_json=tmp_path / "s.json",
        summary_md=tmp_path / "s.md",
        batch_report_md=tmp_path / "b.md",
    )

    def boom(_ns: SimpleNamespace) -> dict:
        raise RuntimeError("no chroma")

    summ = bgr.run_batch(ns, retrieve_fn=boom)
    assert summ["counts"]["failed"] == 1
    assert summ["per_ticker"][0]["error"]
