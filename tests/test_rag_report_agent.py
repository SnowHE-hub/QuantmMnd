"""Tests for quantmind.agents.rag_report_agent.RAGReportAgent.

覆盖：
1. provider=none 能生成 Markdown
2. report_context 为空时不崩溃
3. news 被标注为 current_sentiment
4. 300750 report 缺失会写 warning
5. snapshot 五类能进入 evidence summary
6. JSON 外的数据不会被引用（模板只用 context 字段）
7. LLM client 缺 key 时 fallback none
8. 输出包含"不构成投资建议"声明
9. 不读取或打印密钥
10. mock LLM 能被调用并返回文本
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quantmind.agents.llm_client import LLMClient, LLMClientResponse, build_client
from quantmind.agents.rag_report_agent import RAGReportAgent


# ── 测试数据工厂 ─────────────────────────────────────────────────────────────


def _make_context(
    ticker: str = "600519.SH",
    as_of: str = "2024-12-31",
    include_news: bool = True,
    include_report: bool = True,
    include_snapshot: bool = True,
    include_market: bool = True,
) -> dict[str, Any]:
    def _snap_item(doc_type: str) -> dict:
        return {
            "rank": 1,
            "score": 0.9,
            "ticker": ticker,
            "source_type": "snapshot",
            "doc_type": doc_type,
            "as_of": as_of,
            "published_date": as_of,
            "source": f"snapshot:{ticker}:{doc_type}",
            "title": f"{ticker} {doc_type}",
            "text_preview": f"{doc_type} 测试文本",
            "text": (
                f"{doc_type} 测试文本\n"
                "roe：0.35\ngrossprofit_margin：0.92\ndebt_to_assets：0.28\n"
                "pe=28.5\npb=9.2\ntotal_mv=1000000\n名称：贵州茅台\n行业：白酒\n上市日期：2001-08-27"
            ),
        }

    ctx: dict[str, Any] = {
        "ticker": ticker,
        "as_of": as_of,
        "collection_count": 20966,
        "news_context": [],
        "news_count": 0,
        "report_context": [],
        "report_count": 0,
        "market_context": [],
        "market_count": 0,
    }

    doc_types = [
        "company_profile",
        "latest_market_metrics",
        "financial_indicator_summary",
        "northbound_summary",
        "margin_summary",
    ]
    for dt in doc_types:
        ctx[f"snapshot_{dt}"] = [_snap_item(dt)] if include_snapshot else []
        ctx[f"snapshot_{dt}_count"] = 1 if include_snapshot else 0

    if include_news:
        ctx["news_context"] = [
            {
                "rank": 1,
                "score": 0.8,
                "ticker": ticker,
                "source_type": "news",
                "doc_type": "",
                "as_of": "",
                "published_date": "2026-05-10",
                "source": "http://example.com/news/1",
                "title": "测试新闻",
                "text_preview": "测试新闻内容",
                "text": "测试新闻内容全文",
            }
        ]
        ctx["news_count"] = 1

    if include_report:
        ctx["report_context"] = [
            {
                "rank": 1,
                "score": 0.7,
                "ticker": "600519",
                "source_type": "report",
                "doc_type": "",
                "as_of": "",
                "published_date": "",
                "source": "data/reports/test_report.md",
                "title": "agent_research_600519_SH_2024-06-30",
                "text_preview": "测试研究报告内容摘要",
                "text": "测试研究报告完整内容",
            }
        ]
        ctx["report_count"] = 1

    if include_market:
        ctx["market_context"] = [
            {
                "rank": 1,
                "score": 0.9,
                "ticker": "__MARKET__",
                "source_type": "snapshot",
                "doc_type": "market_index_context",
                "as_of": as_of,
                "published_date": as_of,
                "source": "snapshot:__MARKET__:market_index_context",
                "title": "指数环境",
                "text_preview": "沪深300 最新收盘 close=3800",
                "text": "主要指数市场环境\n000300.SH 最新(2024-12-31): close=3800, pct_chg=0.5",
            }
        ]
        ctx["market_count"] = 1

    return ctx


# ── 1. provider=none 能生成 Markdown ────────────────────────────────────────


def test_generate_report_no_llm() -> None:
    agent = RAGReportAgent()
    ctx = _make_context()
    result = agent.generate_report(ctx, use_llm=False)
    assert isinstance(result["report_markdown"], str)
    assert len(result["report_markdown"]) > 100
    assert result["llm_used"] is False
    assert result["provider"] == "none"


def test_report_markdown_is_valid_markdown() -> None:
    agent = RAGReportAgent()
    ctx = _make_context()
    result = agent.generate_report(ctx, use_llm=False)
    md = result["report_markdown"]
    assert md.startswith("#")
    assert "## " in md


# ── 2. report_context 为空时不崩溃 ──────────────────────────────────────────


def test_empty_report_context_no_crash() -> None:
    agent = RAGReportAgent()
    ctx = _make_context(include_report=False)
    result = agent.generate_report(ctx, use_llm=False)
    assert result is not None
    assert isinstance(result["report_markdown"], str)


# ── 3. news 被标注为 current_sentiment ──────────────────────────────────────


def test_news_marked_as_current_sentiment() -> None:
    agent = RAGReportAgent()
    ctx = _make_context(include_news=True)
    result = agent.generate_report(ctx, use_llm=False)
    md = result["report_markdown"]
    assert "当前市场舆情" in md or "当前舆情" in md


def test_news_not_described_as_historical() -> None:
    """news 不应被描述为 as_of 时刻的历史事实."""
    agent = RAGReportAgent()
    ctx = _make_context(include_news=True)
    result = agent.generate_report(ctx, use_llm=False)
    md = result["report_markdown"]
    assert "当前" in md


# ── 4. 300750 report 缺失会写 warning ────────────────────────────────────────


def test_missing_report_generates_warning() -> None:
    agent = RAGReportAgent()
    ctx = _make_context(ticker="300750.SZ", include_report=False)
    result = agent.generate_report(ctx, use_llm=False)
    warnings = result["warnings"]
    assert any("研究报告" in w or "report" in w.lower() for w in warnings)


def test_missing_report_in_markdown() -> None:
    agent = RAGReportAgent()
    ctx = _make_context(ticker="300750.SZ", include_report=False)
    result = agent.generate_report(ctx, use_llm=False)
    md = result["report_markdown"]
    assert "未检索到" in md


# ── 5. snapshot 五类能进入 evidence summary ───────────────────────────────────


def test_all_snapshot_doc_types_in_evidence() -> None:
    agent = RAGReportAgent()
    ctx = _make_context(include_snapshot=True)
    warnings: list[str] = []
    used_sources: list[str] = []
    evidence = agent._build_evidence_summary(ctx, warnings, used_sources)

    assert evidence.get("company_profile") is not None
    assert evidence.get("market_metrics") is not None
    assert evidence.get("financial_summary") is not None
    assert evidence.get("northbound") is not None
    assert evidence.get("margin") is not None


def test_snapshot_in_report_markdown() -> None:
    agent = RAGReportAgent()
    ctx = _make_context(include_snapshot=True)
    result = agent.generate_report(ctx, use_llm=False)
    md = result["report_markdown"]
    for section in ["公司概况", "行情与估值", "财务分析", "北向资金", "融资融券"]:
        assert section in md, f"Missing section: {section}"


# ── 6. JSON 外的数据不会被引用 ────────────────────────────────────────────────


def test_no_fabricated_data() -> None:
    """报告中不应出现 context 之外的具体数字。"""
    agent = RAGReportAgent()
    ctx = _make_context(include_snapshot=True)
    result = agent.generate_report(ctx, use_llm=False)
    md = result["report_markdown"]
    # "9999" 不在任何 context 文本中
    assert "9999" not in md
    # 模板不应编造研究报告内容（当 report 为空时）
    ctx_no_report = _make_context(include_report=False)
    result2 = agent.generate_report(ctx_no_report, use_llm=False)
    assert "agent_research" not in result2["report_markdown"]


# ── 7. LLM client 缺 key 时 fallback none ────────────────────────────────────


def test_llm_client_missing_key_fallback() -> None:
    import os
    with patch.dict(os.environ, {}, clear=False):
        # 确保 key 不存在
        for k in ["OPENAI_API_KEY", "DASHSCOPE_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY"]:
            os.environ.pop(k, None)

        for provider in ("openai", "dashscope", "anthropic", "deepseek"):
            client = build_client(provider=provider)
            assert client.provider == "none", f"{provider} should fallback to none"
            assert client.fallback_used is True


def test_llm_client_none_provider() -> None:
    client = build_client(provider="none")
    assert client.provider == "none"
    assert client.is_none_provider
    result = client.chat(system="", user="test")
    assert result is None


# ── 8. 输出包含"不构成投资建议"声明 ─────────────────────────────────────────


def test_disclaimer_in_report() -> None:
    agent = RAGReportAgent()
    ctx = _make_context()
    result = agent.generate_report(ctx, use_llm=False)
    md = result["report_markdown"]
    assert "不构成" in md
    assert "投资建议" in md


# ── 9. 不读取或打印密钥 ───────────────────────────────────────────────────────


def test_no_secret_access_in_agent() -> None:
    import quantmind.agents.rag_report_agent as mod_agent
    import quantmind.agents.llm_client as mod_client

    for mod in (mod_agent, mod_client):
        src = inspect.getsource(mod)
        forbidden = ["print.*KEY", "print.*key", "print.*secret"]
        for pat in forbidden:
            import re
            assert not re.search(pat, src), f"Secret print found in {mod.__name__}"


def test_api_key_never_printed(capsys) -> None:
    """build_client 不应打印 API key 内容。"""
    import os
    fake_key = "sk-SUPER_SECRET_FAKE_KEY_12345"
    with patch.dict(os.environ, {"OPENAI_API_KEY": fake_key}):
        client = build_client(provider="openai")
        _ = client.is_none_provider

    captured = capsys.readouterr()
    assert fake_key not in captured.out
    assert fake_key not in captured.err


# ── 10. mock LLM 能被调用并返回文本 ──────────────────────────────────────────


def test_mock_llm_called_and_used() -> None:
    """mock LLM 返回文本时，report_markdown 应包含 LLM 返回内容。"""
    mock_response = LLMClientResponse(
        content="# 600519.SH 测试报告\n\n## 执行摘要\n- 这是一段 LLM 生成的测试内容\n\n## 不构成投资建议\n免责声明。",
        provider="openai",
        model="gpt-4o-mini",
    )
    mock_client = MagicMock(spec=LLMClient)
    mock_client.provider = "openai"
    mock_client.model = "gpt-4o-mini"
    mock_client.is_none_provider = False
    mock_client.chat.return_value = mock_response

    agent = RAGReportAgent(llm_client=mock_client)
    ctx = _make_context()
    result = agent.generate_report(ctx, use_llm=True)

    mock_client.chat.assert_called_once()
    assert result["llm_used"] is True
    assert "LLM 生成的测试内容" in result["report_markdown"]


def test_mock_llm_returns_none_fallback_template() -> None:
    """mock LLM 返回 None 时，自动回退模板。"""
    mock_client = MagicMock(spec=LLMClient)
    mock_client.provider = "openai"
    mock_client.model = "gpt-4o-mini"
    mock_client.is_none_provider = False
    mock_client.chat.return_value = None

    agent = RAGReportAgent(llm_client=mock_client)
    ctx = _make_context()
    result = agent.generate_report(ctx, use_llm=True)

    assert result["llm_used"] is False
    assert result["provider"] == "none"
    assert result["model"] == "template"
    assert isinstance(result["report_markdown"], str)
    assert len(result["report_markdown"]) > 100


# ── 额外：data_coverage 结构稳定 ─────────────────────────────────────────────


def test_data_coverage_structure() -> None:
    agent = RAGReportAgent()
    ctx = _make_context()
    result = agent.generate_report(ctx, use_llm=False)
    cov = result["data_coverage"]
    required = {
        "collection_count",
        "news_count",
        "report_count",
        "snapshot_company_profile_count",
        "snapshot_latest_market_metrics_count",
        "snapshot_financial_indicator_summary_count",
        "snapshot_northbound_summary_count",
        "snapshot_margin_summary_count",
        "market_context_count",
    }
    assert required.issubset(cov.keys())


def test_result_json_serializable() -> None:
    agent = RAGReportAgent()
    ctx = _make_context()
    result = agent.generate_report(ctx, use_llm=False)
    j = json.dumps(result, ensure_ascii=False)
    parsed = json.loads(j)
    assert parsed["ticker"] == "600519.SH"
    assert "report_markdown" in parsed


# ── _pick_latest ──────────────────────────────────────────────────────────────


def test_pick_latest_selects_newest() -> None:
    agent = RAGReportAgent()
    items = [
        {"as_of": "2024-06-30", "text": "old"},
        {"as_of": "2024-12-31", "text": "latest"},
        {"as_of": "2024-09-30", "text": "mid"},
    ]
    picked = agent._pick_latest(items)
    assert picked is not None
    assert picked["as_of"] == "2024-12-31"


def test_pick_latest_empty() -> None:
    agent = RAGReportAgent()
    assert agent._pick_latest([]) is None


# ── _extract_field ────────────────────────────────────────────────────────────


def test_extract_field_colon_format() -> None:
    agent = RAGReportAgent()
    text = "roe：0.35\ngrossprofit_margin：0.92"
    assert agent._extract_field(text, "roe") == "0.35"
    assert agent._extract_field(text, "grossprofit_margin") == "0.92"


def test_extract_field_eq_format() -> None:
    agent = RAGReportAgent()
    text = "pe=28.5, pb=9.2, total_mv=1000000"
    assert agent._extract_field(text, "pe") == "28.5"
    assert agent._extract_field(text, "pb") == "9.2"


def test_extract_field_missing() -> None:
    agent = RAGReportAgent()
    assert agent._extract_field("no data here", "roe") is None


# ── 市场环境进入报告 ──────────────────────────────────────────────────────────


def test_market_context_in_report() -> None:
    agent = RAGReportAgent()
    ctx = _make_context(include_market=True)
    result = agent.generate_report(ctx, use_llm=False)
    md = result["report_markdown"]
    assert "市场环境" in md
    assert "沪深300" in md or "指数" in md or "close=3800" in md
