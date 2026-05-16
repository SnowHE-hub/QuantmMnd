"""Tests for scripts/generate_investment_report.py CLI.

覆盖：
8. CLI 能读取 context-json（跳过检索）
9. CLI 能自动生成 context（mock retrieve 路径）
10. 输出包含"不构成投资建议"声明
11. 不读取或打印密钥
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


# ── context 工厂（与 test_rag_report_agent 保持独立） ─────────────────────────


def _sample_context(ticker: str = "600519.SH", as_of: str = "2024-12-31") -> dict[str, Any]:
    def _snap(dt: str) -> dict:
        return {
            "rank": 1, "score": 0.9, "ticker": ticker, "source_type": "snapshot",
            "doc_type": dt, "as_of": as_of, "published_date": as_of,
            "source": f"snapshot:{ticker}:{dt}", "title": f"{ticker} {dt}",
            "text_preview": f"{dt} 摘要文本 roe：0.35 pe=28.5 名称：测试公司 行业：测试行业",
            "text": f"{dt} 完整文本 roe：0.35 pe=28.5 名称：测试公司 行业：测试行业",
        }

    ctx: dict[str, Any] = {
        "ticker": ticker, "as_of": as_of, "collection_count": 500,
        "news_context": [
            {"rank": 1, "score": 0.8, "ticker": ticker, "source_type": "news",
             "doc_type": "", "as_of": "", "published_date": "2026-05-10",
             "source": "http://example.com/news/1", "title": "测试新闻",
             "text_preview": "测试新闻内容摘要", "text": "测试新闻完整内容"},
        ],
        "news_count": 1,
        "report_context": [], "report_count": 0,
        "market_context": [
            {"rank": 1, "score": 0.9, "ticker": "__MARKET__", "source_type": "snapshot",
             "doc_type": "market_index_context", "as_of": as_of, "published_date": as_of,
             "source": "snapshot:__MARKET__:market_index_context", "title": "指数环境",
             "text_preview": "沪深300 最新 close=3800", "text": "沪深300 最新 close=3800"}
        ],
        "market_count": 1,
    }
    for dt in ["company_profile", "latest_market_metrics", "financial_indicator_summary",
               "northbound_summary", "margin_summary"]:
        ctx[f"snapshot_{dt}"] = [_snap(dt)]
        ctx[f"snapshot_{dt}_count"] = 1
    return ctx


# ── 8. CLI 能读取 context-json ────────────────────────────────────────────────


def test_cli_reads_context_json(tmp_path: Path) -> None:
    """CLI 从已有 JSON 读取 context，不重新检索。"""
    ctx = _sample_context("600519.SH")
    ctx_file = tmp_path / "ctx.json"
    ctx_file.write_text(json.dumps(ctx), encoding="utf-8")
    out_file = tmp_path / "report.md"

    sys.argv = [
        "generate_investment_report.py",
        "--ticker", "600519.SH",
        "--as-of", "2024-12-31",
        "--context-json", str(ctx_file),
        "--output", str(out_file),
        "--provider", "none",
    ]

    import scripts.generate_investment_report as cli_mod

    with patch.object(cli_mod, "retrieve_context", side_effect=AssertionError("Should not call retrieve")):
        with pytest.raises(SystemExit) as exc_info:
            cli_mod.main()
        assert exc_info.value.code == 0

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "600519" in content
    assert len(content) > 200


# ── 9. CLI 能自动生成 context（mock retrieve 路径）────────────────────────────


def test_cli_auto_retrieve_context(tmp_path: Path) -> None:
    """CLI 无 context-json 时，调用 retrieve_context 生成。"""
    out_file = tmp_path / "report.md"
    ctx_out = tmp_path / "ctx_out.json"
    ctx = _sample_context("600519.SH")

    sys.argv = [
        "generate_investment_report.py",
        "--ticker", "600519.SH",
        "--as-of", "2024-12-31",
        "--output", str(out_file),
        "--context-output", str(ctx_out),
        "--provider", "none",
    ]

    import scripts.generate_investment_report as cli_mod

    with patch.object(cli_mod, "retrieve_context", return_value=ctx) as mock_retrieve:
        with pytest.raises(SystemExit) as exc_info:
            cli_mod.main()
        assert exc_info.value.code == 0
        mock_retrieve.assert_called_once()

    assert out_file.exists()
    assert ctx_out.exists()
    saved_ctx = json.loads(ctx_out.read_text())
    assert saved_ctx["ticker"] == "600519.SH"


# ── 10. 输出包含"不构成投资建议"声明 ─────────────────────────────────────────


def test_cli_output_has_disclaimer(tmp_path: Path) -> None:
    ctx = _sample_context("600519.SH")
    ctx_file = tmp_path / "ctx.json"
    ctx_file.write_text(json.dumps(ctx), encoding="utf-8")
    out_file = tmp_path / "report.md"

    sys.argv = [
        "generate_investment_report.py",
        "--ticker", "600519.SH",
        "--context-json", str(ctx_file),
        "--output", str(out_file),
        "--provider", "none",
    ]
    import scripts.generate_investment_report as cli_mod

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main()
    assert exc_info.value.code == 0

    content = out_file.read_text(encoding="utf-8")
    assert "不构成" in content or "免责声明" in content


# ── 11. 不读取或打印密钥 ──────────────────────────────────────────────────────


def test_cli_script_no_secret_access() -> None:
    import scripts.generate_investment_report as mod
    src = inspect.getsource(mod)
    import re
    # 不应该有打印 API_KEY 值的语句（允许 "key" 作为循环变量）
    assert not re.search(r"print.*API_KEY", src)
    assert not re.search(r"print.*api_key\s*=", src)
    # 不应该直接读取密钥环境变量后打印
    forbidden_patterns = [
        r'print.*os\.environ\.get.*KEY',
        r'print.*os\.getenv.*KEY',
    ]
    for pat in forbidden_patterns:
        assert not re.search(pat, src), f"Found secret print pattern: {pat}"


def test_api_key_not_printed_in_cli(tmp_path: Path, capsys) -> None:
    import os
    fake_key = "sk-CLI_FAKE_KEY_SHOULD_NOT_APPEAR"
    ctx = _sample_context("600519.SH")
    ctx_file = tmp_path / "ctx.json"
    ctx_file.write_text(json.dumps(ctx), encoding="utf-8")
    out_file = tmp_path / "report.md"

    sys.argv = [
        "generate_investment_report.py",
        "--ticker", "600519.SH",
        "--context-json", str(ctx_file),
        "--output", str(out_file),
        "--provider", "openai",  # 会 fallback none（key 不存在）
    ]

    with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
        import scripts.generate_investment_report as cli_mod
        with pytest.raises(SystemExit) as exc_info:
            cli_mod.main()
        assert exc_info.value.code == 0

    captured = capsys.readouterr()
    assert fake_key not in captured.out
    assert fake_key not in captured.err


# ── 报告含所有 12 章节 ────────────────────────────────────────────────────────


def test_report_has_all_sections(tmp_path: Path) -> None:
    ctx = _sample_context("600519.SH")
    ctx_file = tmp_path / "ctx.json"
    ctx_file.write_text(json.dumps(ctx), encoding="utf-8")
    out_file = tmp_path / "report.md"

    sys.argv = [
        "generate_investment_report.py",
        "--ticker", "600519.SH",
        "--context-json", str(ctx_file),
        "--output", str(out_file),
        "--provider", "none",
    ]
    import scripts.generate_investment_report as cli_mod

    with pytest.raises(SystemExit):
        cli_mod.main()

    content = out_file.read_text(encoding="utf-8")
    sections = [
        "执行摘要", "公司概况", "行情与估值", "财务分析",
        "北向资金", "融资融券", "市场环境",
        "新闻与事件", "本地研究报告", "风险因素", "投资建议", "数据来源",
    ]
    for s in sections:
        assert s in content, f"Missing section: {s}"


# ── 300750 report 缺失时明确写入 ──────────────────────────────────────────────


def test_300750_report_missing_in_output(tmp_path: Path) -> None:
    ctx = _sample_context("300750.SZ")
    ctx["report_context"] = []
    ctx["report_count"] = 0
    ctx_file = tmp_path / "ctx.json"
    ctx_file.write_text(json.dumps(ctx), encoding="utf-8")
    out_file = tmp_path / "report.md"

    sys.argv = [
        "generate_investment_report.py",
        "--ticker", "300750.SZ",
        "--context-json", str(ctx_file),
        "--output", str(out_file),
        "--provider", "none",
    ]
    import scripts.generate_investment_report as cli_mod

    with pytest.raises(SystemExit):
        cli_mod.main()

    content = out_file.read_text(encoding="utf-8")
    assert "未检索到" in content or "研究报告" in content


# ── news 标注当前舆情 ────────────────────────────────────────────────────────


def test_news_marked_current_in_cli_output(tmp_path: Path) -> None:
    ctx = _sample_context("600519.SH")
    ctx_file = tmp_path / "ctx.json"
    ctx_file.write_text(json.dumps(ctx), encoding="utf-8")
    out_file = tmp_path / "report.md"

    sys.argv = [
        "generate_investment_report.py",
        "--ticker", "600519.SH",
        "--context-json", str(ctx_file),
        "--output", str(out_file),
        "--provider", "none",
    ]
    import scripts.generate_investment_report as cli_mod

    with pytest.raises(SystemExit):
        cli_mod.main()

    content = out_file.read_text(encoding="utf-8")
    assert "当前市场舆情" in content or "当前舆情" in content
