"""tests/test_ollama_agent.py — Ollama ReAct Agent 单元测试（≥15 个，全部 mock）.

所有需要真实 Ollama 调用的测试标记为 @pytest.mark.integration，
默认运行 `-m "not integration"` 时跳过。
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_ollama_response(
    content: str = "",
    tool_name: str = "",
    tool_args: dict | None = None,
) -> dict:
    """构造 Ollama /api/chat 格式的响应字典."""
    if tool_name:
        return {
            "message": {
                "role":       "assistant",
                "content":    content,
                "tool_calls": [
                    {"function": {"name": tool_name, "arguments": tool_args or {}}}
                ],
            }
        }
    return {"message": {"role": "assistant", "content": content}}


def _mock_agent_context() -> dict:
    """返回最简 context dict，避免读文件."""
    return {
        "industry": "食品饮料",
        "snapshot_latest_market_metrics": [
            {"text": "pe_ttm: 25.3 pb: 8.5 total_mv: 6500"}
        ],
        "snapshot_financial_indicator_summary": [
            {"text": "roe_ttm: 30.5 netprofit_yoy: 15.2 or_yoy: 12.8"}
        ],
    }


def _make_agent(cls, ticker="000858.SZ"):
    """实例化 Agent，注入 mock context，跳过 DB/文件/注册表初始化.

    通过 patch _init_model（BaseInvestmentAgent 的模型加载方法）跳过 IO，
    然后手动设置必要属性。
    """
    from quantmind.agents.investment_agents.base_agent import BaseInvestmentAgent
    with patch.object(BaseInvestmentAgent, "_init_model", return_value=None):
        agent = cls(ticker, pd.Timestamp("2025-03-31"), _mock_agent_context())
    # 保证 _model_record / _ml_model 为 None（_init_model 被 mock，不会设置）
    agent._model_record = None
    agent._ml_model     = None
    return agent


# ──────────────────────────────────────────────────────────────────────────────
# 1. OllamaReActClient — 解析器单元测试
# ──────────────────────────────────────────────────────────────────────────────

from quantmind.agents.ollama_client import OllamaReActClient


def test_parse_signal_standard_format():
    c = OllamaReActClient()
    assert c._parse_signal_from_text("SIGNAL: 0.75") == pytest.approx(0.75)


def test_parse_signal_chinese_format():
    c = OllamaReActClient()
    assert c._parse_signal_from_text("信号: -0.4") == pytest.approx(-0.4)


def test_parse_signal_equals_format():
    c = OllamaReActClient()
    assert c._parse_signal_from_text("signal=0.5\nCONFIDENCE: 0.8") == pytest.approx(0.5)


def test_parse_signal_clamped_to_range():
    c = OllamaReActClient()
    assert c._parse_signal_from_text("SIGNAL: 2.5") == pytest.approx(1.0)
    assert c._parse_signal_from_text("SIGNAL: -3.0") == pytest.approx(-1.0)


def test_parse_signal_missing_returns_zero():
    c = OllamaReActClient()
    assert c._parse_signal_from_text("No signal here.") == pytest.approx(0.0)


def test_parse_confidence_standard():
    c = OllamaReActClient()
    assert c._parse_confidence_from_text("CONFIDENCE: 0.85") == pytest.approx(0.85)


def test_parse_confidence_percentage_format():
    c = OllamaReActClient()
    # 80 > 1 → treated as percentage → 0.80
    assert c._parse_confidence_from_text("CONFIDENCE: 80") == pytest.approx(0.80)


def test_parse_confidence_clamped():
    c = OllamaReActClient()
    # 1.5 < 10 → no percentage conversion → clamp to 1.0
    assert c._parse_confidence_from_text("CONFIDENCE: 1.5") == pytest.approx(1.0)
    # negative → clamp to 0.0
    assert c._parse_confidence_from_text("CONFIDENCE: -0.2") == pytest.approx(0.0)


def test_parse_confidence_missing_returns_half():
    c = OllamaReActClient()
    assert c._parse_confidence_from_text("no conf here") == pytest.approx(0.5)


def test_parse_both_from_realistic_llm_output():
    c  = OllamaReActClient()
    txt = (
        "综合以上分析：\n"
        "SIGNAL: 0.62\n"
        "CONFIDENCE: 0.78\n"
        "SUMMARY: 五粮液估值合理，长期动量强，建议买入。\n"
        "KEY_RISK: 消费复苏不及预期。"
    )
    assert c._parse_signal_from_text(txt)     == pytest.approx(0.62)
    assert c._parse_confidence_from_text(txt) == pytest.approx(0.78)


# ──────────────────────────────────────────────────────────────────────────────
# 2. OllamaReActClient — chat_with_tools ReAct 循环
# ──────────────────────────────────────────────────────────────────────────────

def test_chat_returns_final_answer_without_tools():
    """LLM 直接给出最终答案（无工具调用）→ fallback=False, 答案正确."""
    c    = OllamaReActClient()
    resp = _make_ollama_response("SIGNAL: 0.5\nCONFIDENCE: 0.7\nSUMMARY: 估值合理")
    with patch.object(c, "_call_ollama", return_value=resp):
        result = c.chat_with_tools("sys", "user", tools=[])
    assert result["fallback"] is False
    assert result["signal"]   == pytest.approx(0.5)
    assert result["confidence"] == pytest.approx(0.7)
    assert len(result["reasoning_trace"]) == 1


def test_chat_executes_tool_and_continues():
    """LLM 先调用工具，再给出最终答案 → tools_called 包含工具名."""
    c = OllamaReActClient()
    tool_resp = _make_ollama_response(tool_name="my_tool", tool_args={"x": 1})
    final_resp = _make_ollama_response("SIGNAL: 0.3\nCONFIDENCE: 0.6\nSUMMARY: ok")

    executor_called = []

    def my_executor(x, **kw):
        executor_called.append(x)
        return {"result": "tool output"}

    with patch.object(c, "_call_ollama", side_effect=[tool_resp, final_resp]):
        result = c.chat_with_tools(
            "sys", "user",
            tools=[{"name": "my_tool", "description": "d", "parameters": {}}],
            tool_executors={"my_tool": my_executor},
        )
    assert result["fallback"] is False
    assert "my_tool" in result["tools_called"]
    assert executor_called == [1]
    assert len(result["reasoning_trace"]) == 2


def test_chat_returns_fallback_on_exception():
    """LLM 抛出异常 → fallback=True."""
    c = OllamaReActClient()
    with patch.object(c, "_call_ollama", side_effect=Exception("timeout")):
        result = c.chat_with_tools("sys", "user", tools=[])
    assert result["fallback"] is True
    assert result["signal"]   == pytest.approx(0.0)


def test_chat_handles_missing_executor_gracefully():
    """LLM 调用未注册工具 → 不崩溃，tools_called 仍有记录."""
    c = OllamaReActClient()
    tool_resp  = _make_ollama_response(tool_name="unknown_tool", tool_args={})
    final_resp = _make_ollama_response("SIGNAL: 0.1\nCONFIDENCE: 0.5\nSUMMARY: ok")
    with patch.object(c, "_call_ollama", side_effect=[tool_resp, final_resp]):
        result = c.chat_with_tools("sys", "user", tools=[], tool_executors={})
    assert "unknown_tool" in result["tools_called"]
    assert result["fallback"] is False


def test_reasoning_trace_has_step_entries():
    """reasoning_trace 每一步都有 step 字段."""
    c = OllamaReActClient()
    resp = _make_ollama_response("SIGNAL: 0.2\nCONFIDENCE: 0.6\nSUMMARY: ok")
    with patch.object(c, "_call_ollama", return_value=resp):
        result = c.chat_with_tools("sys", "user", tools=[])
    for entry in result["reasoning_trace"]:
        assert "step" in entry


def test_is_available_returns_bool():
    """is_available 在正常/异常情况下都返回布尔值."""
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "models": [{"name": "qwen2.5:7b"}]
        }
        assert OllamaReActClient.is_available() is True

    with patch("requests.get", side_effect=Exception("refused")):
        assert OllamaReActClient.is_available() is False


# ──────────────────────────────────────────────────────────────────────────────
# 3. ValuationAgent — mode 路由和降级
# ──────────────────────────────────────────────────────────────────────────────

from quantmind.agents.investment_agents.valuation_agent import ValuationAgent
from quantmind.agents.investment_agents.base_agent import AgentSignal


def test_valuation_fast_mode_skips_llm():
    """fast 模式不调用 LLM."""
    agent = _make_agent(ValuationAgent)
    with patch.object(agent, "_analyze_with_llm") as mock_llm, \
         patch.object(agent, "_rule_based_analyze",
                      return_value=AgentSignal("ValuationAgent", agent.ticker, 0.1, 0.5, "ok")):
        agent.analyze(mode="fast")
    mock_llm.assert_not_called()


def test_valuation_auto_mode_calls_llm_when_available():
    """auto 模式 + Ollama 可用 → 调用 _analyze_with_llm."""
    agent = _make_agent(ValuationAgent)
    mock_sig = AgentSignal("ValuationAgent", agent.ticker, 0.4, 0.7, "llm ok",
                            llm_mode=True)
    with patch.object(ValuationAgent, "_ollama_available", return_value=True), \
         patch.object(agent, "_analyze_with_llm", return_value=mock_sig) as mock_llm:
        result = agent.analyze(mode="auto")
    mock_llm.assert_called_once()
    assert result.llm_mode is True


def test_valuation_auto_mode_falls_back_when_unavailable():
    """auto 模式 + Ollama 不可用 → 降级规则，不崩溃."""
    agent = _make_agent(ValuationAgent)
    with patch.object(ValuationAgent, "_ollama_available", return_value=False), \
         patch.object(agent, "_rule_based_analyze",
                      return_value=AgentSignal("ValuationAgent", agent.ticker, -0.1, 0.4, "rules")) as mock_rules:
        result = agent.analyze(mode="auto")
    mock_rules.assert_called_once()
    assert result.signal == pytest.approx(-0.1)


def test_valuation_full_mode_falls_back_gracefully_on_llm_error():
    """full 模式但 _analyze_with_llm 抛异常 → 降级规则，不崩溃."""
    agent = _make_agent(ValuationAgent)
    with patch.object(ValuationAgent, "_ollama_available", return_value=True), \
         patch.object(agent, "_analyze_with_llm", side_effect=RuntimeError("oops")), \
         patch.object(agent, "_rule_based_analyze",
                      return_value=AgentSignal("ValuationAgent", agent.ticker, 0.0, 0.4, "fallback")):
        result = agent.analyze(mode="full")
    assert result.signal is not None   # 有输出，不崩溃


def test_valuation_signal_has_new_fields():
    """AgentSignal 包含 reasoning_trace / tools_called / llm_mode 字段."""
    sig = AgentSignal("ValuationAgent", "600519.SH", 0.5, 0.7, "ok")
    assert isinstance(sig.reasoning_trace, list)
    assert isinstance(sig.tools_called, list)
    assert sig.llm_mode is False


def test_valuation_llm_result_parsed_into_agent_signal():
    """_analyze_with_llm 正确把 OllamaReActClient 结果映射到 AgentSignal."""
    agent = _make_agent(ValuationAgent)
    mock_result = {
        "final_answer":    "SIGNAL: 0.65\nCONFIDENCE: 0.75\nSUMMARY: 低估",
        "signal":          0.65,
        "confidence":      0.75,
        "reasoning_trace": [{"step": 0, "content": "thought"}],
        "tools_called":    ["get_industry_peers_valuation"],
        "fallback":        False,
    }
    with patch("quantmind.agents.ollama_client.OllamaReActClient.chat_with_tools",
               return_value=mock_result):
        sig = agent._analyze_with_llm()
    assert sig.signal     == pytest.approx(0.65)
    assert sig.confidence == pytest.approx(0.75)
    assert sig.llm_mode   is True
    assert "get_industry_peers_valuation" in sig.tools_called


# ──────────────────────────────────────────────────────────────────────────────
# 4. 其他 Agent — mode 参数和降级
# ──────────────────────────────────────────────────────────────────────────────

from quantmind.agents.investment_agents.momentum_agent  import MomentumAgent
from quantmind.agents.investment_agents.quality_agent   import QualityAgent
from quantmind.agents.investment_agents.sentiment_agent import SentimentAgent
from quantmind.agents.investment_agents.risk_agent      import RiskAgent


@pytest.mark.parametrize("cls,fallback_method", [
    (MomentumAgent,  "_analyze_rules"),
    (SentimentAgent, "_analyze_rules"),
    (RiskAgent,      "_rule_based_analyze"),
])
def test_other_agents_fast_mode_skips_llm(cls, fallback_method):
    """fast 模式各 Agent 不调用 LLM."""
    agent = _make_agent(cls)
    fallback_sig = AgentSignal(cls.__name__, agent.ticker, 0.0, 0.4, "rules")
    with patch.object(agent, "_analyze_with_llm", side_effect=AssertionError("should not be called")), \
         patch.object(agent, fallback_method, return_value=fallback_sig):
        result = agent.analyze(mode="fast")
    assert result is not None


def test_quality_agent_fast_mode_skips_llm():
    """QualityAgent fast 模式不调用 LLM."""
    agent = _make_agent(QualityAgent)
    fallback_sig = AgentSignal("QualityAgent", agent.ticker, 0.0, 0.4, "rules")
    with patch.object(agent, "_analyze_with_llm", side_effect=AssertionError("no llm")), \
         patch.object(agent, "_analyze_piotroski", return_value=fallback_sig), \
         patch.object(agent, "_analyze_lgbm_v2", return_value=None):
        result = agent.analyze(mode="fast")
    assert result is not None


# ──────────────────────────────────────────────────────────────────────────────
# 5. DebateOrchestrator — agent_mode 传递
# ──────────────────────────────────────────────────────────────────────────────

from quantmind.agents.debate_orchestrator import DebateOrchestrator


def test_orchestrator_accepts_agent_mode_param():
    """DebateOrchestrator 接受 agent_mode 参数且存储。"""
    orch = DebateOrchestrator("600519.SH", "2025-03-31", {}, agent_mode="full")
    assert orch._agent_mode == "full"


def test_orchestrator_passes_mode_to_agents():
    """_run_single_agent 调用 agent.analyze(mode=self._agent_mode)."""
    orch = DebateOrchestrator("600519.SH", "2025-03-31", {}, agent_mode="fast")
    mock_sig = AgentSignal("ValuationAgent", "600519.SH", 0.2, 0.6, "ok")

    with patch.object(ValuationAgent, "__init__", return_value=None), \
         patch.object(ValuationAgent, "analyze", return_value=mock_sig) as mock_analyze:
        ValuationAgent.ticker = "600519.SH"
        ValuationAgent.as_of  = None
        ValuationAgent.context = {}
        orch._run_single_agent(ValuationAgent)

    mock_analyze.assert_called_once_with(mode="fast")


# ──────────────────────────────────────────────────────────────────────────────
# 6. 修复验证：数据注入 / level='ticker' / 中文 signal 解析
# ──────────────────────────────────────────────────────────────────────────────

def test_tool_uses_ticker_level_not_ts_code():
    """工具函数中不得以 level='ts_code' 方式调用 xs()，必须用 level='ticker'.

    注意：只检查可执行语句中的 level= 参数，注释里出现 'ts_code' 字符串是允许的。
    """
    import re
    import inspect
    from quantmind.agents.investment_agents.valuation_agent import ValuationAgent

    # 精确模式：检查 level='ts_code' 或 level="ts_code" 的可执行语句
    bad_pattern  = re.compile(r'level\s*=\s*["\']ts_code["\']')
    good_pattern = re.compile(r'level\s*=\s*["\']ticker["\']|\.loc\[ticker|in latest\.index')

    for method_name in ("_tool_industry_peers", "_tool_historical_band"):
        src = inspect.getsource(getattr(ValuationAgent, method_name))
        assert not bad_pattern.search(src), (
            f"{method_name} 仍含 level='ts_code'（可执行语句），会导致 KeyError"
        )
        assert good_pattern.search(src), (
            f"{method_name} 中未找到正确的 ticker 定位方式"
        )


def test_user_message_contains_pe_value():
    """_build_user_message() 能把 context 里的 pe_ttm / pb 注入到消息体中.

    不管 key 是直接格式（pe_ttm）还是 snapshot_ 前缀格式（snapshot_pe_ttm），
    都应该在 user_message 里出现数值。其余字段缺失时出现 N/A 是正常行为。
    """
    from quantmind.agents.investment_agents.valuation_agent import ValuationAgent

    # 两种 key 格式都要支持
    cases = [
        {"pe_ttm": 0.109, "pb": 1.04},                   # 直接格式
        {"snapshot_pe_ttm": 0.109, "snapshot_pb": 1.04}, # snapshot_ 前缀格式
    ]
    for context in cases:
        agent = ValuationAgent.__new__(ValuationAgent)
        msg = agent._build_user_message("600519.SH", context)

        # pe_ttm 值必须出现在消息中（以任意精度格式）
        pe_val_present = any(v in msg for v in ("0.109", "0.1090", "0.1100"))
        assert pe_val_present, (
            f"context={list(context.keys())} → msg 里找不到 pe_ttm 数值\n{msg[:400]}"
        )
        # pb 值必须出现
        pb_val_present = any(v in msg for v in ("1.04", "1.0400"))
        assert pb_val_present, (
            f"context={list(context.keys())} → msg 里找不到 pb 数值\n{msg[:400]}"
        )
        # 消息不应该为空
        assert len(msg) > 50, "user_message 太短，可能构建失败"


def test_signal_parsing_from_chinese_text():
    """中文情感描述（无结构化数字）也能解析出非零 signal."""
    c = OllamaReActClient()

    # 看多文字 → 正值
    bullish = "综合来看，该股票估值偏低，具有一定的投资价值，建议买入。"
    sig_bull = c._parse_signal_from_text(bullish)
    assert sig_bull > 0, f"看多文字应解析出正 signal，实际={sig_bull}"

    # 看空文字 → 负值
    bearish = "目前估值偏高，存在较大下行风险，建议卖出观望。"
    sig_bear = c._parse_signal_from_text(bearish)
    assert sig_bear < 0, f"看空文字应解析出负 signal，实际={sig_bear}"

    # 结构化格式优先级高于关键词（即使同时含关键词）
    structured = "该股低估，但综合判断：SIGNAL: -0.3"
    sig_struct = c._parse_signal_from_text(structured)
    assert sig_struct == pytest.approx(-0.3), (
        f"结构化 SIGNAL 应覆盖关键词 fallback，实际={sig_struct}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 7. Integration tests (需要真实 Ollama，默认跳过)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_valuation_agent_full_mode_real_ollama():  # noqa: F811
    """真实 Ollama 调用 ValuationAgent（需要 qwen2.5:7b 运行中）."""
    agent = _make_agent(ValuationAgent, ticker="000858.SZ")
    result = agent.analyze(mode="full")
    assert -1.0 <= result.signal <= 1.0
    assert 0.0  <= result.confidence <= 1.0
    assert result.summary
    # 至少应调用了一个工具
    print(f"\n五粮液 signal={result.signal:+.2f} conf={result.confidence:.2f}")
    print(f"工具调用: {result.tools_called}")
    print(f"推理步骤: {len(result.reasoning_trace)}")
