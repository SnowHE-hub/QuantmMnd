"""quantmind.agents.technical_agent — TechnicalAgent.

计算技术指标 + 获取 LGBM 量化信号，LLM 综合输出技术观点。
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from typing import Any

from loguru import logger

from quantmind.agents.base import BaseAgent
from quantmind.agents.tools import (
    fetch_price_history,
    compute_technical_indicators,
    get_factor_signal,
)
from quantmind.core.state import AgentState, AgentType, TechnicalAnalysis

__all__ = ["TechnicalAgent"]

_SYSTEM_PROMPT = """你是一位专业的技术分析师和量化研究员。
你将收到：1）技术指标计算结果，2）LightGBM 量化模型信号

请综合技术面和量化面给出判断，注意：
- 技术信号（均线/MACD/RSI/布林带）揭示短期趋势和超买超卖
- 量化信号（LGBM 排名）揭示多因子综合评分
- 两者共振时信号更可靠

输出 JSON：
{
  "signal": "bullish|neutral|bearish",
  "summary": "技术观点（100字内）",
  "key_patterns": ["识别的关键形态或信号"],
  "support_level": null或数字（支撑位）,
  "resistance_level": null或数字（压力位）,
  "confidence": 0.0-1.0
}"""


class TechnicalAgent(BaseAgent):
    agent_type = AgentType.TECHNICAL
    description = "技术分析 Agent：价格指标 + LGBM 量化信号 + LLM 综合判断"
    system_prompt = _SYSTEM_PROMPT

    def format_input(self, state: AgentState) -> str:
        as_of = state.query.as_of or date.today()
        lookback_start = as_of - timedelta(days=120)
        sections: list[str] = []

        for ticker in state.query.tickers:
            # 获取价格历史
            prices_res = self.call_tool(
                "fetch_price_history", fetch_price_history,
                ticker, lookback_start, as_of, as_of
            )
            prices = prices_res.data

            indicators: dict[str, float] = {}
            if prices is not None and not prices.empty:
                ind_res = self.call_tool(
                    "compute_technical_indicators", compute_technical_indicators, prices
                )
                indicators = ind_res.data or {}

            # 量化信号
            quant_res = self.call_tool("get_factor_signal", get_factor_signal, ticker, as_of)
            quant = quant_res.data or {}

            # 缓存到 state
            if not hasattr(state, "_technical_cache"):
                state._technical_cache = {}  # type: ignore[attr-defined]
            state._technical_cache[ticker] = {  # type: ignore[attr-defined]
                "indicators": indicators,
                "quant": quant,
                "prices": prices,
            }

            section = (
                f"=== {ticker} 技术数据 ===\n"
                f"技术指标：{json.dumps(indicators, ensure_ascii=False)}\n"
                f"量化信号：score={quant.get('score')}, rank={quant.get('rank')}\n"
                f"主要正面因子：{quant.get('top_positive_factors', [])}\n"
                f"主要负面因子：{quant.get('top_negative_factors', [])}\n"
            )
            sections.append(section)

        return "\n".join(sections) + "\n\n请对每只股票进行技术分析，输出 JSON。"

    def parse_output(self, llm_response: str) -> dict[str, Any]:
        text = llm_response.strip()
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return {}
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            return {}

    def _execute(self, state: AgentState) -> AgentState:
        as_of = state.query.as_of or date.today()
        prompt = self.format_input(state)
        resp = self.llm_chat(prompt, max_tokens=1024)
        parsed = self.parse_output(resp.content)

        for ticker in state.query.tickers:
            cache = getattr(state, "_technical_cache", {}).get(ticker, {})
            indicators = cache.get("indicators", {})
            quant = cache.get("quant", {})

            signal_str = parsed.get("signal", "neutral")
            if signal_str not in ("bullish", "neutral", "bearish"):
                signal_str = "neutral"

            ta = TechnicalAnalysis(
                ticker=ticker,
                as_of=as_of,
                indicators=indicators,
                quant_score=quant.get("score"),
                quant_explanation=quant.get("shap_values", {}),
                summary=parsed.get("summary", ""),
                signal=signal_str,  # type: ignore[arg-type]
                confidence=float(parsed.get("confidence", 0.5)),
            )
            state.technicals[ticker] = ta

        logger.info(f"[TechnicalAgent] analyzed {list(state.technicals.keys())}")
        return state
