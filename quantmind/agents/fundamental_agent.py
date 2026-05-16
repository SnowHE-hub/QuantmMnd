"""quantmind.agents.fundamental_agent — FundamentalAgent.

从 state 读财务数据，调用 analysis_tools 计算比率/DCF/可比，LLM 解读结果。
输出 FundamentalAnalysis（盈利/偿债/成长/估值/风险5个维度）。
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

from loguru import logger

from quantmind.agents.base import BaseAgent
from quantmind.agents.tools import (
    fetch_financials_pit,
    fetch_industry_peers,
    compute_financial_ratios,
    compute_dcf_valuation,
    compute_comparable_multiples,
)
from quantmind.core.state import AgentState, AgentType, FundamentalAnalysis

__all__ = ["FundamentalAgent"]

_SYSTEM_PROMPT = """你是一位专业的基本面分析师。
你将收到一家公司的财务数据和量化指标，请从5个维度进行分析：
1. 盈利质量（ROE/ROA/净利率，现金流质量）
2. 偿债能力（流动比率，资产负债率）
3. 成长潜力（收入增速，利润增速）
4. 估值水平（PE/PB，DCF公允价值，与同行比较）
5. 主要风险（列举3-5个具体风险点）

严格遵守：
- 引用具体财务数字，不泛泛而谈
- LLM 只负责解读，不重新计算已有数字
- 给出明确的综合评级（strong_buy/buy/hold/sell/strong_sell）和置信度（0-1）

输出 JSON 格式：
{
  "profitability_analysis": "盈利质量分析（100字内）",
  "solvency_analysis": "偿债能力分析（100字内）",
  "growth_analysis": "成长潜力分析（100字内）",
  "valuation_analysis": "估值分析（100字内）",
  "key_risks": ["风险1", "风险2", "风险3"],
  "summary": "综合评价（50字内）",
  "rating": "strong_buy|buy|hold|sell|strong_sell",
  "confidence": 0.0-1.0,
  "dcf_value_per_share": null或数字
}"""


class FundamentalAgent(BaseAgent):
    agent_type = AgentType.FUNDAMENTAL
    description = "基本面分析 Agent：财务比率 + DCF + 可比估值 + LLM 解读"
    system_prompt = _SYSTEM_PROMPT

    def format_input(self, state: AgentState) -> str:
        tickers = state.query.tickers
        as_of = state.query.as_of or date.today()
        sections: list[str] = []

        for ticker in tickers:
            # 获取财务数据
            fin_res = self.call_tool("fetch_financials_pit", fetch_financials_pit, ticker, as_of)
            financials = fin_res.data or {}

            # 计算比率
            ratios = compute_financial_ratios(financials)

            # DCF
            dcf_res = self.call_tool("compute_dcf_valuation", compute_dcf_valuation, financials)
            dcf = dcf_res.data or {}

            # 获取同行并计算可比估值
            peers_res = self.call_tool("fetch_industry_peers", fetch_industry_peers, ticker, as_of, 8)
            peers = peers_res.data or []

            # 目标公司基本信息
            basics = state.snapshot.tickers if state.snapshot else []
            target_basics: dict[str, Any] = {}
            if basics:
                for td in basics:
                    if td.ticker == ticker:
                        target_basics = {"pe_ttm": td.pe_ttm, "pb": td.pb, "market_cap": td.market_cap}
                        break

            multiples = compute_comparable_multiples(target_basics, peers)

            # 存储计算结果到 state（供 LLM 不重复计算）
            section = (
                f"=== {ticker} 财务数据 ===\n"
                f"财务比率：{json.dumps(ratios, ensure_ascii=False)}\n"
                f"DCF 估值：{json.dumps(dcf, ensure_ascii=False)}\n"
                f"可比估值：{json.dumps(multiples, ensure_ascii=False)}\n"
                f"同行数量：{len(peers)}\n"
            )
            sections.append(section)

            # 把计算结果缓存到 state（parse_output 会重新从 state 读）
            if not hasattr(state, "_fundamental_cache"):
                state._fundamental_cache = {}  # type: ignore[attr-defined]
            state._fundamental_cache[ticker] = {  # type: ignore[attr-defined]
                "ratios": ratios,
                "dcf": dcf,
                "multiples": multiples,
                "peers": peers,
                "financials": financials,
            }

        return "\n".join(sections) + "\n\n请对每只股票进行基本面分析，输出 JSON。"

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
        resp = self.llm_chat(prompt, max_tokens=2048)
        parsed = self.parse_output(resp.content)

        for ticker in state.query.tickers:
            cache = getattr(state, "_fundamental_cache", {}).get(ticker, {})
            ratios = cache.get("ratios", {})
            dcf = cache.get("dcf", {})
            multiples = cache.get("multiples", {})

            fa = FundamentalAnalysis(
                ticker=ticker,
                as_of=as_of,
                ratios=ratios,
                dcf_value=dcf.get("per_share_value") or dcf.get("intrinsic_value"),
                dcf_assumptions=dcf.get("assumptions", {}),
                comparable_value=multiples.get("peer_median_pe"),
                peer_comparison={
                    "pe_premium": multiples.get("premium_discount_pe", 0),
                    "pb_premium": multiples.get("premium_discount_pb", 0),
                    "peer_median_pe": multiples.get("peer_median_pe"),
                    "peer_median_pb": multiples.get("peer_median_pb"),
                },
                profitability_analysis=parsed.get("profitability_analysis", ""),
                solvency_analysis=parsed.get("solvency_analysis", ""),
                growth_analysis=parsed.get("growth_analysis", ""),
                valuation_analysis=parsed.get("valuation_analysis", ""),
                key_risks=parsed.get("key_risks", []),
                summary=parsed.get("summary", ""),
                confidence=float(parsed.get("confidence", 0.5)),
            )
            state.fundamentals[ticker] = fa

        logger.info(f"[FundamentalAgent] analyzed {list(state.fundamentals.keys())}")
        return state
