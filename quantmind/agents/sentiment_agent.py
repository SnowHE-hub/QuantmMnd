"""quantmind.agents.sentiment_agent — SentimentAgent.

调用 kb_tools 检索研报（stub）、data_tools 取近期新闻，LLM 提取事件和情绪。
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from typing import Any

from loguru import logger

from quantmind.agents.base import BaseAgent
from quantmind.agents.tools import (
    fetch_recent_news,
    search_research_reports,
    search_news,
)
from quantmind.core.state import (
    AgentState,
    AgentType,
    NewsItem,
    SentimentAnalysis,
)

__all__ = ["SentimentAgent"]

_SYSTEM_PROMPT = """你是一位专业的市场情绪分析师。
你将收到公司的最新新闻和研究报告摘要，请分析：

1. 主要催化剂（近期利好事件）
2. 主要风险点（近期利空因素）
3. 情绪倾向（-1.0 极度悲观 到 1.0 极度乐观）
4. 重要事件列表（标注影响程度 1-5）

注意：
- 严格区分事实和预测
- 标注信息来源（新闻/研报/公告）
- 当前知识库未初始化，以新闻摘要为主

输出 JSON：
{
  "aggregated_sentiment": -1.0到1.0,
  "summary": "情绪摘要（100字内）",
  "catalysts": ["催化剂1", "催化剂2"],
  "risks": ["风险1", "风险2"],
  "key_events": [
    {"title": "事件标题", "impact": 1-5, "sentiment": -1.0到1.0}
  ],
  "confidence": 0.0-1.0
}"""


class SentimentAgent(BaseAgent):
    agent_type = AgentType.SENTIMENT
    description = "情绪分析 Agent：研报检索（stub）+ 新闻 + LLM 情绪提取"
    system_prompt = _SYSTEM_PROMPT

    def format_input(self, state: AgentState) -> str:
        as_of = state.query.as_of or date.today()
        start = as_of - timedelta(days=30)
        sections: list[str] = []

        for ticker in state.query.tickers:
            # 研报检索（当前为 stub）
            reports_res = self.call_tool(
                "search_research_reports", search_research_reports,
                ticker, (start, as_of), 5
            )
            reports = reports_res.data or []

            # 新闻
            news_res = self.call_tool("fetch_recent_news", fetch_recent_news, ticker, 30, as_of)
            news = news_res.data or []

            # kb_tools 新闻搜索（stub）
            kb_news_res = self.call_tool("search_news", search_news, ticker, (start, as_of), 10)
            kb_news = kb_news_res.data or []

            all_news = news + kb_news

            # 缓存
            if not hasattr(state, "_sentiment_cache"):
                state._sentiment_cache = {}  # type: ignore[attr-defined]
            state._sentiment_cache[ticker] = {  # type: ignore[attr-defined]
                "reports": reports,
                "news": all_news,
            }

            section = (
                f"=== {ticker} 情绪数据 ===\n"
                f"研究报告：{len(reports)} 篇（知识库 stub，当前为空）\n"
                f"新闻条数：{len(all_news)}\n"
            )
            if reports:
                for r in reports[:3]:
                    section += f"  - 研报：{r.get('title', '')} ({r.get('date', '')})\n"
            if all_news:
                for n in all_news[:5]:
                    section += f"  - 新闻：{n.get('title', '')} ({n.get('published_at', '')})\n"
            else:
                section += "  （无新闻数据，请基于公司基本面和行业背景进行情绪判断）\n"

            sections.append(section)

        return "\n".join(sections) + "\n\n请对每只股票进行情绪分析，输出 JSON。"

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
            cache = getattr(state, "_sentiment_cache", {}).get(ticker, {})
            news_raw = cache.get("news", [])

            news_items: list[NewsItem] = []
            for n in news_raw:
                try:
                    from datetime import datetime
                    pub = n.get("published_at")
                    if isinstance(pub, str):
                        try:
                            pub = datetime.fromisoformat(pub)
                        except ValueError:
                            pub = datetime.now()
                    elif pub is None:
                        pub = datetime.now()
                    news_items.append(NewsItem(
                        title=n.get("title", ""),
                        published_at=pub,
                        source=n.get("source", ""),
                        summary=n.get("summary", ""),
                        sentiment=float(n.get("sentiment", 0.0)),
                    ))
                except Exception:  # noqa: BLE001
                    pass

            sentiment_val = parsed.get("aggregated_sentiment", 0.0)
            try:
                sentiment_val = max(-1.0, min(1.0, float(sentiment_val)))
            except (ValueError, TypeError):
                sentiment_val = 0.0

            sa = SentimentAnalysis(
                ticker=ticker,
                as_of=as_of,
                news_items=news_items,
                aggregated_sentiment=sentiment_val,
                summary=parsed.get("summary", ""),
                catalysts=parsed.get("catalysts", []),
                risks=parsed.get("risks", []),
                confidence=float(parsed.get("confidence", 0.5)),
            )
            state.sentiments[ticker] = sa

        logger.info(f"[SentimentAgent] analyzed {list(state.sentiments.keys())}")
        return state
