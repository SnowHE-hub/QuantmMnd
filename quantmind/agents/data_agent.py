"""quantmind.agents.data_agent — DataAgent.

接收 Planner 派发的 data 类任务，调用 data_tools 取数，写入 state.snapshot。
失败自动重试（最多3次）。
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Any

from loguru import logger

from quantmind.agents.base import BaseAgent
from quantmind.agents.tools import (
    fetch_stock_basics,
    fetch_financials_pit,
    fetch_price_history,
    fetch_industry_peers,
    fetch_recent_news,
)
from quantmind.core.state import (
    AgentState,
    AgentType,
    DataSnapshot,
    TickerData,
)

__all__ = ["DataAgent"]

_MAX_RETRY = 3


class DataAgent(BaseAgent):
    agent_type = AgentType.DATA
    description = "数据收集 Agent，从 snapshot 中提取所需数据并写入 state"
    system_prompt = ""  # DataAgent 不需要 LLM
    max_iterations = _MAX_RETRY

    def format_input(self, state: AgentState) -> str:
        return f"tickers={state.query.tickers}, as_of={state.query.as_of}"

    def parse_output(self, llm_response: str) -> dict[str, Any]:
        return {}  # DataAgent 不用 LLM

    def _execute(self, state: AgentState) -> AgentState:
        query = state.query
        tickers = query.tickers
        as_of: date = query.as_of or date.today()

        ticker_data_list: list[TickerData] = []

        for ticker in tickers:
            td = self._fetch_one_ticker(ticker, as_of, state)
            if td:
                ticker_data_list.append(td)

        # 更新 state.snapshot
        state.snapshot = DataSnapshot(
            as_of=as_of,
            universe="custom",
            tickers=ticker_data_list,
            data_sources=["tushare_snapshot"],
            coverage_ratio=len(ticker_data_list) / max(len(tickers), 1),
        )

        logger.info(
            f"[DataAgent] fetched {len(ticker_data_list)}/{len(tickers)} tickers "
            f"for {as_of}"
        )
        return state

    def _fetch_one_ticker(
        self, ticker: str, as_of: date, state: AgentState
    ) -> TickerData | None:
        for attempt in range(1, _MAX_RETRY + 1):
            try:
                basics = self.call_tool("fetch_stock_basics", fetch_stock_basics, ticker, as_of)
                if basics.ok:
                    b = basics.data
                    return TickerData(
                        ticker=ticker,
                        name=b.get("name"),
                        industry=b.get("industry"),
                        sector=b.get("sector"),
                        is_tradable=b.get("is_tradable", True),
                        latest_close=b.get("latest_close"),
                        market_cap=b.get("market_cap"),
                        pe_ttm=b.get("pe_ttm"),
                        pb=b.get("pb"),
                    )
                else:
                    logger.warning(
                        f"[DataAgent] fetch_stock_basics({ticker}) attempt {attempt} failed"
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[DataAgent] ticker {ticker} attempt {attempt}: {e}")
                if attempt < _MAX_RETRY:
                    time.sleep(0.5 * attempt)

        return TickerData(ticker=ticker)  # 返回空壳，避免完全跳过
