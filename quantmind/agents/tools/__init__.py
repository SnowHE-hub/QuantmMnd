"""quantmind.agents.tools — 工具集."""

from quantmind.agents.tools.data_tools import (
    fetch_stock_basics,
    fetch_financials_pit,
    fetch_price_history,
    fetch_industry_peers,
    fetch_recent_news,
)
from quantmind.agents.tools.analysis_tools import (
    compute_financial_ratios,
    compute_dcf_valuation,
    compute_comparable_multiples,
    compute_technical_indicators,
    run_factor_screening,
)
from quantmind.agents.tools.kb_tools import (
    search_research_reports,
    search_news,
    search_company_filings,
)
from quantmind.agents.tools.quant_tools import (
    get_factor_signal,
    get_llm_rerank_thesis,
    get_backtest_performance,
)
from quantmind.agents.tools.registry import (
    register_tool,
    get_tool,
    get_tools_for_agent,
    list_tools,
    REGISTRY,
)

__all__ = [
    "fetch_stock_basics", "fetch_financials_pit", "fetch_price_history",
    "fetch_industry_peers", "fetch_recent_news",
    "compute_financial_ratios", "compute_dcf_valuation", "compute_comparable_multiples",
    "compute_technical_indicators", "run_factor_screening",
    "search_research_reports", "search_news", "search_company_filings",
    "get_factor_signal", "get_llm_rerank_thesis", "get_backtest_performance",
    "register_tool", "get_tool", "get_tools_for_agent", "list_tools", "REGISTRY",
]
