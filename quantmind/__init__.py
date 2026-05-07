"""QuantMind: AI Agent-Driven Quantitative Investment Research System.

Three subsystems that mutually validate each other:
    1. Multi-Agent Research (LangGraph orchestrated)
    2. Generative Quant Selection (LightGBM + LLM Listwise Rerank + DPO)
    3. Rigorous Backtest (PIT-strict, Walk-Forward, Deflated Sharpe)
"""

__version__ = "0.1.0"
__author__ = "QuantMind Author"

__all__ = ["__version__"]
