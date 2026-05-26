"""End-to-end test: ValuationAgent with 五粮液 (000858.SZ) mode='full'."""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

# ── 构建 snapshot（五粮液 000858.SZ）──
snapshot = {
    "ticker":          "000858.SZ",
    "as_of":           pd.Timestamp("2025-03-31"),
    "pe_ttm":          25.4,
    "pb":              7.2,
    "ps_ttm":          8.1,
    "ev_ebitda":       19.3,
    "roe":             0.285,
    "roa":             0.195,
    "net_margin":      0.289,
    "revenue_growth":  0.082,
    "earnings_growth": 0.075,
    "debt_to_equity":  0.12,
    "current_ratio":   1.95,
    "fcf_yield":       0.038,
    "industry":        "白酒",
    "market_cap":      5.2e11,
    "beta":            0.78,
    "vol_63d":         0.218,
}

from quantmind.agents.investment_agents.valuation_agent import ValuationAgent

agent = ValuationAgent(
    ticker  = "000858.SZ",
    as_of   = pd.Timestamp("2025-03-31"),
    context = snapshot,
)

print("══════════════════════════════════════════════════════")
print("   五粮液 000858.SZ  ValuationAgent E2E  (mode=full) ")
print("══════════════════════════════════════════════════════")

sig = agent.analyze(mode="full")

print(f"Signal:       {sig.signal:+.3f}")
print(f"Confidence:   {sig.confidence:.3f}")
print(f"LLM mode:     {sig.llm_mode}")
print(f"Tools called: {sig.tools_called}")
print(f"Steps taken:  {len(sig.reasoning_trace)}")
print()
print("── Summary ──────────────────────────────────────────")
print(sig.summary[:500])
print()

if sig.reasoning_trace:
    print("── Reasoning Trace ──────────────────────────────────")
    for step in sig.reasoning_trace:
        step_id = step.get("step", "?")
        tools   = step.get("tool_calls", [])
        content = step.get("content", "")
        print(f"  [step {step_id}] tools={tools}")
        if content:
            print(f"    {content[:150]}")
        result = step.get("tool_result", "")
        if result:
            print(f"    → result: {result[:100]}")
