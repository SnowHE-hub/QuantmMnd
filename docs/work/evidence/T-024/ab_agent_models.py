"""T-024：日更 6-Agent 用 qwen2.5:7b 与 qwen3.5:9b 的对比（同一输入；只读，不写生产产物）.

输入：生产特征 data/features/alpha_2026-09-30.parquet；10 只票 = 流通市值最大的 5 只 + 其余随机 5 只（种子 0）。
每个模型：预热一次；对每只票依次运行 5 个辩论 Agent（Valuation / Momentum / Quality / Sentiment / Risk），mode = full。
记录：是否真正走了 LLM（llm_mode，否则为降级到规则）、工具调用次数、耗时、信号、置信度、摘要。
对比：LLM 成功率、耗时、两模型信号方向一致率（|signal| ≤ 0.1 记为中性）。
用法：PYTHONPATH=. python ab_agent_models.py <data_root> <out_dir>
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

root, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
MODELS = ["qwen2.5:7b", "qwen3.5:9b"]
KEYS = ["pe_ttm", "pb", "roe_ttm", "gross_margin", "revenue_yoy", "net_profit_yoy", "debt_to_assets",
        "momentum_1m", "momentum_3m", "momentum_6m", "volatility_3m", "rsi_14", "exposure_industry"]
AS_OF = "2026-09-30"

f = pd.read_parquet(root / f"data/features/alpha_{AS_OF}.parquet")
big = list(f.sort_values("log_circ_market_cap", ascending=False).index[:5])
rest = [t for t in f.index if t not in big]
tickers = big + list(np.random.default_rng(0).choice(rest, 5, replace=False))


def context(t: str) -> dict:
    snap = {k: (None if pd.isna(f.at[t, k]) else (f.at[t, k] if isinstance(f.at[t, k], str) else float(f.at[t, k])))
            for k in KEYS}
    ctx = {"ticker": t, "as_of": AS_OF, "news": [], "reports": [], "snapshot": snap,
           "lgbm_score": 0.0, "composite_score": 0.0, "regime": "neutral"}
    for k, v in snap.items():
        ctx[f"snapshot_{k}"] = v
    return ctx


rows = []
for model in MODELS:
    os.environ["QM_AGENT_MODEL"] = model
    import importlib

    import quantmind.agents.investment_agents as ia
    import quantmind.agents.ollama_client as oc
    importlib.reload(oc)
    ok, secs, msg = oc.OllamaReActClient.warm_up(model=model, timeout=600)
    print(f"[{model}] warm-up {ok} {secs:.1f}s {msg}", flush=True)
    agents = [ia.ValuationAgent, ia.MomentumAgent, ia.QualityAgent, ia.SentimentAgent, ia.RiskAgent]
    for t in tickers:
        for cls in agents:
            t0 = time.perf_counter()
            try:
                sig = cls(t, AS_OF, context(t)).analyze(mode="full")
                rec = {"model": model, "ticker": t, "agent": cls.__name__, "llm_mode": bool(sig.llm_mode),
                       "n_tools": len(sig.tools_called), "signal": float(sig.signal), "confidence": float(sig.confidence),
                       "summary": (sig.summary or "")[:120], "error": None}
            except Exception as e:  # noqa: BLE001
                rec = {"model": model, "ticker": t, "agent": cls.__name__, "llm_mode": False, "n_tools": 0,
                       "signal": np.nan, "confidence": np.nan, "summary": "", "error": f"{type(e).__name__}: {str(e)[:100]}"}
            rec["seconds"] = round(time.perf_counter() - t0, 2)
            rows.append(rec)
            print(f"  {model} {t} {cls.__name__} llm={rec['llm_mode']} tools={rec['n_tools']} {rec['seconds']}s "
                  f"sig={rec['signal']:+.2f}" if rec["error"] is None else f"  {model} {t} {cls.__name__} ERROR {rec['error']}",
                  flush=True)
df = pd.DataFrame(rows)
df.to_csv(out_dir / "ab_rows.csv", index=False)
summ = {}
for m, g in df.groupby("model"):
    summ[m] = {"n": int(len(g)), "llm_success_rate": round(float(g.llm_mode.mean()), 3),
               "errors": int(g.error.notna().sum()), "mean_seconds": round(float(g.seconds.mean()), 2),
               "median_seconds": round(float(g.seconds.median()), 2), "mean_tools": round(float(g.n_tools.mean()), 2),
               "by_agent_llm_success": g.groupby("agent").llm_mode.mean().round(2).to_dict()}
w = df.pivot_table(index=["ticker", "agent"], columns="model", values="signal")


def stance(x):
    return np.where(x > 0.1, 1, np.where(x < -0.1, -1, 0))


both = w.dropna()
summ["agreement"] = {"n_pairs": int(len(both)),
                     "stance_agreement": round(float((stance(both[MODELS[0]]) == stance(both[MODELS[1]])).mean()), 3),
                     "mean_abs_signal_diff": round(float((both[MODELS[0]] - both[MODELS[1]]).abs().mean()), 3)}
summ["tickers"] = tickers
(out_dir / "ab_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(summ, ensure_ascii=False, indent=1))
