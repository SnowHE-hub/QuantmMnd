"""scripts/line_a/run_x7_x8.py — X7 后端选型与 X8 单次调用对照（T-041；规则在运行前登记）.

三种运行方式，都在 A1 留出集（种子 20261022，四个时点，40 个案例）上，用同一份标准事实和打分：
- loop：DeepSeek 工具循环（基线；用当前代码和数据重跑，请求完全相同的会命中缓存，不重复付费）；
- local：本机 Ollama 的 Qwen3.5-9B 工具循环（--model，默认 qwen3.5:9b-16k；角色墙钟放宽到 1,800 秒）；
- single：DeepSeek 单次调用。代码先用默认参数调用 A1 的四个工具，把结果一次性放进消息，模型不再调用工具。
用法：python scripts/line_a/run_x7_x8.py --mode loop|local|single --lake <lake> --out <dir> [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

AS_OFS = ["2024-05-10", "2024-09-10", "2024-11-11", "2025-05-12"]
SEED = 20261022
A1_TOOLS = ("financial_snapshot", "forecasts_and_express", "disclosure_schedule", "audit_and_pledge")


def single_call(llm, dt, ticker: str) -> dict:
    """X8 单次调用：同样的工具结果、同样的 Schema 与修复机会，只是不让模型自己调用工具。"""
    from quantmind.line_a.harness.llm import canonical
    from quantmind.line_a.harness.loop import (
        DATA_NOTICE,
        MAX_TOOL_CHARS,
        RoleSpec,
        collect_evidence,
        run_role,
    )
    from quantmind.line_a.harness.roles import A1_SYSTEM, RoleOutput
    from quantmind.line_a.harness.verifier import verify
    evidence: dict = {}
    blocks = []
    for name in A1_TOOLS:
        try:
            out = getattr(dt, name)(ticker)
            collect_evidence(out, evidence)
        except Exception as e:  # noqa: BLE001
            out = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        body = canonical(out)
        blocks.append(f"[{name}] " + DATA_NOTICE + (body[:MAX_TOOL_CHARS] + "…[truncated]" if len(body) > MAX_TOOL_CHARS
                                                    else body))
    task = (f"研究对象：{ticker}；as_of = {dt.as_of}（只能使用这一天及之前已发布的信息）。下面已经给出四个工具的结果，"
            "不能再调用工具，请直接输出 JSON。\n\n" + "\n\n".join(blocks))
    spec = RoleSpec("A1", A1_SYSTEM, [], RoleOutput, max_steps=3, thinking="high", max_tokens=32000, max_tool_calls=0)
    rec = run_role(llm, spec, task, meta={"ticker": ticker, "as_of": dt.as_of, "x8": "single"})
    rec.evidence.update(evidence)
    checks = [asdict(c) for c in verify(rec.output, rec.evidence)] if rec.output and "claims" in rec.output else []
    return {"ticker": ticker, "as_of": dt.as_of, "role": "A1", "status": rec.status, "output": rec.output,
            "checks": checks, "steps": rec.steps, "errors": rec.errors, "cost_rmb": rec.cost_rmb,
            "evidence": rec.evidence, "model": llm.model}


def main() -> None:
    from quantmind.line_a.harness import slices
    from quantmind.line_a.harness.evalset import gold_facts_a1, sample_cases, score_a1
    from quantmind.line_a.harness.llm import (
        LLM,
        BudgetExceededError,
        TLSInterceptionError,
        canonical,
    )
    from quantmind.line_a.harness.tools_more import ResearchTools
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["loop", "local", "single"], required=True)
    ap.add_argument("--lake", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="qwen3.5:9b-16k")
    ap.add_argument("--budget", type=float, default=8.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cases = sample_cases(a.lake, AS_OFS, per_date=10, seed=SEED)
    if a.limit:
        cases = cases.head(a.limit)
    if a.mode == "local":
        llm = LLM(model=a.model, local=True, timeout_s=1800, ledger=a.out / "llm_ledger.jsonl",
                  cache_dir=a.out / "_llm_cache")
        base_spec = slices.role_spec

        def slow_spec(*args, **kw):                         # 本机生成慢：角色墙钟 900 → 1,800 秒（已登记）
            s = base_spec(*args, **kw)
            s.wall_clock_s = 1800.0
            return s
        slices.role_spec = slow_spec
    else:
        llm = LLM(ledger=a.out / "llm_ledger.jsonl", cache_dir=a.out / "_llm_cache", budget_rmb=a.budget)
    res_path = a.out / "results.jsonl"
    done = {json.loads(x)["case_id"] for x in res_path.read_text(encoding="utf-8").splitlines() if x.strip()} \
        if res_path.exists() else set()
    for _, c in cases.iterrows():
        if c["case_id"] in done:
            continue
        t0 = time.time()
        dt = ResearchTools(a.lake, c["as_of"])
        try:
            if a.mode == "single":
                r = single_call(llm, dt, c["ticker"])
                p = a.out / "runs" / c["as_of"] / c["ticker"] / "A1_single.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(canonical(r), encoding="utf-8")
                r["path"] = str(p)
            else:
                r = slices.run_slice(llm, dt, c["ticker"], "A1", a.out / "runs")
        except (BudgetExceededError, TLSInterceptionError) as e:
            print(f"stop: {type(e).__name__}: {e}", flush=True)
            break
        gold = gold_facts_a1(a.lake, c["ticker"], c["as_of"])
        s = score_a1(r["output"], r["checks"], gold, r["status"])
        rec = {"case_id": c["case_id"], "as_of": c["as_of"], "ticker": c["ticker"], "board": c["board"],
               "status": r["status"], "cost_rmb": r["cost_rmb"], "secs": round(time.time() - t0, 1), **s,
               "errors": r.get("errors", [])[:3], "path": r.get("path")}
        with res_path.open("a", encoding="utf-8") as f:
            f.write(canonical(rec) + "\n")
        print(json.dumps({k: rec[k] for k in ("case_id", "ticker", "as_of", "status", "recall", "secs")},
                         ensure_ascii=False), flush=True)
    rows = [json.loads(x) for x in res_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    ok = [r for r in rows if not r.get("abstained")]
    summ = {"mode": a.mode, "model": llm.model, "n_cases": len(rows), "n_ok": len(ok),
            "abstain_rate": 1 - len(ok) / max(1, len(rows)),
            "mean_recall": sum(r["recall"] for r in ok) / max(1, len(ok)),
            "mean_verified_share": sum(r["verified_share"] for r in ok) / max(1, len(ok)),
            "mean_problem_share": sum(r["problem_share"] for r in ok) / max(1, len(ok)),
            "cost_rmb_total": sum(r["cost_rmb"] for r in rows), "secs_mean": sum(r["secs"] for r in rows) / max(1, len(rows)),
            "status_counts": {s: sum(1 for r in rows if r["status"] == s) for s in {r["status"] for r in rows}},
            "seed": SEED, "as_ofs": AS_OFS}
    missed: dict = {}
    for r in ok:
        for m in r.get("missed", []):
            missed[m] = missed.get(m, 0) + 1
    summ["missed_fact_counts"] = missed
    (a.out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print("SUMMARY", json.dumps(summ, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
