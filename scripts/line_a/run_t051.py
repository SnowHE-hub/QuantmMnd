"""scripts/line_a/run_t051.py — T-051：A1 harness v3 在留出集 2 上的三组对照（规则在运行前登记）.

- B：DeepSeek，单次调用（当前默认，v3 关闭）；
- V：DeepSeek，单次调用 + v3（可读金额、A1 完整性检查）；
- L：本机 llama-server 上的 Qwen3.5-9B，单次调用 + v3，关闭思考，约束解码（角色墙钟 1,800 秒）；
- V31 / L31：同 V / L，A1 改用不带引文的证据 Schema（v3.1，在留出集 3 上评测）。
三组用同一个校验器（含 v3 的文字字段核对）、同一份标准事实和打分函数。
用法：python scripts/line_a/run_t051.py --arm B|V|L --lake <lake> --out <dir> [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

SETS = {"holdout2": (20261026, ["2025-09-10", "2025-11-10", "2026-05-11", "2026-09-10"]),
        "holdout3": (20261028, ["2023-03-10", "2025-01-10", "2025-04-10", "2026-03-10"])}


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
    ap.add_argument("--arm", choices=["B", "V", "L", "V31", "L31"], required=True)
    ap.add_argument("--set", choices=list(SETS), default="holdout2")
    ap.add_argument("--lake", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="qwen3.5-9b-q4km")
    ap.add_argument("--budget", type=float, default=8.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    SEED, AS_OFS = SETS[a.set]
    cases = sample_cases(a.lake, AS_OFS, per_date=10, seed=SEED)
    (a.out / "cases.json").write_text(cases.to_json(orient="records", force_ascii=False), encoding="utf-8")
    if a.limit:
        cases = cases.head(a.limit)
    if a.arm in ("L", "L31"):
        llm = LLM(model=a.model, local=True, timeout_s=1800, ledger=a.out / "llm_ledger.jsonl",
                  cache_dir=a.out / "_llm_cache")
        base_spec = slices.role_spec

        def slow_spec(*args, **kw):
            s = base_spec(*args, **kw)
            s.wall_clock_s, s.thinking = 1800.0, None
            return s
        slices.role_spec = slow_spec
    else:
        llm = LLM(ledger=a.out / "llm_ledger.jsonl", cache_dir=a.out / "_llm_cache", budget_rmb=a.budget)
    v3 = a.arm != "B"
    structured = a.arm in ("V31", "L31")
    res_path = a.out / "results.jsonl"
    done = {json.loads(x)["case_id"] for x in res_path.read_text(encoding="utf-8").splitlines() if x.strip()} \
        if res_path.exists() else set()
    for _, c in cases.iterrows():
        if c["case_id"] in done:
            continue
        t0 = time.time()
        try:
            r = slices.run_slice(llm, ResearchTools(a.lake, c["as_of"]), c["ticker"], "A1", a.out / "runs",
                                 single=True, v3=v3, structured_a1=structured)
        except (BudgetExceededError, TLSInterceptionError) as e:
            print(f"stop: {type(e).__name__}: {e}", flush=True)
            break
        except Exception as e:  # noqa: BLE001  单个案例出错：记为 error，接着跑下一个
            if type(e).__name__ in ("ConnectError", "ConnectTimeout") and llm.local:
                # 本机后端挂了（例如 llama-server 的 CUDA 错误）：这是基础设施故障，不记成案例结果，停下等重启
                print(f"stop: local backend unreachable: {type(e).__name__}", flush=True)
                sys.exit(3)
            r = {"output": None, "checks": [], "status": f"error: {type(e).__name__}", "cost_rmb": 0.0,
                 "errors": [str(e)[:300]], "steps": []}
        gold = gold_facts_a1(a.lake, c["ticker"], c["as_of"])
        s = score_a1(r["output"], r["checks"], gold, r["status"])
        rec = {"case_id": c["case_id"], "as_of": c["as_of"], "ticker": c["ticker"], "board": c["board"],
               "status": r["status"], "cost_rmb": r["cost_rmb"], "secs": round(time.time() - t0, 1), **s,
               "completeness_asked": any("completeness_missing" in st for st in r.get("steps", [])),
               "errors": r.get("errors", [])[:3], "path": r.get("path")}
        with res_path.open("a", encoding="utf-8") as f:
            f.write(canonical(rec) + "\n")
        print(json.dumps({k: rec[k] for k in ("case_id", "ticker", "as_of", "status", "recall", "secs")},
                         ensure_ascii=False), flush=True)
    rows = [json.loads(x) for x in res_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    ok = [r for r in rows if not r.get("abstained")]
    summ = {"arm": a.arm, "set": a.set, "model": llm.model, "v3": v3, "structured_a1": structured, "n_cases": len(rows), "n_ok": len(ok),
            "abstain_rate": 1 - len(ok) / max(1, len(rows)),
            "mean_recall": sum(r["recall"] for r in ok) / max(1, len(ok)),
            "mean_verified_share": sum(r["verified_share"] for r in ok) / max(1, len(ok)),
            "mean_problem_share": sum(r["problem_share"] for r in ok) / max(1, len(ok)),
            "completeness_asked": sum(r.get("completeness_asked", False) for r in rows),
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
