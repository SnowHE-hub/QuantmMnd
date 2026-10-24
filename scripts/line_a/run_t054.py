"""scripts/line_a/run_t054.py — T-054：本机小模型的代码辅助（S1–S4）与 DeepSeek 参照组（R），留出集 4 / 开发集.

- R：DeepSeek，v3.1 单次调用（生产默认）；
- S1：本机 9B，v3.1；S2：+ 紧凑证据视图；S3：+ 校验器反馈修复；S4：+ 代码预填事实表；
- D2：2B 底座 + A1 适配器（本机 peft 服务，--model 指定），代码辅助用 --aids 指定。
用法：python scripts/line_a/run_t054.py --arm R|S1|S2|S3|S4|D2 --set holdout4|dev --lake <lake> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

SETS = {"holdout4": (20261029, ["2023-07-10", "2024-07-10", "2025-03-10", "2026-01-12"]),
        "dev": (20261031, ["2022-09-09", "2023-12-11", "2025-06-10"])}       # 蒸馏开发集时点，用来选 D2 的代码辅助
ARMS = {"R": ("deepseek", frozenset()), "S1": ("local", frozenset()), "S2": ("local", frozenset({"compact"})),
        "S3": ("local", frozenset({"compact", "repair"})), "S4": ("local", frozenset({"compact", "repair", "prefill"}))}


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
    ap.add_argument("--arm", required=True)
    ap.add_argument("--set", choices=list(SETS), default="holdout4")
    ap.add_argument("--lake", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="qwen3.5-9b-q4km")
    ap.add_argument("--url", default=None, help="本机后端地址（D2 用 2B 服务）")
    ap.add_argument("--aids", default=None, help="逗号分隔，覆盖该组默认的代码辅助")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    seed, as_ofs = SETS[a.set]
    cases = sample_cases(a.lake, as_ofs, per_date=10, seed=seed)
    (a.out / "cases.json").write_text(cases.to_json(orient="records", force_ascii=False), encoding="utf-8")
    if a.limit:
        cases = cases.head(a.limit)
    backend, aids = ARMS.get(a.arm, ("local", frozenset()))
    if a.aids is not None:
        aids = frozenset(x for x in a.aids.split(",") if x)
    if backend == "local":
        import os
        if a.url:
            os.environ["QM_LOCAL_LLM_URL"] = a.url
        llm = LLM(model=a.model, local=True, timeout_s=1800, ledger=a.out / "llm_ledger.jsonl", cache_dir=a.out / "_llm_cache")
        base_spec = slices.role_spec

        def slow_spec(*args, **kw):
            s = base_spec(*args, **kw)
            s.wall_clock_s, s.thinking = 1800.0, None
            s.max_tokens = 8192          # 本机输出上限（T-054 变更：补充轮会退化成重复输出、写满 32K，A1 正常输出约 3K）
            return s
        slices.role_spec = slow_spec
    else:
        llm = LLM(ledger=a.out / "llm_ledger.jsonl", cache_dir=a.out / "_llm_cache", budget_rmb=8.0)
    res_path = a.out / "results.jsonl"
    done = {json.loads(x)["case_id"] for x in res_path.read_text(encoding="utf-8").splitlines() if x.strip()} \
        if res_path.exists() else set()
    for _, c in cases.iterrows():
        if c["case_id"] in done:
            continue
        t0 = time.time()
        try:
            r = slices.run_slice(llm, ResearchTools(a.lake, c["as_of"]), c["ticker"], "A1", a.out / "runs", single=True,
                                 v3=True, structured_a1=True, aids=aids)
        except (BudgetExceededError, TLSInterceptionError) as e:
            print(f"stop: {type(e).__name__}: {e}", flush=True)
            break
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ in ("ConnectError", "ConnectTimeout") and llm.local:
                print(f"stop: local backend unreachable: {type(e).__name__}", flush=True)
                sys.exit(3)
            r = {"output": None, "checks": [], "status": f"error: {type(e).__name__}", "cost_rmb": 0.0,
                 "errors": [str(e)[:300]], "steps": []}
        s = score_a1(r["output"], r["checks"], gold_facts_a1(a.lake, c["ticker"], c["as_of"]), r["status"])
        rec = {"case_id": c["case_id"], "as_of": c["as_of"], "ticker": c["ticker"], "status": r["status"],
               "cost_rmb": r["cost_rmb"], "secs": round(time.time() - t0, 1), **s, "repair": r.get("repair"),
               "errors": r.get("errors", [])[:3], "path": r.get("path")}
        with res_path.open("a", encoding="utf-8") as f:
            f.write(canonical(rec) + "\n")
        print(json.dumps({k: rec[k] for k in ("case_id", "status", "recall", "secs")}, ensure_ascii=False), flush=True)
    rows = [json.loads(x) for x in res_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    ok = [r for r in rows if not r.get("abstained")]
    summ = {"arm": a.arm, "set": a.set, "aids": sorted(aids), "model": llm.model, "n_cases": len(rows), "n_ok": len(ok),
            "abstain_rate": 1 - len(ok) / max(1, len(rows)),
            "mean_recall": sum(r["recall"] for r in ok) / max(1, len(ok)),
            "mean_verified_share": sum(r["verified_share"] for r in ok) / max(1, len(ok)),
            "mean_problem_share": sum(r["problem_share"] for r in ok) / max(1, len(ok)),
            "repair_used": sum(bool((r.get("repair") or {}).get("used")) for r in rows),
            "secs_mean": sum(r["secs"] for r in rows) / max(1, len(rows)), "cost_rmb_total": sum(r["cost_rmb"] for r in rows),
            "seed": seed, "as_ofs": as_ofs}
    (a.out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print("SUMMARY", json.dumps(summ, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
