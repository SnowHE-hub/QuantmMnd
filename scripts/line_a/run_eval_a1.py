"""scripts/line_a/run_eval_a1.py — A1 研究质量评测（T-037；评测集定义在运行前提交）.

评测集：
- quantmind.line_a.harness.evalset.sample_cases(lake, AS_OFS, per_date=10, seed=20261021)，共 40 个历史案例；
- 4 个时点：2023-05-10、2023-09-10、2023-11-10、2024-04-10，都在财报披露季之后。

每个案例：
- 跑一次 A1（DeepSeek deepseek-flash，思考模式 high）；
- 代码校验器逐条核对；
- 用结构化数据生成的标准事实打分。

汇总：
- 事实召回；
- 已核实占比、有问题声明的比例；
- 弃答率、每案例费用。

只检验"能否把材料里的事实准确取出来"，与收益无关。
用法：python scripts/line_a/run_eval_a1.py --lake <lake> --out <dir> [--budget 8] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

AS_OFS = ["2023-05-10", "2023-09-10", "2023-11-10", "2024-04-10"]
SEED = 20261021


def main() -> None:
    from quantmind.line_a.harness.evalset import gold_facts_a1, sample_cases, score_a1
    from quantmind.line_a.harness.llm import (
        LLM,
        BudgetExceededError,
        TLSInterceptionError,
        canonical,
    )
    from quantmind.line_a.harness.slices import run_slice
    from quantmind.line_a.harness.tools_more import ResearchTools
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--budget", type=float, default=8.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=SEED, help="v1 开发集 20261021；留出集用别的种子")
    ap.add_argument("--as-ofs", default=",".join(AS_OFS))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    as_ofs = a.as_ofs.split(",")
    cases = sample_cases(a.lake, as_ofs, per_date=10, seed=a.seed)
    if a.limit:
        cases = cases.head(a.limit)
    llm = LLM(ledger=a.out / "llm_ledger.jsonl", cache_dir=a.out / "_llm_cache", budget_rmb=a.budget)
    done = set()
    res_path = a.out / "results.jsonl"
    if res_path.exists():
        done = {json.loads(x)["case_id"] for x in res_path.read_text(encoding="utf-8").splitlines() if x.strip()}
    for _, c in cases.iterrows():
        if c["case_id"] in done:
            continue
        t0 = time.time()
        try:
            r = run_slice(llm, ResearchTools(a.lake, c["as_of"]), c["ticker"], "A1", a.out / "runs")
        except (BudgetExceededError, TLSInterceptionError) as e:
            print(f"stop: {type(e).__name__}: {e}", flush=True)
            break
        gold = gold_facts_a1(a.lake, c["ticker"], c["as_of"])
        s = score_a1(r["output"], r["checks"], gold, r["status"])
        rec = {"case_id": c["case_id"], "as_of": c["as_of"], "ticker": c["ticker"], "board": c["board"],
               "status": r["status"], "cost_rmb": r["cost_rmb"], "secs": round(time.time() - t0, 1), **s,
               "path": r.get("path")}
        with res_path.open("a", encoding="utf-8") as f:
            f.write(canonical(rec) + "\n")
        print(json.dumps({k: rec[k] for k in ("case_id", "ticker", "as_of", "status", "recall", "cost_rmb")},
                         ensure_ascii=False), flush=True)
    rows = [json.loads(x) for x in res_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    ok = [r for r in rows if not r.get("abstained")]
    summ = {"n_cases": len(rows), "n_ok": len(ok), "abstain_rate": 1 - len(ok) / max(1, len(rows)),
            "mean_recall": sum(r["recall"] for r in ok) / max(1, len(ok)),
            "mean_verified_share": sum(r["verified_share"] for r in ok) / max(1, len(ok)),
            "mean_problem_share": sum(r["problem_share"] for r in ok) / max(1, len(ok)),
            "cost_rmb_total": sum(r["cost_rmb"] for r in rows), "seed": a.seed, "as_ofs": as_ofs}
    missed: dict = {}
    for r in ok:
        for m in r.get("missed", []):
            missed[m] = missed.get(m, 0) + 1
    summ["missed_fact_counts"] = missed
    (a.out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print("SUMMARY", json.dumps(summ, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
