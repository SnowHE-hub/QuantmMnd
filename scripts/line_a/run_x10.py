"""scripts/line_a/run_x10.py — X10 前视偏差诊断 v1（T-046；规则在运行前登记）.

D1 数值指纹扰动：A1 工具结果里单位为元 / 万元的金额统一乘 K；看模型写出的金额跟着扰动值走，还是照抄真实值（泄漏）。
D2 闭卷对照：不给工具，问 as_of 之前最近报告期的营业收入和归母净利润，看凭记忆的命中率。
评测集：A1 留出集（种子 20261022，四个时点，40 个案例），与 T-041 相同。
用法：python scripts/line_a/run_x10.py --mode d1|d2 --lake <lake> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

K = 1.37
SCALE = {"元": 1.0, "万元": 1e4, "亿元": 1e8, "千元": 1e3}
AS_OFS = ["2024-05-10", "2024-09-10", "2024-11-11", "2025-05-12"]
SEED = 20261022
A1_TOOLS = ("financial_snapshot", "forecasts_and_express", "disclosure_schedule", "audit_and_pledge")


def scale_money(obj, k: float):
    """带 _units 的记录里，单位是元 / 万元的字段乘 k；其他字段和嵌套结构照搬。"""
    if isinstance(obj, list):
        return [scale_money(x, k) for x in obj]
    if not isinstance(obj, dict):
        return obj
    units = obj.get("_units") or {}
    out = {}
    for key, v in obj.items():
        if key != "_units" and units.get(key) in ("元", "万元") and isinstance(v, int | float) and not isinstance(v, bool):
            out[key] = v * k
        else:
            out[key] = scale_money(v, k)
    return out


def perturbed_tools(lake: Path, as_of: str, k: float):
    from quantmind.line_a.harness.tools_more import ResearchTools

    class Perturbed(ResearchTools):
        pass
    for name in A1_TOOLS:
        base = getattr(ResearchTools, name)

        def wrapped(self, *a, _base=base, **kw):
            return scale_money(_base(self, *a, **kw), k)
        setattr(Perturbed, name, wrapped)
    return Perturbed(lake, as_of)


def leak_stats(output: dict | None, evidence: dict, k: float) -> dict:
    n = tool = leak = 0
    rows = []
    for c in (output or {}).get("claims", []):
        if c.get("claim_type") != "reported_fact" or c.get("value") is None or c.get("unit") not in SCALE:
            continue
        v = float(c["value"]) * SCALE[c["unit"]]
        for e in c.get("evidence", []):
            rec = evidence.get(e.get("evidence_id"))
            f = e.get("field")
            if not rec or f not in rec or (rec.get("_units") or {}).get(f) not in ("元", "万元"):
                continue
            p = float(rec[f]) * SCALE[rec["_units"][f]]
            if p == 0:
                continue
            n += 1
            m_tool = abs(v - p) / abs(p) <= 0.005
            m_true = abs(v - p / k) / abs(p / k) <= 0.005
            tool += m_tool
            leak += (m_true and not m_tool)
            if m_true and not m_tool:
                rows.append({"text": c.get("text"), "value": c.get("value"), "unit": c.get("unit"), "field": f})
            break
    return {"n_money_facts": n, "n_match_tool": tool, "n_match_true_only": leak, "leaks": rows}


def main() -> None:
    import pandas as pd

    from quantmind.line_a.harness.evalset import gold_facts_a1, sample_cases, score_a1
    from quantmind.line_a.harness.llm import (
        LLM,
        BudgetExceededError,
        TLSInterceptionError,
        canonical,
    )
    from quantmind.line_a.harness.slices import run_slice
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["d1", "d2"], required=True)
    ap.add_argument("--lake", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--budget", type=float, default=8.0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cases = sample_cases(a.lake, AS_OFS, per_date=10, seed=SEED)
    llm = LLM(ledger=a.out / "llm_ledger.jsonl", cache_dir=a.out / "_llm_cache", budget_rmb=a.budget)
    res_path = a.out / "results.jsonl"
    done = {json.loads(x)["case_id"] for x in res_path.read_text(encoding="utf-8").splitlines() if x.strip()} \
        if res_path.exists() else set()
    names = {}
    if a.mode == "d2":
        sb = pd.read_parquet(sorted(a.lake.glob("stock_basic_full_merged_2*.parquet"))[-1], columns=["ts_code", "name"])
        names = dict(zip(sb["ts_code"], sb["name"], strict=True))
    for _, c in cases.iterrows():
        if c["case_id"] in done:
            continue
        t0 = time.time()
        gold = gold_facts_a1(a.lake, c["ticker"], c["as_of"])
        try:
            if a.mode == "d1":
                r = run_slice(llm, perturbed_tools(a.lake, c["as_of"], K), c["ticker"], "A1", a.out / "runs")
                s = score_a1(r["output"], r["checks"], gold, r["status"])
                rec = {"case_id": c["case_id"], "ticker": c["ticker"], "as_of": c["as_of"], "status": r["status"],
                       "cost_rmb": r["cost_rmb"], **s, **leak_stats(r["output"], r["evidence"], K)}
            else:
                truth = {g["metric"]: g for g in gold if g["fact_id"].endswith(".latest") and g["unit"] == "元"}
                q = (f"不使用任何工具，只凭你已有的知识回答：A 股上市公司 {names.get(c['ticker'], '')}（{c['ticker']}）"
                     f"在 {c['as_of']} 之前已披露的最近一个报告期，营业收入和归属于母公司股东的净利润各是多少元？"
                     '不知道就填 null。只输出 JSON：{"period": "YYYYMMDD 或 null", "revenue": 数字或 null, '
                     '"n_income_attr_p": 数字或 null}')
                out = llm.chat([{"role": "user", "content": q}], json_mode=True, thinking=None, max_tokens=300,
                               meta={"x10": "d2", "ticker": c["ticker"]})
                try:
                    ans = json.loads(out.content)
                except json.JSONDecodeError:
                    ans = {"unparseable": out.content[:200]}      # 记为没有回答数字
                if not isinstance(ans, dict):
                    ans = {"unparseable": out.content[:200]}
                rec = {"case_id": c["case_id"], "ticker": c["ticker"], "as_of": c["as_of"], "cost_rmb": out.cost_rmb,
                       "answer": ans}
                for m in ("revenue", "n_income_attr_p"):
                    v, g = ans.get(m), truth.get(m)
                    rec[f"{m}_answered"] = isinstance(v, int | float)
                    rec[f"{m}_hit"] = bool(g and isinstance(v, int | float) and g["value"]
                                           and abs(v - g["value"]) / abs(g["value"]) <= 0.05)
                    rec[f"{m}_true"] = g["value"] if g else None
        except (BudgetExceededError, TLSInterceptionError) as e:
            print(f"stop: {type(e).__name__}: {e}", flush=True)
            break
        rec["secs"] = round(time.time() - t0, 1)
        with res_path.open("a", encoding="utf-8") as f:
            f.write(canonical(rec) + "\n")
        print(json.dumps({k: rec.get(k) for k in ("case_id", "ticker", "status", "recall", "n_match_true_only",
                                                  "revenue_hit")}, ensure_ascii=False), flush=True)
    rows = [json.loads(x) for x in res_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if a.mode == "d1":
        ok = [r for r in rows if not r.get("abstained")]
        n = sum(r["n_money_facts"] for r in rows)
        summ = {"n_cases": len(rows), "abstain_rate": 1 - len(ok) / max(1, len(rows)),
                "mean_recall": sum(r["recall"] for r in ok) / max(1, len(ok)),
                "mean_verified_share": sum(r["verified_share"] for r in ok) / max(1, len(ok)),
                "mean_problem_share": sum(r["problem_share"] for r in ok) / max(1, len(ok)),
                "n_money_facts": n, "share_match_tool": sum(r["n_match_tool"] for r in rows) / max(1, n),
                "share_leak": sum(r["n_match_true_only"] for r in rows) / max(1, n),
                "leaks": [dict(x, ticker=r["ticker"], as_of=r["as_of"]) for r in rows for x in r["leaks"]],
                "cost_rmb": sum(r["cost_rmb"] for r in rows), "k": K}
    else:
        summ = {"n_cases": len(rows), **{f"{m}_{s}": sum(r[f"{m}_{s}"] for r in rows) / max(1, len(rows))
                                         for m in ("revenue", "n_income_attr_p") for s in ("answered", "hit")},
                "cost_rmb": sum(r["cost_rmb"] for r in rows)}
    (a.out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print("SUMMARY", json.dumps(summ, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
