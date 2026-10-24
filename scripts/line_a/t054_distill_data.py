"""scripts/line_a/t054_distill_data.py — T-054 D2 的蒸馏数据：DeepSeek v3.1（生产口径）作教师.

- 时点：2021-01 到 2026-08 每月第二个交易日，去掉与开发集、留出集 1–4 相差 20 天以内的时点；
- 每个时点 10 个案例（evalset.sample_cases，种子 20261030）；
- 每个案例存：四个工具的原始结果（之后可以按任意代码辅助重建学生的输入）、教师输出、校验结果；
- 6 路并行，每路一个 LLM 对象；已完成的案例跳过（断点续跑）。
用法：python scripts/line_a/t054_distill_data.py --lake <lake> --out <dir> [--workers 6] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

EXCLUDE = ["2023-05-10", "2023-09-10", "2023-11-10", "2024-04-10", "2024-05-10", "2024-09-10", "2024-11-11", "2025-05-12",
           "2025-09-10", "2025-11-10", "2026-05-11", "2026-09-10", "2023-03-10", "2025-01-10", "2025-04-10", "2026-03-10",
           "2023-07-10", "2024-07-10", "2025-03-10", "2026-01-12"]


def as_ofs(lake: Path) -> list[str]:
    cal = pd.read_parquet(lake / "trade_cal_sse.parquet")
    if "is_open" in cal:
        cal = cal[cal["is_open"].astype(int) == 1]
    d = pd.DatetimeIndex(pd.to_datetime(cal["cal_date"].astype(str))).sort_values()
    d = d[(d >= "2021-01-01") & (d <= "2026-08-31")]
    picks = [g.iloc[1] for _, g in pd.Series(d, index=d).groupby([d.year, d.month]) if len(g) > 1]
    ex = pd.DatetimeIndex(pd.to_datetime(EXCLUDE))
    return [str(p.date()) for p in picks if (abs(ex - p).days > 20).all()]


def main() -> None:
    from quantmind.line_a.harness import slices
    from quantmind.line_a.harness.evalset import sample_cases
    from quantmind.line_a.harness.llm import (
        LLM,
        BudgetExceededError,
        TLSInterceptionError,
        canonical,
    )
    from quantmind.line_a.harness.tools_more import ResearchTools
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget", type=float, default=40.0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dates = as_ofs(a.lake)
    cases = sample_cases(a.lake, dates, per_date=10, seed=20261030)
    cases.to_parquet(a.out / "cases.parquet")
    if a.limit:
        cases = cases.head(a.limit)
    done = {p.stem for p in (a.out / "cases").glob("*.json")} if (a.out / "cases").exists() else set()
    (a.out / "cases").mkdir(exist_ok=True)
    todo = [r for _, r in cases.iterrows() if r["case_id"] not in done]
    print(f"{len(dates)} as_ofs, {len(cases)} cases, {len(todo)} to do", flush=True)
    lock, stop = threading.Lock(), threading.Event()
    spent = [0.0]

    def work(c) -> None:
        if stop.is_set():
            return
        llm = LLM(ledger=a.out / "llm_ledger.jsonl", cache_dir=a.out / "_llm_cache", budget_rmb=a.budget)
        raw: dict = {}
        try:
            r = slices.run_slice(llm, ResearchTools(a.lake, c["as_of"]), c["ticker"], "A1", None, raw_out=raw)
        except (BudgetExceededError, TLSInterceptionError) as e:
            print(f"stop: {type(e).__name__}", flush=True)
            stop.set()
            return
        except Exception as e:  # noqa: BLE001
            print(f"case {c['case_id']} error {type(e).__name__}: {str(e)[:160]}", flush=True)
            return
        rec = {"case": c.to_dict(), "raw_tools": raw, "teacher": r["output"], "checks": r["checks"], "status": r["status"],
               "cost_rmb": r["cost_rmb"]}
        (a.out / "cases" / f"{c['case_id']}.json").write_text(canonical(rec), encoding="utf-8")
        with lock:
            spent[0] += r["cost_rmb"]
            if spent[0] > a.budget:
                stop.set()
        print(json.dumps({"case": c["case_id"], "status": r["status"], "n_claims": len((r["output"] or {}).get("claims", [])),
                          "spent": round(spent[0], 2)}), flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, todo))
    print("DONE", len(list((a.out / "cases").glob("*.json"))), "cases; spent", round(spent[0], 2), flush=True)


if __name__ == "__main__":
    main()
