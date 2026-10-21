"""scripts/line_a/run_x6.py — X6 检索评测（T-038）：build 生成开发集和测试集（运行前提交），eval 跑指定方法.

用法：
  python scripts/line_a/run_x6.py build --data <data_root>
  python scripts/line_a/run_x6.py eval  --data <data_root> --split dev|test --method bm25|dense:<模型>|hybrid:<模型>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from quantmind.line_a.evidence.fetch import fetch_pdf  # noqa: E402
from quantmind.line_a.p4 import x6_retrieval as x6  # noqa: E402


def build(data: Path) -> None:
    lake, p4 = data / "lake", data / "line_a_v2/p4"
    cache = lake / "text/cninfo_pdf"
    out = p4 / "x6_v1"
    out.mkdir(parents=True, exist_ok=True)
    x13 = {s: (pd.read_csv(p4 / f"x13_{s}_v1/cases.csv", dtype={"announcementId": str}),
               pd.read_csv(p4 / f"x13_{s}_v1/gold.csv", dtype={"ann_id": str})) for s in ("dev", "test")}
    used = set(x13["dev"][0]["ts_code"]) | set(x13["test"][0]["ts_code"])
    risk = x6.risk_docs(lake, 20, x6.SEED, used)
    risk["split"] = ["dev"] * 10 + ["test"] * (len(risk) - 10)
    for r in risk.itertuples(index=False):
        fetch_pdf(str(r.announcementId), r.adjunctUrl, cache)
    risk.to_csv(out / "risk_docs.csv", index=False, encoding="utf-8-sig")
    fa = pd.read_parquet(sorted((lake / "events").glob("fina_audit_2*.parquet"))[-1])
    fa = fa[fa["end_date"].astype(str).str.replace("-", "").str[4:8] == "1231"]
    audit = {(t, int(str(e).replace("-", "")[:4])): r
             for t, e, r in fa[["ts_code", "end_date", "audit_result"]].itertuples(index=False)}
    man = {"version": "x6-v1", "seed": x6.SEED, "fact_queries": x6.FACT_QUERIES, "risk_query": x6.RISK_QUERY,
           "opinion_patterns": x6.OPINION_PATTERNS, "splits": {}}
    for s in ("dev", "test"):
        cases, gold = x13[s]
        docs = pd.concat([cases[["announcementId", "ts_code", "year"]],
                          risk[risk["split"] == s][["announcementId", "ts_code", "year"]].astype({"announcementId": str})],
                         ignore_index=True)
        paras = x6.parse_docs(docs, cache)
        (out / f"paras_{s}.json").write_text(json.dumps(paras, ensure_ascii=False), encoding="utf-8")
        q = x6.build_queries(docs, gold, paras, audit)
        sha = x6.save_queries(q, out / f"queries_{s}.jsonl")
        man["splits"][s] = {"n_docs": len(paras), "n_queries": len(q), "by_kind": q["kind"].value_counts().to_dict(),
                            "dropped": q.attrs["dropped"], "queries_sha256": sha,
                            "mean_relevant": float(q["relevant"].map(len).mean())}
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(man["splits"], ensure_ascii=False))


def evaluate(data: Path, split: str, method: str) -> None:
    out = data / "line_a_v2/p4/x6_v1"
    paras = {k: [tuple(x) for x in v] for k, v in json.loads((out / f"paras_{split}.json").read_text(encoding="utf-8")).items()}
    q = pd.DataFrame([json.loads(x) for x in (out / f"queries_{split}.jsonl").read_text(encoding="utf-8").splitlines()])
    kind, _, model = method.partition(":")
    dense = x6.Dense(model) if kind in ("dense", "hybrid") else None
    rerank = x6.Rerank(model) if kind == "rerank" else None
    t0 = time.time()
    r = x6.evaluate(q, paras, method, dense=dense, rerank=rerank)
    r["seconds"] = round(time.time() - t0, 1)
    name = method.replace(":", "_").replace("/", "_")
    (out / f"result_{split}_{name}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    print("RESULT", split, method, json.dumps({k: r[k] for k in ("all", "by_kind", "seconds")}, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "eval"])
    ap.add_argument("--data", type=Path, default=Path("/home/lenovo/projects/quantmind/data"))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--method", default="bm25")
    a = ap.parse_args()
    build(a.data) if a.cmd == "build" else evaluate(a.data, a.split, a.method)
