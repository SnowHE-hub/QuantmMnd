"""T-007 v3 补充核对（描述性，不改预登记判读）：各组输出的 Top-10 是否就是 LGBM 的前 10（且顺序相同）.

用法：python t007_v3_copy_check.py <frozen_dir> <out.json>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

d, out = Path(sys.argv[1]), Path(sys.argv[2])


def jl(name):
    return [json.loads(x) for x in (d / name).read_text(encoding="utf-8").splitlines() if x]


reqs = {r["request_id"]: r for r in jl("requests.jsonl")}
mp = defaultdict(dict)
for m in jl("private_mapping.jsonl"):
    mp[(m["as_of"], m["repeat"])][m["id"]] = m["lgbm_rank"]
res = defaultdict(lambda: {"n": 0, "exact_lgbm_top10_order": 0, "same_set_as_lgbm_top10": 0, "mean_overlap": 0.0})
for r in jl("responses.jsonl"):
    if r.get("status") != "ok":
        continue
    q = reqs[r["request_id"]]
    if q["group"] == "PROBE":
        continue
    ids = [x["id"] for x in json.loads(r["content"])["rankings"]]
    ranks = [mp[(q["as_of"], q["repeat"])][i] for i in ids]
    g = res[q["group"]]
    g["n"] += 1
    g["exact_lgbm_top10_order"] += int(ranks == list(range(1, 11)))
    g["same_set_as_lgbm_top10"] += int(sorted(ranks) == list(range(1, 11)))
    g["mean_overlap"] += len(set(ranks) & set(range(1, 11))) / 10
for g in res.values():
    g["mean_overlap"] = round(g["mean_overlap"] / g["n"], 4)
out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
