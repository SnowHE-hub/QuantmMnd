"""scripts/line_a/x4_hard_examples.py — X4 v2.1 难例挖掘（T-049 后续）：MacBERT 与标题规则判断不一致的标题.

不加标题规则（用户 2026-10-23），改进放在训练数据上：
- 输入：X4 年度表（x4_<年>.parquet，keep = True，已按模板去重），默认 2026 年；
- 排除金标准 v1 / v2 用过的公告，以及标题与金标准完全相同的公告（防止泄漏到验收集）；
- 对每条标题取模型的 14 类判断和标题规则的类型（只算 X4 的 14 类），两者不同的记为难例：
  kind = model_only（模型多判）/ rule_only（规则多判）/ both_differ（两边各有对方没有的类型）；
- 输出 hard_pool.parquet 与 summary.json。之后按登记分成训练补充和难例验收集。
用法：python scripts/line_a/x4_hard_examples.py --years 2026 --out data/line_a_v2/p4/x4_v21 [--limit N]
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
P4 = Path("/home/lenovo/projects/quantmind/data/line_a_v2/p4")


def hard_examples(df: pd.DataFrame, model_types, rule_type) -> pd.DataFrame:
    """df：announcementId, title。model_types(titles) → 每条的类型元组；rule_type(title) → 规则类型。"""
    from quantmind.line_a.evidence.ann_classifier import X4_TYPES
    titles = df["title"].astype(str).tolist()
    mt = model_types(titles)
    rows = []
    for aid, t, m in zip(df["announcementId"], titles, mt, strict=True):
        r = rule_type(t)
        rs, ms = ({r} if r in X4_TYPES else set()), set(m)
        if rs == ms:
            continue
        kind = "model_only" if rs < ms else "rule_only" if ms < rs else "both_differ"
        rows.append({"announcementId": str(aid), "title": t, "model_types": sorted(ms), "rule_types": sorted(rs),
                     "kind": kind})
    return pd.DataFrame(rows, columns=["announcementId", "title", "model_types", "rule_types", "kind"])


def main() -> None:
    from quantmind.line_a.evidence.ann_classifier import TitleClassifier
    from quantmind.line_a.evidence.ann_types import classify_title
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2026")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_parquet(P4 / "x4_v1" / f"x4_{y}.parquet", columns=["announcementId", "date", "title", "keep"])
                    for y in a.years.split(",")], ignore_index=True)
    df = df[df["keep"]].drop(columns="keep")
    gold_ids, gold_titles = set(), set()
    for f in (P4 / "x4_v1/gold_final.parquet", P4 / "x4_v2/gold_a.parquet", P4 / "x4_v2/gold_b.parquet"):
        g = pd.read_parquet(f)
        gold_ids |= set(g["announcementId"].astype(str))
        gold_titles |= set(g["title"].astype(str)) if "title" in g else set()
    n0 = len(df)
    df = df[~df["announcementId"].astype(str).isin(gold_ids) & ~df["title"].astype(str).isin(gold_titles)]
    df = df.drop_duplicates("title")
    if a.limit:
        df = df.sample(min(a.limit, len(df)), random_state=0)
    clf = TitleClassifier.from_adoption()
    t0 = time.time()
    out = []
    for i in range(0, len(df), 5000):
        out.append(hard_examples(df.iloc[i:i + 5000], clf.model_types, lambda t: classify_title(t).event_type))
        print(f"{min(i + 5000, len(df))}/{len(df)} titles, {sum(len(x) for x in out)} hard, {time.time() - t0:.0f}s",
              flush=True)
    hard = pd.concat(out, ignore_index=True).merge(df[["announcementId", "date"]].astype({"announcementId": str}),
                                                   on="announcementId", how="left")
    hard.to_parquet(a.out / "hard_pool.parquet")
    by_type: dict = {}
    for _, r in hard.iterrows():
        for t in set(r["model_types"]) ^ set(r["rule_types"]):
            by_type[t] = by_type.get(t, 0) + 1
    summ = {"years": a.years, "n_rows_keep": n0, "n_titles": len(df), "n_hard": len(hard),
            "by_kind": hard["kind"].value_counts().to_dict(), "by_type_disagreement": by_type,
            "excluded_gold_ids": len(gold_ids), "model": clf.version, "secs": round(time.time() - t0)}
    (a.out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print("SUMMARY", json.dumps(summ, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
