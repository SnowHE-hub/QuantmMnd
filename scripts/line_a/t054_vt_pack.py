"""scripts/line_a/t054_vt_pack.py — T-054 §5：2B 底座 VT 适配器的题包（指南 v2.1 标签，x4_v22 的 DeepSeek 标注）.

格式与 T-038 的 x4_cloud 题包相同（cells.jsonl / train.jsonl / manifest.json，每行 user 文本带 sha256），所以本机直接用
quantmind/line_a/p4/x4_cloud.py 训练和打分。
- train：v2 训练集 + 难例训练补充的每条标题，正类全取，负类每条标题抽 2 个（1 个"易混"——标题规则判出的类型或同族类型，1 个随机）；
- cells：
  - calib：v2 校准集 + 校准补充，每条标题 × 14 类（用来选阈值和早停）；
  - gold_v2A：金标准 v2 的 A 部分（单类判断）；gold_v2B、gold_hard：每条标题 × 14 类。
- 题面用指南 v2.1 的定义（含排除项），只回答 Yes 或 No。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd

from quantmind.line_a.evidence.ann_types import classify_title
from quantmind.line_a.p4.x4v2 import EXCLUSIONS_V21, TYPE_KEYS, TYPES

P4 = Path("/home/lenovo/projects/quantmind/data/line_a_v2/p4")
FAMILY = {"performance_forecast": ["performance_express"], "performance_express": ["performance_forecast"],
          "repurchase": ["unlock", "dividend"], "holder_reduction": ["holder_increase", "unlock"],
          "holder_increase": ["holder_reduction"], "unlock": ["repurchase", "holder_reduction"],
          "investigation": ["penalty", "regulatory_letter"], "penalty": ["investigation", "regulatory_letter"],
          "regulatory_letter": ["penalty", "investigation"], "restructuring": ["control_change"],
          "control_change": ["restructuring", "holder_reduction"], "st_change": ["regulatory_letter"],
          "dividend": ["repurchase"], "litigation": ["penalty"]}


def message(t: str, title: str) -> str:
    return (f"判断下面这条 A 股上市公司公告的标题是否属于这一类事件：{TYPES[t][0]}。{EXCLUSIONS_V21[t]}。\n"
            f"只根据标题判断，只回答 Yes 或 No。\n公告标题：{title}")


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _labels(*files) -> dict:
    out = {}
    for f in files:
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[str(r["announcementId"])] = r["types"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    v2, v21, v22 = P4 / "x4_v2", P4 / "x4_v21", P4 / "x4_v22"
    lt = _labels(v22 / "distill_train.jsonl", v22 / "distill_train_add.jsonl")
    lc = _labels(v22 / "distill_calib.jsonl", v22 / "distill_calib_add.jsonl")
    tr = pd.concat([pd.read_parquet(v2 / "train.parquet"), pd.read_parquet(v21 / "train_add.parquet")])[["announcementId", "title"]]
    ca = pd.concat([pd.read_parquet(v2 / "calib.parquet"), pd.read_parquet(v21 / "calib_add.parquet")])[["announcementId", "title"]]
    tr["announcementId"], ca["announcementId"] = tr["announcementId"].astype(str), ca["announcementId"].astype(str)
    rng = random.Random(20261032)
    train = []
    for r in tr.drop_duplicates("announcementId").itertuples():
        if r.announcementId not in lt:
            continue
        pos = set(lt[r.announcementId])
        rule = classify_title(r.title).event_type
        conf = [t for t in ([rule] if rule in TYPE_KEYS else []) + sum((FAMILY.get(p, []) for p in pos), []) if t not in pos]
        rest = [t for t in TYPE_KEYS if t not in pos]
        negs = list(dict.fromkeys(([rng.choice(conf)] if conf else []) + rng.sample(rest, k=min(2, len(rest)))))[:2]
        for t in [*sorted(pos), *negs]:
            u = message(t, r.title)
            train.append({"cell": f"tr:{r.announcementId}:{t}", "ann_id": r.announcementId, "type": t, "split": "train",
                          "label": int(t in pos), "user": u, "sha": _sha(u)})
    cells = []
    for r in ca.drop_duplicates("announcementId").itertuples():
        if r.announcementId not in lc:
            continue
        for t in TYPE_KEYS:
            u = message(t, r.title)
            cells.append({"cell": f"ca:{r.announcementId}:{t}", "ann_id": r.announcementId, "type": t, "split": "calib",
                          "label": int(t in lc[r.announcementId]), "user": u, "sha": _sha(u)})
    ga, gb = pd.read_parquet(v2 / "gold_v2_a_labeled.parquet"), pd.read_parquet(v2 / "gold_v2_b_labeled.parquet")
    for r in ga.itertuples():
        u = message(r.gold_for, r.title)
        cells.append({"cell": f"gA:{r.announcementId}:{r.gold_for}", "ann_id": str(r.announcementId), "type": r.gold_for,
                      "split": "gold_v2A", "label": int(r.y), "user": u, "sha": _sha(u)})
    gh = pd.read_parquet(v21 / "gold_hard_labeled.parquet")
    for name, g in (("gold_v2B", gb), ("gold_hard", gh)):
        for r in g.itertuples():
            for t in TYPE_KEYS:
                u = message(t, r.title)
                cells.append({"cell": f"{name}:{r.announcementId}:{t}", "ann_id": str(r.announcementId), "type": t,
                              "split": name, "label": int(t in list(r.types)), "user": u, "sha": _sha(u)})
    overlap = {r["ann_id"] for r in train} & {c["ann_id"] for c in cells}
    if overlap:
        train = [r for r in train if r["ann_id"] not in overlap]        # 训练集里去掉任何出现在打分集合里的公告
    for name, rows in (("train", train), ("cells", cells)):
        (a.out / f"{name}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")
    man = {"protocol": "VT Yes/No，指南 v2.1", "rows": {"train": len(train), "cells": len(cells)},
           "files": {f: {"sha256": hashlib.sha256((a.out / f).read_bytes()).hexdigest()} for f in ("train.jsonl", "cells.jsonl")},
           "n_train": len(train), "n_train_pos": sum(r["label"] for r in train), "n_cells": len(cells),
           "cells_by_split": pd.Series([c["split"] for c in cells]).value_counts().to_dict(), "removed_overlap": len(overlap),
           "guide": "v2.1"}
    (a.out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(man, ensure_ascii=False))


if __name__ == "__main__":
    main()
