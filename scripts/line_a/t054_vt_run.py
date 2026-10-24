"""scripts/line_a/t054_vt_run.py — T-054 §5：本机训练 2B 底座的 VT 适配器并打分，再与 MacBERT v2.2 在金标准 v2 / gold-hard 上比较.

训练与打分复用 T-038 的 quantmind/line_a/p4/x4_cloud.py（同一协议：p(Yes) / (p(Yes) + p(No))，关闭思考，断点续跑）。
底座：本机 D:/RAGScope_Runtime/models/Qwen3.5-2B（revision 15852e8c）。
用法：PYTHONPATH=. python scripts/line_a/t054_vt_run.py train|eval
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from quantmind.line_a.p4 import x4_cloud as xc  # noqa: E402
from quantmind.line_a.p4 import x4v2  # noqa: E402

BASE = "/mnt/d/RAGScope_Runtime/models/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc"
D = Path("/home/lenovo/projects/quantmind/data/line_a_v2/p4/t054_vt")
PACK, OUT = D / "pack", D / "out"
CFG = {"max_train": 40000, "eval_every": 200, "dev_cells": 1400, "epochs": 1}
V22 = Path("/home/lenovo/projects/quantmind/data/line_a_v2/p4/x4_v22")


def train() -> None:
    xc.MODELS["q35-2b"] = BASE
    xc.verify_pack(PACK)
    xc.run_finetune("q35-2b", PACK, OUT, cfg=CFG)
    xc.score_finetuned("q35-2b", PACK, OUT)


def evaluate() -> None:
    cells = pd.DataFrame(xc._read_jsonl(PACK / "cells.jsonl"))
    sc = pd.DataFrame(xc._read_jsonl(OUT / "ft__q35-2b" / "scores.jsonl"))
    df = cells.merge(sc[["cell", "p_yes"]], on="cell", how="left")
    cal = df[df["split"] == "calib"]
    thr = {}
    for t in x4v2.TYPE_KEYS:
        g = cal[cal["type"] == t]
        p, y = g["p_yes"].to_numpy(dtype=float)[:, None], g["label"].to_numpy()[:, None].astype(float)
        thr[t] = float(x4v2.thresholds(p, y)[0])
    df["vt"] = df.apply(lambda r: bool(r["p_yes"] >= thr[r["type"]]) if pd.notna(r["p_yes"]) else False, axis=1)
    res = {"thresholds": thr, "coverage": float(df["p_yes"].notna().mean())}
    for name, file, splits in (("gold_v2", "predictions_gold_v2.parquet", ("gold_v2A", "gold_v2B")),
                               ("gold_hard", "predictions_gold_hard.parquet", ("gold_hard",))):
        pairs = pd.read_parquet(V22 / file)
        vt = df[df["split"].isin(splits)].copy()
        vt["announcementId"] = vt["ann_id"].astype(str)
        key = vt.set_index(["announcementId", "type"])["vt"]
        pairs["announcementId"] = pairs["announcementId"].astype(str)
        pairs["VT"] = [bool(key.get((a, t), False)) for a, t in zip(pairs["announcementId"], pairs["type"], strict=True)]
        preds = {"R_rule": pairs["R_rule"], "MacBERT_v22": pairs["M21"], "VT_2B": pairs["VT"]}
        sc2 = x4v2.score(pairs, preds)
        res[name] = {k: {"f1": round(v["all"]["f1"], 4), "precision": round(v["all"]["precision"], 4),
                         "recall": round(v["all"]["recall"], 4), "severe_recall": round(v["severe"]["recall"], 4)}
                     for k, v in sc2.items()}
    h, g = res["gold_hard"], res["gold_v2"]
    res["verdict"] = {"hard_gain": round(h["VT_2B"]["f1"] - h["MacBERT_v22"]["f1"], 4),
                      "v2_drop": round(g["MacBERT_v22"]["f1"] - g["VT_2B"]["f1"], 4),
                      "severe": [g["MacBERT_v22"]["severe_recall"], g["VT_2B"]["severe_recall"]]}
    res["verdict"]["adopt"] = bool(res["verdict"]["hard_gain"] >= 0.02 and res["verdict"]["v2_drop"] <= 0.005
                                   and g["VT_2B"]["severe_recall"] >= g["MacBERT_v22"]["severe_recall"])
    (D / "vt_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    _ = np


if __name__ == "__main__":
    {"train": train, "eval": evaluate}[sys.argv[1]]()
