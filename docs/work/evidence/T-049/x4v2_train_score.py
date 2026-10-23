"""T-049：用 DeepSeek 蒸馏标签训练 MacBERT 多标签分类器，在金标准 v2（Claude 盲标，af59a8b 已提交）上比较 R / D / M.

- R：标题规则 classify_title（只作对照）；
- D：DeepSeek 直接分类（与蒸馏同一个提示）；
- M：MacBERT 多标签（hfl/chinese-macbert-base），阈值在校准集（2023 下半年，DeepSeek 标签）上按 F1 选。
判定按 docs/work/tasks/T-049.md：总体 F1 比规则高 0.05 以上，且严重类型召回率不低于规则；满足的候选里，
F1 与最好者相差 0.02 以内的选最便宜的（M < D）。
用法：PYTHONPATH=. python docs/work/evidence/T-049/x4v2_train_score.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from quantmind.line_a.evidence.ann_types import classify_title  # noqa: E402
from quantmind.line_a.p4 import x4v2  # noqa: E402

O = Path("/home/lenovo/projects/quantmind/data/line_a_v2/p4/x4_v2")


def load_labels(p: Path) -> dict:
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["announcementId"]] = r["types"]
    return out


def main() -> None:
    train, calib = pd.read_parquet(O / "train.parquet"), pd.read_parquet(O / "calib.parquet")
    lt, lc = load_labels(O / "distill_train.jsonl"), load_labels(O / "distill_calib.jsonl")
    train = train[train["announcementId"].isin(lt)].reset_index(drop=True)
    calib = calib[calib["announcementId"].isin(lc)].reset_index(drop=True)
    yt, yc = x4v2.label_matrix(train, pd.Series(lt)), x4v2.label_matrix(calib, pd.Series(lc))
    res: dict = {"n_train": len(train), "n_calib": len(calib),
                 "train_pos_rate": dict(zip(x4v2.TYPE_KEYS, yt.mean(axis=0).round(4).tolist(), strict=True))}
    mdir = O / "macbert_multilabel"
    if not (mdir / "x4v2_config.json").exists():
        x4v2.train_macbert(train["title"].tolist(), yt, mdir, log=lambda m: print(m, flush=True))
    pc = x4v2.predict_macbert(mdir, calib["title"].tolist())
    thr = x4v2.thresholds(pc, yc)
    res["thresholds"] = dict(zip(x4v2.TYPE_KEYS, thr.round(3).tolist(), strict=True))
    a = pd.read_parquet(O / "gold_v2_a_labeled.parquet")
    b = pd.read_parquet(O / "gold_v2_b_labeled.parquet")
    pairs = x4v2.gold_pairs(a, b, dict(zip(a["announcementId"], a["y"], strict=True)),
                            dict(zip(b["announcementId"], b["types"], strict=True)))
    titles = sorted(set(pairs["title"]))
    pm = x4v2.predict_macbert(mdir, titles)
    pm_map = {t: p for t, p in zip(titles, pm, strict=True)}
    ti = {t: i for i, t in enumerate(x4v2.TYPE_KEYS)}
    m_pred = pd.Series([bool(pm_map[t][ti[k]] >= thr[ti[k]]) for t, k in zip(pairs["title"], pairs["type"], strict=True)],
                       index=pairs.index)
    d_lab = load_labels(O / "D_gold_predictions_DO_NOT_OPEN_BEFORE_LABELS.jsonl")
    d_pred = pd.Series([k in (d_lab.get(aid) or []) for aid, k in zip(pairs["announcementId"], pairs["type"], strict=True)],
                       index=pairs.index)
    r_pred = pd.Series([classify_title(t).event_type == k for t, k in zip(pairs["title"], pairs["type"], strict=True)],
                       index=pairs.index)
    res["d_coverage"] = float(pairs["announcementId"].isin(d_lab).mean())
    sc = x4v2.score(pairs, {"R_rule": r_pred, "D_deepseek": d_pred, "M_macbert": m_pred})
    rule = sc["R_rule"]
    verdict = {}
    for k in ("D_deepseek", "M_macbert"):
        v = sc[k]
        verdict[k] = {"f1": round(v["all"]["f1"], 4), "gain": round(v["all"]["f1"] - rule["all"]["f1"], 4),
                      "severe_recall": round(v["severe"]["recall"], 4),
                      "cond1": v["all"]["f1"] - rule["all"]["f1"] >= 0.05,
                      "cond2": v["severe"]["recall"] >= rule["severe"]["recall"]}
    ok = [k for k, v in verdict.items() if v["cond1"] and v["cond2"]]
    best = max((sc[k]["all"]["f1"] for k in ok), default=None)
    pick = None
    for k in ("M_macbert", "D_deepseek"):                       # 便宜的在前
        if k in ok and sc[k]["all"]["f1"] >= best - 0.02:
            pick = k
            break
    res |= {"score": {k: {"all": v["all"], "macro_f1": v["macro_f1"], "severe": v["severe"], "part_A": v["part_A"],
                          "part_B": v["part_B"], "by_type_f1": {t: round(b["f1"], 3) for t, b in v["by_type"].items()}}
                      for k, v in sc.items()},
            "verdict": verdict, "adopted": pick, "rule_f1": round(rule["all"]["f1"], 4),
            "rule_severe_recall": round(rule["severe"]["recall"], 4)}
    pairs.assign(R=r_pred, D=d_pred, M=m_pred).to_parquet(O / "gold_v2_predictions.parquet")
    (O / "x4v2_score.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("n_train", "n_calib", "d_coverage", "rule_f1", "rule_severe_recall", "verdict",
                                          "adopted")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
