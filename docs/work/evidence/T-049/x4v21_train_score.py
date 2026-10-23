"""T-049 X4 v2.1：v2 训练集 + 难例训练补充重训 MacBERT，与 v2 在金标准 v2 和 gold-hard 上比较（登记见 T-049.md）.

- 超参数与 v2 相同（x4v2.CFG）；阈值在"v2 校准集 + 校准补充"上按 F1 选；
- gold-hard：Claude 盲标（41d06c0），每条标题 × 14 类展开成判断表，part = H；
- 系统：R 标题规则、D DeepSeek 直接分类、M2 = v2 模型（原阈值）、M21 = v2.1；
- 采用条件：gold-hard F1 比 M2 高 ≥ 0.03；金标准 v2 F1 不比 M2 低 0.005 以上；金标准 v2 严重类型召回不低于 M2。
用法：PYTHONPATH=. python docs/work/evidence/T-049/x4v21_train_score.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from quantmind.line_a.evidence.ann_types import classify_title  # noqa: E402
from quantmind.line_a.p4 import x4v2  # noqa: E402

V2 = Path("/home/lenovo/projects/quantmind/data/line_a_v2/p4/x4_v2")
O = Path("/home/lenovo/projects/quantmind/data/line_a_v2/p4/x4_v21")


def load_labels(p: Path) -> dict:
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[str(r["announcementId"])] = r["types"]
    return out


def labeled(df: pd.DataFrame, lab: dict) -> pd.DataFrame:
    df = df.assign(announcementId=df["announcementId"].astype(str))
    return df[df["announcementId"].isin(lab)].reset_index(drop=True)


def preds_for(pairs: pd.DataFrame, mdir: Path, thr: np.ndarray) -> pd.Series:
    titles = sorted(set(pairs["title"]))
    pm = dict(zip(titles, x4v2.predict_macbert(mdir, titles), strict=True))
    ti = {t: i for i, t in enumerate(x4v2.TYPE_KEYS)}
    return pd.Series([bool(pm[t][ti[k]] >= thr[ti[k]]) for t, k in zip(pairs["title"], pairs["type"], strict=True)],
                     index=pairs.index)


def main() -> None:
    lt = load_labels(V2 / "distill_train_g2.jsonl") | load_labels(O / "distill_train_add.jsonl")
    lc = load_labels(V2 / "distill_calib_g2.jsonl") | load_labels(O / "distill_calib_add.jsonl")
    train = pd.concat([labeled(pd.read_parquet(V2 / "train.parquet")[["announcementId", "title"]], lt),
                       labeled(pd.read_parquet(O / "train_add.parquet")[["announcementId", "title"]], lt)],
                      ignore_index=True).drop_duplicates("announcementId")
    calib = pd.concat([labeled(pd.read_parquet(V2 / "calib.parquet")[["announcementId", "title"]], lc),
                       labeled(pd.read_parquet(O / "calib_add.parquet")[["announcementId", "title"]], lc)],
                      ignore_index=True).drop_duplicates("announcementId")
    yt, yc = x4v2.label_matrix(train, pd.Series(lt)), x4v2.label_matrix(calib, pd.Series(lc))
    res: dict = {"n_train": len(train), "n_calib": len(calib), "n_train_add": int(train["announcementId"].isin(
        set(pd.read_parquet(O / "train_add.parquet")["announcementId"].astype(str))).sum())}
    mdir = O / "macbert_multilabel_v21"
    if not (mdir / "x4v2_config.json").exists():
        x4v2.train_macbert(train["title"].tolist(), yt, mdir, log=lambda m: print(m, flush=True))
    thr21 = x4v2.thresholds(x4v2.predict_macbert(mdir, calib["title"].tolist()), yc)
    s2 = json.loads((V2 / "x4v2_score_g2.json").read_text(encoding="utf-8"))
    thr2 = np.array([s2["thresholds"][k] for k in x4v2.TYPE_KEYS])
    res["thresholds"] = dict(zip(x4v2.TYPE_KEYS, thr21.round(3).tolist(), strict=True))
    # 金标准 v2
    a, b = pd.read_parquet(V2 / "gold_v2_a_labeled.parquet"), pd.read_parquet(V2 / "gold_v2_b_labeled.parquet")
    pv2 = x4v2.gold_pairs(a, b, dict(zip(a["announcementId"], a["y"], strict=True)),
                          dict(zip(b["announcementId"], b["types"], strict=True)))
    # gold-hard
    gh = pd.read_parquet(O / "gold_hard_labeled.parquet")
    ph = pd.DataFrame([{"announcementId": str(r.announcementId), "title": r.title, "type": k, "y": k in list(r.types),
                        "part": "H"} for r in gh.itertuples() for k in x4v2.TYPE_KEYS])
    d_lab = load_labels(O / "D_gold_hard_predictions_DO_NOT_OPEN_BEFORE_LABELS.jsonl")
    out = {}
    for name, pairs in (("gold_v2", pv2), ("gold_hard", ph)):
        preds = {"R_rule": pd.Series([classify_title(t).event_type == k for t, k in zip(pairs["title"], pairs["type"],
                                                                                         strict=True)], index=pairs.index),
                 "M2": preds_for(pairs, V2 / "macbert_multilabel_g2", thr2),
                 "M21": preds_for(pairs, mdir, thr21)}
        if name == "gold_hard":
            preds["D_deepseek"] = pd.Series([k in (d_lab.get(aid) or []) for aid, k in
                                             zip(pairs["announcementId"], pairs["type"], strict=True)], index=pairs.index)
            res["d_coverage_hard"] = float(pairs["announcementId"].isin(d_lab).mean())
        sc = x4v2.score(pairs, preds)
        out[name] = {k: {"all": v["all"], "macro_f1": v["macro_f1"], "severe": v["severe"],
                         "by_type_f1": {t: round(x["f1"], 3) for t, x in v["by_type"].items() if x["n_pos"]}}
                     for k, v in sc.items()}
        pairs.assign(**preds).to_parquet(O / f"predictions_{name}.parquet")
    h, g = out["gold_hard"], out["gold_v2"]
    cond = {"hard_gain": round(h["M21"]["all"]["f1"] - h["M2"]["all"]["f1"], 4),
            "v2_drop": round(g["M2"]["all"]["f1"] - g["M21"]["all"]["f1"], 4),
            "v2_severe_recall": [round(g["M2"]["severe"]["recall"], 4), round(g["M21"]["severe"]["recall"], 4)]}
    cond["adopt"] = bool(cond["hard_gain"] >= 0.03 and cond["v2_drop"] <= 0.005
                         and g["M21"]["severe"]["recall"] >= g["M2"]["severe"]["recall"])
    res |= {"score": out, "verdict": cond}
    (O / "x4v21_score.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"n_train": res["n_train"], "verdict": cond,
                      "f1": {n: {k: round(v["all"]["f1"], 4) for k, v in s.items()} for n, s in out.items()},
                      "severe_recall": {n: {k: round(v["severe"]["recall"], 4) for k, v in s.items()}
                                        for n, s in out.items()}}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
