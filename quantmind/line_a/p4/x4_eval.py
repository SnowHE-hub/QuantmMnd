"""quantmind.line_a.p4.x4_eval — 导入 Colab 结果并评估 X4（本地；按哈希核对，不重新渲染消息）.

- 导入：每一行的 cell 与 sha 必须和题包里的一致，否则拒收；
- 指标（按类型和总体）：
  - 测试集 AUC；
  - 在校准集上按 F1 最大选阈值，测试集用这个阈值报精确率、召回率、F1；
  - 难负例上的误报率；
- 对照：零样本 vs 微调，按模型规模列出。
注意：
- 银标签要求"标题规则与结构化表或类别代码一致"，所以用银标签评估规则基线是循环论证，这里不报规则基线；
- 模型和规则谁更好，以人工金标准（gold_sample）为准。
用法：python -m quantmind.line_a.p4.x4_eval <pack_dir> <results_dir> <out.json>
"""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _read(p: Path) -> list[dict]:
    out = []
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        if line.strip():
            with contextlib.suppress(json.JSONDecodeError):      # 断线时写了一半的最后一行
                out.append(json.loads(line))
    return out


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def _best_threshold(y: np.ndarray, s: np.ndarray) -> float:
    best, thr = -1.0, 0.5
    for t in np.unique(np.quantile(s, np.linspace(0.02, 0.98, 49))):
        p = s >= t
        tp = (p & (y == 1)).sum()
        f1 = 2 * tp / max(1, p.sum() + (y == 1).sum())
        if f1 > best:
            best, thr = f1, float(t)
    return thr


def evaluate(pack: Path, results: Path) -> dict:
    cells = pd.DataFrame(_read(Path(pack) / "cells.jsonl")).set_index("cell")
    out: dict = {}
    for f in sorted(Path(results).rglob("*.jsonl")):
        if "_smoke" in f.parts or f.name == "train_log.jsonl":
            continue
        rows = pd.DataFrame(_read(f))
        if rows.empty or "p_yes" not in rows:
            continue
        rows = rows[rows["status"] == "ok"].set_index("cell")
        bad = rows.index[~rows.index.isin(cells.index)].tolist()
        mism = (rows["sha"] != cells["sha"].reindex(rows.index)).sum()
        if bad or mism:
            out[f.name] = {"error": f"rejected: {len(bad)} unknown cells, {int(mism)} sha mismatches"}
            continue
        d = cells.join(rows[["p_yes"]], how="inner")
        name = f"{rows['mode'].iloc[0]}__{rows['model_key'].iloc[0]}"
        res = {"n_scored": int(len(d)), "types": {}}
        for t, g in d.groupby("type"):
            cal, te = g[g["split"] == "calib"], g[g["split"] == "test"]
            if te.empty or cal.empty:
                continue
            thr = _best_threshold(cal["label"].to_numpy(), cal["p_yes"].to_numpy())
            y, s = te["label"].to_numpy(), te["p_yes"].to_numpy()
            pred = s >= thr
            tp = int((pred & (y == 1)).sum())
            hn = te[te["stratum"] == "hard_neg"]
            res["types"][t] = {"test_auc": _auc(y, s), "threshold": thr,
                               "precision": tp / max(1, int(pred.sum())), "recall": tp / max(1, int(y.sum())),
                               "hard_neg_fpr": float((hn["p_yes"] >= thr).mean()) if len(hn) else None,
                               "n_test": int(len(te)), "n_pos": int(y.sum())}
        ty = res["types"].values()
        res["macro_test_auc"] = float(np.nanmean([v["test_auc"] for v in ty])) if ty else None
        res["macro_hard_neg_fpr"] = float(np.nanmean([v["hard_neg_fpr"] for v in ty if v["hard_neg_fpr"] is not None]))
        out[name] = res
    return out


if __name__ == "__main__":
    r = evaluate(Path(sys.argv[1]), Path(sys.argv[2]))
    Path(sys.argv[3]).write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: {"macro_test_auc": v.get("macro_test_auc"), "macro_hard_neg_fpr": v.get("macro_hard_neg_fpr")}
                      for k, v in r.items()}, ensure_ascii=False))
