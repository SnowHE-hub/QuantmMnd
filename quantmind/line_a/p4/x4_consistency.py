"""quantmind.line_a.p4.x4_consistency — X4 的云端与本地一致性验收（03_final_plan_v2 第 9.3 节；P4）.

做法：
- 本地重算：用题包里同一份 x4_cloud.py（不用仓库里的副本，避免两边代码不一致），在本机 GPU 上加载同一个基座和
  Colab 训练出的 adapter_best，对 cells.jsonl 逐题打分（`run_local`）；
- 比对（`compare`）：只比两边都是 status = ok 且 cell、sha 一致的题。
  - 门槛（沿用 5703）：logit 差（lp_yes − lp_no）的 Pearson ≥ 0.98，平均 |Δp| ≤ 0.02；
  - 另外报告（ChatGPT 评审）：
    - 阈值附近的翻转：用云端校准集上按 F1 选的阈值（每类一个），两边判定不同的比例；
    - 少数类召回：测试集正例上两边召回率之差；
    - 严重事件漏判：严重类型的测试正例里，云端判对、本地判漏的条数；
    - 拒答率：status 不是 ok 的比例；最终判定一致率。
用法：python -m quantmind.line_a.p4.x4_consistency compare <pack_dir> <cloud_scores.jsonl> <local_scores.jsonl> <out.json>
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quantmind.line_a.p4.x4_eval import _best_threshold, _read

GATES = {"pearson_margin": 0.98, "mean_abs_dp": 0.02}
SEVERE = ("investigation", "penalty", "st_change", "regulatory_letter", "restructuring", "control_change")


def _cloud_module(pack: Path):
    spec = importlib.util.spec_from_file_location("x4_cloud_pack", Path(pack) / "x4_cloud.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_local(pack: Path, run_dir: Path, out_path: Path, model_key: str, *, limit: int | None = None,
              local_path: str | None = None) -> Path:
    """run_dir：Colab 输出里的 ft__<model_key>（含 adapter_best、best.json、config.json）。
    local_path：基座的本地目录（固定 revision 下载的文件）；不给就按模型名从 Hub 缓存加载。"""
    from peft import PeftModel
    xc = _cloud_module(pack)
    xc.verify_pack(Path(pack))
    best = json.loads((Path(run_dir) / "best.json").read_text())
    q4 = json.loads((Path(run_dir) / "config.json").read_text()).get("load_in_4bit", False)
    base, tok = xc.load_model(model_key, load_in_4bit=q4, local_path=local_path)
    model = PeftModel.from_pretrained(base, Path(run_dir) / "adapter_best")
    model.eval()
    rows = xc._read_jsonl(Path(pack) / "cells.jsonl")
    xc._check_rows(rows)
    rows = rows[:limit] if limit else rows
    xc.score_rows(model, tok, rows, Path(out_path), {"model_key": model_key, "mode": "ft_local",
                                                     "adapter_step": best["step"],
                                                     "precision": "nf4" if q4 else "bf16", "env": xc.env_identity()})
    return Path(out_path)


def compare(pack: Path, cloud: Path, local: Path) -> dict:
    cells = pd.DataFrame(_read(Path(pack) / "cells.jsonl")).set_index("cell")
    a = pd.DataFrame(_read(cloud)).set_index("cell")
    b = pd.DataFrame(_read(local)).set_index("cell")
    n_all = len(cells)
    refuse = {"cloud": float((a["status"] != "ok").mean()), "local": float((b["status"] != "ok").mean())}
    a, b = a[a["status"] == "ok"], b[b["status"] == "ok"]
    common = a.index.intersection(b.index)
    bad = int((a.loc[common, "sha"] != cells["sha"].reindex(common)).sum()
              + (b.loc[common, "sha"] != cells["sha"].reindex(common)).sum())
    if bad:
        return {"error": f"{bad} sha mismatches against the pack"}
    d = cells.loc[common, ["type", "split", "label", "stratum"]].join(
        a.loc[common, ["p_yes", "lp_yes", "lp_no"]]).join(
        b.loc[common, ["p_yes", "lp_yes", "lp_no"]], rsuffix="_l")
    ma, mb = d["lp_yes"] - d["lp_no"], d["lp_yes_l"] - d["lp_no_l"]
    res = {"n_cells": n_all, "n_compared": int(len(d)), "refusal_rate": refuse,
           "pearson_margin": float(np.corrcoef(ma, mb)[0, 1]),
           "mean_abs_dp": float((d["p_yes"] - d["p_yes_l"]).abs().mean()), "types": {}}
    flips = agree = n_dec = 0
    severe_miss = 0
    for t, g in d.groupby("type"):
        cal, te = g[g["split"] == "calib"], g[g["split"] == "test"]
        if cal.empty or te.empty:
            continue
        thr = _best_threshold(cal["label"].to_numpy(), cal["p_yes"].to_numpy())
        pa, pb = te["p_yes"] >= thr, te["p_yes_l"] >= thr
        pos = te["label"] == 1
        flips += int((pa != pb).sum())
        agree += int((pa == pb).sum())
        n_dec += len(te)
        miss = int((pos & pa & ~pb).sum())
        if t in SEVERE:
            severe_miss += miss
        res["types"][t] = {"threshold": thr, "flip_rate": float((pa != pb).mean()),
                           "recall_cloud": float(pa[pos].mean()) if pos.any() else None,
                           "recall_local": float(pb[pos].mean()) if pos.any() else None,
                           "local_only_misses": miss}
    res["flip_rate"] = flips / max(1, n_dec)
    res["decision_agreement"] = agree / max(1, n_dec)
    res["severe_local_only_misses"] = severe_miss
    res["gates"] = {k: (res[k] >= v if k == "pearson_margin" else res[k] <= v) for k, v in GATES.items()}
    res["pass"] = all(res["gates"].values())
    return res


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        r = compare(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
        Path(sys.argv[5]).write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({k: r.get(k) for k in ("n_compared", "pearson_margin", "mean_abs_dp", "flip_rate",
                                                 "severe_local_only_misses", "pass", "error")}, ensure_ascii=False))
