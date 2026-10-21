"""P4 X4：题包导出（抽样、切分不重叠、哈希）、云端运行器的纯函数、结果导入与评估（不需要 GPU）。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from quantmind.line_a.p4.x4_dataset import TYPES, template


def _fake_corpus(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        split = ["train", "calib", "test"][i % 3]
        t = "repurchase" if i % 4 == 0 else "dividend"
        title = "关于回购股份的进展公告" if t == "repurchase" else "2023年度利润分配方案公告"
        if i % 7 == 0:
            title = "关于回购注销限制性股票的公告"                    # 含关键词的难负例
        row = {"announcementId": str(1_000_000 + i), "date": f"2023{1 + i % 12:02d}15", "title": f"{title}{i}",
               "split": split, "keep": True}
        for tt in TYPES:
            row[f"label_{tt}"] = "neg"
        if i % 7 != 0:
            row[f"label_{t}"] = "pos_strong" if rng.random() > 0.1 else "uncertain"
        rows.append(row)
    return pd.DataFrame(rows)


def test_template_dedup_key():
    assert template("2023年度利润分配方案公告") == template("2024年度利润分配方案公告")
    assert template("关于回购股份的进展公告（2024-031）") == template("关于回购股份的进展公告")


def test_pack_rows_and_eval(tmp_path):
    from quantmind.line_a.p4 import x4_eval
    from quantmind.line_a.p4.x4_pack import make_rows, message
    rows = make_rows(_fake_corpus())
    df = pd.DataFrame(rows)
    assert set(df["split"]) == {"train", "calib", "test"} and set(df["label"]) == {0, 1}
    assert not (set(df[df.split == "train"].ann_id) & set(df[df.split != "train"].ann_id))
    hn = df[(df.type == "repurchase") & (df.stratum == "hard_neg")]
    assert len(hn) and hn["user"].str.contains("回购注销").all() and (hn["label"] == 0).all()
    assert message("repurchase", "x").endswith("公告标题：x") and "Yes 或 No" in message("repurchase", "x")
    pack = tmp_path / "pack"
    pack.mkdir()
    cells = df[df.split != "train"]
    (pack / "cells.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in cells.to_dict("records")),
                                      encoding="utf-8")
    res = tmp_path / "out"
    res.mkdir()
    good = [{"cell": r["cell"], "sha": r["sha"], "p_yes": 0.9 if r["label"] else 0.1, "status": "ok",
             "mode": "zs", "model_key": "q35-2b"} for r in cells.to_dict("records")]
    (res / "zs__q35-2b.jsonl").write_text("\n".join(json.dumps(r) for r in good), encoding="utf-8")
    bad = [dict(good[0], sha="0" * 64)]
    (res / "zs__bad.jsonl").write_text(json.dumps(bad[0] | {"model_key": "bad"}), encoding="utf-8")
    ev = x4_eval.evaluate(pack, res)
    assert ev["zs__q35-2b"]["macro_test_auc"] == 1.0 and "error" in ev["zs__bad.jsonl"]


def test_cloud_pure_functions(tmp_path):
    import hashlib
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("xc", Path(__file__).resolve().parents[1] / "quantmind/line_a/p4/x4_cloud.py")
    xc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(xc)                                  # 云端运行器不能依赖 quantmind 包
    assert xc._auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.75
    (tmp_path / "a.jsonl").write_text('{"x": 1}\n{"x": 2', encoding="utf-8")
    assert xc._read_jsonl(tmp_path / "a.jsonl") == [{"x": 1}]
    (tmp_path / "f.txt").write_text("hi", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"protocol": "t", "files": {"f.txt": {
        "sha256": hashlib.sha256(b"hi").hexdigest()}}}), encoding="utf-8")
    assert xc.verify_pack(tmp_path)["protocol"] == "t"
