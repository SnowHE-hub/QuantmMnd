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


def test_consistency_compare(tmp_path):
    from quantmind.line_a.p4.x4_consistency import compare
    from quantmind.line_a.p4.x4_pack import make_rows
    cells = [r for r in make_rows(_fake_corpus()) if r["split"] != "train"]
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "cells.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in cells), encoding="utf-8")
    rng = np.random.default_rng(1)

    def rows(noise):
        out = []
        for r in cells:
            m = (2.0 if r["label"] else -2.0) + rng.normal(0, 0.5) + noise * rng.normal()
            p = 1 / (1 + np.exp(-m))
            out.append({"cell": r["cell"], "sha": r["sha"], "status": "ok", "p_yes": p,
                        "lp_yes": float(np.log(p)), "lp_no": float(np.log(1 - p))})
        return out
    cloud = rows(0.0)
    near = [dict(c, lp_yes=c["lp_yes"] + 1e-4, p_yes=c["p_yes"]) for c in cloud]
    far = rows(3.0)
    for name, rs in (("cloud", cloud), ("near", near), ("far", far)):
        (tmp_path / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rs), encoding="utf-8")
    ok = compare(pack, tmp_path / "cloud.jsonl", tmp_path / "near.jsonl")
    assert ok["pass"] and ok["flip_rate"] == 0 and ok["n_compared"] == len(cells)
    bad = compare(pack, tmp_path / "cloud.jsonl", tmp_path / "far.jsonl")
    assert not bad["pass"] and bad["pearson_margin"] < 0.98


def test_gold_prelabel_review_import(tmp_path):
    from quantmind.line_a.harness.llm import LLM
    from quantmind.line_a.p4 import x4_gold

    def transport(url, headers, body):
        title = body["messages"][1]["content"]
        lab = "是" if "回购股份" in title else "否"
        return 200, {"choices": [{"message": {"content": json.dumps({"label": lab, "reason": "测试"}, ensure_ascii=False)},
                                  "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    gold = pd.DataFrame([
        {"gold_id": "a1", "gold_for": "repurchase", "stratum": "pos_strong", "title": "关于回购股份的进展公告"},
        {"gold_id": "a2", "gold_for": "repurchase", "stratum": "hard_neg", "title": "关于回购注销限制性股票的公告"},
        {"gold_id": "a3", "gold_for": "repurchase", "stratum": "pos_rule", "title": "关于注销部分股票的公告"},
        {"gold_id": "a4", "gold_for": "repurchase", "stratum": "uncertain", "title": "关于回购股份方案的公告"}])
    llm = LLM(transport=transport, ledger=tmp_path / "l.jsonl", cache_dir=tmp_path / "c")
    pre = x4_gold.prelabel(gold, llm, log=lambda m: None)
    flags = x4_gold.review_sheet(pre, tmp_path / "review.csv")
    s = pd.read_csv(tmp_path / "review.csv", encoding="utf-8-sig", dtype=str)
    assert set(s["编号"]) >= {"a3", "a4"}                       # 不一致、规则不确定一定复核
    assert dict(zip(flags.gold_id, flags.review_reason, strict=True))["a3"] == "DeepSeek 与规则不一致"
    s["你的判断（是/否）"] = "否"
    s.to_csv(tmp_path / "review.csv", index=False, encoding="utf-8-sig")
    fin = x4_gold.import_review(flags, tmp_path / "review.csv").set_index("gold_id")
    assert fin.loc["a3", "gold_label"] == "否" and fin.loc["a3", "gold_source"] == "user"
    assert fin.loc["a1", "gold_label"] in ("是", "否") and fin["gold_label"].notna().all()


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
