"""quantmind.line_a.p4.x4_gold — X4 人工金标准：DeepSeek 预标 + 用户复核（方式由用户 2026-10-22 选定）.

流程：
1. `prelabel`：对 gold_sample_todo.csv 的每一条（gold_id、gold_for、stratum、title），让 DeepSeek（deepseek-flash，
   关闭思考，JSON 输出）判断标题是否属于该类事件，给"是 / 否"和一句理由。只发送公开的公告标题和类别说明；
2. `review_sheet`：需要用户复核的条目 =
   - DeepSeek 与规则判断不一致；
   - 规则本身不确定（uncertain）；
   - 其余条目里按种子随机抽 20%（检查"两边一致"是否可靠）。
   写成 UTF-8-BOM 的 CSV（Excel 可直接打开）。用户只填"你的判断"列（是 / 否），可以在"备注"里写说明；
3. `import_review`：读回复核表。最终标签：复核过的用用户判断，没复核的用"DeepSeek 与规则一致"的判断。每行记录来源。
规则判断：stratum 是 pos_strong / pos_rule 记为"是"，hard_neg 记为"否"，uncertain 记为无。
复核时能看到 DeepSeek 的判断和理由，这会带来锚定。这一点在任务卡里说明，抽查部分用来估计它的影响。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from quantmind.line_a.p4.x4_dataset import TYPES

SEED = 20261022
REVIEW_SHARE = 0.20
SYSTEM = ("你是 A 股上市公司公告的分类标注员。给你一类事件的定义和一条公告标题，只根据标题判断这条公告是否属于这类事件。"
          "标题信息不足以判断时，按最可能的情况判断，并在理由里写明不确定。"
          '只输出 JSON：{"label": "是" 或 "否", "reason": "不超过 40 字的理由"}。')


def _rule(stratum: str) -> str | None:
    return {"pos_strong": "是", "pos_rule": "是", "hard_neg": "否"}.get(stratum)


def prelabel(gold: pd.DataFrame, llm, *, log=print) -> pd.DataFrame:
    from quantmind.line_a.harness.llm import BudgetExceededError, TLSInterceptionError
    rows = []
    for i, r in enumerate(gold.itertuples(index=False), 1):
        user = f"事件类别：{TYPES[r.gold_for][0]}\n公告标题：{r.title}"
        try:
            res = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], json_mode=True,
                           max_tokens=200, meta={"task": "x4_gold_prelabel", "gold_id": r.gold_id})
            out = json.loads(res.content)
            lab = out.get("label") if out.get("label") in ("是", "否") else None
            rows.append({"gold_id": r.gold_id, "ds_label": lab, "ds_reason": str(out.get("reason", ""))[:80],
                         "ds_cost": res.cost_rmb, "ds_model_version": res.model_version})
        except (TLSInterceptionError, BudgetExceededError):
            raise                      # 网络被解密或超预算：整步停下，不生成一张全是失败的复核表
        except Exception as e:  # noqa: BLE001  单条失败留空，进入复核
            rows.append({"gold_id": r.gold_id, "ds_label": None, "ds_reason": f"调用失败：{str(e)[:60]}",
                         "ds_cost": 0.0, "ds_model_version": ""})
        if i % 50 == 0:
            log(f"  prelabel {i}/{len(gold)}")
    return gold.merge(pd.DataFrame(rows), on="gold_id", how="left")


def review_sheet(pre: pd.DataFrame, path: Path, seed: int = SEED) -> pd.DataFrame:
    d = pre.copy()
    d["rule_label"] = d["stratum"].map(_rule)
    disagree = d["ds_label"].isna() | d["rule_label"].isna() | (d["ds_label"] != d["rule_label"])
    rng = random.Random(seed)
    agree_ids = sorted(d.loc[~disagree, "gold_id"])
    spot = set(rng.sample(agree_ids, round(REVIEW_SHARE * len(agree_ids)))) if agree_ids else set()
    d["review_reason"] = ""
    d.loc[d["rule_label"].isna(), "review_reason"] = "规则不确定"
    d.loc[d["rule_label"].notna() & disagree, "review_reason"] = "DeepSeek 与规则不一致"
    d.loc[d["gold_id"].isin(spot), "review_reason"] = "随机抽查"
    d["needs_review"] = d["review_reason"] != ""
    sheet = d[d["needs_review"]].sort_values(["review_reason", "gold_for", "gold_id"])
    out = pd.DataFrame({"编号": sheet["gold_id"], "需要复核的原因": sheet["review_reason"],
                        "事件类别": sheet["gold_for"].map(lambda t: TYPES[t][0]), "公告标题": sheet["title"],
                        "规则判断": sheet["rule_label"].fillna("不确定"), "DeepSeek 判断": sheet["ds_label"].fillna("（失败）"),
                        "DeepSeek 理由": sheet["ds_reason"], "你的判断（是/否）": "", "备注": ""})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return d


def import_review(pre_with_flags: pd.DataFrame, sheet_path: Path) -> pd.DataFrame:
    s = pd.read_csv(sheet_path, dtype=str, encoding="utf-8-sig").fillna("")
    s = s.rename(columns={"编号": "gold_id", "你的判断（是/否）": "user_label", "备注": "user_note"})
    s["user_label"] = s["user_label"].str.strip()
    bad = s[(s["user_label"] != "") & ~s["user_label"].isin(["是", "否"])]
    if len(bad):
        raise ValueError(f"{len(bad)} rows have a judgement other than 是/否, e.g. {bad['gold_id'].head(3).tolist()}")
    missing = s[s["user_label"] == ""]
    d = pre_with_flags.merge(s[["gold_id", "user_label", "user_note"]], on="gold_id", how="left")
    d["gold_label"] = d["user_label"].where(d["user_label"].isin(["是", "否"]))
    agree = ~d["needs_review"]
    d.loc[agree, "gold_label"] = d.loc[agree, "ds_label"]
    d["gold_source"] = "user"
    d.loc[agree, "gold_source"] = "deepseek=rule"
    d.loc[d["gold_label"].isna(), "gold_source"] = "missing"
    d.attrs["n_unreviewed"] = int(len(missing))
    return d


SEVERE = ("investigation", "penalty", "st_change", "regulatory_letter", "restructuring", "control_change")


def _prf(y: pd.Series, p: pd.Series) -> dict:
    tp = int((y & p).sum())
    prec = tp / max(1, int(p.sum()))
    rec = tp / max(1, int(y.sum()))
    return {"n": int(len(y)), "n_pos": int(y.sum()), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(1e-9, prec + rec)}


def score_gold(final: pd.DataFrame, pack: Path, results: Path) -> dict:
    """金标准上比较规则和各模型。final：import_review 的输出（gold_id、gold_for、announcementId、title、gold_label）。
    - 规则：标题规则 classify_title 判出的类型是不是 gold_for；
    - 模型：每类的阈值在银标签校准集上按 F1 选（与 x4_eval 相同），用到金标准条目（cell = sha(公告编号:类型)）上；
    - 指标：每类和宏平均的精确率、召回率、F1；严重事件类型（SEVERE）合在一起的召回率。"""
    import hashlib

    from quantmind.line_a.evidence.ann_types import classify_title
    from quantmind.line_a.p4.x4_eval import _best_threshold, _read
    f = final[final["gold_label"].isin(["是", "否"])].copy()
    f["y"] = f["gold_label"] == "是"
    f["cell"] = [hashlib.sha256(f"{a}:{t}".encode()).hexdigest()[:16] for a, t in zip(f["announcementId"], f["gold_for"],
                                                                                      strict=True)]
    systems = {"rule": pd.Series([classify_title(t).event_type == g for t, g in zip(f["title"], f["gold_for"], strict=True)],
                                 index=f.index)}
    cells = pd.DataFrame(_read(Path(pack) / "cells.jsonl")).set_index("cell")
    for fp in sorted(Path(results).rglob("*.jsonl")):
        if "_smoke" in fp.parts or fp.name == "train_log.jsonl":
            continue
        rows = pd.DataFrame(_read(fp))
        if rows.empty or "p_yes" not in rows:
            continue
        rows = rows[rows["status"] == "ok"].set_index("cell")
        name = f"{rows['mode'].iloc[0]}__{rows['model_key'].iloc[0]}"
        d = cells.join(rows[["p_yes"]], how="inner")
        pred = pd.Series(False, index=f.index)
        for t, g in d[d["split"] == "calib"].groupby("type"):
            thr = _best_threshold(g["label"].to_numpy(), g["p_yes"].to_numpy())
            m = f["gold_for"] == t
            p = f.loc[m, "cell"].map(rows["p_yes"])
            pred.loc[m] = (p >= thr).fillna(False).to_numpy()
        systems[name] = pred
    out = {}
    for name, pred in systems.items():
        by = {t: _prf(f.loc[f["gold_for"] == t, "y"], pred[f["gold_for"] == t]) for t in sorted(f["gold_for"].unique())}
        sev = f["gold_for"].isin(SEVERE)
        out[name] = {"all": _prf(f["y"], pred), "macro_f1": float(pd.Series([v["f1"] for v in by.values()]).mean()),
                     "severe": _prf(f.loc[sev, "y"], pred[sev]), "by_type": by}
    return out
