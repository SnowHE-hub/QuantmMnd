"""quantmind.line_a.p4.x5_entity — X5 实体识别与证券链接：开发集、规则基线与打分（03_final_plan_v2 第 9.1、10.4 节 X5；P4）.

开发集 v1（300 句，种子 20261022）：
- 材料：购买的 research_report 摘要（T-038），trade_date 2019–2025；按"。；"和换行切句，长度 15–160 字；
- 分层（用规则基线 evidence.entity.AliasTable 的结果分层，只用来抽样，不当标签）：
  - multi：行业研报里规则认出 ≥ 2 个提及的句子，100 句；
  - other：个股研报里提到的不是本报告覆盖公司的句子，75 句；
  - miss：规则一个都没认出、但含"××集团 / 股份 / 科技…"这类公司名样式的句子，75 句；
  - code：含 6 位数字代码的句子（包括北交所 2024 年换码前的旧代码），50 句；
- 链接时点 = 研报日期（trade_date）。
标注（与 X4 不同，用户 2026-10-22 让 Claude 做复核）：
- DeepSeek 与 Claude 各自独立标注，互相看不到对方的结果；分歧由 Claude 逐条裁定，裁定时可以查别名表和股票基础表；
- 每个提及：原文片段、类型（listed = A 股上市公司 / unlisted = 其他公司或机构）、ts_code（listed 必填）、关系（self / subsidiary / parent / other）。
打分（两项分开报告，方案要求）：
- 识别：公司提及的片段级精确率、召回率、F1（片段文本完全相同才算对）；
- 链接：在标准片段上，上市公司提及链接到正确 ts_code 的比例；ambiguous 算没链接。
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from quantmind.line_a.evidence.entity import AliasTable

SEED = 20261022
STRATA = {"multi": 100, "other": 75, "miss": 75, "code": 50}
_SPLIT = re.compile(r"[。；;\n]|　　")
_ORGLIKE = re.compile(r"[一-龥]{2,10}(?:集团|股份|科技|电子|能源|汽车|药业|医药|银行|证券|控股|材料|电气|新材|智能|半导体|光电)")
_CODE6 = re.compile(r"(?<!\d)\d{6}(?!\d)")


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def alias_table(lake: Path, *, full_history: bool = False) -> AliasTable:
    """full_history：用按股票拉的名称变更全历史（namechange_full_*.parquet），旧名也能认出。"""
    lake = Path(lake)
    sb = pd.read_parquet(sorted(lake.glob("stock_basic_full_merged_2*.parquet"))[-1], columns=["ts_code", "name", "fullname"])
    bse = sorted((lake / "bse").glob("stock_basic_bse_*.parquet"))
    if bse:
        b = pd.read_parquet(bse[-1])
        b = b.assign(fullname=b.get("fullname", None))[["ts_code", "name", "fullname"]]
        sb = pd.concat([sb, b], ignore_index=True).drop_duplicates("ts_code")
    pat = "namechange_full_2*.parquet" if full_history else "namechange_2*.parquet"
    nc = pd.read_parquet(sorted((lake / "events").glob(pat))[-1], columns=["ts_code", "name", "start_date", "end_date"])
    return AliasTable.from_frames(sb, nc)


def sentences(rr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in rr.itertuples(index=False):
        for i, s in enumerate(_SPLIT.split(str(r.abstr))):
            s = s.strip()
            if 15 <= len(s) <= 160:
                rows.append({"sid": _sha(f"{r.row_hash}:{i}")[:12], "trade_date": r.trade_date, "report_type": r.report_type,
                             "ts_code": r.ts_code, "name": r.name, "sentence": s, "row_hash": r.row_hash})
    return pd.DataFrame(rows)


def build_devset(lake: Path, rr_path: Path, out_dir: Path, *, pool: int = 6000, seed: int = SEED,
                 version: str = "x5-dev-v1", exclude_reports: set | None = None) -> pd.DataFrame:
    """先按种子抽 pool 份研报切句，再用规则基线分层抽样。测试集换种子，并排除开发集用过的研报（exclude_reports：row_hash）。"""
    rr = pd.read_parquet(rr_path, columns=["trade_date", "ts_code", "name", "report_type", "abstr", "row_hash"])
    rr = rr[(rr["trade_date"] >= "20190101") & (rr["trade_date"] <= "20251231") & (rr["abstr"].str.len() > 30)]
    if exclude_reports:
        rr = rr[~rr["row_hash"].isin(exclude_reports)]
    rr = rr.sample(min(pool, len(rr)), random_state=seed)
    st = sentences(rr)
    at = alias_table(lake)
    ms = [at.link(s, d) for s, d in zip(st["sentence"], st["trade_date"], strict=True)]
    st["n_rule"] = [len(m) for m in ms]
    st["rule_codes"] = [sorted({c for x in m for c in x.candidates}) for m in ms]
    is_ind = st["report_type"] == "行业研报"
    other = [bool(c) and any(x != t for x in c) for c, t in zip(st["rule_codes"], st["ts_code"], strict=True)]
    cand = {
        "code": st[st["sentence"].str.contains(_CODE6)],
        "multi": st[is_ind & (st["n_rule"] >= 2)],
        "other": st[~is_ind & pd.Series(other, index=st.index) & (st["ts_code"] != "")],
        "miss": st[(st["n_rule"] == 0) & st["sentence"].str.contains(_ORGLIKE)],
    }
    picks, used = [], set()
    for k, n in STRATA.items():
        c = cand[k][~cand[k]["sid"].isin(used)]
        p = c.sample(min(n, len(c)), random_state=seed).assign(stratum=k)
        used |= set(p["sid"])
        picks.append(p)
    dev = pd.concat(picks, ignore_index=True)
    dev = dev.drop(columns=["rule_codes", "n_rule"]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev.to_csv(out_dir / "sentences.csv", index=False, encoding="utf-8-sig")
    man = {"version": version, "seed": seed, "pool_reports": int(len(rr)), "pool_sentences": int(len(st)),
           "n_excluded_reports": len(exclude_reports or ()),
           "strata": dev["stratum"].value_counts().to_dict(),
           "candidates": {k: int(len(v)) for k, v in cand.items()},
           "sentences_sha256": _sha((out_dir / "sentences.csv").read_text(encoding="utf-8-sig"))}
    (out_dir / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    return dev


GUIDE = """标注规范（DeepSeek 与 Claude 用同一份）：
1. 只标企业：公司、集团、银行、券商、基金公司等，不论是否上市；外国公司也标。
   不标：政府部门、监管机构、交易所、行业协会、国家和地区、产品或型号、指数、人名。
2. 片段必须是句子里原样出现的连续文字；同一企业在一句里出现多次，只标第一次；简称、品牌式简称（例如"茅台"指公司时）也标。
3. type：listed = 研报日期当天在沪深北交易所上市的 A 股公司；其他一律 unlisted（包括只在港股、美股上市的公司，
   以及上市公司的子公司——子公司本身没有上市，不能链接到母公司）。
4. listed 必须给 ts_code（6 位代码加 .SH / .SZ / .BJ，按研报日期当时的代码）。"""


def llm_label(dev: pd.DataFrame, llm, *, log=print) -> dict[str, list[dict]]:
    """DeepSeek 独立标注（不给它规则基线的结果）。片段不在原句里的提及丢掉并计数。"""
    from quantmind.line_a.harness.llm import BudgetExceededError, TLSInterceptionError
    out, dropped = {}, 0
    sys_msg = ("你是 A 股研报文本的实体标注员。" + GUIDE +
               '\n只输出 JSON：{"mentions": [{"text": "...", "type": "listed" 或 "unlisted", "ts_code": "600519.SH" 或 null}]}')
    for i, r in enumerate(dev.itertuples(index=False), 1):
        user = f"研报日期：{r.trade_date}\n句子：{r.sentence}"
        ms = None
        for mt in (600, 3000):          # 名单很长的句子 600 token 会被截断、JSON 不完整，就放大上限再试一次
            try:
                res = llm.chat([{"role": "system", "content": sys_msg}, {"role": "user", "content": user}],
                               json_mode=True, max_tokens=mt, meta={"task": "x5_label", "sid": r.sid})
                ms = json.loads(res.content).get("mentions", [])
                break
            except (TLSInterceptionError, BudgetExceededError):
                raise
            except Exception:  # noqa: BLE001
                ms = None
        if ms is None:
            out[r.sid] = None
            continue
        keep = []
        for m in ms:
            if isinstance(m, dict) and m.get("text") and m["text"] in r.sentence:
                keep.append({"text": m["text"], "type": m.get("type"), "ts_code": m.get("ts_code")})
            else:
                dropped += 1
        out[r.sid] = keep
        if i % 50 == 0:
            log(f"  labeled {i}/{len(dev)}")
    out["_dropped_not_substring"] = dropped
    return out


# ---------------------------------------------------------------- 规则基线与打分
def rule_predict(at: AliasTable, dev: pd.DataFrame) -> dict[str, list[dict]]:
    out = {}
    for r in dev.itertuples(index=False):
        out[r.sid] = [{"text": m.text, "ts_code": m.candidates[0] if m.status == "linked" else None,
                       "status": m.status} for m in at.link(r.sentence, r.trade_date)]
    return out


def gliner_predict(at: AliasTable, dev: pd.DataFrame, model_id: str = "fastino/gliner2.5-multi-v1", *,
                   labels: tuple = ("company",), threshold: float = 0.5, batch_size: int = 16) -> dict[str, list[dict]]:
    """GLiNER2 只做识别；识别出的片段再用规则链接器（别名表，按研报日期）链接，链接不上的 ts_code 为空。"""
    from gliner2 import GLiNER2
    m = GLiNER2.from_pretrained(model_id)
    with contextlib.suppress(Exception):                     # 没有 GPU 就用 CPU
        m = m.to("cuda")
    res = m.batch_extract_entities(dev["sentence"].tolist(), list(labels), batch_size=batch_size, threshold=threshold)
    out = {}
    for r, x in zip(dev.itertuples(index=False), res, strict=True):
        ents = (x.get("entities") or {}) if isinstance(x, dict) else {}
        spans = [s if isinstance(s, str) else s.get("text", "") for lab in labels for s in ents.get(lab, [])]
        ms = []
        for s in dict.fromkeys(spans):                       # 去重、保序
            if not s or s not in r.sentence:
                continue
            lk = [x for x in at.link(s, r.trade_date) if x.text == s and x.status == "linked"]
            ms.append({"text": s, "ts_code": lk[0].candidates[0] if lk else None})
        out[r.sid] = ms
    return out


def dedupe(ms: list[dict]) -> list[dict]:
    """同一句里链接到同一代码的提及只留第一个（规范第 2 条"同一企业只标第一次"；对所有系统一样处理，打分前定下）。"""
    seen, out = set(), []
    for m in ms:
        c = m.get("ts_code")
        if c and c in seen:
            continue
        if c:
            seen.add(c)
        out.append(m)
    return out


def score(pred: dict[str, list[dict]], gold: dict[str, list[dict]]) -> dict:
    """识别：公司提及（listed + unlisted）的片段 P/R/F1；链接：标准 listed 片段上，预测里同一片段链接到同一证券的比例
    （标准里的 alt_codes 是同一证券的其他代码，例如北交所换码前的旧代码，也算对）。"""
    tp = fp = fn = 0
    link_ok = link_n = 0
    for sid, g in gold.items():
        p = dedupe(pred.get(sid) or [])
        gs = {(m["text"]) for m in g}
        ps = {(m["text"]) for m in p}
        tp += len(gs & ps)
        fp += len(ps - gs)
        fn += len(gs - ps)
        pmap = {m["text"]: m.get("ts_code") for m in p}
        for m in g:
            if m.get("type") == "listed":
                link_n += 1
                link_ok += int(pmap.get(m["text"]) in {m.get("ts_code"), *m.get("alt_codes", [])})
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return {"ner": {"precision": p, "recall": r, "f1": 2 * p * r / max(1e-9, p + r), "tp": tp, "fp": fp, "fn": fn},
            "linking": {"accuracy": link_ok / max(1, link_n), "n": link_n}}
