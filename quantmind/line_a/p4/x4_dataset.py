"""quantmind.line_a.p4.x4_dataset — X4 公告分类的数据集（03_final_plan_v2 第 9.1、10.4 节 X4；P4）.

单位是一条公告（第一版只用标题）。每条公告对每个目标事件类型给一个银标签（弱标签的组合）：
三个相互独立的来源：
1. 结构化事件表：同一股票、公告日前后 1 天内有对应记录（业绩预告 / 快报 / 回购 / 增减持 / 解禁 / 分红 / 审计 / ST 名称变更）；
2. 标题规则（ann_types.classify_title）；
3. 巨潮自带的类别代码：先用规则标签在训练期统计每个代码的多数类型，作为第三个投票者。
标签：
- pos_strong：规则 = T 且（结构化匹配 T，或该类型没有结构化表时类别代码多数 = T）；
- neg：规则 ≠ T 且没有结构化匹配 T；
- uncertain：其余（来源之间有冲突）。不进训练集，单独作为"难例"子集报告。
切分按时间：train 到 2023-06-30，calib 2023-07-01～2023-12-31，test 2024-01-01 起；
标题先归一（去掉数字、日期、括号里的编号）再按模板去重，模板只出现在一个切分里，避免模板泄漏。
这些都是银标签：方案要求人工金标准覆盖稀有的严重事件，见 gold_sample()。
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

from quantmind.line_a.evidence.ann_types import classify_title

# 目标类型（中文说明会进入 VT 问题）与可用的结构化表
TYPES = {
    "performance_forecast": ("业绩预告（预增、预减、扭亏、首亏、续亏等对报告期业绩的预先披露）", "forecast"),
    "performance_express": ("业绩快报（正式报告前披露的主要财务数据）", "express"),
    "repurchase": ("公司回购本公司股份（预案、进展、实施、完成、终止都算；不含限制性股票的回购注销）", "repurchase"),
    "holder_reduction": ("股东或董监高减持公司股份（计划、进展、结果）", "stk_holdertrade"),
    "holder_increase": ("股东或董监高增持公司股份（计划、进展、结果）", "stk_holdertrade"),
    "unlock": ("限售股份解除限售、上市流通", "share_float"),
    "dividend": ("利润分配、权益分派、分红", "dividend"),
    "st_change": ("被实施或撤销风险警示（ST、*ST）", "namechange"),
    "investigation": ("公司或控股股东、实控人被立案调查", None),
    "penalty": ("收到行政处罚、市场禁入、纪律处分、公开谴责", None),
    "regulatory_letter": ("收到交易所或监管机构的问询函、关注函、监管函、警示函", None),
    "litigation": ("涉及诉讼或仲裁", None),
    "restructuring": ("重大资产重组、发行股份购买资产、吸收合并", None),
    "control_change": ("控股股东或实际控制人变更、控制权变更", None),
}
TRAIN_END, CALIB_END = "20230630", "20231231"


def load_corpus(lake: Path) -> pd.DataFrame:
    parts = sorted((Path(lake) / "text/cninfo/_parts").glob("*.parquet"))
    df = pd.concat([pd.read_parquet(p, columns=["announcementId", "secCode", "announcementTitle", "announcementType",
                                                "publish_ts_bj"]) for p in parts], ignore_index=True)
    df = df.drop_duplicates("announcementId")
    code = df["secCode"].astype(str).str.zfill(6)
    suf = code.str[0].map({"6": ".SH", "0": ".SZ", "3": ".SZ", "4": ".BJ", "8": ".BJ", "9": ".BJ"})
    df = df.assign(ts_code=code + suf, title=df["announcementTitle"].astype(str).str.strip(),
                   date=df["publish_ts_bj"].astype(str).str.slice(0, 10).str.replace("-", ""))
    df = df[suf.notna() & (df["title"].str.len() > 3)]
    return df[["announcementId", "ts_code", "date", "title", "announcementType"]].reset_index(drop=True)


def _struct_keys(lake: Path, api: str, extra=None) -> set[str]:
    """结构化表的 "股票_公告日" 键集合。"""
    fs = sorted((Path(lake) / "events").glob(f"{api}_2*.parquet"))
    if not fs:
        return set()
    cols = ["ts_code", "ann_date"] + ([] if extra is None else extra[0])
    df = pd.read_parquet(fs[-1], columns=cols)
    if extra is not None:
        df = df[extra[1](df)]
    d = df["ann_date"].astype(str).str.replace("-", "").str.slice(0, 8)
    return set((df["ts_code"].astype(str) + "_" + d).tolist())


_KEYS: dict[str, dict] = {}


def struct_matches(lake: Path, corpus: pd.DataFrame) -> pd.DataFrame:
    if str(lake) not in _KEYS:
        _KEYS[str(lake)] = _load_keys(lake)
    keys = _KEYS[str(lake)]
    d = pd.to_datetime(corpus["date"], format="%Y%m%d", errors="coerce")
    near = [d.dt.strftime("%Y%m%d"), (d - pd.Timedelta(days=1)).dt.strftime("%Y%m%d"),
            (d + pd.Timedelta(days=1)).dt.strftime("%Y%m%d")]
    out = pd.DataFrame(index=corpus.index)
    near_keys = [corpus["ts_code"] + "_" + dd for dd in near]
    for t, ks in keys.items():
        hit = pd.Series(False, index=corpus.index)
        for k in near_keys:
            hit |= k.isin(ks)
        out[f"struct_{t}"] = hit
    return out


def _load_keys(lake: Path) -> dict:
    return {
        "performance_forecast": _struct_keys(lake, "forecast"),
        "performance_express": _struct_keys(lake, "express"),
        "repurchase": _struct_keys(lake, "repurchase"),
        "holder_reduction": _struct_keys(lake, "stk_holdertrade", (["in_de"], lambda x: x["in_de"].astype(str) == "DE")),
        "holder_increase": _struct_keys(lake, "stk_holdertrade", (["in_de"], lambda x: x["in_de"].astype(str) == "IN")),
        "unlock": _struct_keys(lake, "share_float"),
        "dividend": _struct_keys(lake, "dividend"),
        "st_change": _struct_keys(lake, "namechange", (["name"], lambda x: x["name"].astype(str).str.contains("ST"))),
    }


def template(title: str) -> str:
    t = re.sub(r"(19|20)\d{2}\s*年?(度|半年度|第[一二三四]季度)?", "<Y>", title)
    t = re.sub(r"[（(][^）)]{0,20}[）)]", "", t)
    t = re.sub(r"\d+(\.\d+)?", "<N>", t)
    return re.sub(r"\s+", "", t)


def build(lake: Path) -> pd.DataFrame:
    c = load_corpus(lake)
    rule = c["title"].map(classify_title)
    c["rule_type"] = [r.event_type for r in rule]
    c["rule_stage"] = [r.stage for r in rule]
    c["is_meta"] = [r.is_summary_or_meta for r in rule]
    c = pd.concat([c, struct_matches(lake, c)], axis=1)
    c["split"] = pd.cut(c["date"].astype(int), [0, int(TRAIN_END), int(CALIB_END), 99999999],
                        labels=["train", "calib", "test"]).astype(str)
    # 巨潮类别代码的多数类型（只用训练期统计，避免泄漏）
    codes = c["announcementType"].astype(str).str.split(r"\|\|")
    tr = c[c["split"] == "train"].assign(code=codes).explode("code")
    major = tr.groupby("code")["rule_type"].agg(lambda s: s.value_counts().index[0])
    c["code_major"] = codes.map(lambda cs: [major.get(x) for x in cs])
    for t, (_, api) in TYPES.items():
        rule_t = c["rule_type"] == t
        if api is not None:
            agree = c[f"struct_{t}"]
        else:
            agree = c["code_major"].map(lambda m, tt=t: tt in m)
        lab = pd.Series("neg", index=c.index)
        lab[rule_t & agree] = "pos_strong"
        lab[(rule_t & ~agree) | (~rule_t & (c[f"struct_{t}"] if api is not None else False))] = "uncertain"
        c[f"label_{t}"] = lab
    # 模板去重：每个模板只保留在它第一次出现的切分里（时间最早），后面切分里的同模板公告剔除
    c["tpl"] = c["title"].map(template)
    first = c.sort_values("date").drop_duplicates("tpl")[["tpl", "split"]].set_index("tpl")["split"]
    c["tpl_first_split"] = c["tpl"].map(first)
    c["keep"] = (c["split"] == c["tpl_first_split"]) & ~c["is_meta"]
    return c


def build_chunked(lake: Path, out_dir: Path) -> pd.DataFrame:
    """与 build() 相同的规则，按年分块处理（全量约 370 万条公告，一次处理内存不够）。每年写一个 parquet，返回汇总。"""
    from collections import Counter
    lake, out_dir = Path(lake), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = sorted((lake / "text/cninfo/_parts").glob("*.parquet"))
    years = sorted({p.stem[:4] for p in parts})

    def year_corpus(y: str) -> pd.DataFrame:
        ps = [p for p in parts if p.stem.startswith(y)]
        df = pd.concat([pd.read_parquet(p, columns=["announcementId", "secCode", "announcementTitle",
                                                    "announcementType", "publish_ts_bj"]) for p in ps], ignore_index=True)
        df = df.drop_duplicates("announcementId")
        code = df["secCode"].astype(str).str.zfill(6)
        suf = code.str[0].map({"6": ".SH", "0": ".SZ", "3": ".SZ", "4": ".BJ", "8": ".BJ", "9": ".BJ"})
        df = df.assign(ts_code=code + suf, title=df["announcementTitle"].astype(str).str.strip(),
                       date=df["publish_ts_bj"].astype(str).str.slice(0, 10).str.replace("-", ""))
        df = df[suf.notna() & (df["title"].str.len() > 3)]
        df = df[["announcementId", "ts_code", "date", "title", "announcementType"]].reset_index(drop=True)
        rule = df["title"].map(classify_title)
        df["rule_type"] = [r.event_type for r in rule]
        df["rule_stage"] = [r.stage for r in rule]
        df["is_meta"] = [r.is_summary_or_meta for r in rule]
        df["split"] = pd.cut(df["date"].astype(int), [0, int(TRAIN_END), int(CALIB_END), 99999999],
                             labels=["train", "calib", "test"]).astype(str)
        return df

    # 第一遍：训练期里每个巨潮类别代码的多数规则类型
    cnt: Counter = Counter()
    for y in years:
        if y > TRAIN_END[:4]:
            break
        df = year_corpus(y)
        df = df[df["split"] == "train"]
        ex = df.assign(code=df["announcementType"].astype(str).str.split(r"\|\|")).explode("code")
        cnt.update(zip(ex["code"], ex["rule_type"], strict=True))
    by_code: dict[str, Counter] = {}
    for (code, t), n in cnt.items():
        by_code.setdefault(code, Counter())[t] = n
    major = {code: c.most_common(1)[0][0] for code, c in by_code.items()}
    # 第二遍：逐年打标签、模板去重（按时间顺序，模板只留在第一次出现的切分）
    first_split: dict[str, str] = {}
    rows = []
    for y in years:
        c = year_corpus(y)
        c = pd.concat([c, struct_matches(lake, c)], axis=1)
        codes = c["announcementType"].astype(str).str.split(r"\|\|")
        c["code_major"] = codes.map(lambda cs: ",".join(sorted({major.get(x, "") for x in cs} - {""})))
        for t, (_, api) in TYPES.items():
            rule_t = c["rule_type"] == t
            agree = c[f"struct_{t}"] if api is not None else c["code_major"].str.split(",").map(lambda m, tt=t: tt in m)
            lab = pd.Series("neg", index=c.index)
            lab[rule_t & agree] = "pos_strong"
            conflict = (rule_t & ~agree) | (~rule_t & c[f"struct_{t}"]) if api is not None else (rule_t & ~agree)
            lab[conflict] = "uncertain"
            c[f"label_{t}"] = lab
        c = c.sort_values("date")
        c["tpl"] = c["title"].map(template)
        for tpl, s in zip(c["tpl"], c["split"], strict=True):
            first_split.setdefault(tpl, s)
        c["keep"] = (c["split"] == c["tpl"].map(first_split)) & ~c["is_meta"]
        c.drop(columns=["announcementType"]).to_parquet(out_dir / f"x4_{y}.parquet")
        rows.append(summary(c).assign(year=y))
        print(f"  {y}: {len(c)} announcements, kept {int(c['keep'].sum())}", flush=True)
    s = pd.concat(rows, ignore_index=True)
    return s.groupby(["type", "split"], as_index=False)[["pos_strong", "uncertain", "neg"]].sum()


def summary(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    k = c[c["keep"]]
    for t in TYPES:
        for s in ("train", "calib", "test"):
            v = k.loc[k["split"] == s, f"label_{t}"].value_counts()
            rows.append({"type": t, "split": s, "pos_strong": int(v.get("pos_strong", 0)),
                         "uncertain": int(v.get("uncertain", 0)), "neg": int(v.get("neg", 0))})
    return pd.DataFrame(rows)


def gold_sample(c: pd.DataFrame, per_type: int = 40, seed: int = 20261021) -> pd.DataFrame:
    """人工金标准的分层样本（测试期）：每类正例、不确定、难负例（标题含关键词但规则判为别的类型）各取若干。"""
    k = c[c["keep"] & (c["split"] == "test")]
    picks = []
    for i, t in enumerate(TYPES):
        for lab, n in (("pos_strong", per_type // 2), ("uncertain", per_type // 4)):
            g = k[k[f"label_{t}"] == lab]
            picks.append(g.sample(min(n, len(g)), random_state=seed + i).assign(gold_for=t, stratum=lab))
        kw = TYPES[t][0][:2]
        hard = k[(k[f"label_{t}"] == "neg") & k["title"].str.contains(kw, regex=False)]
        picks.append(hard.sample(min(per_type // 4, len(hard)), random_state=seed + 100 + i).assign(gold_for=t,
                                                                                                  stratum="hard_neg"))
    out = pd.concat(picks, ignore_index=True)
    out["gold_id"] = [hashlib.sha256(f"{a}:{t}".encode()).hexdigest()[:12] for a, t in zip(out["announcementId"],
                                                                                         out["gold_for"], strict=True)]
    return out[["gold_id", "gold_for", "stratum", "announcementId", "ts_code", "date", "title"]]
