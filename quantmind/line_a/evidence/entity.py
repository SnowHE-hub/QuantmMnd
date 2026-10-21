"""quantmind.line_a.evidence.entity — 证券身份链接（03_final_plan_v2 第 5 节 F5(b)2；P2 第一版）.

两步分开，便于分别评估（第 14 节"主体误链接"）：
1. 识别：在文本里找出可能指向上市主体的提及——证券代码、证券简称（含曾用名）、公司全称。
   第一版用别名词典（规则）；GLiNER 等识别模型是 P4 的实验，必须在这个基线上对比。
2. 链接：把提及映射到 ts_code，并带上时点——简称按名称变更表的生效区间判断，as_of 当天不在生效区间的别名不链接。
   - 同一别名在 as_of 指向多只股票：标为 ambiguous，不自动选；
   - 提及前后出现"子公司 / 参股 / 控股股东 / 母公司"等字样：标出关系上下文，提醒校验器这里可能不是上市主体本身。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

CONTEXT = {
    "subsidiary": r"(?:全资|控股)?子公司|参股公司|孙公司",
    "parent": r"控股股东|母公司|实际控制人|集团公司",
}
_CODE = re.compile(r"(?<!\d)(\d{6})(?:\.(SH|SZ|BJ))?(?!\d)")


@dataclass(frozen=True)
class Alias:
    text: str
    ts_code: str
    kind: str              # code / short_name / former_name / full_name
    start: str = ""        # 生效区间 [start, end]，YYYYMMDD；空 = 不限
    end: str = ""


@dataclass(frozen=True)
class Mention:
    span: tuple[int, int]
    text: str
    candidates: tuple[str, ...]
    kind: str
    status: str            # linked / ambiguous / out_of_period
    context: str           # self / subsidiary / parent


def _suffix(code: str) -> str:
    return {"6": "SH", "0": "SZ", "3": "SZ", "4": "BJ", "8": "BJ", "9": "BJ"}.get(code[0], "")


class AliasTable:
    def __init__(self, aliases: list[Alias]):
        self.by_text: dict[str, list[Alias]] = {}
        for a in aliases:
            if len(a.text) >= 2:
                self.by_text.setdefault(a.text, []).append(a)
        names = sorted(self.by_text, key=len, reverse=True)       # 长名优先，避免"平安"截走"中国平安"
        self.pattern = re.compile("|".join(map(re.escape, names))) if names else None

    @classmethod
    def from_frames(cls, stock_basic, namechange=None) -> AliasTable:
        """stock_basic：ts_code、name，可选 fullname；namechange：ts_code、name、start_date、end_date。"""
        al: list[Alias] = []
        for r in stock_basic.itertuples(index=False):
            al.append(Alias(str(r.name), r.ts_code, "short_name"))
            fn = getattr(r, "fullname", None)
            if isinstance(fn, str) and fn:
                al.append(Alias(fn, r.ts_code, "full_name"))
        if namechange is not None:
            for r in namechange.itertuples(index=False):
                st = "" if r.start_date is None else str(r.start_date)[:8]
                en = "" if r.end_date is None or str(r.end_date) in ("None", "nan", "NaT") else str(r.end_date)[:8]
                al.append(Alias(str(r.name), r.ts_code, "former_name", st, en))
        return cls(al)

    @classmethod
    def from_lake(cls, lake, *, full_history: bool = True) -> AliasTable:
        """从 lake 构建：沪深股票基础表 + 北交所基础表 + 名称变更。
        默认用按股票拉的名称变更全历史（namechange_full_*.parquet；T-038 X5 测试集上链接正确率 95.2% → 97.4%），
        没有全历史文件时退回 lake 的 namechange 快照。"""
        from pathlib import Path

        import pandas as pd
        lake = Path(lake)
        sb = pd.read_parquet(sorted(lake.glob("stock_basic_full_merged_2*.parquet"))[-1],
                             columns=["ts_code", "name", "fullname"])
        bse = sorted((lake / "bse").glob("stock_basic_bse_*.parquet"))
        if bse:
            b = pd.read_parquet(bse[-1])
            b = b.assign(fullname=b.get("fullname", None))[["ts_code", "name", "fullname"]]
            sb = pd.concat([sb, b], ignore_index=True).drop_duplicates("ts_code")
        full = sorted((lake / "events").glob("namechange_full_2*.parquet")) if full_history else []
        ncs = full or sorted((lake / "events").glob("namechange_2*.parquet"))
        nc = pd.read_parquet(ncs[-1], columns=["ts_code", "name", "start_date", "end_date"]) if ncs else None
        return cls.from_frames(sb, nc)

    def link(self, text: str, as_of: str) -> list[Mention]:
        out: list[Mention] = []
        for m in _CODE.finditer(text):
            code, suf = m.group(1), m.group(2) or _suffix(m.group(1))
            if suf:
                out.append(Mention(m.span(), m.group(), (f"{code}.{suf}",), "code", "linked", _context(text, *m.span())))
        if self.pattern is None:
            return out
        for m in self.pattern.finditer(text):
            al = self.by_text[m.group()]
            valid = [a for a in al if (not a.start or a.start <= as_of) and (not a.end or as_of <= a.end)]
            kinds = {a.kind for a in al}
            cands = tuple(sorted({a.ts_code for a in valid}))
            if not cands:
                status = "out_of_period"
                cands = tuple(sorted({a.ts_code for a in al}))
            else:
                status = "linked" if len(cands) == 1 else "ambiguous"
            kind = "full_name" if "full_name" in kinds else ("short_name" if "short_name" in kinds else "former_name")
            out.append(Mention(m.span(), m.group(), cands, kind, status, _context(text, *m.span())))
        return sorted(out, key=lambda x: x.span)


def _context(text: str, s: int, e: int, width: int = 8) -> str:
    near = text[max(0, s - width):s] + text[e:e + width]
    for k, pat in CONTEXT.items():
        if re.search(pat, near):
            return k
    return "self"
