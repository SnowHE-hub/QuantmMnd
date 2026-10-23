"""quantmind.line_a.harness.completeness — 角色输出的完整性检查（T-051；代码实现）.

只检查"必须覆盖"的项目有没有写到，不检查数值（数值由校验器核对）。缺了就把清单发回去让模型补一次。
A1 的必须覆盖项来自 A1 提示词 v2（2026-10-21）：最新报告期与上年同期的营业收入、归母净利润；as_of 前 120 天内
最近一次业绩预告的类型；最近一份年报的审计意见。工具没有返回对应记录的项目不要求。
"""
from __future__ import annotations

import pandas as pd


def _parts(eid: str) -> list[str]:
    return eid.split(":")


def a1_missing(output: dict, evidence: dict[str, dict], as_of: str) -> list[str]:
    cited: dict[tuple[str, str], set[str]] = {}          # (表, 报告期) → 引用过的字段
    cited_tables: set[str] = set()
    for c in output.get("claims", []):
        for e in c.get("evidence") or []:
            p = _parts(e.get("evidence_id", ""))
            if len(p) >= 4 and p[0] == "tushare":
                cited_tables.add(p[1])
                cited.setdefault((p[1], p[3]), set()).add(e.get("field") or "")
    inc = sorted({_parts(k)[3] for k in evidence if k.startswith("tushare:income:") and len(_parts(k)) >= 5})
    miss = []
    if inc:
        latest = inc[-1]
        prev = str(int(latest[:4]) - 1) + latest[4:]
        periods = [("最新报告期", latest)] + ([("上年同期", prev)] if prev in inc else [])
        for name, end in periods:
            got = cited.get(("income", end), set())
            for f, label in (("revenue", "营业收入"), ("n_income_attr_p", "归母净利润")):
                if f not in got:
                    miss.append(f"{name}（{end}）的{label}：引用利润表记录 tushare:income:…:{end}:… 的 {f} 字段")
    lo = (pd.Timestamp(as_of) - pd.Timedelta(days=120)).strftime("%Y%m%d")
    fc = [k for k in evidence if k.startswith("tushare:forecast:") and len(_parts(k)) >= 5 and lo < _parts(k)[4] <= as_of]
    if fc and "forecast" not in cited_tables:
        miss.append("as_of 前 120 天内最近一次业绩预告的类型（引用 tushare:forecast 记录的 type 字段）")
    au = [k for k in evidence if k.startswith("tushare:fina_audit:") and len(_parts(k)) >= 4 and _parts(k)[3].endswith("1231")]
    if au and "fina_audit" not in cited_tables:
        miss.append("最近一份年报的审计意见（引用 tushare:fina_audit 记录的 audit_result 字段）")
    return miss
