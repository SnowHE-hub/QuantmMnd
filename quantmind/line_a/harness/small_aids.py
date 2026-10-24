"""quantmind.line_a.harness.small_aids — 给本机小模型的代码辅助（T-054）.

思路是"代码多做一点，模型少做一点"，不加长提示词：
- compact：紧凑证据视图。只给 A1 需要的表、期数和字段，证据编号与字段名不变；
- prefill：代码预填事实表。把"必须覆盖"的事实直接写成声明草稿（数值、单位、可读金额、证据编号、字段名都由代码给出）；
- repair：校验器反馈修复。第一次输出后，把没通过校验的声明和原因发回去修一轮；修复后更差就退回；
- render（S5）：数字归代码。必须覆盖的事实和同比增速由代码写入，模型声明里的数值和金额写法按所引记录改正。
"""
from __future__ import annotations

import json
import math
import re

import pandas as pd

A1_FIELDS = {
    "income": ["revenue", "total_revenue", "n_income_attr_p", "n_income", "operate_profit"],
    "balancesheet": ["total_assets", "total_liab", "total_cur_assets", "total_cur_liab", "money_cap"],
    "cashflow": ["n_cashflow_act", "free_cashflow", "c_paid_invest"],
    "fina_indicator_full": ["grossprofit_margin", "netprofit_margin", "debt_to_assets", "current_ratio", "or_yoy",
                            "netprofit_yoy", "roe"],
}
KEEP_PERIODS = {"income": 5, "balancesheet": 2, "cashflow": 2, "fina_indicator_full": 2}
BASE = ("evidence_id", "end_date", "ann_date", "_units", "_display")


def _slim(rec: dict, fields: list[str]) -> dict:
    out = {k: rec[k] for k in BASE if k in rec}
    out |= {f: rec[f] for f in fields if f in rec}
    if "_units" in out:
        out["_units"] = {k: v for k, v in out["_units"].items() if k in fields}
    if "_display" in out:
        out["_display"] = {k: v for k, v in out["_display"].items() if k in fields}
    return out


def compact_a1(tool: str, out: dict) -> dict:
    """单个 A1 工具结果的紧凑视图。利润表另外保证"最新期的上年同期"在里面（即使不在最近 5 期）。"""
    if tool != "financial_snapshot" or not isinstance(out, dict) or "tables" not in out:
        return out
    tabs = {}
    for t, rows in out["tables"].items():
        if t not in A1_FIELDS or not rows:
            continue
        rows = sorted(rows, key=lambda r: (str(r.get("end_date")), str(r.get("ann_date"))), reverse=True)
        keep = rows[: KEEP_PERIODS[t]]
        if t == "income":
            latest = str(rows[0].get("end_date"))
            prev = str(int(latest[:4]) - 1) + latest[4:]
            if not any(str(r.get("end_date")) == prev for r in keep):
                keep += [r for r in rows if str(r.get("end_date")) == prev][:1]
        tabs[t] = [_slim(r, A1_FIELDS[t]) for r in keep]
    return {k: v for k, v in out.items() if k != "tables"} | {"tables": tabs, "_view": "compact（T-054）"}


def prefill_a1(evidence: dict[str, dict], as_of: str) -> list[dict]:
    """必须覆盖事实的声明草稿；只用已经给模型看的记录。"""
    inc = [r for k, r in evidence.items() if k.startswith("tushare:income:")]
    drafts = []
    if inc:
        last = {}
        for r in sorted(inc, key=lambda r: (str(r.get("end_date")), str(r.get("ann_date")))):
            last[str(r.get("end_date"))] = r                          # 每期取最新的一版
        latest = max(last)
        prev = str(int(latest[:4]) - 1) + latest[4:]
        for end, label in ((latest, "最新报告期"), (prev, "上年同期")):
            r = last.get(end)
            if not r:
                continue
            for f, name in (("revenue", "营业收入"), ("n_income_attr_p", "归母净利润")):
                if r.get(f) is None:
                    continue
                disp = (r.get("_display") or {}).get(f, "")
                drafts.append({"claim_type": "reported_fact", "text": f"{end[:4]}年{end[4:6]}月{end[6:]}日止（{label}）{name}{disp}",
                               "period": end, "metric": f, "value": r[f], "unit": "元",
                               "evidence": [{"evidence_id": r["evidence_id"], "field": f}]})
    lo = (pd.Timestamp(as_of) - pd.Timedelta(days=120)).strftime("%Y%m%d")
    fc = sorted((r for k, r in evidence.items() if k.startswith("tushare:forecast:") and lo < str(r.get("ann_date")) <= as_of),
                key=lambda r: str(r.get("ann_date")))
    if fc and fc[-1].get("type"):
        r = fc[-1]
        drafts.append({"claim_type": "reported_fact", "text": f"{r['ann_date']}披露的{r.get('end_date')}期业绩预告类型为{r['type']}",
                       "period": str(r.get("end_date")), "metric": "type",
                       "evidence": [{"evidence_id": r["evidence_id"], "field": "type"}]})
    au = sorted((r for k, r in evidence.items() if k.startswith("tushare:fina_audit:") and str(r.get("end_date")).endswith("1231")),
                key=lambda r: str(r.get("end_date")))
    if au and au[-1].get("audit_result"):
        r = au[-1]
        drafts.append({"claim_type": "reported_fact", "text": f"{str(r['end_date'])[:4]}年年报审计意见为{r['audit_result']}",
                       "period": str(r["end_date"]), "metric": "audit_result",
                       "evidence": [{"evidence_id": r["evidence_id"], "field": "audit_result"}]})
    return drafts


PREFILL_NOTE = ("\n代码已经从上面的记录里取出必须覆盖的事实，写成了声明草稿（数值和证据编号已核对）。"
                "请直接放进 claims（补上 local_id），再按需要补充其他事实、计算和推断：\n")


def repair_task(task: str, output: dict, checks: list[dict]) -> str | None:
    """校验没通过的声明及原因；全部通过时返回 None。"""
    bad = [c for c in checks if c["status"] not in ("verified", "minor_mismatch")]
    if not bad:
        return None
    by = {c["local_id"]: c for c in output.get("claims", [])}
    lines = [f"- {c['local_id']}（{c['status']}）：{'；'.join(c['notes'])[:200]}；原声明：{by.get(c['local_id'], {}).get('text', '')[:80]}"
             for c in bad]
    return (task + "\n\n你上一次的输出：\n" + json.dumps(output, ensure_ascii=False)[:12000]
            + "\n\n代码校验发现下面这些声明有问题：\n" + "\n".join(lines)
            + "\n请修正这些声明：数值必须等于记录里该字段的原值，单位按 _units；计算声明写清 formula 和 inputs；"
              "确实无法核对的删掉。其余声明保持不变，输出完整的 JSON 对象。")


_AMT = re.compile(r"(-?\d+(?:\.\d+)?)\s*(亿元|万元|亿|万|元)")
_SCALE = {"亿元": 1e8, "亿": 1e8, "万元": 1e4, "万": 1e4, "元": 1.0}
_RU = {"元": 1.0, "万元": 1e4, "亿元": 1e8}
RENDER_NOTE = ("\n营业收入、归母净利润（最新期与上年同期）、它们的同比增速、最近的业绩预告类型和年报审计意见，"
               "代码会直接写进结果（编号 k1、k2……，可以在 inputs 里引用）。你不用再写这些，也不要自己算同比增速；"
               "请把篇幅用在其他事实（现金流与利润的关系、负债与流动性、预告与实际的差异、质押）和推断上。"
               "computed 的数值由代码按 formula 计算，每条 computed 只算一个数。\n")


def _fmt_amount(v: float) -> str:
    return f"{v / 1e8:.2f}亿元" if abs(v) >= 1e8 else (f"{v / 1e4:.2f}万元" if abs(v) >= 1e4 else f"{v:.2f}元")


def _fix_amounts(text: str, cands: list[float]) -> tuple[str, int]:
    """正文里的金额写法由代码按所引记录重写：与某个记录值相差 0.5% 以内，或正好差 10 的整数次方（单位换算错）。"""
    n = 0

    def sub(m):
        nonlocal n
        x = float(m.group(1)) * _SCALE[m.group(2)]
        for cv in cands:
            if not cv or not x or (x > 0) != (cv > 0):
                continue
            r = abs(x / cv)
            k = round(math.log10(r))
            if abs(k) <= 4 and abs(r / 10 ** k - 1) <= 5e-3:
                new = _fmt_amount(cv)
                if new != m.group(0).replace(" ", ""):
                    n += 1
                return new
        return m.group(0)
    return _AMT.sub(sub, text), n


def code_facts_a1(evidence: dict[str, dict], as_of: str) -> list[dict]:
    """必须覆盖的事实 + 收入、归母净利润的同比增速，全部由代码写成声明（编号 k1、k2……）。"""
    facts = [d | {"local_id": f"k{i + 1}"} for i, d in enumerate(prefill_a1(evidence, as_of))]
    by = {(d["metric"], d["period"]): d for d in facts if d.get("value") is not None}
    for f, name in (("revenue", "营业收入"), ("n_income_attr_p", "归母净利润")):
        cur = [d for (m, p), d in by.items() if m == f]
        if len(cur) < 2:
            continue
        a, b = sorted(cur, key=lambda d: d["period"], reverse=True)[:2]
        if int(a["period"][:4]) - int(b["period"][:4]) != 1 or not b["value"]:
            continue
        g = (a["value"] - b["value"]) / abs(b["value"]) * 100
        facts.append({"local_id": f"k{len(facts) + 1}", "claim_type": "computed",
                      "text": f"{a['period'][:4]}年{a['period'][4:6]}月{a['period'][6:]}日止{name}同比变化{g:.2f}%",
                      "period": a["period"], "metric": f"{f}_yoy", "value": round(g, 4), "unit": "%",
                      "formula": f"({a['local_id']}-{b['local_id']})/abs({b['local_id']})*100",
                      "inputs": [a["local_id"], b["local_id"]], "evidence": []})
    return facts


def render_a1(output: dict, evidence: dict[str, dict], as_of: str) -> tuple[dict, dict]:
    """S5（T-054）：数字归代码。
    1. 模型的 reported_fact：value 改成所引记录的原值，正文里的金额按记录重写（只改相差 0.5% 以内或正好差 10^k 的写法）；
    2. 必须覆盖的事实和同比增速由代码写入（k1…），模型写了同一个事实（同一证据编号 + 字段）就保留模型那条；
    3. 模型自己算的同比（单位 %、两个输入正好是代码那两条事实）换成代码的结果，local_id 不变；
    其余计算、推断原样保留，照常校验。返回 (新输出, 处理记录)。"""
    out = json.loads(json.dumps(output))
    claims = out.get("claims", [])
    info = {"value_fixed": 0, "text_fixed": 0, "code_added": [], "yoy_replaced": [], "code_owned": []}
    for c in claims:
        if c.get("claim_type") != "reported_fact":
            continue
        cands = []
        refs = c.get("evidence") or []
        for r in refs:
            rec = evidence.get(r.get("evidence_id")) or {}
            raw, f = rec.get(r.get("field")) if r.get("field") else None, r.get("field")
            if not isinstance(raw, (int, float)) or isinstance(raw, bool):
                continue
            ru = (rec.get("_units") or {}).get(f, "元")
            if ru in _RU:
                cands.append(float(raw) * _RU[ru])
            if r.get("value") is not None:
                if r["value"] != raw or r.get("unit") != ru:
                    info["value_fixed"] += 1
                r["value"], r["unit"] = raw, ru
            elif len(refs) == 1:
                if c.get("value") != raw or c.get("unit") != ru:
                    info["value_fixed"] += 1
                c["value"], c["unit"] = raw, ru
        if cands:
            c["text"], n = _fix_amounts(c.get("text", ""), cands)
            info["text_fixed"] += n
    # 已有的事实：(证据编号, 字段) → 模型声明的 local_id
    have = {}
    for c in claims:
        if c.get("claim_type") == "reported_fact" and len(c.get("evidence") or []) == 1:
            r = c["evidence"][0]
            have.setdefault((r.get("evidence_id"), r.get("field")), c["local_id"])
    used = {c["local_id"] for c in claims}
    alias = {}
    for k in code_facts_a1(evidence, as_of):
        k, orig = dict(k), k["local_id"]
        while k["local_id"] in used:
            k["local_id"] += "c"
        if k["claim_type"] == "reported_fact":
            key = (k["evidence"][0]["evidence_id"], k["evidence"][0]["field"])
            if key in have:
                alias[orig] = have[key]
                continue
            alias[orig] = k["local_id"]
        else:
            ins = [alias.get(i) for i in k["inputs"]]
            if None in ins:
                continue
            k["inputs"] = ins
            k["formula"] = f"({ins[0]}-{ins[1]})/abs({ins[1]})*100"
            twin = next((c for c in claims if c.get("claim_type") == "computed" and c.get("unit") == "%"
                         and sorted(c.get("inputs") or []) == sorted(ins)), None)
            if twin is not None:
                twin.update({x: k[x] for x in ("text", "value", "unit", "formula", "inputs")})
                info["yoy_replaced"].append(twin["local_id"])
                info["code_owned"].append(twin["local_id"])
                continue
        used.add(k["local_id"])
        k.pop("period", None)
        k.pop("metric", None)
        claims.append(k)
        info["code_added"].append(k["local_id"])
        info["code_owned"].append(k["local_id"])
    _calculate(claims, set(info["code_owned"]), info)
    out["claims"] = claims
    return out, info


_PCT = re.compile(r"(增长|增加|上升|提升|下降|减少|降低|下滑)?(约|了)?\s*(-?\d+(?:\.\d+)?)\s*%")
_UP, _DOWN = ("增长", "增加", "上升", "提升"), ("下降", "减少", "降低", "下滑")


def _period(c: dict) -> tuple[str, str] | None:
    """单证据 reported_fact 的 (字段, 报告期)，报告期取自证据编号 tushare:<表>:<代码>:<报告期>:<公告日>。"""
    refs = c.get("evidence") or []
    if c.get("claim_type") != "reported_fact" or len(refs) != 1:
        return None
    parts = str(refs[0].get("evidence_id", "")).split(":")
    return (refs[0].get("field"), parts[3]) if len(parts) >= 5 and refs[0].get("field") else None


def _calculate(claims: list[dict], skip: set[str], info: dict) -> None:
    """计算器：模型写公式，代码求值并把结果写回正文（只在正文里恰好有一个同类数字时改写；有几个就不动，照常校验）。
    同比：单位 %、两个输入是同一字段相隔一年的两期 → 公式统一成 (新-旧)/abs(旧)*100，防止把比值当增速。"""
    from quantmind.line_a.harness.verifier import UNIT, _clean_formula, _eval
    by = {c["local_id"]: c for c in claims}
    info.setdefault("calc_fixed", [])
    for c in claims:
        if c.get("claim_type") != "computed" or c["local_id"] in skip or not c.get("formula"):
            continue
        ins = [by.get(i) for i in c.get("inputs") or []]
        if not ins or None in ins or any(x.get("value") is None for x in ins):
            continue
        unit = c.get("unit") or ""
        if unit in ("", "百分比", "pct") and len(_PCT.findall(c.get("text", ""))) == 1 and not _AMT.search(c.get("text", "")):
            unit = c["unit"] = "%"                        # 没写单位，但正文里只有一个百分数
        if unit == "%" and len(ins) == 2:
            p = [_period(x) for x in ins]
            if None not in p and p[0][0] == p[1][0] and p[0][1][4:] == p[1][1][4:] and \
                    abs(int(p[0][1][:4]) - int(p[1][1][:4])) == 1:
                new, old = (ins[0], ins[1]) if p[0][1] > p[1][1] else (ins[1], ins[0])
                c["formula"] = f"({new['local_id']}-{old['local_id']})/abs({old['local_id']})*100"
        try:
            val = _eval(_clean_formula(c["formula"]), {x["local_id"]: float(x["value"]) for x in ins})
        except (ValueError, ZeroDivisionError, SyntaxError, KeyError, TypeError):
            continue
        text = c.get("text", "")
        if unit == "%":
            toks = list(_PCT.finditer(text))
            if len(toks) != 1:
                continue
            m = toks[0]
            if m.group(1):
                word = (_UP if val >= 0 else _DOWN)[0] if m.group(1) in _UP + _DOWN else m.group(1)
                if (m.group(1) in _UP) == (val >= 0):
                    word = m.group(1)
                rep = f"{word}{m.group(2) or ''}{abs(val):.2f}%"
            else:
                rep = f"{val:.2f}%"
        elif unit in UNIT and unit in ("元", "万元", "亿元"):
            toks = list(_AMT.finditer(text))
            if len(toks) != 1:
                continue
            m, rep = toks[0], _fmt_amount(val * UNIT[unit])
        else:
            continue
        if c.get("value") is None or abs(float(c["value"]) - val) > 1e-9 * max(1.0, abs(val)) or rep != m.group(0):
            info["calc_fixed"].append(c["local_id"])
        c["text"] = text[:m.start()] + rep + text[m.end():]
        c["value"] = round(val, 6)


def score_checks(checks: list[dict]) -> tuple[int, int]:
    """(有问题的条数, 已核实的条数)：修复后用来判断是否更好。"""
    bad = sum(c["status"] in ("mismatch", "missing_evidence", "unverifiable") for c in checks)
    return bad, sum(c["status"] == "verified" for c in checks)
