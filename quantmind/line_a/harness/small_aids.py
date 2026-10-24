"""quantmind.line_a.harness.small_aids — 给本机小模型的代码辅助（T-054）.

思路是"代码多做一点，模型少做一点"，不加长提示词：
- compact：紧凑证据视图。只给 A1 需要的表、期数和字段，证据编号与字段名不变；
- prefill：代码预填事实表。把"必须覆盖"的事实直接写成声明草稿（数值、单位、可读金额、证据编号、字段名都由代码给出）；
- repair：校验器反馈修复。第一次输出后，把没通过校验的声明和原因发回去修一轮；修复后更差就退回。
"""
from __future__ import annotations

import json

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


def score_checks(checks: list[dict]) -> tuple[int, int]:
    """(有问题的条数, 已核实的条数)：修复后用来判断是否更好。"""
    bad = sum(c["status"] in ("mismatch", "missing_evidence", "unverifiable") for c in checks)
    return bad, sum(c["status"] == "verified" for c in checks)
