"""quantmind.line_a.harness.slices — 角色运行与案例编排（P2 垂直切片 → P3 多角色；工具循环 + 独立校验 + 产物落盘）.

用法（WSL）：
  python -m quantmind.line_a.harness.slices --lake data/lake --as-of 20240410 --ticker 600519.SH --role A1 --out …
  python -m quantmind.line_a.harness.slices --lake data/lake --as-of 20240410 --ticker 600519.SH --case A1,A2,A3,A4 --out …
案例模式：先跑 A1–A4（互相独立），再把它们的声明与实际取得的证据交给 A6 审查，最后由代码校验器逐条核对。
产物：<out>/<as_of>/<ticker>/<role 或 case>_<请求哈希前 12 位>.json，包含：
- 角色输出和逐条校验；
- 过程记录（每步请求哈希、工具名、参数与结果哈希）；
- 证据记录与费用。
只发公开数据（财报、公告、行情与资金数据），不发送任何账户信息。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from quantmind.line_a.harness.completeness import a1_missing
from quantmind.line_a.harness.llm import LLM, canonical
from quantmind.line_a.harness.loop import RoleSpec, Tool, run_role
from quantmind.line_a.harness.roles import (
    A1_SYSTEM,
    A2_SYSTEM,
    A3_SYSTEM,
    A4_SYSTEM,
    A5_SYSTEM,
    A6_SYSTEM,
    ReviewOutput,
    RoleOutput,
)
from quantmind.line_a.harness.tools_more import ResearchTools
from quantmind.line_a.harness.verifier import verify

_T = {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}
_LB = {"lookback_days": {"type": "integer", "minimum": 7, "maximum": 730}}
_ANN = {"ann_id": {"type": "string", "description": "公告编号（来自 list_announcements 的 cninfo:ann:<编号>）"}}


def role_spec(role: str, dt: ResearchTools, subject: str, *, thinking: str | None = "high") -> RoleSpec:
    def _tool(name: str, desc: str, fn, extra: dict | None = None, required: tuple = ()) -> Tool:
        params = json.loads(json.dumps(_T))
        params["properties"].update(extra or {})
        params["required"] = ["ticker", *required]

        def guarded(ticker: str, **kw):          # 只允许查本次研究对象，防止被文本诱导去查别的股票
            t = str(ticker).strip().upper()
            if t == subject.split(".")[0]:          # 不带交易所后缀的 6 位代码（本机 9B 常这样写）：补成研究对象本身
                t = subject
            if t != subject:
                return {"error": f"this run is restricted to {subject}"}
            return fn(subject, **kw)
        return Tool(name, desc, params, guarded)

    ann = [_tool("list_announcements", "按时点列出公告标题、发布时间、规则分类（类型 / 阶段 / 重要性）",
                 dt.list_announcements, {**_LB, "limit": {"type": "integer", "minimum": 10, "maximum": 200},
                                         "event_types": {"type": "array", "items": {"type": "string"}},
                                         "min_importance": {"type": "string", "enum": ["routine", "watch", "major"]}}),
           _tool("event_chains", "同类公告串成的事件链与当前状态（计划 / 进展 / 实施 / 完成 / 终止 / 撤回 / 解除）",
                 dt.event_chains, _LB)]
    docs = [_tool("read_document", "取得并解析一份公告的 PDF：页数、需 OCR 的页、表格数、前 12 段开头", dt.read_document,
                  _ANN, ("ann_id",)),
            _tool("locate_section", "在一份公告里按问题定位最相关的段落（BM25），返回段落证据编号和原文", dt.locate_section,
                  {**_ANN, "query": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1, "maximum": 8}},
                  ("ann_id", "query"))]
    big = {"max_steps": 8, "thinking": thinking, "max_tokens": 32000, "max_tool_calls": 8,
           "defaults": {"ticker": subject, "as_of": dt.as_of}}       # 编排器已知的字段，模型漏写时补上（T-048）
    if role == "A1":
        tools = [_tool("financial_snapshot", "按时点取最近若干期利润表、资产负债表、现金流量表与财务指标（金额单位：元）",
                       dt.financial_snapshot, {"periods": {"type": "integer", "minimum": 1, "maximum": 12}}),
                 _tool("forecasts_and_express", "按时点取业绩预告与业绩快报", dt.forecasts_and_express),
                 _tool("disclosure_schedule", "财报预约披露日与实际披露日（as_of 之后的实际日期不可见）",
                       dt.disclosure_schedule),
                 _tool("audit_and_pledge", "审计意见与股权质押统计", dt.audit_and_pledge)]
        return RoleSpec("A1", A1_SYSTEM, tools, RoleOutput, **big)
    if role == "A2":
        return RoleSpec("A2", A2_SYSTEM, ann + docs, RoleOutput, **big)
    if role == "A3":
        tools = [_tool("holder_trade_records", "股东与高管增减持记录（方向、比例、持股变化）", dt.holder_trade_records, _LB),
                 _tool("share_float_schedule", "已公告的限售股解禁安排（过去 90 天到未来 180 天）", dt.share_float_schedule,
                       {"ahead_days": {"type": "integer", "minimum": 30, "maximum": 365}}),
                 _tool("repurchase_records", "回购的各阶段记录（预案、实施、完成、停止）", dt.repurchase_records, _LB),
                 _tool("audit_and_pledge", "审计意见与股权质押统计", dt.audit_and_pledge),
                 _tool("top_float_holders", "最近一期已公告的十大流通股东，以及与上一期相比新进、退出的股东", dt.top_float_holders),
                 _tool("current_managers", "as_of 当天在任的董监高（姓名、职务、任职开始日）", dt.current_managers),
                 *ann, docs[1]]
        return RoleSpec("A3", A3_SYSTEM, tools, RoleOutput, **big)
    if role == "A4":
        tools = [_tool("analyst_revisions", "卖方盈利预测：同机构同年度的前后修正、各年度中位数与覆盖说明",
                       dt.analyst_revisions, _LB),
                 _tool("flow_snapshot", "资金流向（大单与特大单净额）与融资余额的 5 / 20 日变化", dt.flow_snapshot),
                 _tool("top_list_records", "龙虎榜上榜原因与机构净买入", dt.top_list_records, _LB)]
        return RoleSpec("A4", A4_SYSTEM, tools, RoleOutput, **big)
    if role == "A5":
        tools = [_tool("business_profile", "公司的行业、地区、板块（业务映射目前只到行业分类）", dt.business_profile),
                 _tool("industry_pack", "同行业共享的证据包：窗口内的行业新闻与行业成分股相对全市场的表现", dt.industry_pack,
                       {"days": {"type": "integer", "minimum": 1, "maximum": 30}}),
                 _tool("policy_search", "在窗口内的新闻里检索政策与监管信息", dt.policy_search,
                       {"query": {"type": "string"}, "days": {"type": "integer", "minimum": 1, "maximum": 30}},
                       ("query",))]
        return RoleSpec("A5", A5_SYSTEM, tools, RoleOutput, **big)
    if role == "A6":
        tools = [_tool("read_evidence", "一次读回多条（最多 20 条）上游角色引用的原始证据记录", dt.read_evidence,
                       {"evidence_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 20}},
                       ("evidence_ids",)),
                 _tool("search_counterevidence", "在公告标题和已解析的公告段落里检索可能的反证（更正、撤回、澄清、终止）",
                       dt.search_counterevidence, {"query": {"type": "string"}, **_LB}, ("query",)), *ann]
        return RoleSpec("A6", A6_SYSTEM, tools, ReviewOutput, **{**big, "max_tool_calls": 12, "max_steps": 10})
    raise ValueError(role)


def _task(ticker: str, as_of: str, extra: str = "") -> str:
    return (f"研究对象：{ticker}；as_of = {as_of}（只能使用这一天及之前已发布的信息）。请先调用工具取数，再输出 JSON。"
            + extra)


# X8（T-041，2026-10-23）：A1 的输入是固定的四个结构化工具，单次调用不比工具循环差
# （留出集召回 96.3% vs 95.4%，已核实 97.1% vs 92.5%，有问题声明 0.5% vs 2.1%，费用相近 ¥1.70 vs ¥1.64），
# 所以 A1 默认单次调用；其他角色仍用工具循环。
SINGLE_CALL_ROLES = frozenset({"A1"})


def _single_call_task(spec: RoleSpec, ticker: str, as_of: str, extra: str) -> tuple[str, dict]:
    """代码先用默认参数调用角色的全部工具，结果作为数据一次性放进消息；返回 (任务文本, 证据)。"""
    from quantmind.line_a.harness.loop import DATA_NOTICE, collect_shown, fit_tool_body
    evidence: dict = {}
    blocks = []
    for t in spec.tools:
        try:
            out = t.fn(ticker=ticker)
        except Exception as e:  # noqa: BLE001  工具错误照样作为数据给模型
            out = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        body = fit_tool_body(out)
        collect_shown(body, out, evidence)
        blocks.append(f"[{t.name}] " + DATA_NOTICE + body)
    task = (f"研究对象：{ticker}；as_of = {as_of}（只能使用这一天及之前已发布的信息）。下面已经给出全部工具的结果，"
            "不能再调用工具，请直接输出 JSON。" + extra + "\n\n" + "\n\n".join(blocks))
    return task, evidence


V3_NOTE = ("\n金额字段旁的 _display 是代码换算好的可读写法（例如 54.33亿元），正文写金额时直接照抄 _display，不要自己换算单位；"
           "value 和 unit 仍按原始字段填写。")


def run_slice(llm: LLM, dt: ResearchTools, ticker: str, role: str, out_root: Path | None = None,
              extra: str = "", single: bool | None = None, v3: bool = False) -> dict:
    """v3：T-051 harness v3（可读金额 + A1 完整性检查）；判定之前默认关闭。"""
    if v3:
        dt.display_amounts = True
        extra = V3_NOTE + extra
    spec = role_spec(role, dt, ticker)
    if v3 and role == "A1":
        spec.completeness = lambda out, ev: a1_missing(out, ev, dt.as_of)
    single = role in SINGLE_CALL_ROLES if single is None else single
    if single:
        task, pre = _single_call_task(spec, ticker, dt.as_of, extra)
        spec1 = RoleSpec(spec.name, spec.system, [], spec.output_model, max_steps=4, thinking=spec.thinking,
                         max_tokens=spec.max_tokens, max_tool_calls=0, defaults=spec.defaults,
                         completeness=spec.completeness)
        rec = run_role(llm, spec1, task, meta={"ticker": ticker, "as_of": dt.as_of, "mode": "single"}, evidence=pre)
    else:
        rec = run_role(llm, spec, _task(ticker, dt.as_of, extra), meta={"ticker": ticker, "as_of": dt.as_of})
    rec.evidence.update(getattr(dt, "side_evidence", {}))     # 公告全文与逐段记录（覆盖工具里截短的版本）
    checks =[asdict(c) for c in verify(rec.output, rec.evidence)] if rec.output and "claims" in rec.output else []
    res = {"ticker": ticker, "as_of": dt.as_of, "role": role, "status": rec.status, "output": rec.output,
           "checks": checks, "steps": rec.steps, "errors": rec.errors, "cost_rmb": rec.cost_rmb,
           "evidence": rec.evidence, "model": llm.model}
    if out_root is not None:
        key = rec.steps[0]["request_hash"][:12] if rec.steps else "none"
        p = Path(out_root) / dt.as_of / ticker / f"{role}_{key}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(canonical(res), encoding="utf-8")
        res["path"] = str(p)
    return res


def review_checks(review: dict, upstream: dict[str, dict], registry: dict) -> list[dict]:
    """A6 的审查结果也要机械核对：引用的声明必须存在；引用的证据必须是本案例里实际取得过的记录。"""
    out = []
    for j in review.get("judgments", []):
        role, _, cid = j["claim_ref"].partition(":")
        ok_ref = role in upstream and any(c["local_id"] == cid for c in (upstream[role].get("output") or {}).get("claims", []))
        missing = [e["evidence_id"] for e in j.get("evidence", []) if e["evidence_id"] not in registry]
        out.append({"claim_ref": j["claim_ref"], "verdict": j["verdict"], "claim_exists": ok_ref,
                    "missing_evidence": missing,
                    "status": "ok" if ok_ref and not missing else ("unknown_claim" if not ok_ref else "missing_evidence")})
    return out


def run_case(llm: LLM, lake: Path, as_of: str, ticker: str, roles: list[str], out_root: Path | None = None,
             reuse: dict[str, dict] | None = None) -> dict:
    """reuse：{角色: 以前某天 run_slice 的结果}，这些角色本次不重跑，声明和证据照样交给 A6（T-040 按角色缺口调用）。"""
    up: dict[str, dict] = {}
    registry: dict = {}
    for r, prev in (reuse or {}).items():
        up[r] = prev
        registry.update(prev.get("evidence", {}))
    for r in roles:
        dt = ResearchTools(lake, as_of)
        up[r] = run_slice(llm, dt, ticker, r, out_root)
        registry.update(up[r]["evidence"])
    brief = {r: [{k: c.get(k) for k in ("local_id", "claim_type", "text", "value", "unit", "period")} |
                 {"evidence_ids": [e["evidence_id"] for e in c.get("evidence", [])]}
                 for c in (v.get("output") or {}).get("claims", [])] for r, v in up.items()}
    dt6 = ResearchTools(lake, as_of, registry=registry)
    a6 = run_slice(llm, dt6, ticker, "A6", out_root,
                   extra="\n上游角色的声明（用 <角色>:<local_id> 引用）：\n" + canonical(brief))
    registry.update(a6["evidence"])
    a6["review_checks"] = review_checks(a6["output"] or {}, up, registry) if a6["output"] else []
    case = {"ticker": ticker, "as_of": as_of, "roles": roles,
            "reused": {r: v.get("as_of") for r, v in (reuse or {}).items()},
            "cost_rmb": sum(up[r]["cost_rmb"] for r in roles) + a6["cost_rmb"],
            "upstream": {r: {"status": v["status"], "path": v.get("path"), "as_of": v.get("as_of"),
                             "check_status": _counts(v["checks"])} for r, v in up.items()},
            "a6": {"status": a6["status"], "path": a6.get("path"),
                   "verdicts": _counts([{"status": j["verdict"]} for j in (a6["output"] or {}).get("judgments", [])]),
                   "review_checks": _counts(a6["review_checks"])}}
    if out_root is not None:
        p = Path(out_root) / as_of / ticker / f"case_{a6['steps'][0]['request_hash'][:12] if a6['steps'] else 'none'}.json"
        p.write_text(canonical(case), encoding="utf-8")
        case["path"] = str(p)
    return case


def _counts(rows: list[dict]) -> dict:
    out: dict = {}
    for r in rows:
        out[r["status"]] = out.get(r["status"], 0) + 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake", type=Path, required=True)
    ap.add_argument("--as-of", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--role", choices=["A1", "A2", "A3", "A4", "A5"])
    ap.add_argument("--case", help="逗号分隔的上游角色，例如 A1,A2,A3,A4；之后自动运行 A6")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--budget", type=float, default=2.0)
    a = ap.parse_args()
    llm = LLM(ledger=a.out / "llm_ledger.jsonl", cache_dir=a.out / "_llm_cache", budget_rmb=a.budget)
    if a.case:
        print(json.dumps(run_case(llm, a.lake, a.as_of, a.ticker, a.case.split(","), a.out), ensure_ascii=False))
        return
    res = run_slice(llm, ResearchTools(a.lake, a.as_of), a.ticker, a.role, a.out)
    summ = {k: res[k] for k in ("ticker", "as_of", "role", "status", "cost_rmb")}
    summ["n_claims"] = len((res["output"] or {}).get("claims", []))
    summ["check_status"] = _counts(res["checks"])
    summ["path"] = res.get("path")
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main()
