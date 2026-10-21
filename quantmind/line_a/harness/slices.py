"""quantmind.line_a.harness.slices — P2 的两个垂直切片：A1 财务、A2 公告事件（工具循环 + 独立校验 + 产物落盘）.

用法（WSL）：
  python -m quantmind.line_a.harness.slices --lake data/lake --as-of 20261020 --ticker 600000.SH --role A1 \
      --out data/line_a_v2/research
产物：<out>/<as_of>/<ticker>/<role>_<请求哈希前 12 位>.json，包含：
- 角色输出；
- 逐条校验结果；
- 过程记录（每步的请求哈希、工具名、参数与结果哈希）；
- 本次工具返回的证据记录、费用。
只发公开数据（财报、公告标题），不发送任何账户信息。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from quantmind.line_a.harness.llm import LLM, canonical
from quantmind.line_a.harness.loop import RoleSpec, Tool, run_role
from quantmind.line_a.harness.roles import A1_SYSTEM, A2_SYSTEM, RoleOutput
from quantmind.line_a.harness.tools_data import DataTools
from quantmind.line_a.harness.verifier import verify

_T = {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}


def role_spec(role: str, dt: DataTools, subject: str, *, thinking: str | None = "high") -> RoleSpec:
    def _tool(name: str, desc: str, fn, extra: dict | None = None) -> Tool:
        params = json.loads(json.dumps(_T))
        params["properties"].update(extra or {})

        def guarded(ticker: str, **kw):          # 只允许查本次研究对象，防止被文本诱导去查别的股票
            if ticker != subject:
                return {"error": f"this run is restricted to {subject}"}
            return fn(ticker, **kw)
        return Tool(name, desc, params, guarded)

    if role == "A1":
        tools = [_tool("financial_snapshot", "按时点取最近若干期利润表、资产负债表、现金流量表与财务指标（金额单位：元）",
                       dt.financial_snapshot, {"periods": {"type": "integer", "minimum": 1, "maximum": 12}}),
                 _tool("forecasts_and_express", "按时点取业绩预告与业绩快报", dt.forecasts_and_express),
                 _tool("disclosure_schedule", "财报预约披露日与实际披露日（as_of 之后的实际日期不可见）",
                       dt.disclosure_schedule),
                 _tool("audit_and_pledge", "审计意见与股权质押统计", dt.audit_and_pledge)]
        return RoleSpec("A1", A1_SYSTEM, tools, RoleOutput, max_steps=8, thinking=thinking, max_tokens=32000)
    if role == "A2":
        tools = [_tool("list_announcements", "按时点列出公告标题、发布时间、规则分类（类型 / 阶段 / 重要性）",
                       dt.list_announcements, {"lookback_days": {"type": "integer", "minimum": 7, "maximum": 730},
                                               "limit": {"type": "integer", "minimum": 10, "maximum": 200},
                                               "event_types": {"type": "array", "items": {"type": "string"}},
                                               "min_importance": {"type": "string",
                                                                  "enum": ["routine", "watch", "major"]}}),
                 _tool("event_chains", "同类公告串成的事件链与当前状态（计划 / 进展 / 实施 / 完成 / 终止 / 撤回 / 解除）",
                       dt.event_chains, {"lookback_days": {"type": "integer", "minimum": 30, "maximum": 730}})]
        return RoleSpec("A2", A2_SYSTEM, tools, RoleOutput, max_steps=8, thinking=thinking, max_tokens=32000)
    raise ValueError(role)


def run_slice(llm: LLM, dt: DataTools, ticker: str, role: str, out_root: Path | None = None) -> dict:
    spec = role_spec(role, dt, ticker)
    task = (f"研究对象：{ticker}；as_of = {dt.as_of}（只能使用这一天及之前已发布的信息）。"
            f"请先调用工具取数，再输出 JSON。")
    rec = run_role(llm, spec, task, meta={"ticker": ticker, "as_of": dt.as_of})
    checks = [asdict(c) for c in verify(rec.output, rec.evidence)] if rec.output else []
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake", type=Path, required=True)
    ap.add_argument("--as-of", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--role", choices=["A1", "A2"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--budget", type=float, default=2.0)
    a = ap.parse_args()
    llm = LLM(ledger=a.out / "llm_ledger.jsonl", cache_dir=a.out / "_llm_cache", budget_rmb=a.budget)
    res = run_slice(llm, DataTools(a.lake, a.as_of), a.ticker, a.role, a.out)
    summ = {k: res[k] for k in ("ticker", "as_of", "role", "status", "cost_rmb")}
    summ["n_claims"] = len((res["output"] or {}).get("claims", []))
    summ["check_status"] = {s: sum(c["status"] == s for c in res["checks"]) for s in {c["status"] for c in res["checks"]}}
    summ["path"] = res.get("path")
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main()
