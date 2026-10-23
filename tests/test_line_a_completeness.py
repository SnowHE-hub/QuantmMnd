"""A1 完整性检查（T-051 harness v3）：缺失项清单与工具循环里的一次补充。"""
from __future__ import annotations

import json

from quantmind.line_a.harness.completeness import a1_missing
from quantmind.line_a.harness.llm import LLM
from quantmind.line_a.harness.loop import RoleSpec, run_role
from quantmind.line_a.harness.roles import RoleOutput

T = "600000.SH"
EV = {f"tushare:income:{T}:20260630:20260828": {"revenue": 1e10, "n_income_attr_p": 2e9},
      f"tushare:income:{T}:20250630:20250829": {"revenue": 9e9, "n_income_attr_p": 1.5e9},
      f"tushare:forecast:{T}:20260930:20261010": {"type": "预增"},
      f"tushare:fina_audit:{T}:20251231:20260420": {"audit_result": "标准无保留意见"}}


def _claim(i, eid, field):
    return {"local_id": f"c{i}", "claim_type": "reported_fact", "text": "x", "value": 1.0,
            "evidence": [{"evidence_id": eid, "field": field}]}


def _full():
    ks = list(EV)
    return [_claim(1, ks[0], "revenue"), _claim(2, ks[0], "n_income_attr_p"), _claim(3, ks[1], "revenue"),
            _claim(4, ks[1], "n_income_attr_p"), _claim(5, ks[2], "type"), _claim(6, ks[3], "audit_result")]


def test_a1_missing_lists_uncovered_items():
    assert a1_missing({"claims": _full()}, EV, "20261020") == []
    miss = a1_missing({"claims": _full()[:2]}, EV, "20261020")
    assert len(miss) == 4 and any("上年同期" in m and "营业收入" in m for m in miss)
    assert any("业绩预告" in m for m in miss) and any("审计意见" in m for m in miss)
    assert not any("业绩预告" in m for m in a1_missing({"claims": _full()[:2]}, EV, "20270301"))   # 超过 120 天不要求


def _llm(replies, seen):
    def transport(url, headers, body):
        seen.append(list(body["messages"]))
        return 200, {"model": "m", "choices": [{"finish_reason": "stop", "message": {"content": replies.pop(0)}}],
                     "usage": {"prompt_tokens": 10, "completion_tokens": 10}}
    return LLM(transport=transport, budget_rmb=1.0)


def _spec():
    return RoleSpec("A1", "s", [], RoleOutput, max_steps=4, thinking=None, max_tool_calls=0,
                    defaults={"ticker": T, "as_of": "20261020"},
                    completeness=lambda out, ev: a1_missing(out, ev, "20261020"))


def test_completeness_follow_up_once():
    first = json.dumps({"ticker": T, "as_of": "20261020", "claims": _full()[:2]})
    second = json.dumps({"ticker": T, "as_of": "20261020", "claims": _full()})
    seen: list = []
    rec = run_role(_llm([first, second], seen), _spec(), "task", evidence=EV)
    assert rec.status == "ok" and len(rec.output["claims"]) == 6 and len(seen) == 2
    assert "必须覆盖" in seen[1][-1]["content"] and rec.steps[0]["completeness_missing"]
    assert set(EV) <= set(rec.evidence)


def test_completeness_bad_follow_up_keeps_first_output():
    first = json.dumps({"ticker": T, "as_of": "20261020", "claims": _full()[:2]})
    seen: list = []
    rec = run_role(_llm([first, "not json at all"], seen), _spec(), "task", evidence=EV)
    assert rec.status == "ok" and len(rec.output["claims"]) == 2 and len(seen) == 2


def test_fit_tool_body_structural_truncation():
    from quantmind.line_a.harness.loop import collect_shown, fit_tool_body
    out = {"tables": {"a": [{"evidence_id": f"a{i}", "x": "y" * 50} for i in range(10)],
                      "income": [{"evidence_id": f"i{i}", "x": "y" * 50} for i in range(3)]}}
    body = fit_tool_body(out, limit=600)
    o = json.loads(body)                                  # 仍是合法 JSON
    assert len(body) <= 600 and "_truncated" in o and len(o["tables"]["income"]) == 3
    assert [r["evidence_id"] for r in o["tables"]["a"]] == [f"a{i}" for i in range(len(o["tables"]["a"]))]
    ev: dict = {}
    collect_shown(body, out, ev)
    assert "i2" in ev and "a9" not in ev
