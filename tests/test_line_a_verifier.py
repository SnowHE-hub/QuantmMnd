"""独立校验器 F7：事实与证据、单位换算、计算重算、依据链、正文数字溯源。"""
from __future__ import annotations

from quantmind.line_a.harness.roles import RoleOutput
from quantmind.line_a.harness.verifier import verify

EV = {"tushare:income:600000.SH:20260630:20260828": {"evidence_id": "tushare:income:600000.SH:20260630:20260828",
                                                     "n_income_attr_p": 2.5e9, "end_date": "20260630"},
      "tushare:income:600000.SH:20250630:20250829": {"evidence_id": "tushare:income:600000.SH:20250630:20250829",
                                                     "n_income_attr_p": 2.0e9, "end_date": "20250630"},
      "cninfo:ann:1": {"evidence_id": "cninfo:ann:1", "title": "关于终止回购公司股份方案的公告"}}


def _out(claims):
    return RoleOutput.model_validate({"ticker": "600000.SH", "as_of": "20261020", "claims": claims}).model_dump()


def test_facts_units_and_computed():
    out = _out([
        {"local_id": "c1", "claim_type": "reported_fact", "text": "2026 年上半年归母净利润 25 亿元", "value": 25,
         "unit": "亿元", "evidence": [{"evidence_id": "tushare:income:600000.SH:20260630:20260828",
                                      "field": "n_income_attr_p"}]},
        {"local_id": "c2", "claim_type": "reported_fact", "text": "2025 年上半年归母净利润 20 亿元", "value": 20,
         "unit": "亿元", "evidence": [{"evidence_id": "tushare:income:600000.SH:20250630:20250829",
                                      "field": "n_income_attr_p"}]},
        {"local_id": "c3", "claim_type": "computed", "text": "同比增长 25%", "value": 25, "unit": "%",
         "formula": "(c1-c2)/abs(c2)*100", "inputs": ["c1", "c2"]},
        {"local_id": "c4", "claim_type": "computed", "text": "同比增长 30%", "value": 30, "unit": "%",
         "formula": "(c1-c2)/abs(c2)*100", "inputs": ["c1", "c2"]},
        {"local_id": "c5", "claim_type": "inference", "text": "利润增长较快", "inputs": ["c3"]},
        {"local_id": "c6", "claim_type": "inference", "text": "有疑问", "inputs": ["c4"]},
    ])
    st = {c.local_id: c.status for c in verify(out, EV)}
    assert st == {"c1": "verified", "c2": "verified", "c3": "verified", "c4": "mismatch", "c5": "verified",
                  "c6": "weak_basis"}


def test_missing_evidence_wrong_value_quote_and_unsourced_numbers():
    out = _out([
        {"local_id": "c1", "claim_type": "reported_fact", "text": "净利润 26 亿元", "value": 26, "unit": "亿元",
         "evidence": [{"evidence_id": "tushare:income:600000.SH:20260630:20260828", "field": "n_income_attr_p"}]},
        {"local_id": "c2", "claim_type": "reported_fact", "text": "有一笔担保",
         "evidence": [{"evidence_id": "tushare:guarantee:x"}]},
        {"local_id": "c3", "claim_type": "reported_fact", "text": "公司终止了回购方案",
         "evidence": [{"evidence_id": "cninfo:ann:1", "quote": "终止回购公司股份方案"}]},
        {"local_id": "c4", "claim_type": "reported_fact", "text": "公司完成了回购",
         "evidence": [{"evidence_id": "cninfo:ann:1", "quote": "回购完成"}]},
        {"local_id": "c5", "claim_type": "reported_fact", "text": "回购金额 3000 万元",
         "evidence": [{"evidence_id": "cninfo:ann:1"}]},
        {"local_id": "c6", "claim_type": "computed", "text": "x", "value": 1, "formula": "__import__('os')",
         "inputs": []},
    ])
    ch = {c.local_id: c for c in verify(out, EV)}
    assert ch["c1"].status == "mismatch" and ch["c2"].status == "missing_evidence"
    assert ch["c3"].status == "verified" and ch["c4"].status == "mismatch"
    assert any("unsourced_number 3000" in n for n in ch["c5"].notes)
    assert ch["c6"].status in ("unverifiable", "mismatch")
