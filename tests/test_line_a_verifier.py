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


def test_dates_and_ids_are_not_unsourced_numbers():
    out = _out([{"local_id": "c1", "claim_type": "reported_fact",
                 "text": "2023年11月21日披露方案（规则 T22/S08），事件链 600519.SH:dividend:1，回购 3000 万元",
                 "evidence": [{"evidence_id": "cninfo:ann:1"}]}])
    notes = verify(out, EV)[0].notes
    assert notes == ["unsourced_number 3000"]


def test_multi_field_facts_units_rounding_and_formula_cleanup():
    ev = {"tushare:income:x:1": {"evidence_id": "tushare:income:x:1", "revenue": 147693604994.14,
                                 "total_revenue": 150560330316.45},
          "tushare:forecast:x:1": {"evidence_id": "tushare:forecast:x:1", "net_profit_min": 7350000.0,
                                   "_units": {"net_profit_min": "万元"}}}
    out = _out([
        {"local_id": "c1", "claim_type": "reported_fact", "text": "营业收入 1476.94 亿元、营业总收入 1505.60 亿元",
         "value": 1476.94, "unit": "亿元",
         "evidence": [{"evidence_id": "tushare:income:x:1", "field": "revenue"},
                      {"evidence_id": "tushare:income:x:1", "field": "total_revenue", "value": 1505.6, "unit": "亿元"}]},
        {"local_id": "c2", "claim_type": "reported_fact", "text": "预告净利润下限 735 亿元", "value": 735, "unit": "亿元",
         "evidence": [{"evidence_id": "tushare:forecast:x:1", "field": "net_profit_min"}]},
        {"local_id": "c3", "claim_type": "reported_fact", "text": "营业收入 1200 亿元", "value": 1200, "unit": "亿元",
         "evidence": [{"evidence_id": "tushare:income:x:1", "field": "revenue"}]},
        {"local_id": "c4", "claim_type": "computed", "text": "两者差约 2.0%", "value": 1.9, "unit": "%",
         "formula": "(c1-c2*1)/c1*100，即两者的相对差", "inputs": ["c1", "c2"]},
    ])
    st = {c.local_id: c.status for c in verify(out, ev)}
    assert st["c1"] == "verified" and st["c2"] == "verified" and st["c3"] == "mismatch"
    assert st["c4"] == "mismatch"                     # 公式能解析（去掉了解释），但 (1476.94-735)/1476.94*100 = 50.2 ≠ 1.9
