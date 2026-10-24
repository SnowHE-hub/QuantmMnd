"""T-054 S5：render_a1（数字归代码）的单元测试。"""
from dataclasses import asdict

from quantmind.line_a.harness.small_aids import _fix_amounts, render_a1
from quantmind.line_a.harness.verifier import verify

T = "000001.SZ"
E_CUR, E_PREV = f"tushare:income:{T}:20240331:20240425", f"tushare:income:{T}:20230331:20230428"
EV = {E_CUR: {"evidence_id": E_CUR, "end_date": "20240331", "ann_date": "20240425", "revenue": 578749365.4,
              "n_income_attr_p": -2e7, "_units": {"revenue": "元", "n_income_attr_p": "元"}},
      E_PREV: {"evidence_id": E_PREV, "end_date": "20230331", "ann_date": "20230428", "revenue": 504000000.0,
               "n_income_attr_p": 5e7, "_units": {"revenue": "元", "n_income_attr_p": "元"}}}


def test_fix_amounts_magnitude_and_rounding():
    t, n = _fix_amounts("营业收入 57.87 亿元，另有 3 家子公司", [578749365.4])
    assert t.startswith("营业收入 5.79亿元") and n == 1
    t, n = _fix_amounts("营业收入 12.00 亿元", [578749365.4])     # 对不上任何记录：不动
    assert n == 0 and "12.00" in t


def test_render_adds_code_facts_and_fixes_yoy():
    out = {"claims": [
        {"local_id": "c1", "claim_type": "reported_fact", "text": "2024年一季度营业收入57.87亿元", "value": 5787493654,
         "unit": "元", "evidence": [{"evidence_id": E_CUR, "field": "revenue"}]},
        {"local_id": "c2", "claim_type": "reported_fact", "text": "2023年一季度营业收入5.04亿元", "value": 504000000,
         "unit": "元", "evidence": [{"evidence_id": E_PREV, "field": "revenue"}]},
        {"local_id": "c3", "claim_type": "computed", "text": "营业收入同比增长 99.9%", "value": 99.9, "unit": "%",
         "formula": "c1/c2*100", "inputs": ["c2", "c1"]},
        {"local_id": "c4", "claim_type": "computed", "text": "两期营收差额为 1 亿元", "value": 1, "unit": "亿元",
         "formula": "(c1-c2)/100000000", "inputs": ["c1", "c2"]},
    ]}
    new, info = render_a1(out, EV, "20240430")
    by = {c["local_id"]: c for c in new["claims"]}
    assert by["c1"]["value"] == 578749365.4 and "5.79亿元" in by["c1"]["text"]
    assert by["c3"]["formula"] == "(c1-c2)/abs(c2)*100" and abs(by["c3"]["value"] - 14.831) < 1e-2
    assert "c3" in info["yoy_replaced"] or "c3" in info["calc_fixed"]
    assert abs(by["c4"]["value"] - 0.7475) < 1e-3                 # 计算器：按公式求值
    # 代码补上归母净利润两期与同比；营收两期模型已写，不重复
    fields = [(c["evidence"][0]["evidence_id"], c["evidence"][0]["field"]) for c in new["claims"]
              if c["claim_type"] == "reported_fact"]
    assert len(fields) == len(set(fields)) == 4
    checks = {c.local_id: c.status for c in verify(new, EV)}
    assert all(s in ("verified", "minor_mismatch") for s in checks.values()), checks
    loss = next(c for c in new["claims"] if c["claim_type"] == "computed" and "归母净利润" in c["text"])
    assert loss["value"] == -140.0
