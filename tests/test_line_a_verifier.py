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


def test_minor_mismatch_for_truncation():
    out = _out([{"local_id": "a", "claim_type": "reported_fact", "text": "x", "value": 66593247721.09, "unit": "元",
                 "evidence": [{"evidence_id": "tushare:income:600000.SH:20260630:20260828", "field": "n_income_attr_p",
                               "value": 2.5e9, "unit": "元"}]},
                {"local_id": "b", "claim_type": "reported_fact", "text": "y", "value": 2.0e9, "unit": "元",
                 "evidence": [{"evidence_id": "tushare:income:600000.SH:20250630:20250829", "field": "n_income_attr_p"}]},
                {"local_id": "c", "claim_type": "computed", "text": "约 1.25 倍", "value": 1.2497, "unit": "倍",
                 "formula": "a/b", "inputs": ["a", "b"]},
                {"local_id": "d", "claim_type": "inference", "text": "z", "inputs": ["c"]}])
    out["claims"][0]["value"] = 2.5e9
    st = {c.local_id: c.status for c in verify(out, EV)}
    assert st == {"a": "verified", "b": "verified", "c": "minor_mismatch", "d": "verified"}


def test_eval_score_a1():
    from quantmind.line_a.harness.evalset import score_a1
    gold = [{"fact_id": "income.revenue.latest", "metric": "revenue", "evidence_id": "e1"},
            {"fact_id": "audit.result.latest_annual", "metric": "audit_result", "evidence_id": "e2"}]
    out = {"claims": [{"local_id": "c1", "evidence": [{"evidence_id": "e1", "field": "revenue"}]},
                      {"local_id": "c2", "evidence": [{"evidence_id": "e2", "field": "audit_result"}]}]}
    s = score_a1(out, [{"local_id": "c1", "status": "verified"}, {"local_id": "c2", "status": "mismatch"}], gold, "ok")
    assert s["recall"] == 0.5 and s["missed"] == ["audit.result.latest_annual"] and s["problem_share"] == 0.5
    assert score_a1(None, [], gold, "abstain")["abstained"]


def test_tls_interception_is_not_retried(monkeypatch):
    import httpx
    import pytest as _pt

    from quantmind.line_a.harness import llm as m

    def boom(*a, **k):
        raise httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
    monkeypatch.setattr(httpx, "post", boom)
    x = m.LLM()
    x._key = "dummy"
    with _pt.raises(m.TLSInterceptionError):
        x.chat([{"role": "user", "content": "公开问题"}])


def test_document_quote_values():
    ev = {"doc:1219506496": {"evidence_id": "doc:1219506496",
                             "full_text": "公司拟向全体股东每10 股派发现金红利308.76 元（含税）。截至2023 年12 月31 日，公司总股本1,256,197,800 股。"}}
    out = _out([
        {"local_id": "a", "claim_type": "reported_fact", "text": "每 10 股派 308.76 元", "value": 308.76, "unit": "元",
         "evidence": [{"evidence_id": "doc:1219506496", "field": "现金红利", "quote": "每10 股派发现金红利308.76 元"}]},
        {"local_id": "b", "claim_type": "reported_fact", "text": "总股本 125,619.78 万股", "value": 125619.78, "unit": "万股",
         "evidence": [{"evidence_id": "doc:1219506496", "quote": "公司总股本1,256,197,800 股"}]},
        {"local_id": "c", "claim_type": "reported_fact", "text": "每 10 股派 300 元", "value": 300, "unit": "元",
         "evidence": [{"evidence_id": "doc:1219506496", "field": "现金红利", "quote": "每10 股派发现金红利308.76 元"}]},
        {"local_id": "d", "claim_type": "reported_fact", "text": "x", "value": 1, "unit": "元",
         "evidence": [{"evidence_id": "doc:1219506496", "quote": "每10股派发现金红利500元"}]}])
    st = {c.local_id: c.status for c in verify(out, ev)}
    assert st == {"a": "verified", "b": "verified", "c": "mismatch", "d": "mismatch"}


def test_a6_batch_read_evidence(tmp_path):
    from quantmind.line_a.harness.tools_more import ResearchTools
    t = ResearchTools(tmp_path, "20240410", registry={"e1": {"evidence_id": "e1", "text": "x" * 900}})
    r = t.read_evidence("600519.SH", ["e1", "e2"])
    assert r["n"] == 2 and r["records"]["e1"]["text"].endswith("…") and r["records"]["e2"]["status"] == "not_in_registry"


def test_a3_holders_and_managers_point_in_time(tmp_path):
    import pandas as pd

    from quantmind.line_a.harness.tools_more import ResearchTools
    ev = tmp_path / "events"
    ev.mkdir()
    pd.DataFrame({"ts_code": ["X"] * 5, "ann_date": ["20231028", "20231028", "20240403", "20240403", "20240420"],
                  "end_date": ["20230930", "20230930", "20231231", "20231231", "20231231"],
                  "holder_name": ["甲", "乙", "甲", "丙", "丁"], "holder_type": ["一般企业"] * 5,
                  "hold_amount": [10.0, 5.0, 10.0, 6.0, 9.0], "hold_float_ratio": [1.0] * 5,
                  "hold_change": [0.0] * 5}).to_parquet(ev / "top10_floatholders_20261020.parquet")
    pd.DataFrame({"ts_code": ["X"] * 3, "ann_date": ["20200101", "20200101", "20240501"], "name": ["张", "李", "王"],
                  "title": ["董事长", "董事", "总经理"], "begin_date": ["20200101", "20200101", "20240501"],
                  "end_date": [None, "20230601", None]}).to_parquet(ev / "stk_managers_20261020.parquet")
    t = ResearchTools(tmp_path, "20240410")
    h = t.top_float_holders("X")
    assert h["end_date"] == "20231231" and [x["holder_name"] for x in h["holders"]] == ["甲", "丙"]   # 04-20 的更正还没公告
    assert h["new_entrants"] == ["丙"] and h["exited"] == ["乙"]
    m = t.current_managers("X")
    assert [x["name"] for x in m["managers"]] == ["张"]                                         # 李已离任，王尚未任职


def test_verifier_v2_document_text_and_ellipsis_quotes():
    from quantmind.line_a.harness.verifier import quote_in, verify
    para = "控股股东计划减持本公司股份不超过1,383.4079万股（即不超过公司总股本的3.00%）。"
    ev = {"doc:1:p1:3": {"evidence_id": "doc:1:p1:3", "text": para}}
    out = {"claims": [
        {"local_id": "c1", "claim_type": "reported_fact", "text": "拟减持不超过1383.4079万股", "value": 1383.4079,
         "unit": "万股", "evidence": [{"evidence_id": "doc:1:p1:3", "field": "减持上限"}]},
        {"local_id": "c2", "claim_type": "reported_fact", "text": "拟减持不超过2000万股", "value": 2000, "unit": "万股",
         "evidence": [{"evidence_id": "doc:1:p1:3", "field": "减持上限"}]},
        {"local_id": "c3", "claim_type": "reported_fact", "text": "公告原文", "evidence": [
            {"evidence_id": "doc:1:p1:3", "quote": "控股股东计划减持……不超过公司总股本的3.00%"}]}]}
    st = {c.local_id: c.status for c in verify(out, ev)}
    assert st == {"c1": "verified", "c2": "mismatch", "c3": "verified"}      # 原文里找数；数不对仍是不一致
    assert quote_in("（即不超过公司总股本的3.00%）", "(即不超过公司总股本的3.00%)")       # 全角半角统一
    assert not quote_in("总股本的5.00%", para)


def test_verifier_v2_magnitude_error_and_thousands_separator():
    from quantmind.line_a.harness.verifier import verify
    ev = {"tushare:income:X:20231231:20240416": {"evidence_id": "tushare:income:X:20231231:20240416",
                                                  "revenue": 2468563737.92, "_units": {"revenue": "元"}}}

    def claim(text):
        return {"claims": [{"local_id": "c1", "claim_type": "reported_fact", "text": text, "value": 2468563737.92,
                            "unit": "元", "evidence": [{"evidence_id": "tushare:income:X:20231231:20240416",
                                                       "field": "revenue"}]}]}
    bad = verify(claim("2023年营业收入为246.86亿元。"), ev)[0]
    assert bad.status == "mismatch" and any("magnitude" in n for n in bad.notes)
    good = verify(claim("2023年营业收入为24.69亿元（2,468,563,737.92元）。"), ev)[0]
    assert good.status == "verified" and not any("unsourced" in n for n in good.notes)   # 千分位不再被拆开


def test_verifier_v3_text_fields():
    ev = {"tushare:fina_audit:600000.SH:20251231:20260420": {
        "evidence_id": "tushare:fina_audit:600000.SH:20251231:20260420", "end_date": "20251231",
        "audit_result": "标准无保留意见", "audit_agency": "某会计师事务所"}}
    ref = [{"evidence_id": "tushare:fina_audit:600000.SH:20251231:20260420", "field": "audit_result"}]
    out = _out([
        {"local_id": "c1", "claim_type": "reported_fact", "text": "2025 年年报审计意见为标准无保留意见", "value": 1.0,
         "evidence": ref},
        {"local_id": "c2", "claim_type": "reported_fact", "text": "2025 年年报审计意见为保留意见", "evidence": ref},
        {"local_id": "c3", "claim_type": "reported_fact", "text": "年报审计结论如引文",
         "evidence": [ref[0] | {"quote": "标准无保留意见"}]},
        {"local_id": "c4", "claim_type": "reported_fact", "text": "审计期间为 2025 年",
         "evidence": [ref[0] | {"field": "end_date"}]},
    ])
    st = {c.local_id: c.status for c in verify(out, ev)}
    assert st == {"c1": "verified", "c2": "mismatch", "c3": "verified", "c4": "verified"}
