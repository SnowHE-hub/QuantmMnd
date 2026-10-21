"""P2 证据层第一版的单元测试：标题分类、事件状态机、BM25、证券身份链接、PDF 定位。"""
from __future__ import annotations

import pandas as pd

from quantmind.line_a.evidence.ann_types import classify_title
from quantmind.line_a.evidence.entity import AliasTable
from quantmind.line_a.evidence.event_chain import Ann, build_chains, state_as_of
from quantmind.line_a.evidence.retrieval import BM25, tokenize


def test_classify_titles():
    c = classify_title("关于公司收到中国证券监督管理委员会立案告知书的公告")
    assert (c.event_type, c.importance, c.type_rule) == ("investigation", "major", "T01")
    c = classify_title("关于以集中竞价交易方式回购公司股份的进展公告")
    assert (c.event_type, c.stage) == ("repurchase", "progress")
    c = classify_title("关于终止回购公司股份方案的公告")
    assert (c.event_type, c.stage, c.importance) == ("repurchase", "terminated", "watch")
    c = classify_title("关于持股5%以上股东减持股份预披露的公告")
    assert (c.event_type, c.stage) == ("holder_reduction", "plan")
    c = classify_title("关于2026年半年度报告的更正公告")
    assert (c.event_type, c.stage) == ("periodic_report", "correction")
    assert classify_title("2026年半年度报告摘要").is_summary_or_meta
    c = classify_title("关于股票复牌的公告")
    assert (c.event_type, c.stage) == ("suspension", "lifted")
    assert classify_title("关于撤销其他风险警示的公告").event_type == "st_change"


def test_event_chain_states_and_corrections():
    a = [Ann("1", "600000.SH", "repurchase", "plan", "20260101"),
         Ann("2", "600000.SH", "repurchase", "implemented", "20260110"),
         Ann("3", "600000.SH", "repurchase", "correction", "20260111"),
         Ann("4", "600000.SH", "repurchase", "progress", "20260201"),
         Ann("5", "600000.SH", "repurchase", "completed", "20260301"),
         Ann("6", "600000.SH", "repurchase", "plan", "20260401"),
         Ann("7", "600000.SH", "repurchase", "terminated", "20260501"),
         Ann("8", "000001.SZ", "regulatory_letter", "single", "20260105")]
    ch = build_chains(a)
    rp = [c for c in ch if c.event_type == "repurchase"]
    assert [c.state for c in rp] == ["completed", "terminated"]
    assert rp[0].corrections == [("3", "2")] and rp[0].ann_ids == ["1", "2", "4", "5"]
    assert rp[0].anomalies and "implemented -> progress" in rp[0].anomalies[0]
    # 时点：2026-02-15 时第一条链进行中（progress），第二条还不存在
    mid = [c for c in state_as_of(a, "20260215") if c.event_type == "repurchase"]
    assert len(mid) == 1 and mid[0].state == "progress" and not mid[0].closed
    assert [c.state for c in ch if c.event_type == "regulatory_letter"] == ["single"]


def test_bm25_finds_amount_paragraph():
    paras = [("p1", "公司拟以自有资金回购股份，回购金额不低于5000万元，不超过1亿元。"),
             ("p2", "本次担保额度为2亿元，担保对象为全资子公司。"),
             ("p3", "董事会审议通过了2026年半年度报告。")]
    bm = BM25(paras)
    assert bm.search("回购金额")[0].para_id == "p1"
    assert bm.search("担保 子公司")[0].para_id == "p2"
    assert "5000" in tokenize("回购金额不低于5000万元")


def test_entity_linking_periods_and_context():
    sb = pd.DataFrame({"ts_code": ["600000.SH", "000001.SZ"], "name": ["浦发银行", "平安银行"],
                       "fullname": ["上海浦东发展银行股份有限公司", "平安银行股份有限公司"]})
    nc = pd.DataFrame({"ts_code": ["000001.SZ"], "name": ["深发展A"], "start_date": ["19910403"], "end_date": ["20120801"]})
    t = AliasTable.from_frames(sb, nc)
    ms = t.link("浦发银行（600000）公告：其全资子公司平安银行股份有限公司……深发展A", "20261020")
    by = {m.text: m for m in ms}
    assert by["浦发银行"].candidates == ("600000.SH",) and by["浦发银行"].status == "linked"
    assert by["600000"].candidates == ("600000.SH",) and by["600000"].kind == "code"
    assert by["平安银行股份有限公司"].context == "subsidiary" and by["平安银行股份有限公司"].kind == "full_name"
    assert by["深发展A"].status == "out_of_period"                       # 曾用名不在 2026 年的生效区间
    assert t.link("深发展A", "20100101")[0].status == "linked"


def test_pdf_parse_and_quote_check(tmp_path):
    import fitz

    from quantmind.line_a.evidence.pdf_locate import parse_pdf, verify_quote
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((50, 72), "关于回购股份的进展公告。", fontname="china-s")
    p.insert_text((50, 120), "截至本公告日，公司已回购股份1000000股。", fontname="china-s")
    doc.new_page()                                                   # 空白页 → 需要 OCR
    f = tmp_path / "a.pdf"
    doc.save(f)
    d = parse_pdf(f, "ann-1")
    assert d.n_pages == 2 and d.needs_ocr == [2] and len(d.file_sha256) == 64
    hit = [x for x in d.paras if "1000000" in x.text]
    assert hit and verify_quote(d, hit[0].page, hit[0].idx, "已回购股份 1000000股")
    assert not verify_quote(d, hit[0].page, hit[0].idx, "已回购股份2000000股")


def test_subject_tag_separates_concurrent_dividends():
    from quantmind.line_a.evidence.ann_types import subject_tag
    assert subject_tag("2023年度回报股东特别分红方案公告", "dividend") == "特别-2023年度"
    assert subject_tag("2023年度利润分配方案公告", "dividend") == "2023年度"
    assert subject_tag("2022年年度权益分派实施公告", "dividend") == "2022年度"
    assert subject_tag("关于回购股份的进展公告", "repurchase") == ""
    a = [Ann("1", "600519.SH", "dividend", "plan", "20231121", subject="特别-2023年度"),
         Ann("2", "600519.SH", "dividend", "implemented", "20231214", subject="特别-2023年度"),
         Ann("3", "600519.SH", "dividend", "plan", "20240403", subject="2023年度")]
    ch = build_chains(a)
    assert [(c.subject, c.state) for c in ch] == [("特别-2023年度", "implemented"), ("2023年度", "plan")]
    assert not any(c.anomalies for c in ch) and ch[0].chain_id == "600519.SH:dividend:特别-2023年度:1"


def test_list_announcements_truncation_and_chain_ids(tmp_path):
    from quantmind.line_a.harness.tools_data import DataTools
    parts = tmp_path / "text/cninfo/_parts"
    parts.mkdir(parents=True)
    rows = [{"announcementId": str(i), "secCode": "600519", "announcementTitle": t, "publish_ts_bj": f"2024-04-0{d}T00:00:00+08:00"}
            for i, (t, d) in enumerate([("2023年度利润分配方案公告", 3), ("关于回购股份的进展公告", 2),
                                        ("关于收到问询函的公告", 1)])]
    pd.DataFrame(rows).to_parquet(parts / "20240403.parquet")
    dt = DataTools(tmp_path, "20240410")
    r = dt.list_announcements("600519.SH", lookback_days=30, limit=2)
    assert r["total"] == 3 and r["truncated"] and len(r["announcements"]) == 2 and "共 3 条" in r["note"]
    r = dt.list_announcements("600519.SH", lookback_days=30, min_importance="watch")
    assert [a["event_type"] for a in r["announcements"]] == ["regulatory_letter"]
    ch = dt.event_chains("600519.SH", lookback_days=30)["chains"]
    assert all(c["evidence_id"].startswith("chain:") for c in ch)
    assert {c["subject"] for c in ch if c["event_type"] == "dividend"} == {"2023年度"}
