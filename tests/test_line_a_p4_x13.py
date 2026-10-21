"""P4 X13：年报解析基线（报表段定位、标签与金额、单位、负数、母公司报表排除）和打分类别。用合成的 PDF，不联网。"""
from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz")

from quantmind.line_a.p4 import x13_pdf as x  # noqa: E402


def _page(doc, lines):
    p = doc.new_page()
    y = 60
    for row in lines:
        cells = row if isinstance(row, tuple) else (row,)
        xs = {1: [40], 2: [40, 300], 3: [40, 300, 420], 4: [40, 230, 300, 420]}[len(cells)]
        for xx, c in zip(xs, cells, strict=True):
            p.insert_text((xx, y), c, fontname="china-s", fontsize=9)
        y += 16


def _make(path):
    doc = fitz.open()
    _page(doc, ["目录", "合并资产负债表", "合并利润表"])                       # 目录里的标题不应被选中
    _page(doc, ["合并资产负债表", "单位：元", ("流动资产合计", "七、1", "1,000.00", "900.00"),
                ("资产总计", "5,000.00", "4,000.00"), ("负债合计", "2,000.00", "1,500.00"),
                ("归属于母公司所有者权益（或股", ), ("东权益）合计", "2,800.00", "2,300.00"),
                ("流动负债合计", "1,200.00", "1,100.00")])
    doc[-1].insert_text((40, 300), "非流动资产合计", fontname="china-s", fontsize=9)
    doc[-1].insert_text((300, 300), "7,000.006,000.00", fontname="china-s", fontsize=9)   # 粘连的两个金额
    _page(doc, ["母公司资产负债表", "单位：元", ("资产总计", "9,999.00", "8,888.00")])
    _page(doc, ["合并利润表", "单位：万元", ("一、营业总收入", "七、61", "12,345.67", "10,000.00"),
                ("其中：营业收入", "12,345.67", "10,000.00"),
                ("三、营业利润（亏损以“－”号填列）", "(1,234.50)", "800.00"),
                ("五、净利润（净亏损以“－”号填列）", "-1,000.00", "600.00"),
                ("1.归属于母公司股东的净利润", "-900.00", "500.00")])
    _page(doc, ["合并现金流量表", "单位：元", ("经营活动产生的现金流量净额", "3,000.00", "2,000.00")])
    doc.save(str(path))


def test_extract_and_score(tmp_path):
    p = tmp_path / "ar.pdf"
    _make(p)
    r = x.extract(p)
    assert r["total_assets"]["value"] == 5000.0 and r["total_assets"]["page"] == 2      # 不取母公司的 9,999
    assert r["total_cur_assets"]["value"] == 1000.0                                      # 附注编号"七、1"跳过
    assert r["total_hldr_eqy_exc_min_int"]["value"] == 2800.0                            # 折行标签
    assert r["total_revenue"]["value"] == pytest.approx(123_456_700.0)                  # 万元换算
    assert r["operate_profit"]["value"] == pytest.approx(-12_345_000.0)                 # 括号负数
    assert r["n_income"]["value"] == pytest.approx(-10_000_000.0)
    assert r["n_income_attr_p"]["value"] == pytest.approx(-9_000_000.0)
    assert r["n_cashflow_act"]["value"] == 3000.0 and r["n_cashflow_act"]["prior"] == 2000.0
    g = {"value": 5000.0, "prior": 4000.0, "excluded": None}
    assert x.score_field(r["total_assets"], g) == "correct"
    assert x.score_field(r["total_assets"], {**g, "value": 5_000_000.0}) == "unit_error"
    assert x.score_field(r["total_assets"], {**g, "value": -5000.0}) == "sign_error"
    assert x.score_field({**r["total_assets"], "value": 4000.0}, g) == "period_error"
    assert x.score_field(None, g) == "not_found"
    assert x.score_field(r["total_assets"], {**g, "excluded": "restated_or_missing"}) == "skipped"
    # 万元两位小数：精度是 100 元，差 40 元仍算正确
    assert x.score_field(r["total_revenue"], {"value": 123_456_740.0, "prior": None, "excluded": None}) == "correct"


def test_merged_amounts_split():
    row = [(0, 0, 10, 10, "非流动资产合计"), (300, 0, 360, 10, "7,000.00(6,000.00)")]
    assert [w[4] for w in x._tokens(row)] == ["非流动资产合计", "7,000.00", "(6,000.00)"]
    assert x._num("(6,000.00)") == (-6000.0, 2) and x._num("61") is None and x._num("-") == (0.0, 2)


def test_title_filter():
    assert x._TITLE.search("贵州茅台2023年年度报告").group(1) == "2023"
    assert x._TITLE.search("某公司2023年度报告（修订版）").group(1) == "2023"
    assert x._TITLE.search("某公司2023年年度报告摘要") is None
    assert x._EXCLUDE.search("关于某公司2023年年度报告的审计报告")
