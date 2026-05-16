"""Tests for scripts/compare_stock_reports.py — no API keys."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import compare_stock_reports as csr


def _minimal_ctx(
    ticker: str = "999999.SZ",
    *,
    pe: str = "10.5",
    pb: str = "2",
    total_mv: str = "1e+08",
    roe: str = "20.0",
    gm: str = "30.0",
    npm: str = "15.0",
    dta: str = "40.0",
    or_yoy: str = "5.5",
    hold: str = "12.0",
    rzye: str = "2e9",
    name: str = "测试公司",
    industry: str = "测试行业",
) -> dict:
    return {
        "ticker": ticker,
        "news_count": 3,
        "report_count": 0,
        "snapshot_company_profile": [
            {
                "as_of": "2024-12-31",
                "text": f"名称：{name}\n行业：{industry}\n",
            }
        ],
        "snapshot_latest_market_metrics": [
            {
                "as_of": "2024-12-31",
                "text": f"股票：{ticker}\ntotal_mv={total_mv}, pe={pe}, pb={pb}\n",
            }
        ],
        "snapshot_financial_indicator_summary": [
            {
                "as_of": "2024-12-31",
                "text": (
                    f"roe：{roe}\ngrossprofit_margin：{gm}\n"
                    f"netprofit_margin：{npm}\ndebt_to_assets：{dta}\nor_yoy：{or_yoy}\n"
                ),
            }
        ],
        "snapshot_northbound_summary": [
            {"as_of": "2024-12-31", "text": f"hold_ratio：{hold}\n"},
        ],
        "snapshot_margin_summary": [
            {"as_of": "2024-12-31", "text": f"rzye：{rzye}\n"},
        ],
    }


def test_extract_metrics_from_mock_context() -> None:
    ctx = _minimal_ctx()
    row = csr.extract_row_from_context(ctx)
    assert row["ticker"] == "999999.SZ"
    assert row["pe"] == pytest.approx(10.5)
    assert row["pb"] == pytest.approx(2.0)
    assert row["total_mv_yi"] == pytest.approx(1e8 / 10000.0)
    assert row["roe"] == pytest.approx(20.0)
    assert row["grossprofit_margin"] == pytest.approx(30.0)
    assert row["netprofit_margin"] == pytest.approx(15.0)
    assert row["debt_to_assets"] == pytest.approx(40.0)
    assert row["or_yoy"] == pytest.approx(5.5)
    assert row["hold_ratio"] == pytest.approx(12.0)
    assert row["rzye_yi"] == pytest.approx(2e9 / 1e8)
    assert row["company_name"] == "测试公司"
    assert row["industry"] == "测试行业"


def test_missing_fields_none_not_zero() -> None:
    row = csr.extract_row_from_context({"ticker": "X", "snapshot_company_profile": []})
    assert row["pe"] is None
    assert row["company_name"] is None
    assert row["news_count"] is None


def test_output_json_structure(tmp_path: Path) -> None:
    ctx_dir = tmp_path / "ctx"
    ctx_dir.mkdir()
    for i in range(14):
        p = ctx_dir / f"STOCK{i:02d}_SZ_context.json"
        ctx = _minimal_ctx(
            ticker=f"STOCK{i:02d}.SZ",
            pe=str(10 + i * 0.1),
            hold=str(i % 3 * 4 + 1),
        )
        p.write_text(json.dumps(ctx, ensure_ascii=False), encoding="utf-8")
    out_dir = tmp_path / "out"
    rows = csr.run_compare(context_dir=ctx_dir, output_dir=out_dir, as_of_label="2024-12-31")
    assert len(rows) == 14
    data = json.loads((out_dir / "comparison_table.json").read_text(encoding="utf-8"))
    assert len(data) == 14
    for item in data:
        assert "ticker" in item and "industry" in item and "pe" in item


def test_valuation_md_pe_ascending(tmp_path: Path) -> None:
    ctx_dir = tmp_path / "ctx"
    ctx_dir.mkdir()
    for i, pe in enumerate([50.0, 10.0, 30.0]):
        ctx = _minimal_ctx(ticker=f"A{i}.SZ", pe=str(pe), hold="2")
        (ctx_dir / f"A{i}_SZ_context.json").write_text(
            json.dumps(ctx, ensure_ascii=False), encoding="utf-8"
        )
    out = tmp_path / "o"
    csr.run_compare(context_dir=ctx_dir, output_dir=out, as_of_label="2024-12-31")
    md = (out / "valuation_ranking.md").read_text(encoding="utf-8")
    assert "A1.SZ" in md  # PE 10 lowest
    lines = [
        ln
        for ln in md.splitlines()
        if ln.startswith("| ")
        and not ln.startswith("| ---")
        and not ln.startswith("| 排名")
    ]
    first_data = lines[0]
    assert "A1.SZ" in first_data


def test_northbound_md_hold_descending(tmp_path: Path) -> None:
    ctx_dir = tmp_path / "ctx"
    ctx_dir.mkdir()
    for i, h in enumerate([1.0, 15.0, 8.0]):
        ctx = _minimal_ctx(ticker=f"B{i}.SZ", hold=str(h), pe="20")
        (ctx_dir / f"B{i}_SZ_context.json").write_text(
            json.dumps(ctx, ensure_ascii=False), encoding="utf-8"
        )
    out = tmp_path / "o2"
    csr.run_compare(context_dir=ctx_dir, output_dir=out, as_of_label="2024-12-31")
    md = (out / "northbound_ranking.md").read_text(encoding="utf-8")
    lines = [
        ln
        for ln in md.splitlines()
        if ln.startswith("| ")
        and not ln.startswith("| ---")
        and not ln.startswith("| 排名")
    ]
    assert lines[0].startswith("| 1 | B1.SZ")


def test_comparison_table_md_14_rows(tmp_path: Path) -> None:
    ctx_dir = tmp_path / "ctx"
    ctx_dir.mkdir()
    for i in range(14):
        (ctx_dir / f"T{i}_SZ_context.json").write_text(
            json.dumps(_minimal_ctx(ticker=f"T{i}.SZ"), ensure_ascii=False),
            encoding="utf-8",
        )
    out = tmp_path / "o3"
    csr.run_compare(context_dir=ctx_dir, output_dir=out)
    md = (out / "comparison_table.md").read_text(encoding="utf-8")
    body_lines = [ln for ln in md.splitlines() if ln.startswith("| T")]
    assert len(body_lines) == 14
