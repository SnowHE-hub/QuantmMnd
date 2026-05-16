"""Unit tests for parquet snapshot → KB document generation."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from quantmind.kb.builder import KBBuilder
from quantmind.kb.chunker import SmartChunker
from quantmind.kb.snapshot_parquet import generate_snapshot_documents


def _write_min_snapshot(root: Path, *, with_hk: bool = True, with_margin: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"ts_code": "600519.SH", "name": "贵州茅台", "industry": "白酒", "area": "贵州", "list_date": "19910827"},
        {"ts_code": "300750.SZ", "name": "宁德时代", "industry": "电池", "area": "福建", "list_date": "20180611"},
    ]).to_parquet(root / "stock_basic.parquet")

    pd.DataFrame([
        {"ts_code": "600519.SH", "trade_date": "20241230", "close": 1500.0, "pct_chg": 1.2, "vol": 100.0, "amount": 1e6},
        {"ts_code": "300750.SZ", "trade_date": "20241230", "close": 200.0, "pct_chg": -0.5, "vol": 200.0, "amount": 2e6},
    ]).to_parquet(root / "prices.parquet")

    pd.DataFrame([
        {"ts_code": "600519.SH", "trade_date": "20241230", "pe": 25.0, "pb": 8.0, "total_mv": 1.8e7, "circ_mv": 1.8e7},
        {"ts_code": "300750.SZ", "trade_date": "20241230", "pe": 18.0, "pb": 4.0, "total_mv": 8e6, "circ_mv": 8e6},
    ]).to_parquet(root / "daily_basic.parquet")

    pd.DataFrame([
        {"ts_code": "600519.SH", "end_date": "20240930", "roe": 0.28, "grossprofit_margin": 0.91},
        {"ts_code": "300750.SZ", "end_date": "20240930", "roe": 0.15, "grossprofit_margin": 0.22},
    ]).to_parquet(root / "financial_indicators.parquet")

    if with_hk:
        pd.DataFrame([
            {"ts_code": "600519.SH", "trade_date": "20241230", "hold_ratio": 6.5, "hold_vol": 8000.0},
            {"ts_code": "300750.SZ", "trade_date": "20241230", "hold_ratio": 4.1, "hold_vol": 9000.0},
        ]).to_parquet(root / "hk_hold.parquet")

    if with_margin:
        pd.DataFrame([
            {"ts_code": "600519.SH", "trade_date": "20241230", "rzye": 1e8, "rzmre": 1e6, "rqye": 2e6, "rqyl": 100.0},
            {"ts_code": "300750.SZ", "trade_date": "20241230", "rzye": 2e8, "rzmre": 2e6, "rqye": 3e6, "rqyl": 200.0},
        ]).to_parquet(root / "margin.parquet")

    pd.DataFrame([
        {"ts_code": "000300.SH", "trade_date": "20241230", "close": 3500.0, "pct_chg": 0.5},
        {"ts_code": "000300.SH", "trade_date": "20241202", "close": 3400.0, "pct_chg": 0.1},
        {"ts_code": "000905.SH", "trade_date": "20241230", "close": 5500.0, "pct_chg": 0.3},
        {"ts_code": "399006.SZ", "trade_date": "20241230", "close": 2100.0, "pct_chg": -0.2},
        {"ts_code": "000001.SH", "trade_date": "20241230", "close": 3000.0, "pct_chg": 0.15},
    ]).to_parquet(root / "index_daily.parquet")


def test_company_profile_document(tmp_path: Path) -> None:
    d = tmp_path / "s"
    _write_min_snapshot(d, with_hk=False, with_margin=False)
    docs, _ = generate_snapshot_documents(d, ["600519.SH"], as_of=date(2024, 12, 31))
    prof = [x for x in docs if x.get("doc_type") == "company_profile"]
    assert len(prof) == 1
    assert "贵州茅台" in prof[0]["text"]
    assert prof[0]["source_type"] == "snapshot"
    assert prof[0]["ticker"] == "600519.SH"


def test_latest_market_metrics(tmp_path: Path) -> None:
    d = tmp_path / "s"
    _write_min_snapshot(d, with_hk=False, with_margin=False)
    docs, _ = generate_snapshot_documents(d, ["600519.SH"], as_of=date(2024, 12, 31))
    m = [x for x in docs if x.get("doc_type") == "latest_market_metrics"]
    assert len(m) == 1
    assert "close" in m[0]["text"].lower() or "1500" in m[0]["text"]
    assert m[0]["table_name"] == "prices,daily_basic"


def test_financial_indicator_summary(tmp_path: Path) -> None:
    d = tmp_path / "s"
    _write_min_snapshot(d, with_hk=False, with_margin=False)
    docs, _ = generate_snapshot_documents(d, ["600519.SH"], as_of=date(2024, 12, 31))
    f = [x for x in docs if x.get("doc_type") == "financial_indicator_summary"]
    assert len(f) == 1
    assert "roe" in f[0]["text"].lower()


def test_hk_hold_missing_no_crash(tmp_path: Path) -> None:
    d = tmp_path / "s"
    _write_min_snapshot(d, with_hk=False, with_margin=True)
    docs, _ = generate_snapshot_documents(d, ["600519.SH"], as_of=date(2024, 12, 31))
    assert not any(x.get("doc_type") == "northbound_summary" for x in docs)


def test_company_profile_fallback_from_universe_without_stock_basic(tmp_path: Path) -> None:
    d = tmp_path / "s_legacy"
    d.mkdir(parents=True)
    pd.DataFrame([
        {"ts_code": "600519.SH", "name": "贵州茅台"},
        {"ts_code": "300750.SZ", "name": "宁德时代"},
    ]).to_parquet(d / "universe.parquet")
    pd.DataFrame([
        {"ts_code": "600519.SH", "trade_date": "20241230", "close": 1500.0},
    ]).to_parquet(d / "prices.parquet")
    pd.DataFrame([
        {"ts_code": "600519.SH", "trade_date": "20241230", "pe": 25.0},
    ]).to_parquet(d / "daily_basic.parquet")
    pd.DataFrame([
        {"ts_code": "600519.SH", "end_date": "20240930", "roe": 0.28},
    ]).to_parquet(d / "financial_indicators.parquet")
    docs, _ = generate_snapshot_documents(d, ["600519.SH"], as_of=date(2024, 12, 31))
    prof = [x for x in docs if x.get("doc_type") == "company_profile"]
    assert len(prof) == 1
    assert prof[0].get("table_name") == "universe"
    assert "贵州茅台" in prof[0]["text"]


def test_margin_missing_no_crash(tmp_path: Path) -> None:
    d = tmp_path / "s"
    _write_min_snapshot(d, with_hk=True, with_margin=False)
    docs, _ = generate_snapshot_documents(d, ["600519.SH"], as_of=date(2024, 12, 31))
    assert not any(x.get("doc_type") == "margin_summary" for x in docs)


def test_market_index_context(tmp_path: Path) -> None:
    d = tmp_path / "s"
    _write_min_snapshot(d, with_hk=False, with_margin=False)
    docs, _ = generate_snapshot_documents(d, [], as_of=date(2024, 12, 31))
    ctx = [x for x in docs if x.get("doc_type") == "market_index_context"]
    assert len(ctx) == 1
    assert ctx[0]["ticker"] == "__MARKET__"
    assert "000300" in ctx[0]["text"]


def test_dry_run_no_retriever(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = tmp_path / "s"
    _write_min_snapshot(d)
    called: list[str] = []

    def boom(*_a, **_k):
        called.append("retriever")
        raise AssertionError("HybridRetriever should not be used in dry_run")

    monkeypatch.setattr("quantmind.kb.builder.KBBuilder._get_retriever", boom)
    b = KBBuilder(chunker=SmartChunker())
    res = b.build_from_snapshot_dir(d, tickers=["600519.SH"], as_of=date(2024, 12, 31), dry_run=True)
    assert res["dry_run"] is True
    assert res["chunks_written"] == 0
    assert res["n_docs"] >= 1
    assert not called


def test_metadata_required_keys(tmp_path: Path) -> None:
    d = tmp_path / "s"
    _write_min_snapshot(d)
    docs, _ = generate_snapshot_documents(d, ["600519.SH"], as_of=date(2024, 12, 31))
    for doc in docs:
        if doc.get("ticker") == "__MARKET__":
            assert doc.get("source_type") == "snapshot"
            assert doc.get("doc_type") == "market_index_context"
            continue
        for k in ("ticker", "as_of", "source_type", "doc_type", "table_name"):
            assert k in doc and str(doc[k]).strip(), f"missing {k} in {doc.get('doc_type')}"
        assert doc["source_type"] == "snapshot"


def test_build_kb_cli_calls_parquet_builder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = tmp_path / "snap_cli"
    _write_min_snapshot(d)
    calls: list[dict] = []

    def capture_snapshot_dir(self, snapshot_dir, tickers=None, as_of=None, dry_run=False):
        calls.append({
            "snapshot_dir": snapshot_dir,
            "tickers": tickers,
            "as_of": as_of,
        })
        return {
            "snapshot_dir": str(snapshot_dir),
            "as_of": "2024-12-31",
            "parquet_loaded": [],
            "parquet_missing": [],
            "n_docs": 1,
            "per_ticker": {},
            "n_chunks": 1,
            "chunks_written": 1,
            "dry_run": dry_run,
        }

    monkeypatch.setattr("quantmind.kb.builder.KBBuilder.build_from_snapshot_dir", capture_snapshot_dir)
    monkeypatch.setattr(sys, "argv", [
        "build_kb",
        "--source", "snapshot",
        "--snapshot-dir", str(d),
        "--snapshot-as-of", "2024-12-31",
        "--snapshot-tickers", "600519.SH",
    ])
    import scripts.build_kb as bk
    bk.main()
    assert len(calls) == 1
    assert Path(calls[0]["snapshot_dir"]) == d
    assert calls[0]["tickers"] == ["600519.SH"]


def test_snapshot_skip_explicit_when_no_pkl_no_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    real_glob = Path.glob

    def wrapped(self: Path, pattern: str):
        if self.name == "snapshots" and self.parent.name == "data" and pattern == "*.pkl":
            return []
        return real_glob(self, pattern)

    monkeypatch.setattr(Path, "glob", wrapped)

    msgs: list[str] = []

    class _Log:
        @staticmethod
        def info(m, *a, **k):
            msgs.append(str(m))

        @staticmethod
        def warning(m, *a, **k):
            msgs.append(str(m))

    import scripts.build_kb as bk

    monkeypatch.setattr(bk, "logger", _Log())
    monkeypatch.setattr(sys, "argv", ["build_kb", "--source", "snapshot"])
    bk.main()
    blob = "\n".join(msgs).lower()
    assert "snapshot skipped" in blob or "no data/snapshots/*.pkl" in blob
