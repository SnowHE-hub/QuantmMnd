"""Tests for scripts/build_kb_all_snapshots batch planner and manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from quantmind.kb.snapshot_inventory import (
    ingest_readiness,
    iter_snapshot_date_dirs,
    manifest_success_as_ofs,
)


def _minimal_snapshot_dir(root: Path, name: str, *, univ_rows: int) -> Path:
    d = root / name
    d.mkdir(parents=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        sym = pa.table({"ts_code": pa.array(["600519.SH"] * univ_rows)})
        pq.write_table(sym, d / "universe.parquet")
        sb = pa.table({"ts_code": pa.array(["600519.SH"]), "name": pa.array(["X"])})
        pq.write_table(sb, d / "stock_basic.parquet")
        pq.write_table(
            pa.table({
                "ts_code": pa.array(["000300.SH"]),
                "trade_date": pa.array(["2024-01-02"]),
                "close": pa.array([1.0]),
            }),
            d / "index_daily.parquet",
        )
        for fn in (
            "prices.parquet",
            "daily_basic.parquet",
            "financial_indicators.parquet",
            "hk_hold.parquet",
            "margin.parquet",
        ):
            pq.write_table(pa.table({"ts_code": pa.array(["600519"])}), d / fn)
    except Exception:
        pytest.skip("pyarrow write failed")
    return d


def test_scan_yyyy_mm_dd_dirs(tmp_path: Path) -> None:
    (tmp_path / "2024-06-30").mkdir()
    (tmp_path / "bad").mkdir()
    (tmp_path / "2020-01-01").mkdir()
    got = iter_snapshot_date_dirs(tmp_path)
    assert [p.name for p in got] == ["2020-01-01", "2024-06-30"]


def test_min_universe_skips_small(tmp_path: Path) -> None:
    d = _minimal_snapshot_dir(tmp_path, "2024-01-01", univ_rows=10)
    inv = ingest_readiness(d, min_universe=250, include_small=False)
    assert inv["skip_reason"]
    inv2 = ingest_readiness(d, min_universe=250, include_small=True)
    assert inv2["ingest_eligible"] and not inv2["skip_reason"]


def test_manifest_skip_existing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    man = tmp_path / "m.json"
    man.write_text(json.dumps({"runs": {"2024-01-01": {"status": "success"}}}), encoding="utf-8")
    assert "2024-01-01" in manifest_success_as_ofs(man)

    root = tmp_path / "snap"
    _minimal_snapshot_dir(root, "2024-01-01", univ_rows=300)
    _minimal_snapshot_dir(root, "2024-02-01", univ_rows=300)
    _minimal_snapshot_dir(root, "2024-03-01", univ_rows=300)

    calls: list[str] = []

    def fake_build(self, snapshot_dir, tickers=None, as_of=None, dry_run=False):
        calls.append(str(snapshot_dir))
        return {
            "n_docs": 1,
            "n_chunks": 1,
            "chunks_written": 1,
            "universe_count": 300,
            "n_tickers_ingested": 1,
            "final_collection_count": 9,
            "dry_run": dry_run,
        }

    monkeypatch.setattr("quantmind.kb.builder.KBBuilder.build_from_snapshot_dir", fake_build)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--snapshots-root",
            str(root),
            "--manifest",
            str(man),
            "--execute",
            "--force",
            "--log-file",
            str(tmp_path / "l.log"),
            "--max-dates",
            "2",
        ],
    )
    import scripts.build_kb_all_snapshots as bks

    bks.main()
    assert len(calls) == 2

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--snapshots-root",
            str(root),
            "--manifest",
            str(man),
            "--execute",
            "--log-file",
            str(tmp_path / "l2.log"),
            "--max-dates",
            "2",
        ],
    )
    calls.clear()
    bks.main()
    assert len(calls) == 1
    assert "2024-03-01" in calls[0]


def test_dry_run_no_builder_write(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "snap"
    _minimal_snapshot_dir(root, "2024-02-01", univ_rows=300)

    def boom(*a, **k):
        raise AssertionError("build_from_snapshot_dir should not run in dry-run")

    monkeypatch.setattr("quantmind.kb.builder.KBBuilder.build_from_snapshot_dir", boom)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--snapshots-root",
            str(root),
            "--dry-run",
            "--manifest",
            str(tmp_path / "m.json"),
            "--log-file",
            str(tmp_path / "l.log"),
        ],
    )
    import scripts.build_kb_all_snapshots as bks

    bks.main()


def test_max_dates_and_reverse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "snap"
    _minimal_snapshot_dir(root, "2024-01-01", univ_rows=300)
    _minimal_snapshot_dir(root, "2024-03-01", univ_rows=300)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--snapshots-root",
            str(root),
            "--dry-run",
            "--reverse",
            "--max-dates",
            "1",
            "--manifest",
            str(tmp_path / "m.json"),
            "--log-file",
            str(tmp_path / "l.log"),
        ],
    )
    import scripts.build_kb_all_snapshots as bks

    bks.main()


def test_failure_records_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "snap"
    _minimal_snapshot_dir(root, "2024-01-01", univ_rows=300)
    _minimal_snapshot_dir(root, "2024-02-01", univ_rows=300)

    def flaky(self, snapshot_dir, tickers=None, as_of=None, dry_run=False):
        if "2024-01-01" in str(snapshot_dir):
            raise RuntimeError("boom")
        return {
            "n_docs": 1,
            "n_chunks": 1,
            "chunks_written": 1,
            "universe_count": 300,
            "n_tickers_ingested": 1,
            "final_collection_count": 2,
            "dry_run": False,
        }

    monkeypatch.setattr("quantmind.kb.builder.KBBuilder.build_from_snapshot_dir", flaky)
    man = tmp_path / "m.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--snapshots-root",
            str(root),
            "--execute",
            "--no-skip-existing",
            "--manifest",
            str(man),
            "--log-file",
            str(tmp_path / "l.log"),
        ],
    )
    import scripts.build_kb_all_snapshots as bks

    bks.main()
    data = json.loads(man.read_text(encoding="utf-8"))
    assert data["runs"]["2024-01-01"]["status"] == "failed"
    assert data["runs"]["2024-02-01"]["status"] == "success"


def test_force_reingest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "snap"
    _minimal_snapshot_dir(root, "2024-01-01", univ_rows=300)
    man = tmp_path / "m.json"
    man.write_text(json.dumps({"runs": {"2024-01-01": {"status": "success"}}}), encoding="utf-8")
    n = {"c": 0}

    def count(self, *a, **k):
        n["c"] += 1
        return {
            "n_docs": 1,
            "n_chunks": 1,
            "chunks_written": 1,
            "universe_count": 300,
            "n_tickers_ingested": 1,
            "final_collection_count": 5,
            "dry_run": False,
        }

    monkeypatch.setattr("quantmind.kb.builder.KBBuilder.build_from_snapshot_dir", count)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--snapshots-root",
            str(root),
            "--execute",
            "--force",
            "--manifest",
            str(man),
            "--log-file",
            str(tmp_path / "l.log"),
        ],
    )
    import scripts.build_kb_all_snapshots as bks

    bks.main()
    assert n["c"] == 1
