"""tests/test_patch_snapshot_v1_modules.py — patch_v1_modules 增量补 v1 四表."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from quantmind.data import snapshot as snap


@pytest.fixture
def fake_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "data"
    root.mkdir()
    cfg = SimpleNamespace(
        dir=str(root),
        hk_hold_lookback_calendar_days=30,
        margin_lookback_calendar_days=30,
        index_daily_lookback_calendar_days=60,
        index_daily_codes=["000300.SH", "000905.SH"],
        snapshot_strict=False,
        include_stock_basic=True,
        include_hk_hold=True,
        include_margin=True,
        include_index_daily=True,
    )
    settings = SimpleNamespace(data=cfg)
    monkeypatch.setattr(snap, "get_settings", lambda: settings)
    return root


def _write_min_snapshot(
    snap_dir: Path,
    *,
    meta_extra: dict | None = None,
    with_prices: bool = True,
) -> None:
    snap_dir.mkdir(parents=True, exist_ok=True)
    u = pd.DataFrame({"ticker": ["000001.SZ", "600000.SH"], "weight": [10.0, 10.0]})
    u.to_parquet(snap_dir / "universe.parquet", index=False)
    meta = {
        "as_of": snap_dir.name,
        "snapshot_dir": str(snap_dir),
        "files": ["universe.parquet"],
        "rows_per_table": {"universe": 2},
        "legacy_marker": 42,
    }
    if meta_extra:
        meta.update(meta_extra)
    if with_prices:
        px = pd.DataFrame(
            {
                "ticker": ["000001.SZ"],
                "trade_date": [pd.Timestamp("2024-01-02")],
                "close": [10.0],
            }
        )
        px.to_parquet(snap_dir / "prices.parquet", index=False)
        mf = list(meta["files"])
        mf.append("prices.parquet")
        meta["files"] = mf
        meta.setdefault("rows_per_table", {})["prices"] = len(px)
    (snap_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


class _OkProvider:
    """全部返回最小非空 DataFrame。"""

    def __init__(self) -> None:
        self.hk_calls = 0
        self.sb_calls = 0
        self.mg_calls = 0
        self.ix_calls = 0

    def get_stock_basic(self, include_delisted: bool = True) -> pd.DataFrame:  # noqa: ARG002
        self.sb_calls += 1
        return pd.DataFrame(
            {
                "ticker": ["000001.SZ", "600000.SH"],
                "industry": ["银行", "银行"],
                "area": ["深圳", "上海"],
                "list_date": pd.to_datetime(["2000-01-01", "2000-02-01"]),
            }
        )

    def get_hk_hold(self, ticker: str, start, end, as_of=None) -> pd.DataFrame:  # noqa: ANN001
        self.hk_calls += 1
        return pd.DataFrame(
            {
                "ticker": [ticker],
                "trade_date": [pd.Timestamp("2024-01-02")],
                "hold_ratio": [1.0],
            }
        )

    def get_margin_detail(self, ticker: str, start, end, as_of=None) -> pd.DataFrame:  # noqa: ANN001
        self.mg_calls += 1
        return pd.DataFrame(
            {
                "ticker": [ticker],
                "trade_date": [pd.Timestamp("2024-01-02")],
                "rzye": [100.0],
                "rqye": [10.0],
            }
        )

    def get_index_daily(self, code: str, start, end, as_of=None) -> pd.DataFrame:  # noqa: ANN001
        self.ix_calls += 1
        return pd.DataFrame(
            {
                "ts_code": [code],
                "trade_date": [pd.Timestamp("2024-01-02")],
                "close": [3000.0],
            }
        )


def test_dry_run_writes_nothing(fake_settings: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snap_dir = fake_settings / "snapshots" / "2024-06-30"
    _write_min_snapshot(snap_dir)

    boom = MagicMock(side_effect=AssertionError("Tushare should not initialize"))
    monkeypatch.setattr(snap, "TushareProvider", boom)

    before = sorted(snap_dir.iterdir(), key=lambda x: x.name)
    r = snap.patch_v1_modules(date(2024, 6, 30), dry_run=True)

    assert r["dry_run"] is True
    assert sorted(snap_dir.iterdir(), key=lambda x: x.name) == before
    boom.assert_not_called()


def test_missing_universe_raises(fake_settings: Path) -> None:
    snap_dir = fake_settings / "snapshots" / "2024-06-30"
    snap_dir.mkdir(parents=True)
    (snap_dir / "meta.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="universe.parquet"):
        snap.patch_v1_modules(date(2024, 6, 30), dry_run=False, provider=_OkProvider())


def test_legacy_meta_preserved_and_v1_added(fake_settings: Path) -> None:
    snap_dir = fake_settings / "snapshots" / "2024-06-30"
    _write_min_snapshot(snap_dir)
    prov = _OkProvider()
    snap.patch_v1_modules(
        date(2024, 6, 30),
        dry_run=False,
        provider=prov,
        overwrite_v1_modules=True,
    )
    meta = json.loads((snap_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta.get("legacy_marker") == 42
    assert meta.get("data_expansion_version") == snap.DATA_EXPANSION_VERSION
    assert "stock_basic" in meta.get("modules", {})
    assert meta["modules"]["stock_basic"]["row_count"] == 2
    assert meta["modules"]["stock_basic"]["pit_date_column"] is None


def test_skip_hk_hold_no_provider_call(fake_settings: Path) -> None:
    snap_dir = fake_settings / "snapshots" / "2024-06-30"
    _write_min_snapshot(snap_dir)
    prov = _OkProvider()
    snap.patch_v1_modules(
        date(2024, 6, 30),
        include_hk_hold=False,
        dry_run=False,
        provider=prov,
        overwrite_v1_modules=True,
    )
    assert prov.hk_calls == 0
    assert prov.sb_calls >= 1


def test_strict_false_continues_after_stock_basic_fail(fake_settings: Path) -> None:
    snap_dir = fake_settings / "snapshots" / "2024-06-30"
    _write_min_snapshot(snap_dir)

    class P(_OkProvider):
        def get_stock_basic(self, include_delisted: bool = True) -> pd.DataFrame:  # noqa: ARG002
            self.sb_calls += 1
            raise RuntimeError("sb fail")

    prov = P()
    snap.patch_v1_modules(
        date(2024, 6, 30),
        include_margin=False,
        include_index_daily=False,
        strict=False,
        dry_run=False,
        provider=prov,
        overwrite_v1_modules=True,
    )
    assert prov.hk_calls == 2
    assert not (snap_dir / "stock_basic.parquet").is_file()
    assert (snap_dir / "hk_hold.parquet").is_file()


def test_strict_true_aborts_on_stock_basic_fail(fake_settings: Path) -> None:
    snap_dir = fake_settings / "snapshots" / "2024-06-30"
    _write_min_snapshot(snap_dir)

    class P(_OkProvider):
        def get_stock_basic(self, include_delisted: bool = True) -> pd.DataFrame:  # noqa: ARG002
            self.sb_calls += 1
            raise RuntimeError("sb fail")

    prov = P()
    with pytest.raises(RuntimeError, match="sb fail"):
        snap.patch_v1_modules(
            date(2024, 6, 30),
            include_hk_hold=True,
            strict=True,
            dry_run=False,
            provider=prov,
            overwrite_v1_modules=True,
        )
    assert prov.hk_calls == 0


def test_no_overwrite_skips_second_fetch(fake_settings: Path) -> None:
    snap_dir = fake_settings / "snapshots" / "2024-06-30"
    _write_min_snapshot(snap_dir)
    prov = _OkProvider()
    snap.patch_v1_modules(
        date(2024, 6, 30),
        include_hk_hold=False,
        include_margin=False,
        include_index_daily=False,
        dry_run=False,
        provider=prov,
        overwrite_v1_modules=True,
    )
    assert prov.sb_calls == 1
    snap.patch_v1_modules(
        date(2024, 6, 30),
        include_hk_hold=False,
        include_margin=False,
        include_index_daily=False,
        dry_run=False,
        provider=prov,
        overwrite_v1_modules=False,
    )
    assert prov.sb_calls == 1


def test_overwrite_true_refetches(fake_settings: Path) -> None:
    snap_dir = fake_settings / "snapshots" / "2024-06-30"
    _write_min_snapshot(snap_dir)
    prov = _OkProvider()
    snap.patch_v1_modules(
        date(2024, 6, 30),
        include_hk_hold=False,
        include_margin=False,
        include_index_daily=False,
        dry_run=False,
        provider=prov,
        overwrite_v1_modules=True,
    )
    snap.patch_v1_modules(
        date(2024, 6, 30),
        include_hk_hold=False,
        include_margin=False,
        include_index_daily=False,
        dry_run=False,
        provider=prov,
        overwrite_v1_modules=True,
    )
    assert prov.sb_calls == 2


def test_pit_manifest_columns(fake_settings: Path) -> None:
    snap_dir = fake_settings / "snapshots" / "2024-06-30"
    _write_min_snapshot(snap_dir)
    prov = _OkProvider()
    snap.patch_v1_modules(
        date(2024, 6, 30),
        include_stock_basic=False,
        dry_run=False,
        provider=prov,
        overwrite_v1_modules=True,
    )
    meta = json.loads((snap_dir / "meta.json").read_text(encoding="utf-8"))
    m = meta["modules"]["hk_hold"]
    assert m["pit_date_column"] == "trade_date"
    assert m["row_count"] == 2
    assert "trade_date" in m["columns"]


def test_prices_untouched(fake_settings: Path) -> None:
    snap_dir = fake_settings / "snapshots" / "2024-06-30"
    _write_min_snapshot(snap_dir)
    pq = snap_dir / "prices.parquet"
    before = pq.read_bytes()

    snap.patch_v1_modules(
        date(2024, 6, 30),
        dry_run=False,
        provider=_OkProvider(),
        overwrite_v1_modules=True,
    )
    assert pq.read_bytes() == before
