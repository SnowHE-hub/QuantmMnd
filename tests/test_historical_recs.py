"""tests/test_historical_recs.py — 历史推荐数据层单元测试 (≥12).

所有测试均使用临时目录中的假数据，不依赖真实 data/ 目录。
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
import pytest


# ── 测试夹具 ──────────────────────────────────────────────────────────────────

@pytest.fixture
def rec_dir(tmp_path: Path) -> Path:
    """构造 2 天的推荐 JSON."""
    d = tmp_path / "recommendations"
    d.mkdir()

    for date_str, tickers in [
        ("2026-05-26", [("600707.SH", "A公司"), ("000001.SZ", "B公司")]),
        ("2026-05-27", [("300139.SZ", "C公司"), ("600000.SH", "D公司")]),
    ]:
        items = [
            {
                "rank": i + 1,
                "ticker": t,
                "name": n,
                "lgbm_score": round(0.9 - i * 0.1, 2),
                "lgbm_rank": i + 1,
                "reason": f"理由 {i}",
                "agent_recommendation": "强烈买入" if i == 0 else "买入",
                "industry": "科技" if i % 2 == 0 else "金融",
            }
            for i, (t, n) in enumerate(tickers)
        ]
        rec = {
            "as_of": date_str,
            "generated_at": f"{date_str}T16:00:00",
            "top10": items,
            "market_summary": f"{date_str} 市场概况",
        }
        (d / f"{date_str}.json").write_text(json.dumps(rec), encoding="utf-8")

    return d


@pytest.fixture
def pnl_path(tmp_path: Path) -> Path:
    """构造 realized_pnl.parquet."""
    df = pd.DataFrame({
        "as_of_date":        ["2026-05-26", "2026-05-26"],
        "ticker":            ["600707.SH",  "000001.SZ"],
        "entry_date":        ["2026-05-26", "2026-05-26"],
        "exit_date":         ["2026-08-24", "2026-08-24"],
        "holding_days":      [90, 90],
        "predicted_rank":    [1, 2],
        "predicted_score":   [0.9, 0.8],
        "entry_price":       [10.0, 8.0],
        "exit_price":        [12.5, 7.5],
        "actual_return_63d": [0.25, -0.0625],
        "key_factors":       ["{}", "{}"],
        "actual_rank":       [1, 3],
        "pnl_vs_median":     [0.15, -0.05],
        "hit":               [True, False],
        "panel_return_63d":  [0.10, 0.10],
        "return_diff":       [0.15, -0.16],
    })
    p = tmp_path / "realized_pnl.parquet"
    df.to_parquet(p, index=False)
    return p


@pytest.fixture
def fwd_path(tmp_path: Path) -> Path:
    """构造 forward_positions.json."""
    data = {
        "generated_at": "2026-05-27T17:00:00",
        "tracking_horizon": "3m",
        "n_open_positions": 2,
        "positions": [
            {
                "as_of": "2026-05-27",
                "ticker": "300139.SZ",
                "predicted_rank": 1,
                "holding_period": "3m",
                "estimated_exit_date": "2026-08-25",
                "entry_price": None,
                "exit_price": None,
                "actual_return": None,
                "status": "OPEN",
            },
        ],
    }
    p = tmp_path / "forward_positions.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ── 测试：load_all_recommendations ───────────────────────────────────────────

class TestLoadAllRecommendations:

    def test_returns_list(self, rec_dir):
        from app.utils.rec_data import load_all_recommendations
        recs = load_all_recommendations(rec_dir)
        assert isinstance(recs, list)

    def test_count_matches_files(self, rec_dir):
        from app.utils.rec_data import load_all_recommendations
        recs = load_all_recommendations(rec_dir)
        assert len(recs) == 2

    def test_sorted_descending(self, rec_dir):
        from app.utils.rec_data import load_all_recommendations
        recs = load_all_recommendations(rec_dir)
        dates = [r["as_of"] for r in recs]
        assert dates == sorted(dates, reverse=True)

    def test_top10_normalized_fields(self, rec_dir):
        from app.utils.rec_data import load_all_recommendations
        recs = load_all_recommendations(rec_dir)
        for rec in recs:
            for item in rec["top10"]:
                assert "ticker" in item
                assert "rank" in item
                assert "lgbm_score" in item

    def test_empty_dir_returns_empty_list(self, tmp_path):
        from app.utils.rec_data import load_all_recommendations
        empty = tmp_path / "no_recs"
        empty.mkdir()
        assert load_all_recommendations(empty) == []

    def test_missing_dir_returns_empty_list(self, tmp_path):
        from app.utils.rec_data import load_all_recommendations
        assert load_all_recommendations(tmp_path / "nonexistent") == []


# ── 测试：load_realized_pnl ───────────────────────────────────────────────────

class TestLoadRealizedPnl:

    def test_returns_dataframe(self, pnl_path):
        from app.utils.rec_data import load_realized_pnl
        df = load_realized_pnl(pnl_path)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns_present(self, pnl_path):
        from app.utils.rec_data import load_realized_pnl
        df = load_realized_pnl(pnl_path)
        for col in ("as_of_date", "ticker", "actual_return_63d", "hit"):
            assert col in df.columns

    def test_missing_file_returns_none(self, tmp_path):
        from app.utils.rec_data import load_realized_pnl
        assert load_realized_pnl(tmp_path / "nofile.parquet") is None


# ── 测试：load_forward_positions ─────────────────────────────────────────────

class TestLoadForwardPositions:

    def test_returns_list(self, fwd_path):
        from app.utils.rec_data import load_forward_positions
        pos = load_forward_positions(fwd_path)
        assert isinstance(pos, list)

    def test_positions_have_ticker(self, fwd_path):
        from app.utils.rec_data import load_forward_positions
        pos = load_forward_positions(fwd_path)
        assert len(pos) >= 1
        assert all("ticker" in p for p in pos)

    def test_missing_file_returns_empty(self, tmp_path):
        from app.utils.rec_data import load_forward_positions
        assert load_forward_positions(tmp_path / "nofile.json") == []


# ── 测试：build_flat_records ──────────────────────────────────────────────────

class TestBuildFlatRecords:

    def _load_all(self, rec_dir, pnl_path, fwd_path):
        from app.utils.rec_data import (
            load_all_recommendations, load_realized_pnl,
            load_forward_positions, build_flat_records,
        )
        recs   = load_all_recommendations(rec_dir)
        pnl    = load_realized_pnl(pnl_path)
        fwd    = load_forward_positions(fwd_path)
        flat   = build_flat_records(recs, pnl, fwd, {}, {}, {})
        return flat

    def test_returns_list(self, rec_dir, pnl_path, fwd_path):
        flat = self._load_all(rec_dir, pnl_path, fwd_path)
        assert isinstance(flat, list)

    def test_count_equals_total_top10(self, rec_dir, pnl_path, fwd_path):
        # 2 days × 2 tickers each = 4
        flat = self._load_all(rec_dir, pnl_path, fwd_path)
        assert len(flat) == 4

    def test_settled_status_for_pnl_hit(self, rec_dir, pnl_path, fwd_path):
        flat = self._load_all(rec_dir, pnl_path, fwd_path)
        settled = [r for r in flat if "已结算盈利" in r["状态"]]
        assert len(settled) >= 1

    def test_open_status_for_forward_position(self, rec_dir, pnl_path, fwd_path):
        flat = self._load_all(rec_dir, pnl_path, fwd_path)
        open_pos = [r for r in flat if r["状态"] == "持仓中"]
        assert len(open_pos) >= 1


# ── 测试：compute_summary_stats ───────────────────────────────────────────────

class TestComputeSummaryStats:

    def test_empty_returns_zero_total(self):
        from app.utils.rec_data import compute_summary_stats
        stats = compute_summary_stats([])
        assert stats["total_recs"] == 0

    def test_with_settled_computes_win_rate(self):
        from app.utils.rec_data import compute_summary_stats
        records = [
            {"状态": "已结算盈利 ✅", "3m实际收益%": 10.0},
            {"状态": "已结算亏损 ❌", "3m实际收益%": -5.0},
            {"状态": "已结算盈利 ✅", "3m实际收益%": 8.0},
        ]
        stats = compute_summary_stats(records)
        assert abs(stats["win_rate_3m"] - 2 / 3) < 0.01
        assert stats["n_settled"] == 3
        assert abs(stats["avg_return_3m"] - (10.0 - 5.0 + 8.0) / 3) < 0.01

    def test_with_open_positions_counted(self):
        from app.utils.rec_data import compute_summary_stats
        records = [
            {"状态": "持仓中", "3m实际收益%": None},
            {"状态": "持仓中", "3m实际收益%": None},
        ]
        stats = compute_summary_stats(records)
        assert stats["n_open"] == 2
        assert stats["win_rate_3m"] is None


# ── 测试：load_name_map ───────────────────────────────────────────────────────

class TestLoadNameMap:

    def test_returns_dict(self, tmp_path):
        from app.utils.rec_data import load_name_map
        nm = load_name_map(tmp_path)
        assert isinstance(nm, dict)

    def test_fallback_to_empty_when_no_stock_basic(self, tmp_path):
        from app.utils.rec_data import load_name_map
        # No stock_basic file → empty dict (or dict from forward_positions)
        nm = load_name_map(tmp_path)
        assert isinstance(nm, dict)
