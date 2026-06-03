"""
tests/test_data_writer.py — DataWriter 单元测试

覆盖：
  - parquet_only 模式：DB 不被写
  - dual 模式：parquet 和 DB 都被写
  - DB 写失败时 parquet 不受影响（失败隔离）
  - 各 write_* 方法的 upsert 幂等性
  - 结构一致性（DB 内容 == 写入内容）

运行：
  conda run -n quantmind pytest tests/test_data_writer.py -v
"""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def writer_parquet_only():
    from app.db.writers import DataWriter
    return DataWriter(mode="parquet_only")


@pytest.fixture
def writer_dual():
    from app.db.writers import DataWriter
    return DataWriter(mode="dual")


@pytest.fixture
def sample_recommendations():
    return {
        "as_of": "2026-06-02",
        "generated_at": "2026-06-02T16:00:00",
        "top10": [
            {
                "ticker": "000001.SZ",
                "name": "平安银行",
                "industry": "银行",
                "lgbm_rank": 1,
                "lgbm_score": 0.98,
                "entry_price": 10.5,
                "reason": "测试",
            },
            {
                "ticker": "600519.SH",
                "name": "贵州茅台",
                "industry": "白酒",
                "lgbm_rank": 2,
                "lgbm_score": 0.96,
                "entry_price": 1500.0,
                "reason": "测试",
            },
        ],
        "market_summary": "测试推荐",
    }


@pytest.fixture
def sample_pnl_df():
    return pd.DataFrame([
        {
            "as_of_date": pd.Timestamp("2026-01-01"),
            "ticker": "000001.SZ",
            "entry_date": "2026-01-01",
            "exit_date": "2026-04-01",
            "holding_days": 90,
            "predicted_rank": 1,
            "predicted_score": 0.95,
            "entry_price": 10.0,
            "exit_price": 11.0,
            "actual_return_63d": 0.10,
            "key_factors": "{}",
            "actual_rank": 1,
            "pnl_vs_median": 0.05,
            "hit": True,
            "panel_return_63d": 0.08,
            "return_diff": 0.02,
        }
    ])


@pytest.fixture
def sample_positions():
    return [
        {
            "as_of": "2026-06-02",
            "ticker": "000001.SZ",
            "predicted_rank": 1,
            "predicted_score": 0.98,
            "entry_price": 10.5,
            "status": "OPEN",
        },
        {
            "as_of": "2026-06-02",
            "ticker": "600519.SH",
            "predicted_rank": 2,
            "predicted_score": 0.96,
            "entry_price": 1500.0,
            "status": "OPEN",
        },
    ]


@pytest.fixture
def sample_loss_signals():
    return {
        "run_ts": "2026-06-02T10:00:00",
        "signal_1_ranking_loss": {"value": 0.5, "alert": False},
        "overall_health": "HEALTHY",
    }


@pytest.fixture
def sample_strategies():
    return [
        {
            "ticker": "000001.SZ",
            "rating": "BUY",
            "composite_signal": 0.75,
            "confidence": 0.85,
            "agent_signals": {"ValuationAgent": {"signal": 0.8}},
        }
    ]


# ── Mode 测试 ─────────────────────────────────────────────────────────────────

class TestWriterMode:
    def test_parquet_only_mode(self, writer_parquet_only):
        assert writer_parquet_only.mode == "parquet_only"
        assert writer_parquet_only._parquet_enabled
        assert not writer_parquet_only._db_enabled

    def test_dual_mode(self, writer_dual):
        assert writer_dual.mode == "dual"
        assert writer_dual._parquet_enabled
        assert writer_dual._db_enabled

    def test_db_only_mode(self):
        from app.db.writers import DataWriter
        w = DataWriter(mode="db_only")
        assert not w._parquet_enabled
        assert w._db_enabled

    def test_invalid_mode_defaults_to_parquet_only(self):
        from app.db.writers import DataWriter
        w = DataWriter(mode="invalid_mode")
        assert w.mode == "parquet_only"

    def test_env_var_mode(self, monkeypatch):
        monkeypatch.setenv("WRITE_MODE", "dual")
        from app.db.writers import DataWriter
        w = DataWriter()
        assert w.mode == "dual"


# ── DB 不写测试（parquet_only 模式）─────────────────────────────────────────

class TestParquetOnlyNoDB:
    def test_recommendations_no_db_call(self, writer_parquet_only, sample_recommendations):
        with patch.object(writer_parquet_only, '_mongo') as mock_mongo:
            writer_parquet_only.write_recommendations("2026-06-02", sample_recommendations)
            mock_mongo.assert_not_called()

    def test_realized_pnl_no_db_call(self, writer_parquet_only, sample_pnl_df):
        with patch.object(writer_parquet_only, '_pg') as mock_pg:
            writer_parquet_only.write_realized_pnl(sample_pnl_df)
            mock_pg.assert_not_called()

    def test_forward_positions_no_db_call(self, writer_parquet_only, sample_positions):
        with patch.object(writer_parquet_only, '_mongo') as mock_mongo:
            writer_parquet_only.write_forward_positions(sample_positions)
            mock_mongo.assert_not_called()

    def test_loss_signals_no_db_call(self, writer_parquet_only, sample_loss_signals):
        with patch.object(writer_parquet_only, '_mongo') as mock_mongo:
            writer_parquet_only.write_loss_signals(sample_loss_signals)
            mock_mongo.assert_not_called()


# ── 失败隔离测试（DB 失败，parquet 不受影响）─────────────────────────────────

class TestFailureIsolation:
    @pytest.fixture(autouse=True)
    def _isolate_logs(self, tmp_path, monkeypatch):
        """把双写失败/审计日志重定向到 tmp，避免本类的合成异常
        （'PG down' / 'Mongo down' / 'DB down'）污染真实
        logs/db_write_failures.log（监控页据此判断"生产失败"）。"""
        import app.db.writers as w
        monkeypatch.setattr(w, "_LOG_DIR", tmp_path)
        monkeypatch.setattr(w, "_FAILURE_LOG", tmp_path / "db_write_failures.log")
        monkeypatch.setattr(w, "_AUDIT_LOG", tmp_path / "db_write_audit.log")

    def test_db_failure_does_not_raise(self, writer_dual, sample_recommendations):
        """DB 写失败时 write_recommendations 不抛异常。"""
        with patch.object(writer_dual, '_mongo', side_effect=Exception("DB down")):
            # 不应抛错
            writer_dual.write_recommendations("2026-06-02", sample_recommendations)

    def test_pnl_db_failure_does_not_raise(self, writer_dual, sample_pnl_df):
        with patch.object(writer_dual, '_pg', side_effect=Exception("PG down")):
            writer_dual.write_realized_pnl(sample_pnl_df)  # 不应抛错

    def test_loss_signals_db_failure_does_not_raise(self, writer_dual, sample_loss_signals):
        with patch.object(writer_dual, '_mongo', side_effect=Exception("Mongo down")):
            writer_dual.write_loss_signals(sample_loss_signals)  # 不应抛错

    def test_agent_analysis_db_failure_does_not_raise(self, writer_dual, sample_strategies):
        with patch.object(writer_dual, '_mongo', side_effect=Exception("Mongo down")):
            writer_dual.write_agent_analysis("2026-06-02", sample_strategies)  # 不应抛错


# ── 实际 DB 写入测试（需要 DB 连接）─────────────────────────────────────────

@pytest.mark.integration
class TestDualWriteIntegration:
    """双写写入实际数据库并验证内容一致。需要 PG + Mongo 服务运行。"""

    def test_recommendations_round_trip(self, writer_dual, sample_recommendations):
        date_str = "2026-06-02-test"
        writer_dual.write_recommendations(date_str, sample_recommendations)
        # 验证 Mongo
        coll = writer_dual._mongo("recommendations")
        doc = coll.find_one({"_id": date_str})
        assert doc is not None, f"MongoDB 中找不到 _id={date_str}"
        assert len(doc.get("top10", [])) == len(sample_recommendations["top10"])
        assert doc["as_of"] == date_str
        # 清理
        coll.delete_one({"_id": date_str})

    def test_realized_pnl_round_trip(self, writer_dual, sample_pnl_df):
        from sqlalchemy import text
        writer_dual.write_realized_pnl(sample_pnl_df, full_replace=True)
        # 验证 PG
        eng = writer_dual._pg()
        with eng.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM realized_pnl")).scalar()
        assert count == len(sample_pnl_df), f"PG 行数不一致: {count} != {len(sample_pnl_df)}"

    def test_forward_positions_upsert_idempotent(self, writer_dual, sample_positions):
        # 写两次，结果行数不变（upsert）
        writer_dual.write_forward_positions(sample_positions)
        writer_dual.write_forward_positions(sample_positions)
        coll = writer_dual._mongo("positions")
        # 查询这两个 ticker
        tickers = [p["ticker"] for p in sample_positions]
        count = coll.count_documents({"ticker": {"$in": tickers}, "as_of": "2026-06-02"})
        assert count == len(sample_positions), f"期望 {len(sample_positions)} 条，实际 {count}"

    def test_loss_signals_round_trip(self, writer_dual, sample_loss_signals):
        writer_dual.write_loss_signals(sample_loss_signals)
        coll = writer_dual._mongo("loss_signals")
        doc = coll.find_one({"_id": "2026-06-02"})
        assert doc is not None
        assert doc.get("overall_health") == "HEALTHY"

    def test_agent_analysis_round_trip(self, writer_dual, sample_strategies):
        writer_dual.write_agent_analysis("2026-06-02", sample_strategies)
        coll = writer_dual._mongo("agent_analysis")
        doc = coll.find_one({"_id": "2026-06-02_000001.SZ"})
        assert doc is not None
        assert doc.get("rating") == "BUY"
        # 清理
        coll.delete_one({"_id": "2026-06-02_000001.SZ"})

    def test_strategy_config_round_trip(self, writer_dual):
        config = {"version": "test_v1", "holding_period": {"recommended": "3m"}}
        writer_dual.write_strategy_config(config, version="test_v1")
        coll = writer_dual._mongo("strategy_config")
        doc = coll.find_one({"_id": "test_v1"})
        assert doc is not None
        assert doc["version"] == "test_v1"
        # 清理
        coll.delete_one({"_id": "test_v1"})

    def test_recommendations_upsert_idempotent(self, writer_dual, sample_recommendations):
        date_str = "2026-06-02-idem"
        writer_dual.write_recommendations(date_str, sample_recommendations)
        writer_dual.write_recommendations(date_str, sample_recommendations)
        coll = writer_dual._mongo("recommendations")
        count = coll.count_documents({"_id": date_str})
        assert count == 1, f"重复写入后文档数应为 1，实际为 {count}"
        # 清理
        coll.delete_one({"_id": date_str})


# ── get_writer 单例测试 ──────────────────────────────────────────────────────

class TestGetWriterSingleton:
    def test_singleton_same_mode(self):
        from app.db.writers import get_writer, _writer
        import app.db.writers as w_mod
        w_mod._writer = None  # 重置
        w1 = get_writer("parquet_only")
        w2 = get_writer("parquet_only")
        assert w1 is w2

    def test_new_instance_on_mode_change(self):
        from app.db.writers import get_writer
        import app.db.writers as w_mod
        w_mod._writer = None
        w1 = get_writer("parquet_only")
        w2 = get_writer("dual")
        assert w1 is not w2
        assert w2.mode == "dual"
