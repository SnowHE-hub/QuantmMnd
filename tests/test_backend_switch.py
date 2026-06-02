"""
tests/test_backend_switch.py — Backend 切换与双写监控工具测试

覆盖：
  - app.ops.db_health 各函数返回结构正确
  - .env 读写不破坏其他行（保序、保注释）
  - 失败/审计日志解析正确
  - parity 校验子进程能正常调用
  - overall_db_status 综合状态包含所有必需字段
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── env 读写 ────────────────────────────────────────────────────────────────

class TestEnvReadWrite:
    def test_read_existing_env(self):
        from app.ops.db_health import read_env_value
        # DATA_BACKEND 已在 E1 阶段加入 .env
        val = read_env_value("DATA_BACKEND")
        assert val in ("parquet", "postgres"), f"读到的 DATA_BACKEND={val}"

    def test_read_nonexistent_key(self):
        from app.ops.db_health import read_env_value
        val = read_env_value("THIS_KEY_DOES_NOT_EXIST_XYZ")
        assert val is None

    def test_write_and_restore(self, tmp_path, monkeypatch):
        """写入后恢复原值，确保不污染 .env。"""
        import app.ops.db_health as mod
        # 用临时 .env 测试
        tmp_env = tmp_path / ".env"
        tmp_env.write_text(
            "# comment\nFOO=bar\nDATA_BACKEND=parquet\n# trailing comment\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(mod, "ENV_PATH", tmp_env)

        assert mod.read_env_value("DATA_BACKEND") == "parquet"
        assert mod.write_env_value("DATA_BACKEND", "postgres")
        assert mod.read_env_value("DATA_BACKEND") == "postgres"

        # 其他行保留
        content = tmp_env.read_text(encoding="utf-8")
        assert "# comment" in content
        assert "FOO=bar" in content
        assert "# trailing comment" in content


# ── DB ping ─────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestDBPing:
    def test_pg_ping_returns_structured_dict(self):
        from app.ops.db_health import pg_ping
        result = pg_ping()
        assert "ok" in result
        if result["ok"]:
            assert "tables" in result
            assert "n_tables" in result
            assert "total_rows" in result
            assert result["n_tables"] > 0

    def test_mongo_ping_returns_structured_dict(self):
        from app.ops.db_health import mongo_ping
        result = mongo_ping()
        assert "ok" in result
        if result["ok"]:
            assert "collections" in result
            assert "n_collections" in result
            assert "db_name" in result


# ── 日志解析 ────────────────────────────────────────────────────────────────

class TestLogParsing:
    def test_parse_audit_log_empty(self, tmp_path, monkeypatch):
        import app.ops.db_health as mod
        monkeypatch.setattr(mod, "AUDIT_LOG", tmp_path / "audit.log")
        events = mod.read_audit(hours=24)
        assert events == []

    def test_parse_audit_log_with_entries(self, tmp_path, monkeypatch):
        import app.ops.db_health as mod
        log_file = tmp_path / "audit.log"
        now = datetime.now()
        # 写入 3 条审计记录（含 1 条 25 小时前的）
        lines = [
            f"{(now - timedelta(hours=1)).isoformat(timespec='seconds')}\trecommendations\tOK\tdate=2026-06-01",
            f"{(now - timedelta(hours=2)).isoformat(timespec='seconds')}\trealized_pnl\tOK\trows=80",
            f"{(now - timedelta(hours=25)).isoformat(timespec='seconds')}\told_event\tOK\tstale",
        ]
        log_file.write_text("\n".join(lines), encoding="utf-8")
        monkeypatch.setattr(mod, "AUDIT_LOG", log_file)

        events = mod.read_audit(hours=24)
        assert len(events) == 2  # 25h 那条被过滤
        names = [e["name"] for e in events]
        assert "recommendations" in names
        assert "realized_pnl" in names
        assert "old_event" not in names

    def test_parse_failure_log(self, tmp_path, monkeypatch):
        import app.ops.db_health as mod
        log_file = tmp_path / "failures.log"
        log_file.write_text(
            f"{datetime.now().isoformat(timespec='seconds')}\trealized_pnl\tConnectionError\tPG down\trows=80",
            encoding="utf-8",
        )
        monkeypatch.setattr(mod, "FAILURE_LOG", log_file)

        fails = mod.read_failures(hours=24)
        assert len(fails) == 1
        assert fails[0]["name"] == "realized_pnl"
        assert "ConnectionError" in fails[0]["status"]

    def test_skip_malformed_lines(self, tmp_path, monkeypatch):
        import app.ops.db_health as mod
        log_file = tmp_path / "audit.log"
        log_file.write_text(
            "this is garbage\n"
            "no\ttabs\n"  # 不足 3 列
            f"{datetime.now().isoformat()}\tok_event\tOK\tinfo",
            encoding="utf-8",
        )
        monkeypatch.setattr(mod, "AUDIT_LOG", log_file)
        events = mod.read_audit(hours=24)
        assert len(events) == 1
        assert events[0]["name"] == "ok_event"


# ── dual_write_stats ────────────────────────────────────────────────────────

class TestDualWriteStats:
    def test_stats_with_no_logs(self, tmp_path, monkeypatch):
        import app.ops.db_health as mod
        monkeypatch.setattr(mod, "AUDIT_LOG", tmp_path / "x.log")
        monkeypatch.setattr(mod, "FAILURE_LOG", tmp_path / "y.log")
        stats = mod.dual_write_stats(days=7)
        assert stats["total_ok"] == 0
        assert stats["total_fail"] == 0
        assert stats["success_rate"] is None
        assert stats["by_date"] == []

    def test_stats_aggregate(self, tmp_path, monkeypatch):
        import app.ops.db_health as mod
        now = datetime.now()
        audit_lines = []
        fail_lines = []
        for i in range(5):
            audit_lines.append(
                f"{(now - timedelta(hours=i)).isoformat(timespec='seconds')}"
                f"\trecommendations\tOK\trun{i}")
        fail_lines.append(
            f"{now.isoformat(timespec='seconds')}\trealized_pnl\tConnError\tboom\t")

        (tmp_path / "a.log").write_text("\n".join(audit_lines), encoding="utf-8")
        (tmp_path / "f.log").write_text("\n".join(fail_lines), encoding="utf-8")
        monkeypatch.setattr(mod, "AUDIT_LOG", tmp_path / "a.log")
        monkeypatch.setattr(mod, "FAILURE_LOG", tmp_path / "f.log")

        stats = mod.dual_write_stats(days=7)
        assert stats["total_ok"] == 5
        assert stats["total_fail"] == 1
        assert stats["success_rate"] == pytest.approx(5 / 6, abs=1e-4)
        # by_writer 应有 2 个 writer
        writers = {r["writer"] for r in stats["by_writer"]}
        assert "recommendations" in writers
        assert "realized_pnl" in writers


# ── DataWriter 失败日志集成 ─────────────────────────────────────────────────

class TestDataWriterLogging:
    def test_failure_log_written_on_db_error(self, tmp_path, monkeypatch):
        """触发 DB 失败时确认 _FAILURE_LOG 被写入。"""
        import app.db.writers as wmod
        from unittest.mock import patch

        # 重定向日志文件到临时路径
        monkeypatch.setattr(wmod, "_FAILURE_LOG", tmp_path / "failures.log")
        monkeypatch.setattr(wmod, "_AUDIT_LOG", tmp_path / "audit.log")
        monkeypatch.setattr(wmod, "_LOG_DIR", tmp_path)

        w = wmod.DataWriter(mode="dual")
        # mock _mongo 抛错
        with patch.object(w, "_mongo", side_effect=RuntimeError("simulated")):
            w.write_recommendations("2026-06-02", {"top10": []})

        # 失败日志应有 1 条
        assert (tmp_path / "failures.log").exists()
        content = (tmp_path / "failures.log").read_text(encoding="utf-8")
        assert "recommendations" in content
        assert "RuntimeError" in content
        assert "simulated" in content

    def test_success_log_written_on_db_ok(self, tmp_path, monkeypatch):
        """触发 DB 成功时 _AUDIT_LOG 被写入。"""
        import app.db.writers as wmod
        from unittest.mock import MagicMock, patch

        monkeypatch.setattr(wmod, "_FAILURE_LOG", tmp_path / "failures.log")
        monkeypatch.setattr(wmod, "_AUDIT_LOG", tmp_path / "audit.log")
        monkeypatch.setattr(wmod, "_LOG_DIR", tmp_path)

        w = wmod.DataWriter(mode="dual")
        fake_coll = MagicMock()
        fake_coll.replace_one = MagicMock(return_value=None)
        with patch.object(w, "_mongo", return_value=fake_coll):
            w.write_recommendations("2026-06-02", {"top10": [{"ticker": "X"}]})

        assert (tmp_path / "audit.log").exists()
        content = (tmp_path / "audit.log").read_text(encoding="utf-8")
        assert "recommendations" in content
        assert "OK" in content


# ── parity check 调用 ──────────────────────────────────────────────────────

@pytest.mark.integration
class TestParityCheckRunner:
    def test_run_parity_returns_structured(self):
        from app.ops.db_health import run_parity_check
        result = run_parity_check(timeout_sec=120)
        assert "passed" in result
        assert "failed" in result
        assert "total" in result
        assert "output" in result
        assert "duration_sec" in result


# ── overall_db_status 综合 ──────────────────────────────────────────────────

class TestOverallStatus:
    def test_overall_status_keys(self):
        from app.ops.db_health import overall_db_status
        status = overall_db_status()
        assert "data_backend" in status
        assert "write_mode" in status
        assert "pg" in status
        assert "mongo" in status
        assert "failures_24h" in status
        assert isinstance(status["pg"], dict)
        assert isinstance(status["mongo"], dict)
        assert isinstance(status["failures_24h"], int)
