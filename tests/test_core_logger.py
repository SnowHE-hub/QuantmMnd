"""测试 quantmind.core.logger."""

from __future__ import annotations

from pathlib import Path

import pytest

from quantmind.core.logger import get_logger, operation_logger, setup_logger


class TestSetupLogger:
    def test_setup_creates_log_dir(self, tmp_path: Path) -> None:
        setup_logger(log_dir=tmp_path / "logs", level="DEBUG", force=True)
        log_dir = tmp_path / "logs"
        assert log_dir.is_dir()

    def test_idempotent(self, tmp_path: Path) -> None:
        setup_logger(log_dir=tmp_path / "logs", force=True)
        setup_logger(log_dir=tmp_path / "logs", force=False)  # should be no-op
        # 不抛异常即视为通过

    def test_get_logger_with_name(self, tmp_path: Path) -> None:
        setup_logger(log_dir=tmp_path / "logs", force=True)
        log = get_logger("test_module")
        log.info("hello from named logger")


class TestOperationLogger:
    def test_context_manager_success(self, tmp_path: Path) -> None:
        setup_logger(log_dir=tmp_path / "logs", level="INFO", force=True)
        with operation_logger("test_op", target="x") as op_log:
            op_log.info("doing work")

        # operations.log 应被写入
        ops_files = list((tmp_path / "logs").glob("operations_*.log"))
        assert len(ops_files) == 1
        content = ops_files[0].read_text(encoding="utf-8")
        assert "START test_op" in content
        assert "DONE  test_op" in content

    def test_context_manager_exception(self, tmp_path: Path) -> None:
        setup_logger(log_dir=tmp_path / "logs", level="INFO", force=True)
        with pytest.raises(RuntimeError), operation_logger("fail_op"):
            raise RuntimeError("boom")

        ops_files = list((tmp_path / "logs").glob("operations_*.log"))
        content = ops_files[0].read_text(encoding="utf-8")
        assert "FAIL  fail_op" in content
        assert "RuntimeError" in content

    def test_no_op_context_in_main_log(self, tmp_path: Path) -> None:
        """普通 log 不应该出现在 operations.log."""
        setup_logger(log_dir=tmp_path / "logs", level="INFO", force=True)
        log = get_logger("plain")
        log.info("普通日志，不应进 operations.log")
        with operation_logger("only_this"):
            pass

        ops_content = next((tmp_path / "logs").glob("operations_*.log")).read_text(encoding="utf-8")
        assert "普通日志" not in ops_content
        assert "only_this" in ops_content
