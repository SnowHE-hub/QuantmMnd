"""quantmind.core.logger — 基于 loguru 的统一日志.

特性：
    - 控制台彩色输出
    - 文件按日切分，保留 7 天（可配置）
    - 关键操作（数据下载、模型训练、回测、Agent 调用）单独记录到 ``operations.log``
    - 支持 ``with operation_logger.contextualize(operation="download_data"):``
    - 通过 ``get_logger(__name__)`` 拿一个绑定模块名的 logger

环境变量（覆盖默认）::

    LOG_LEVEL=INFO        # DEBUG / INFO / WARNING / ERROR / CRITICAL
    LOG_DIR=./logs        # 日志目录
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger as _loguru_logger

# loguru 的全局 logger 是单例。我们重新配置它一次，之后大家用 get_logger 拿。
_LOGGER_CONFIGURED = False

DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

OPERATIONS_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[operation]:<20} | {message}"
)


def setup_logger(
    log_dir: str | Path | None = None,
    level: str | None = None,
    rotation: str = "1 day",
    retention: str = "7 days",
    enqueue: bool = False,
    force: bool = False,
) -> None:
    """初始化全局 loguru logger（幂等，重复调用不会重复加 sink）.

    Args:
        log_dir: 日志根目录，默认 ``$LOG_DIR`` 或 ``./logs``
        level: 全局日志级别，默认 ``$LOG_LEVEL`` 或 INFO
        rotation: 文件切分策略，loguru 语法
        retention: 保留时长
        enqueue: 是否用队列异步写文件（多进程安全）
        force: 强制重置（remove 已有 sink）
    """
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED and not force:
        return

    log_dir_path = Path(log_dir or os.getenv("LOG_DIR", "./logs")).resolve()
    log_dir_path.mkdir(parents=True, exist_ok=True)
    lvl = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    _loguru_logger.remove()  # 先 remove 默认 sink

    # 1. 控制台
    _loguru_logger.add(
        sys.stderr,
        level=lvl,
        format=DEFAULT_FORMAT,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # 2. 主日志文件（全部级别 ≥ DEBUG）
    _loguru_logger.add(
        log_dir_path / "quantmind_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        enqueue=enqueue,
        format=DEFAULT_FORMAT,
        backtrace=True,
        diagnose=True,
    )

    # 3. 错误日志文件（仅 ERROR 及以上，方便监控）
    _loguru_logger.add(
        log_dir_path / "errors_{time:YYYY-MM-DD}.log",
        level="ERROR",
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        enqueue=enqueue,
        format=DEFAULT_FORMAT,
        backtrace=True,
        diagnose=True,
    )

    # 4. operations.log：仅记录 extra={"operation": ...} 的 record
    _loguru_logger.add(
        log_dir_path / "operations_{time:YYYY-MM-DD}.log",
        level="INFO",
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        enqueue=enqueue,
        format=OPERATIONS_FORMAT,
        filter=lambda record: "operation" in record["extra"],
    )

    _LOGGER_CONFIGURED = True
    _loguru_logger.debug(f"Logger initialized: level={lvl}, log_dir={log_dir_path}")


def get_logger(name: str | None = None) -> Any:
    """返回 loguru logger，若 name 给了就 bind 进 extra 方便过滤.

    Args:
        name: 模块名，通常传 ``__name__``

    Example::

        from quantmind.core.logger import get_logger
        log = get_logger(__name__)
        log.info("hello")
    """
    if not _LOGGER_CONFIGURED:
        setup_logger()
    if name:
        return _loguru_logger.bind(name=name)
    return _loguru_logger


class operation_logger:  # noqa: N801  (lowercase intentional, used as ctx manager)
    """关键操作日志的上下文管理器.

    自动记录开始/结束/耗时/异常，并写入 operations.log。

    Example::

        from quantmind.core.logger import operation_logger
        with operation_logger("download_data", universe="csi300"):
            ...  # 真正的操作

    日志会出现在 ``logs/operations_YYYY-MM-DD.log``。
    """

    def __init__(self, operation: str, **context: Any) -> None:
        if not _LOGGER_CONFIGURED:
            setup_logger()
        self.operation = operation
        self.context = context
        self._token: Any = None
        self._start_ts: float = 0.0

    def __enter__(self) -> Any:
        import time

        self._start_ts = time.monotonic()
        bound = _loguru_logger.bind(operation=self.operation, **self.context)
        bound.info(f"START {self.operation} | ctx={self.context}")
        self._bound = bound
        return bound

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        import time

        elapsed = time.monotonic() - self._start_ts
        if exc_type is None:
            self._bound.info(f"DONE  {self.operation} | elapsed={elapsed:.2f}s")
        else:
            self._bound.error(
                f"FAIL  {self.operation} | elapsed={elapsed:.2f}s | "
                f"err={exc_type.__name__}: {exc_val}"
            )


__all__ = ["get_logger", "operation_logger", "setup_logger"]


if __name__ == "__main__":
    setup_logger(level="DEBUG", force=True)
    log = get_logger(__name__)
    log.debug("debug-level message")
    log.info("info: project initializing")
    log.warning("warning example")
    log.error("error example (won't crash)")

    with operation_logger("smoke_op", target="demo"):
        log.info("doing the operation...")

    try:
        with operation_logger("intentional_fail"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    print("\nLogger demo finished. Check ./logs/ for output files.")
