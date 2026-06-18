"""quantmind.utils.silence_provider_logging — 一次性屏蔽 provider 日志，防凭证泄漏.

为什么需要
----------
Tushare / HTTP 客户端在 DEBUG/INFO 级别会把**请求 URL、query 参数**写进日志，
其中常含 API token（``?token=xxxx``）。任何走 provider 的离线/数据脚本若开了详细
日志，就可能把凭证写进 ``logs/*.log`` 或终端，进而被误提交 / 被 claude-mem 等工具收录。

用法
----
在任何会调用 Tushare / akshare / requests 的脚本**入口处调用一次**::

    from quantmind.utils.silence_provider_logging import silence_provider_logging
    silence_provider_logging()

比"每个脚本各自记得 ``logger.remove()``"可靠——单点收敛，不会按脚本重犯一遍。
"""

from __future__ import annotations

import logging

#: 可能打印含 token 的请求 URL/参数的 stdlib logger 名
_PROVIDER_LOGGERS: tuple[str, ...] = (
    "tushare",
    "akshare",
    "httpx",
    "httpcore",
    "urllib3",
    "urllib3.connectionpool",
    "requests",
    "requests.packages.urllib3",
)


def silence_provider_logging(
    *,
    drop_loguru: bool = True,
    level: int = logging.CRITICAL,
    hard_disable: bool = False,
) -> None:
    """屏蔽 provider 日志输出（幂等，可多次调用）.

    Args:
        drop_loguru: True 时调用 ``loguru.logger.remove()`` 清空所有 loguru sink。
            对一次性 provider 数据脚本是期望行为（这类脚本不需要 app 日志）；
            如果你的脚本仍需 loguru 业务日志，传 ``False`` 只屏蔽 stdlib provider logger。
        level: stdlib provider logger 抬高到的级别（默认 CRITICAL，等于静音）。
        hard_disable: True 时额外调用 ``logging.disable(level-1)`` 全局压制——
            最钝的兜底，仅在确认脚本不需要任何 stdlib 日志时使用。
    """
    # 1) stdlib：抬高级别 + 摘掉 handler + 断传播，杜绝 token 经请求日志外泄
    for name in _PROVIDER_LOGGERS:
        lg = logging.getLogger(name)
        lg.setLevel(level)
        lg.handlers.clear()
        lg.propagate = False

    # 2) loguru：清空所有 sink（provider 脚本一次性静音）
    if drop_loguru:
        try:
            from loguru import logger as _loguru_logger

            _loguru_logger.remove()
        except Exception:  # noqa: BLE001 — loguru 不存在/已清空时无需处理
            pass

    # 3) 可选全局兜底
    if hard_disable:
        logging.disable(max(0, level - 1))


__all__ = ["silence_provider_logging"]
