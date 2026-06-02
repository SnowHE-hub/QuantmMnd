"""PostgreSQL 连接管理 — 单例 SQLAlchemy engine，QueuePool."""
from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

# 加载 .env（幂等，多次调用无害）
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
except ImportError:
    pass

_engine = None


def get_pg_engine():
    """返回全局 SQLAlchemy engine（懒初始化，线程安全）。"""
    global _engine
    if _engine is None:
        host = os.environ.get("PG_HOST", "localhost")
        db   = os.environ.get("PG_DB",   "quantmind")
        user = os.environ.get("PG_USER", "quantmind")
        pwd  = os.environ.get("PG_PASSWORD", "")
        port = os.environ.get("PG_PORT", "5432")
        dsn = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"
        _engine = create_engine(
            dsn,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
    return _engine


def pg_execute(sql: str, params: dict | None = None):
    """快捷执行 SQL，返回 result。用于脚本/测试。"""
    with get_pg_engine().connect() as conn:
        return conn.execute(text(sql), params or {})
