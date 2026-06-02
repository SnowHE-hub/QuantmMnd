"""MongoDB 连接管理 — 单例 MongoClient，返回 quantmind 数据库。"""
from __future__ import annotations

import os
from pathlib import Path

from pymongo import MongoClient

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
except ImportError:
    pass

_client: MongoClient | None = None


def get_mongo_db():
    """返回 quantmind MongoDB 数据库对象（懒初始化）。"""
    global _client
    if _client is None:
        uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/quantmind")
        _client = MongoClient(
            uri,
            maxPoolSize=10,
            serverSelectionTimeoutMS=5000,
        )
    db_name = os.environ.get("MONGO_DB", "quantmind")
    return _client[db_name]


def get_mongo_collection(name: str):
    """快捷获取 collection。用于脚本/测试。"""
    return get_mongo_db()[name]
