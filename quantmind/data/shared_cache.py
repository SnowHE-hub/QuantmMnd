"""quantmind.data.shared_cache — 跨系统共享数据缓存层.

三个系统（选股/研究/回测）通过同一个 SharedDataCache 实例共享公共数据，
避免重复下载价格、行情、轻量基本面等共用数据集。
"""

from __future__ import annotations

import fnmatch
import json
import time
from pathlib import Path
from threading import Lock
from typing import Union

import pandas as pd
from loguru import logger

_CacheValue = Union[pd.DataFrame, dict, list]

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"


class SharedDataCache:
    """跨系统共享缓存层（单例）.

    同一进程内所有 DataProvider 共享同一实例，避免重复拉取公共数据。
    缓存存储结构：
      data/cache/shared/{key}.parquet   — DataFrame
      data/cache/shared/{key}.json      — dict/list
      data/cache/shared/{key}.meta.json — 元数据（写入时间、TTL）
    """

    _instance: "SharedDataCache | None" = None
    _lock: Lock = Lock()

    def __init__(self, cache_dir: str | Path = _DEFAULT_CACHE_DIR) -> None:
        self._root = Path(cache_dir) / "shared"
        self._root.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0
        self._mem: dict[str, tuple[_CacheValue, float, int]] = {}  # key→(data, ts, ttl_h)

    # ── 单例获取 ─────────────────────────────────────────────────────────────

    @classmethod
    def get_instance(
        cls, cache_dir: str | Path = _DEFAULT_CACHE_DIR
    ) -> "SharedDataCache":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(cache_dir)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """测试用：重置单例."""
        with cls._lock:
            cls._instance = None

    # ── 读写接口 ─────────────────────────────────────────────────────────────

    def read(self, key: str) -> _CacheValue | None:
        """读取缓存，返回 None 表示未命中或已过期."""
        # 1. 内存缓存
        if key in self._mem:
            data, ts, ttl_h = self._mem[key]
            if time.time() - ts < ttl_h * 3600:
                self._hits += 1
                return data
            else:
                del self._mem[key]

        # 2. 磁盘缓存
        meta_path = self._root / f"{key}.meta.json"
        if not meta_path.exists():
            self._misses += 1
            return None

        try:
            meta = json.loads(meta_path.read_text())
            written_at = meta.get("written_at", 0)
            ttl_h = meta.get("ttl_hours", 24)
            if time.time() - written_at > ttl_h * 3600:
                self._misses += 1
                return None

            # 读数据文件
            parquet_path = self._root / f"{key}.parquet"
            json_path = self._root / f"{key}.json"
            if parquet_path.exists():
                data = pd.read_parquet(parquet_path)
            elif json_path.exists():
                data = json.loads(json_path.read_text())
            else:
                self._misses += 1
                return None

            self._mem[key] = (data, written_at, ttl_h)
            self._hits += 1
            return data
        except Exception as e:
            logger.debug(f"[SharedCache] 磁盘读取失败 {key}: {e}")
            self._misses += 1
            return None

    def write(
        self,
        key: str,
        data: _CacheValue,
        ttl_hours: int = 24,
    ) -> None:
        """写入缓存，同时写磁盘和内存."""
        now = time.time()
        try:
            meta = {"written_at": now, "ttl_hours": ttl_hours, "key": key}
            meta_path = self._root / f"{key}.meta.json"
            meta_path.write_text(json.dumps(meta))

            if isinstance(data, pd.DataFrame):
                (self._root / f"{key}.parquet").write_bytes(
                    data.to_parquet(index=True)
                )
            else:
                (self._root / f"{key}.json").write_text(
                    json.dumps(data, ensure_ascii=False, default=str)
                )

            self._mem[key] = (data, now, ttl_hours)
        except Exception as e:
            logger.warning(f"[SharedCache] 写入失败 {key}: {e}")

    def invalidate(self, pattern: str) -> int:
        """手动使匹配 pattern 的缓存失效（支持 shell 风格通配符），返回失效条数."""
        count = 0
        for meta_path in self._root.glob("*.meta.json"):
            key = meta_path.stem.removesuffix(".meta")
            if fnmatch.fnmatch(key, pattern):
                meta_path.unlink(missing_ok=True)
                (self._root / f"{key}.parquet").unlink(missing_ok=True)
                (self._root / f"{key}.json").unlink(missing_ok=True)
                self._mem.pop(key, None)
                count += 1
        logger.info(f"[SharedCache] 使 {count} 条缓存失效（pattern={pattern!r}）")
        return count

    # ── 统计 ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """返回缓存统计."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        # 计算磁盘占用
        size_bytes = sum(
            f.stat().st_size for f in self._root.iterdir() if f.is_file()
        )

        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "size_mb": round(size_bytes / 1024 / 1024, 2),
            "mem_entries": len(self._mem),
        }
