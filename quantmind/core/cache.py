"""quantmind.core.cache — 装饰器化磁盘缓存.

后端：``diskcache`` + 多种序列化（pickle / parquet / json），按数据类型自动选。

设计目标：
    1. 一行装饰加缓存：``@cached(ttl_hours=24)``
    2. key 自动从函数名 + 参数 hash 出来
    3. 大 DataFrame 走 parquet（小、快），其他走 pickle
    4. 支持手动失效：``cached_invalidate(my_func, *args, **kwargs)``
    5. 全局清缓存：``clear_cache()``

用法::

    from quantmind.core.cache import cached

    @cached(ttl_hours=24)
    def expensive_call(ticker: str, as_of: str) -> dict:
        ...

    expensive_call("300750.SZ", "2024-06-30")  # 首次：真跑
    expensive_call("300750.SZ", "2024-06-30")  # 命中缓存
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import pickle
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypeVar

import diskcache  # type: ignore[import-untyped]

from quantmind.core.logger import get_logger

F = TypeVar("F", bound=Callable[..., Any])

log = get_logger(__name__)

Serializer = Literal["auto", "pickle", "parquet", "json"]


# ============================================================================
# 单例 diskcache 实例
# ============================================================================

_cache_instance: diskcache.Cache | None = None
_cache_dir: Path | None = None


def _get_cache_dir() -> Path:
    """惰性获取缓存目录，避免 import 时就触发 settings 加载."""
    global _cache_dir
    if _cache_dir is None:
        try:
            from quantmind.core.config import get_settings

            _cache_dir = get_settings().cache_dir_path() / "diskcache"
        except Exception:
            _cache_dir = Path("./.cache/diskcache").resolve()
        _cache_dir.mkdir(parents=True, exist_ok=True)
    return _cache_dir


def _get_cache() -> diskcache.Cache:
    """惰性单例 diskcache.Cache."""
    global _cache_instance
    if _cache_instance is None:
        cache_path = _get_cache_dir()
        _cache_instance = diskcache.Cache(
            directory=str(cache_path),
            size_limit=int(10 * 1024**3),  # 10 GB 上限，被 settings.cache 覆盖
            eviction_policy="least-recently-used",
        )
    return _cache_instance


# ============================================================================
# Key 生成
# ============================================================================


def _stable_repr(obj: Any) -> str:
    """生成参数的稳定字符串表示（dict 顺序无关）."""
    if isinstance(obj, dict):
        return "{" + ",".join(f"{k}:{_stable_repr(obj[k])}" for k in sorted(obj.keys())) + "}"
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_stable_repr(x) for x in obj) + "]"
    if isinstance(obj, set):
        return "{" + ",".join(sorted(_stable_repr(x) for x in obj)) + "}"
    # pandas DataFrame / numpy array 等：用 hash + shape 做 key（避免太大）
    if hasattr(obj, "shape"):
        return f"<{type(obj).__name__}:{obj.shape}>"
    # 其他类型回退到 repr
    try:
        return repr(obj)
    except Exception:
        return f"<{type(obj).__name__}>"


def _make_key(func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """根据函数签名生成稳定 key."""
    sig = inspect.signature(func)
    try:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        kw = dict(bound.arguments)
    except TypeError:
        kw = dict(zip([f"arg{i}" for i in range(len(args))], args, strict=False))
        kw.update(kwargs)

    payload = {
        "func": f"{func.__module__}.{func.__qualname__}",
        "args": {k: _stable_repr(v) for k, v in kw.items()},
    }
    raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"{func.__module__}.{func.__qualname__}:{digest}"


# ============================================================================
# 序列化
# ============================================================================


def _is_dataframe(obj: Any) -> bool:
    """避免直接 import pandas，只有调用时才查."""
    return type(obj).__module__.startswith("pandas") and hasattr(obj, "to_parquet")


def _detect_serializer(value: Any) -> str:
    if _is_dataframe(value):
        return "parquet"
    return "pickle"


def _serialize(value: Any, serializer: Serializer) -> tuple[bytes, str]:
    """返回 (bytes, used_serializer)."""
    used = _detect_serializer(value) if serializer == "auto" else serializer

    if used == "parquet" and _is_dataframe(value):
        import io

        buf = io.BytesIO()
        value.to_parquet(buf, index=True)
        return buf.getvalue(), "parquet"
    if used == "json":
        return json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"), "json"
    # pickle 兜底
    return pickle.dumps(value), "pickle"


def _deserialize(blob: bytes, used: str) -> Any:
    if used == "parquet":
        import io

        import pandas as pd  # 这里 pandas 一定可用（被 cache 的就是 DataFrame）

        return pd.read_parquet(io.BytesIO(blob))
    if used == "json":
        return json.loads(blob.decode("utf-8"))
    return pickle.loads(blob)


# ============================================================================
# 装饰器
# ============================================================================


def cached(
    ttl_hours: float | None = 24.0,
    key_func: Callable[..., str] | None = None,
    serializer: Serializer = "auto",
    namespace: str | None = None,
) -> Callable[[F], F]:
    """函数级磁盘缓存装饰器.

    Args:
        ttl_hours: 过期时间（小时）。None=永不过期；0=不缓存
        key_func: 自定义 key 生成器，签名 ``f(*args, **kwargs) -> str``
        serializer: ``auto`` | ``pickle`` | ``parquet`` | ``json``
        namespace: 可选 key 前缀

    Example::

        @cached(ttl_hours=24)
        def get_price(ticker, as_of):
            return akshare.stock_zh_a_hist(...)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if ttl_hours == 0:
                return func(*args, **kwargs)

            if key_func is not None:
                key_body = key_func(*args, **kwargs)
                key = f"{func.__module__}.{func.__qualname__}:{key_body}"
            else:
                key = _make_key(func, args, kwargs)
            if namespace:
                key = f"{namespace}::{key}"

            cache = _get_cache()
            entry = cache.get(key, default=None)
            if entry is not None:
                blob, used, stored_at, ttl_s = entry
                if ttl_s is None or (time.time() - stored_at) < ttl_s:
                    log.debug(f"cache hit: {key}")
                    return _deserialize(blob, used)
                log.debug(f"cache stale: {key} (age={time.time() - stored_at:.0f}s)")

            t0 = time.monotonic()
            result = func(*args, **kwargs)
            elapsed = time.monotonic() - t0

            try:
                blob, used = _serialize(result, serializer)
                ttl_s = None if ttl_hours is None else float(ttl_hours) * 3600.0
                cache.set(key, (blob, used, time.time(), ttl_s))
                log.debug(
                    f"cache miss: {key} ({elapsed:.2f}s, {len(blob) / 1024:.1f}KB, "
                    f"serializer={used})"
                )
            except Exception as e:
                # 缓存失败不影响业务返回
                log.warning(f"cache set failed for {key}: {e}")

            return result

        wrapper.__cached__ = True  # type: ignore[attr-defined]
        wrapper.__cache_key_func__ = key_func  # type: ignore[attr-defined]
        wrapper.__wrapped_func__ = func  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


# ============================================================================
# 失效与统计
# ============================================================================


def cached_invalidate(func: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
    """手动失效一个 ``@cached`` 函数对特定参数的缓存项.

    Returns:
        True 如果该 key 之前在缓存里
    """
    real = getattr(func, "__wrapped_func__", func)
    key = _make_key(real, args, kwargs)
    cache = _get_cache()
    return bool(cache.delete(key))


def clear_cache(prefix: str | None = None) -> int:
    """清缓存.

    Args:
        prefix: 仅删 key 以此 prefix 开头的项；None=全清

    Returns:
        删除的条目数
    """
    cache = _get_cache()
    if prefix is None:
        n = len(cache)
        cache.clear()
        return n
    keys_to_delete = [k for k in cache.iterkeys() if str(k).startswith(prefix)]
    for k in keys_to_delete:
        cache.delete(k)
    return len(keys_to_delete)


def cache_stats() -> dict[str, Any]:
    """返回缓存统计."""
    cache = _get_cache()
    return {
        "directory": str(_get_cache_dir()),
        "entries": len(cache),
        "volume_bytes": cache.volume(),
    }


__all__ = ["cache_stats", "cached", "cached_invalidate", "clear_cache"]


if __name__ == "__main__":

    @cached(ttl_hours=1)
    def slow_add(a: int, b: int) -> int:
        time.sleep(0.5)
        return a + b

    print(f"cache dir: {_get_cache_dir()}")
    t0 = time.monotonic()
    print(f"call#1: {slow_add(1, 2)} ({time.monotonic() - t0:.3f}s)")
    t0 = time.monotonic()
    print(f"call#2: {slow_add(1, 2)} ({time.monotonic() - t0:.3f}s)  # should be near 0")
    t0 = time.monotonic()
    print(f"call#3 different args: {slow_add(3, 4)} ({time.monotonic() - t0:.3f}s)")

    print(f"stats: {cache_stats()}")
    print(f"invalidate(1, 2) -> {cached_invalidate(slow_add, 1, 2)}")
    t0 = time.monotonic()
    print(f"call#4 after invalidate: {slow_add(1, 2)} ({time.monotonic() - t0:.3f}s)")
    print(f"clear_cache (this prefix): {clear_cache(prefix='__main__.slow_add')}")
