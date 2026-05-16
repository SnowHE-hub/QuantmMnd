"""tests/test_shared_cache.py — SharedDataCache 单元测试."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from quantmind.data.shared_cache import SharedDataCache


@pytest.fixture(autouse=True)
def reset_singleton():
    """每个测试前后重置单例."""
    SharedDataCache.reset_instance()
    yield
    SharedDataCache.reset_instance()


@pytest.fixture
def cache(tmp_path):
    return SharedDataCache(cache_dir=tmp_path)


@pytest.fixture
def cache2(tmp_path):
    """同目录的第二个缓存实例（不通过单例）."""
    return SharedDataCache(cache_dir=tmp_path)


# ── 测试：两个 Provider 读到同一份缓存数据 ────────────────────────────────────

def test_two_providers_share_same_cache_data(tmp_path):
    """两个不同的 DataProvider 实例读到同一份缓存数据（单例）."""
    # 模拟两个系统的 Provider 通过单例获取同一缓存
    cache_a = SharedDataCache.get_instance(cache_dir=tmp_path)
    cache_b = SharedDataCache.get_instance(cache_dir=tmp_path)

    # 两者应该是同一个对象
    assert cache_a is cache_b

    # 写入数据
    df = pd.DataFrame({"price": [10.0, 11.0]}, index=pd.date_range("2024-12-30", periods=2))
    cache_a.write("test_price_panel", df, ttl_hours=24)

    # 通过 cache_b（同一实例）读取
    result = cache_b.read("test_price_panel")
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["price"]


def test_different_format_write_and_read(tmp_path):
    """正确读写 dict/list 格式数据."""
    cache = SharedDataCache(cache_dir=tmp_path)

    data = {"ticker": "000001.SZ", "pe_ttm": 12.5, "roe_ttm": 15.3}
    cache.write("fund_000001", data, ttl_hours=24)

    result = cache.read("fund_000001")
    assert result is not None
    assert result["ticker"] == "000001.SZ"
    assert result["pe_ttm"] == 12.5


def test_write_and_read_dataframe(cache):
    """写入 DataFrame 后可以正确读取."""
    df = pd.DataFrame(
        {"ticker": ["000001.SZ", "600519.SH"], "price": [10.0, 1800.0]}
    )
    cache.write("spot_2024-12-31", df, ttl_hours=24)

    result = cache.read("spot_2024-12-31")
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert "ticker" in result.columns


# ── 测试：invalidate 正确清除匹配条目 ────────────────────────────────────────

def test_invalidate_pattern_clears_matching_entries(cache):
    """invalidate(pattern) 正确清除匹配条目."""
    df1 = pd.DataFrame({"a": [1]})
    df2 = pd.DataFrame({"b": [2]})
    df3 = pd.DataFrame({"c": [3]})

    cache.write("spot_2024-12-31", df1, ttl_hours=24)
    cache.write("spot_2024-12-30", df2, ttl_hours=24)
    cache.write("prices_2024-12-31", df3, ttl_hours=24)

    # 清除所有 spot_* 条目
    count = cache.invalidate("spot_*")
    assert count == 2

    # spot 条目应该消失
    assert cache.read("spot_2024-12-31") is None
    assert cache.read("spot_2024-12-30") is None
    # prices 条目应该保留
    assert cache.read("prices_2024-12-31") is not None


def test_invalidate_no_match_returns_zero(cache):
    """invalidate 无匹配时返回 0."""
    df = pd.DataFrame({"x": [1]})
    cache.write("spot_2024-12-31", df, ttl_hours=24)
    count = cache.invalidate("nonexistent_*")
    assert count == 0


# ── 测试：缓存 stats 返回正确的命中率 ────────────────────────────────────────

def test_stats_returns_correct_hit_rate(cache):
    """缓存 stats 返回正确的命中率."""
    df = pd.DataFrame({"price": [10.0]})

    # 1 miss
    result = cache.read("nonexistent_key")
    assert result is None

    # 写入后 1 hit（从内存）
    cache.write("known_key", df, ttl_hours=24)
    result = cache.read("known_key")
    assert result is not None

    # 再读 1 hit（内存命中）
    cache.read("known_key")

    stats = cache.stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert abs(stats["hit_rate"] - 2 / 3) < 0.01


def test_stats_size_mb_increases_after_write(cache):
    """写入数据后 size_mb 增加."""
    stats_before = cache.stats()
    size_before = stats_before["size_mb"]

    df = pd.DataFrame({"col": list(range(1000))})
    cache.write("large_data", df, ttl_hours=24)

    stats_after = cache.stats()
    assert stats_after["size_mb"] >= size_before


def test_ttl_expired_cache_returns_none(tmp_path):
    """TTL 过期后读取返回 None."""
    cache = SharedDataCache(cache_dir=tmp_path)
    df = pd.DataFrame({"price": [10.0]})

    # 写入 TTL=0（立即过期）
    cache.write("ttl_test", df, ttl_hours=0)
    time.sleep(0.05)

    # 清除内存缓存（强制走磁盘）
    cache._mem.clear()
    result = cache.read("ttl_test")
    assert result is None


def test_memory_cache_hit_skips_disk(cache):
    """内存缓存命中时不走磁盘."""
    df = pd.DataFrame({"price": [10.0, 11.0]})
    cache.write("mem_test", df, ttl_hours=24)

    # 第一次读（应从内存或磁盘）
    cache.read("mem_test")
    hits_after_first = cache._hits

    # 第二次读（应命中内存）
    cache.read("mem_test")
    assert cache._hits == hits_after_first + 1
