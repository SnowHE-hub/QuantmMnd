"""tests/test_data_handler.py

QuantMindDataHandler 单元测试（≥ 10 个）。

所有测试使用临时目录中的假 parquet 文件，不读取真实项目数据。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantmind.data.handler import (
    QuantMindDataHandler,
    _make_hash,
    get_handler,
    load_panel,
    load_positions,
    load_prices,
    load_returns,
    open_parquet_cached,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_panel_df(n_dates: int = 4, n_tickers: int = 5) -> pd.DataFrame:
    """构建模拟 alpha_panel MultiIndex DataFrame。"""
    dates   = pd.date_range("2024-01-01", periods=n_dates, freq="QE")
    tickers = [f"{i:06d}.SH" for i in range(1, n_tickers + 1)]
    idx     = pd.MultiIndex.from_product([dates, tickers], names=["as_of", "ticker"])
    rng     = np.random.default_rng(42)
    df      = pd.DataFrame(
        rng.standard_normal((len(idx), 5)),
        index=idx,
        columns=["pe_ttm", "pb", "momentum_1m", "sharpe_ratio", "rsi_14"],
    )
    return df


def _make_prices_df(n_days: int = 30, n_tickers: int = 3) -> pd.DataFrame:
    """构建模拟 long-format 日线价格。"""
    dates   = pd.date_range("2024-01-01", periods=n_days, freq="B")
    tickers = ["600519.SH", "000858.SZ", "601318.SH"][:n_tickers]
    rows    = [
        {"trade_date": d, "ts_code": t, "close": float(100 + i * 0.5)}
        for i, d in enumerate(dates)
        for t in tickers
    ]
    return pd.DataFrame(rows)


def _make_positions_df() -> pd.DataFrame:
    return pd.DataFrame({
        "as_of":  ["2024-01-31", "2024-02-29", "2024-02-29"],
        "ticker": ["600519.SH", "000858.SZ", "601318.SH"],
        "predicted_rank": [1, 2, 3],
    })


def _make_returns_df() -> pd.DataFrame:
    return pd.DataFrame({
        "as_of_date":      ["2024-01-31"] * 3,
        "ticker":          ["600519.SH", "000858.SZ", "601318.SH"],
        "actual_return_63d": [0.15, -0.05, 0.08],
        "industry":        ["白酒", "白酒", "保险"],
    })


@pytest.fixture
def tmp_handler(tmp_path: Path) -> QuantMindDataHandler:
    """每个测试使用独立临时目录，文件互不干扰。"""
    panel_path    = tmp_path / "panel.parquet"
    prices_path   = tmp_path / "prices.parquet"
    positions_path = tmp_path / "positions.parquet"
    returns_path  = tmp_path / "returns.parquet"

    _make_panel_df().to_parquet(panel_path)
    _make_prices_df().to_parquet(prices_path, index=False)
    _make_positions_df().to_parquet(positions_path, index=False)
    _make_returns_df().to_parquet(returns_path, index=False)

    return QuantMindDataHandler(paths={
        "panel":     panel_path,
        "prices":    prices_path,
        "positions": positions_path,
        "returns":   returns_path,
    })


# ─────────────────────────────────────────────────────────────────────────────
# 1: fetch_panel 与直接 pd.read_parquet 结果一致
# ─────────────────────────────────────────────────────────────────────────────

def test_fetch_panel_matches_direct_read(tmp_handler, tmp_path):
    """fetch_panel() 与 pd.read_parquet(panel_path) 返回相同 DataFrame。"""
    direct = pd.read_parquet(tmp_handler.panel_path)
    via_h  = tmp_handler.fetch_panel()
    pd.testing.assert_frame_equal(via_h, direct)


# ─────────────────────────────────────────────────────────────────────────────
# 2: 缓存命中 — 第二次调用时间 < 第一次的 5%
# ─────────────────────────────────────────────────────────────────────────────

def test_fetch_panel_cache_speedup(tmp_handler):
    """缓存命中后 fetch_panel 延迟应远小于首次加载。"""
    # 首次（无缓存）
    t0 = time.perf_counter()
    tmp_handler.fetch_panel()
    t1 = time.perf_counter()
    first_ms = t1 - t0

    # 第二次（有缓存）
    t2 = time.perf_counter()
    tmp_handler.fetch_panel()
    t3 = time.perf_counter()
    second_ms = t3 - t2

    # 目标：< 5% 的首次时间（或绝对值 < 10ms）
    assert second_ms < max(first_ms * 0.05, 0.01), (
        f"缓存未命中: 首次={first_ms:.4f}s  缓存={second_ms:.4f}s"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3: as_of 过滤正确
# ─────────────────────────────────────────────────────────────────────────────

def test_fetch_panel_as_of_filter(tmp_handler):
    """fetch_panel(as_of='2024-03-31') 只返回对应季度的行。"""
    as_of_ts = pd.Timestamp("2024-03-31")
    df = tmp_handler.fetch_panel(as_of="2024-03-31")

    assert not df.empty, "过滤结果不应为空（panel 含 2024-03-31 数据）"
    dates = df.index.get_level_values("as_of").unique()
    assert all(d == as_of_ts for d in dates), f"返回了非目标日期: {dates.tolist()}"


# ─────────────────────────────────────────────────────────────────────────────
# 4: ts_codes 过滤正确
# ─────────────────────────────────────────────────────────────────────────────

def test_fetch_panel_ts_codes_filter(tmp_handler):
    """fetch_panel(ts_codes=['000001.SH']) 只返回指定股票的行。"""
    wanted = ["000001.SH"]
    df = tmp_handler.fetch_panel(ts_codes=wanted)

    assert not df.empty
    tickers = df.index.get_level_values("ticker").unique().tolist()
    assert tickers == wanted, f"过滤后包含非预期股票: {tickers}"


# ─────────────────────────────────────────────────────────────────────────────
# 5: as_of + ts_codes 联合过滤
# ─────────────────────────────────────────────────────────────────────────────

def test_fetch_panel_combined_filter(tmp_handler):
    """as_of 和 ts_codes 联合过滤应返回正确子集。"""
    df = tmp_handler.fetch_panel(as_of="2024-03-31", ts_codes=["000001.SH"])
    if not df.empty:
        assert df.index.get_level_values("as_of").unique()[0] == pd.Timestamp("2024-03-31")
        assert "000001.SH" in df.index.get_level_values("ticker")


# ─────────────────────────────────────────────────────────────────────────────
# 6: columns 过滤只返回请求列
# ─────────────────────────────────────────────────────────────────────────────

def test_fetch_panel_columns_filter(tmp_handler):
    """fetch_panel(columns=['pe_ttm', 'pb']) 只返回两列。"""
    df = tmp_handler.fetch_panel(columns=["pe_ttm", "pb"])
    assert set(df.columns) == {"pe_ttm", "pb"}


# ─────────────────────────────────────────────────────────────────────────────
# 7: fetch_prices 过滤正确
# ─────────────────────────────────────────────────────────────────────────────

def test_fetch_prices_ts_code_filter(tmp_handler):
    """fetch_prices(ts_code='600519.SH') 只返回该股票的行。"""
    df = tmp_handler.fetch_prices(ts_code="600519.SH")
    assert not df.empty
    assert (df["ts_code"] == "600519.SH").all()


def test_fetch_prices_date_range(tmp_handler):
    """fetch_prices(start_date, end_date) 只返回日期范围内的行。"""
    df = tmp_handler.fetch_prices(start_date="2024-01-10", end_date="2024-01-20")
    if not df.empty:
        dates = pd.to_datetime(df["trade_date"])
        assert (dates >= pd.Timestamp("2024-01-10")).all()
        assert (dates <= pd.Timestamp("2024-01-20")).all()


# ─────────────────────────────────────────────────────────────────────────────
# 8: invalidate_cache 后下次重新加载
# ─────────────────────────────────────────────────────────────────────────────

def test_invalidate_cache_forces_reload(tmp_handler, tmp_path):
    """invalidate_cache() 后下次调用重新读文件，而非命中缓存。"""
    _ = tmp_handler.fetch_panel()   # 首次加载
    info_before = tmp_handler.cache_info()

    tmp_handler.invalidate_cache()
    info_cleared = tmp_handler.cache_info()
    assert info_cleared["raw_keys"] == 0, "invalidate 后缓存应为空"

    _ = tmp_handler.fetch_panel()   # 重新加载
    info_after = tmp_handler.cache_info()
    # misses 应比清除前多（因为重新加载）
    assert info_after["cache_misses"] > info_before["cache_misses"]


# ─────────────────────────────────────────────────────────────────────────────
# 9: fetch_positions latest_only=True 只返回最新日期
# ─────────────────────────────────────────────────────────────────────────────

def test_fetch_positions_latest_only(tmp_handler):
    """fetch_positions(latest_only=True) 只返回最大 as_of 的记录。"""
    all_pos = tmp_handler.fetch_positions(latest_only=False)
    latest  = tmp_handler.fetch_positions(latest_only=True)

    assert len(latest) <= len(all_pos)
    if not latest.empty and "as_of" in latest.columns:
        max_date = all_pos["as_of"].max()
        assert (latest["as_of"] == max_date).all()


# ─────────────────────────────────────────────────────────────────────────────
# 10: fetch_returns ts_codes 过滤
# ─────────────────────────────────────────────────────────────────────────────

def test_fetch_returns_ts_codes_filter(tmp_handler):
    """fetch_returns(ts_codes=['600519.SH']) 只返回该股票。"""
    df = tmp_handler.fetch_returns(ts_codes=["600519.SH"])
    assert not df.empty
    assert (df["ticker"] == "600519.SH").all()


# ─────────────────────────────────────────────────────────────────────────────
# 11: 单例 get_instance 返回同一对象
# ─────────────────────────────────────────────────────────────────────────────

def test_get_instance_singleton():
    """get_instance() 多次调用应返回完全相同的对象。"""
    # 注意：不 reset 全局单例，只验证 id 一致
    h1 = QuantMindDataHandler.get_instance()
    h2 = QuantMindDataHandler.get_instance()
    assert h1 is h2, "get_instance() 应返回同一实例"


# ─────────────────────────────────────────────────────────────────────────────
# 12: __repr__ 包含 loaded keys
# ─────────────────────────────────────────────────────────────────────────────

def test_repr_after_load(tmp_handler):
    """__repr__ 在加载后应包含已加载的数据键。"""
    tmp_handler.fetch_panel()
    r = repr(tmp_handler)
    assert "panel" in r
    assert "QuantMindDataHandler" in r


# ─────────────────────────────────────────────────────────────────────────────
# 13: cache_info 统计正确
# ─────────────────────────────────────────────────────────────────────────────

def test_cache_info_stats(tmp_handler):
    """cache_info 应正确跟踪命中/未命中次数。"""
    tmp_handler.fetch_panel()       # miss (全量)
    tmp_handler.fetch_panel()       # hit
    tmp_handler.fetch_panel()       # hit
    info = tmp_handler.cache_info()
    assert info["cache_hits"] >= 2
    assert info["cache_misses"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 14: open_parquet_cached 通用缓存 — 第二次比第一次快
# ─────────────────────────────────────────────────────────────────────────────

def test_open_parquet_cached_speedup(tmp_handler, tmp_path):
    """open_parquet_cached 第二次调用应命中缓存。"""
    # 创建临时 parquet 文件
    test_file = tmp_path / "test_generic.parquet"
    df_orig = pd.DataFrame({"a": range(1000), "b": range(1000)})
    df_orig.to_parquet(test_file, index=False)

    # 清除全局 handler 的 raw_cache（用独立 handler 测试）
    h = QuantMindDataHandler(paths={
        "panel": tmp_handler.panel_path,
        "prices": tmp_handler.prices_path,
        "positions": tmp_handler.positions_path,
        "returns": tmp_handler.returns_path,
    })

    t0 = time.perf_counter()
    df1 = open_parquet_cached(test_file)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    df2 = open_parquet_cached(test_file)
    t3 = time.perf_counter()

    pd.testing.assert_frame_equal(df1, df2)
    # 第二次应该极快
    assert (t3 - t2) < (t1 - t0) or (t3 - t2) < 0.005


# ─────────────────────────────────────────────────────────────────────────────
# 15: invalidate_key 只清除指定 key 的缓存
# ─────────────────────────────────────────────────────────────────────────────

def test_invalidate_key_selective(tmp_handler):
    """invalidate_key('panel') 只清除 panel 缓存，不影响 prices。"""
    tmp_handler.fetch_panel()
    tmp_handler.fetch_prices()
    assert "panel" in tmp_handler._raw_cache
    assert "prices" in tmp_handler._raw_cache

    tmp_handler.invalidate_key("panel")
    assert "panel" not in tmp_handler._raw_cache, "panel 缓存应已清除"
    assert "prices" in tmp_handler._raw_cache, "prices 缓存应保留"


# ─────────────────────────────────────────────────────────────────────────────
# 16: setup_data(preload=True) 预热所有 key
# ─────────────────────────────────────────────────────────────────────────────

def test_setup_data_preload(tmp_handler):
    """setup_data(preload=True) 应一次性加载所有 4 个数据源。"""
    assert len(tmp_handler._raw_cache) == 0, "初始缓存应为空"
    tmp_handler.setup_data(preload=True)
    assert len(tmp_handler._raw_cache) == 4, (
        f"预热后应有 4 个数据源，实际: {list(tmp_handler._raw_cache.keys())}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 17: _make_hash 相同参数产生相同 hash，不同参数产生不同 hash
# ─────────────────────────────────────────────────────────────────────────────

def test_make_hash_deterministic():
    h1 = _make_hash("panel", as_of="2024-03-31", ts_codes=["A"])
    h2 = _make_hash("panel", as_of="2024-03-31", ts_codes=["A"])
    assert h1 == h2, "相同参数应产生相同 hash"


def test_make_hash_different_params_differ():
    h1 = _make_hash("panel", as_of="2024-03-31")
    h2 = _make_hash("panel", as_of="2024-06-30")
    assert h1 != h2, "不同参数应产生不同 hash"


# ─────────────────────────────────────────────────────────────────────────────
# 18: 文件不存在时抛出 FileNotFoundError
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_file_raises(tmp_path):
    """指向不存在文件时 fetch_panel 应抛出 FileNotFoundError。"""
    h = QuantMindDataHandler(paths={"panel": tmp_path / "nonexistent.parquet"})
    with pytest.raises(FileNotFoundError):
        h.fetch_panel()
