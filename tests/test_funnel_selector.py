"""tests/test_funnel_selector.py — FunnelSelector 单元测试（使用 mock LazyDataEngine）."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quantmind.selection.funnel_selector import FunnelResult, FunnelSelector, FunnelStats
from quantmind.selection.lazy_data_engine import LazyDataEngine


# ── 测试数据工厂 ──────────────────────────────────────────────────────────────

def _make_universe(n: int = 10, include_st: bool = True) -> pd.DataFrame:
    """构建测试用股票宇宙（不含基本面列，避免merge冲突）."""
    tickers = [f"{str(i).zfill(6)}.SZ" for i in range(1, n + 1)]
    names = [f"ST股票{i}" if include_st and i <= 2 else f"普通股票{i}" for i in range(1, n + 1)]
    return pd.DataFrame({
        "ticker": tickers,
        "name": names,
        "list_date": "2020-01-01",
        "total_mv": [500000.0] * n,
        "circ_mv": [300000.0] * n,
        "is_st": [True if include_st and i <= 2 else False for i in range(1, n + 1)],
    })


def _make_prices(tickers: list[str], n_days: int = 70, trend: str = "up") -> pd.DataFrame:
    """构建测试价格面板.

    使用随机游走 + 漂移，确保 RSI 不会极端（避免 RSI > 80 或 < 20 导致 Layer3 过滤）。
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-10-01", periods=n_days)
    data = {}
    for ticker in tickers:
        if trend == "up":
            # 正漂移 + 较大波动，使 RSI 维持在合理范围
            daily_ret = rng.normal(0.003, 0.02, n_days)  # 日均 +0.3%，波动2%
        elif trend == "down":
            # 负漂移
            daily_ret = rng.normal(-0.006, 0.02, n_days)  # 日均 -0.6%，波动2%
        else:
            daily_ret = rng.normal(0.0, 0.015, n_days)
        prices = 10.0 * np.cumprod(1 + daily_ret)
        data[ticker] = prices
    return pd.DataFrame(data, index=dates)


def _make_fundamentals(tickers: list[str], roe: float = 12.0, pe: float = 20.0) -> pd.DataFrame:
    """构建测试基本面数据（不包含会与 _make_universe 冲突的列）."""
    return pd.DataFrame({
        "ticker": tickers,
        "roe_ttm": [roe] * len(tickers),
        "pe_ttm": [pe] * len(tickers),
        "pb": [2.0] * len(tickers),
        "revenue_yoy": [10.0] * len(tickers),
    })


def _make_mock_engine(
    universe_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    kb_contexts: dict | None = None,
) -> MagicMock:
    """构建 mock LazyDataEngine."""
    engine = MagicMock(spec=LazyDataEngine)
    engine.get_spot_data.return_value = universe_df
    engine.get_price_window.return_value = prices_df
    engine.get_fundamentals_light.return_value = fundamentals_df
    engine.get_kb_context.return_value = kb_contexts or {}
    engine.cache_stats.return_value = {
        "mem_hits": 2,
        "disk_hits": 1,
        "misses": 1,
        "hit_rate": 0.75,
        "mem_hit_rate": 0.5,
    }
    return engine


# ── 测试：Layer1 过滤 ST 股票 ─────────────────────────────────────────────────

def test_layer1_filter_st_stocks():
    """Layer1 正确过滤 ST / *ST 股票."""
    universe = _make_universe(n=8, include_st=True)  # 前2只是ST
    prices = _make_prices(universe["ticker"].tolist())
    fundamentals = _make_fundamentals(universe["ticker"].tolist())
    engine = _make_mock_engine(universe, prices, fundamentals)

    selector = FunnelSelector(
        as_of="2024-12-31",
        data_engine=engine,
        universe="csi300",
    )
    # 直接测试 _filter_basic_quality
    result = selector._filter_basic_quality(universe)

    # ST 股票应被过滤
    st_remaining = result["name"].str.contains("ST", na=False).sum()
    assert st_remaining == 0
    assert len(result) == 6  # 8-2=6


def test_layer1_filter_new_listings():
    """Layer1 过滤上市不足90天的股票."""
    universe = pd.DataFrame({
        "ticker": ["000001.SZ", "000002.SZ"],
        "name": ["老股票", "新股"],
        "list_date": ["2020-01-01", "2024-11-01"],  # 第二只2个月前上市
        "total_mv": [500000.0, 500000.0],
        "is_st": [False, False],
    })
    selector = FunnelSelector(as_of="2024-12-31", universe="csi300")
    result = selector._filter_basic_quality(universe)

    assert "000001.SZ" in result["ticker"].values
    assert "000002.SZ" not in result["ticker"].values


# ── 测试：Layer2 流动性过滤 ───────────────────────────────────────────────────

def test_layer2_filter_illiquid():
    """Layer2 过滤近20日有效价格不足15天的股票."""
    tickers = ["000001.SZ", "000002.SZ"]
    # 000001 有足够价格；000002 价格稀疏（大量0）
    dates = pd.date_range("2024-12-01", periods=25)
    prices_001 = np.ones(25) * 10.0
    prices_002 = np.zeros(25)
    prices_002[:5] = 10.0  # 只有5天有价格

    prices = pd.DataFrame(
        {"000001.SZ": prices_001, "000002.SZ": prices_002},
        index=dates,
    )
    universe = pd.DataFrame({"ticker": tickers, "name": ["股票A", "股票B"]})
    selector = FunnelSelector(as_of="2024-12-31", universe="csi300")
    result = selector._filter_liquidity(universe, prices)

    assert "000001.SZ" in result["ticker"].values
    assert "000002.SZ" not in result["ticker"].values


# ── 测试：Layer3 趋势过滤 MA20 > MA60 ─────────────────────────────────────────

def test_layer3_filter_downtrend():
    """Layer3 正确过滤下降趋势（MA20 < MA60）的股票."""
    tickers = ["up_ticker.SZ", "down_ticker.SZ"]
    up_prices = _make_prices(["up_ticker.SZ"], n_days=70, trend="up")["up_ticker.SZ"]
    down_prices = _make_prices(["down_ticker.SZ"], n_days=70, trend="down")["down_ticker.SZ"]
    prices = pd.DataFrame(
        {"up_ticker.SZ": up_prices, "down_ticker.SZ": down_prices},
        index=pd.date_range("2024-10-01", periods=70),
    )
    universe = pd.DataFrame({
        "ticker": tickers,
        "name": ["上升趋势", "下降趋势"],
    })
    selector = FunnelSelector(as_of="2024-12-31", universe="csi300")
    result = selector._filter_trend(universe, prices)

    assert "up_ticker.SZ" in result["ticker"].values
    assert "down_ticker.SZ" not in result["ticker"].values


def test_layer3_ma20_gt_ma60_condition():
    """Layer3: 验证 close > MA20 且 MA20 > MA60 的逻辑.

    构造一个明确的上升趋势，同时避免 RSI > 80。
    策略：使用随机游走 + 正漂移 + 足够的波动使 RSI 在 20~80 之间。
    """
    tickers = ["000001.SZ"]
    prices_df = _make_prices(tickers, n_days=70, trend="up")

    # 手动验证价格满足 MA20 > MA60 条件
    p = prices_df["000001.SZ"]
    ma20 = p.rolling(20).mean().iloc[-1]
    ma60 = p.rolling(60).mean().iloc[-1] if len(p) >= 60 else p.mean()
    close = p.iloc[-1]

    if not (close > ma20 and ma20 > ma60):
        # 如果随机数据不满足，跳过（此测试依赖随机种子）
        pytest.skip("当前随机种子生成的数据不满足 MA 趋势条件，跳过")

    universe = pd.DataFrame({"ticker": tickers, "name": ["测试股"]})
    selector = FunnelSelector(as_of="2024-12-31", universe="csi300")
    result = selector._filter_trend(universe, prices_df)

    assert "000001.SZ" in result["ticker"].values


# ── 测试：Layer4 基本面过滤 ───────────────────────────────────────────────────

def test_layer4_filter_low_roe():
    """Layer4 过滤 ROE < 8% 的股票."""
    tickers = ["good.SZ", "bad.SZ"]
    universe = pd.DataFrame({"ticker": tickers, "name": ["优质股", "劣质股"]})
    fundamentals = pd.DataFrame({
        "ticker": tickers,
        "roe_ttm": [15.0, 3.0],   # 15% vs 3%
        "pe_ttm": [20.0, 20.0],
        "pb": [2.0, 2.0],
        "revenue_yoy": [10.0, 10.0],
    })
    selector = FunnelSelector(as_of="2024-12-31", universe="csi300")
    result = selector._filter_fundamentals(universe, fundamentals)

    assert "good.SZ" in result["ticker"].values
    assert "bad.SZ" not in result["ticker"].values


def test_layer4_filter_negative_pe():
    """Layer4 过滤 PE < 0（亏损）的股票."""
    tickers = ["profit.SZ", "loss.SZ"]
    universe = pd.DataFrame({"ticker": tickers, "name": ["盈利股", "亏损股"]})
    fundamentals = pd.DataFrame({
        "ticker": tickers,
        "roe_ttm": [10.0, 10.0],
        "pe_ttm": [25.0, -5.0],   # 亏损
        "pb": [2.0, 2.0],
        "revenue_yoy": [5.0, 5.0],
    })
    selector = FunnelSelector(as_of="2024-12-31", universe="csi300")
    result = selector._filter_fundamentals(universe, fundamentals)

    assert "profit.SZ" in result["ticker"].values
    assert "loss.SZ" not in result["ticker"].values


# ── 测试：FunnelResult JSON 格式 ─────────────────────────────────────────────

def test_funnel_result_json_format():
    """FunnelResult.to_json() 包含正确字段."""
    n = 10
    universe = _make_universe(n=n, include_st=True)
    tickers = universe["ticker"].tolist()
    prices = _make_prices(tickers, trend="up")
    fundamentals = _make_fundamentals(tickers)
    engine = _make_mock_engine(universe, prices, fundamentals)

    # 使用 full_a，让 mock 的 get_spot_data() 生效
    selector = FunnelSelector(
        as_of="2024-12-31",
        data_engine=engine,
        universe="full_a",
        provider="none",
    )

    with patch.object(selector, "_score_lgbm", side_effect=lambda df, p, f, top_k=50: (
        df.head(3).copy().reset_index(drop=True).assign(
            lgbm_score=[0.9, 0.8, 0.7],
            layer5_rank=[1, 2, 3],
        )
    )):
        result = selector.run(skip_layers=[6], top_n=3)

    assert isinstance(result, FunnelResult)
    json_out = result.to_json()

    assert "as_of" in json_out
    assert json_out["as_of"] == "2024-12-31"
    assert "funnel_stats" in json_out
    assert "candidates" in json_out
    assert "cache_stats" in json_out

    stats = json_out["funnel_stats"]
    for i in range(1, 7):
        assert f"layer{i}_in" in stats
        assert f"layer{i}_out" in stats

    # layer1 应过滤了 ST 股
    assert stats["layer1_in"] >= stats["layer1_out"]


# ── 测试：跳过某层不崩溃 ─────────────────────────────────────────────────────

def test_skip_layers_does_not_crash():
    """跳过某层时不崩溃."""
    n = 8
    universe = _make_universe(n=n, include_st=False)
    tickers = universe["ticker"].tolist()
    prices = _make_prices(tickers, trend="up")
    fundamentals = _make_fundamentals(tickers)
    engine = _make_mock_engine(universe, prices, fundamentals)

    selector = FunnelSelector(
        as_of="2024-12-31",
        data_engine=engine,
        universe="full_a",  # 使用 full_a 让 mock 生效
    )
    result = selector.run(skip_layers=[4, 5, 6], top_n=5)
    assert isinstance(result, FunnelResult)
    assert result.candidates is not None


def test_skip_all_layers_returns_candidates():
    """跳过所有层时，返回原始 universe（截取 top_n）."""
    n = 8
    universe = _make_universe(n=n, include_st=False)
    tickers = universe["ticker"].tolist()
    prices = _make_prices(tickers, trend="up")
    fundamentals = _make_fundamentals(tickers)
    engine = _make_mock_engine(universe, prices, fundamentals)

    selector = FunnelSelector(
        as_of="2024-12-31",
        data_engine=engine,
        universe="full_a",  # 使用 full_a 让 mock 生效
    )
    result = selector.run(skip_layers=[1, 2, 3, 4, 5, 6], top_n=5)
    assert isinstance(result, FunnelResult)
    assert len(result.candidates) <= 5


# ── 测试：layer_stats 每层进出数量正确 ────────────────────────────────────────

def test_layer_stats_correct_counts():
    """layer_stats 记录每层正确的进出数量."""
    n = 8
    universe = _make_universe(n=n, include_st=True)  # 前2只ST
    tickers = universe["ticker"].tolist()
    prices = _make_prices(tickers, trend="up")
    fundamentals = _make_fundamentals(tickers)
    engine = _make_mock_engine(universe, prices, fundamentals)

    # 使用 full_a，这样 _build_initial_universe 会使用 mock 的 get_spot_data()
    selector = FunnelSelector(
        as_of="2024-12-31",
        data_engine=engine,
        universe="full_a",
    )
    result = selector.run(skip_layers=[5, 6], top_n=10)

    assert isinstance(result, FunnelResult)
    stats = result.layer_stats

    # 每层 n_in >= n_out（只减不增）
    for ls in stats:
        assert ls.n_in >= ls.n_out, f"Layer{ls.layer}: n_in={ls.n_in} < n_out={ls.n_out}"

    # Layer1 应过滤了 ST 股
    layer1 = next(ls for ls in stats if ls.layer == 1)
    assert layer1.n_in == n
    assert layer1.n_out < n  # ST 被过滤


# ── 测试：cache_stats 显示内存命中 ───────────────────────────────────────────

def test_cache_stats_shows_hits():
    """cache_stats 显示缓存命中信息."""
    n = 5
    universe = _make_universe(n=n, include_st=False)
    tickers = universe["ticker"].tolist()
    prices = _make_prices(tickers, trend="up")
    fundamentals = _make_fundamentals(tickers)
    engine = _make_mock_engine(universe, prices, fundamentals)
    engine.cache_stats.return_value = {
        "mem_hits": 3,
        "disk_hits": 1,
        "misses": 1,
        "hit_rate": 0.80,
        "mem_hit_rate": 0.60,
    }

    # 使用 full_a，让 mock 的 get_spot_data() 生效
    selector = FunnelSelector(
        as_of="2024-12-31",
        data_engine=engine,
        universe="full_a",
    )
    result = selector.run(skip_layers=[5, 6], top_n=5)
    assert result.cache_stats["hit_rate"] == 0.80


# ── 测试：基本实例化 ─────────────────────────────────────────────────────────

def test_funnel_selector_instantiation():
    """FunnelSelector 可以正常实例化."""
    selector = FunnelSelector(as_of="2024-12-31")
    assert selector.as_of == "2024-12-31"
    assert selector.universe == "csi300"


def test_funnel_selector_with_custom_engine():
    """FunnelSelector 接受自定义 LazyDataEngine."""
    mock_engine = MagicMock(spec=LazyDataEngine)
    mock_engine.cache_stats.return_value = {"mem_hits": 0, "disk_hits": 0, "misses": 0, "hit_rate": 0}

    selector = FunnelSelector(
        as_of="2024-12-31",
        data_engine=mock_engine,
        universe="csi300",
    )
    assert selector.data_engine is mock_engine
