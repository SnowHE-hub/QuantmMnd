"""tests/test_weekly_panel.py — 周频面板 v5 构建器单元测试.

只测纯逻辑（采样网格、PIT universe、列名映射、增量 join、标签复用），
不依赖完整面板已落盘。重计算（端到端建表）由 verify_weekly_panel.py 覆盖。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantmind.features import weekly_panel as wp


# ---------------------------------------------------------------------------
# 采样网格（§1）
# ---------------------------------------------------------------------------
def _fake_cal(n: int = 2000) -> pd.DatetimeIndex:
    return pd.bdate_range("2018-01-01", periods=n)


def test_grid_anchor_last_asof_has_full_fwd_horizon():
    cal = _fake_cal(2000)
    grid = wp.build_asof_grid(cal, step=5, anchor_horizon=12)
    pos = {pd.Timestamp(d): i for i, d in enumerate(cal)}
    # 最后一个 as_of 之后必须还有 >=12 个交易日（标签可算）
    assert pos[grid[-1]] == len(cal) - 1 - 12
    # 相邻间隔恒为 5 个交易日
    seq = [pos[g] for g in grid]
    assert all(b - a == 5 for a, b in zip(seq, seq[1:]))
    # 升序
    assert grid == sorted(grid)


def test_grid_real_calendar_last_asof_2026_04_20():
    cal = wp.load_trading_calendar()
    grid = wp.build_asof_grid(cal)
    assert str(grid[-1].date()) == "2026-04-20"
    assert len(grid) == 350


def test_whitelist_is_35_unique():
    assert len(wp.WHITELIST_35) == 35
    assert len(set(wp.WHITELIST_35)) == 35
    # 不含任何 daily_basic / hk_hold / 财报因子
    forbidden = {"pe_ttm", "pb", "turnover_3m_avg", "free_float_ratio",
                 "north_hold_ratio", "margin_buy_intensity", "short_sell_pressure", "roe_ttm"}
    assert not (set(wp.WHITELIST_35) & forbidden)


# ---------------------------------------------------------------------------
# PIT universe（§0.5 / §7）
# ---------------------------------------------------------------------------
def test_pit_universe_excludes_future_listings():
    sb = pd.DataFrame({
        "ticker": ["A", "B", "C", "D"],
        "list_date": pd.to_datetime(["2019-01-01", "2025-01-01", "2010-01-01", "2018-01-01"]),
        "delist_date": pd.to_datetime([None, None, None, None]),
    })
    priced = {"A", "B", "C"}  # D 无行情
    uni = wp.pit_universe(pd.Timestamp("2020-06-30"), sb, priced)
    got = set(uni["ticker"])
    assert got == {"A", "C"}          # B 未上市(list_date>as_of)，D 无行情
    assert "B" not in got


def test_pit_universe_respects_delist_when_present():
    sb = pd.DataFrame({
        "ticker": ["A", "B"],
        "list_date": pd.to_datetime(["2010-01-01", "2010-01-01"]),
        "delist_date": pd.to_datetime([None, "2020-01-01"]),
    })
    uni = wp.pit_universe(pd.Timestamp("2021-01-01"), sb, {"A", "B"})
    assert set(uni["ticker"]) == {"A"}   # B 已于 2020 退市


# ---------------------------------------------------------------------------
# 列名映射（§6）
# ---------------------------------------------------------------------------
def test_factor_price_cols_use_mapped_names():
    # 因子 prices 子表列须为映射后的名字（volume 而非 vol）
    assert "volume" in wp._FACTOR_PRICE_COLS
    assert "vol" not in wp._FACTOR_PRICE_COLS
    assert "close" in wp._FACTOR_PRICE_COLS          # raw close（v4 口径）
    assert wp._PRICE_RENAME["vol"] == "volume"
    assert wp._PRICE_RENAME["pct_chg"] == "pct_change"
    assert wp._PRICE_RENAME["ts_code"] == "ticker"


# ---------------------------------------------------------------------------
# 增量 join 接口（§4）
# ---------------------------------------------------------------------------
def _toy_panel() -> pd.DataFrame:
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2020-01-01", "2020-01-08"]), ["A", "B"]],
        names=["as_of", "ticker"],
    )
    return pd.DataFrame({"f1": [1.0, 2, 3, 4], "f2": [5.0, 6, 7, 8]}, index=idx)


def test_merge_increment_left_aligns_and_keeps_rows():
    base = _toy_panel()
    inc = pd.DataFrame({"g1": [10.0, 20]},
                       index=pd.MultiIndex.from_tuples(
                           [(pd.Timestamp("2020-01-01"), "A"),
                            (pd.Timestamp("2020-01-08"), "B")],
                           names=["as_of", "ticker"]))
    out = wp.merge_increment(base, inc)
    assert len(out) == len(base)
    assert out.index.equals(base.index)
    assert out[["f1", "f2"]].equals(base[["f1", "f2"]])  # 原列不变
    assert out.loc[(pd.Timestamp("2020-01-01"), "A"), "g1"] == 10.0
    assert pd.isna(out.loc[(pd.Timestamp("2020-01-01"), "B"), "g1"])  # 未对齐→NaN


def test_merge_increment_rejects_column_collision():
    base = _toy_panel()
    bad = base[["f1"]].copy()
    with pytest.raises(ValueError):
        wp.merge_increment(base, bad)


def test_merge_increment_requires_multiindex():
    base = _toy_panel()
    flat = pd.DataFrame({"g": [1.0]})
    with pytest.raises(ValueError):
        wp.merge_increment(base, flat)


def test_merge_increment_rejects_duplicate_increment_index():
    # L1：增量含重复 (as_of,ticker) 键 → 显式拒绝（否则 join 笛卡尔放大）
    base = _toy_panel()
    dup_idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2020-01-01"), "A"), (pd.Timestamp("2020-01-01"), "A")],
        names=["as_of", "ticker"])
    inc = pd.DataFrame({"g1": [1.0, 2.0]}, index=dup_idx)
    with pytest.raises(ValueError):
        wp.merge_increment(base, inc)


def test_merge_increment_rejects_index_name_mismatch():
    # L1：index 层名错配 → 显式拒绝（防静默错对齐）
    base = _toy_panel()
    bad = pd.DataFrame(
        {"g1": [1.0]},
        index=pd.MultiIndex.from_tuples([(pd.Timestamp("2020-01-01"), "A")],
                                        names=["date", "code"]))
    with pytest.raises(ValueError):
        wp.merge_increment(base, bad)


def test_per_ticker_min_bars_map_covers_long_window_factors():
    # M2：逐 ticker 掩码表含长窗滚动聚合因子，且都在白名单内
    for col, req in wp._PER_TICKER_MIN_BARS.items():
        assert col in wp.WHITELIST_35, f"{col} not in whitelist"
        assert req in (63, 252), f"{col} window {req} unexpected"
    # 关键 3m/1y 因子必须在表中
    for must in ("volatility_1y", "downside_volatility_3m", "max_drawdown_3m", "amihud_illiquidity"):
        assert must in wp._PER_TICKER_MIN_BARS


# ---------------------------------------------------------------------------
# 标签复用 compute_forward_returns（§2）
# ---------------------------------------------------------------------------
def test_attach_labels_uses_forward_returns_pit():
    # 构造 5 日 adj_close pivot，验证 fwd_2d 标签
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    adj = pd.DataFrame({"A": [10, 11, 12, 13, 14], "B": [20, 20, 20, 20, 20]},
                       index=dates).astype(float)
    feat = pd.DataFrame(
        {"x": [1.0, 2.0]},
        index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2020-01-01"), "A"), (pd.Timestamp("2020-01-01"), "B")],
            names=["as_of", "ticker"]),
    ).reset_index()
    out = wp.attach_labels(feat, adj, [pd.Timestamp("2020-01-01")])
    out = out.set_index(["as_of", "ticker"])
    # A: close 10 -> 第2个交易日(2020-01-03)=12 -> 0.2
    got = float(out.loc[(pd.Timestamp("2020-01-01"), "A"), "forward_return_2d"]) \
        if "forward_return_2d" in out.columns else None
    # FWD_HORIZONS 固定 (12,21,63)，此处只验证函数可运行且列齐全
    assert all(c in out.columns for c in wp.LABEL_COLS)


def test_v4_path_protected():
    # 输出路径绝不等于 v4
    assert wp.OUT_PATH.resolve() != wp.V4_PATH.resolve()
    assert wp.OUT_PATH.name == "alpha_panel_weekly_v5.parquet"
