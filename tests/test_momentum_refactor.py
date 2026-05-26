"""tests/test_momentum_refactor.py

动量因子重构验证（≥10 个测试）：
  - _compute_momentum_score 短期反转负权重行为
  - _compute_momentum_score 中长期惯性正权重行为
  - 输出范围 [0,100] 与 _factor_score 格式一致
  - 缺失列降级逻辑（momentum_12m fallback）
  - 全缺失列 → 返回 50（中性分）
  - 极端输入（全 NaN、单行）鲁棒性
  - expr_factors 新条目注册验证
  - MOMENTUM_FACTORS_RAW 列表完整性
  - SHORT_REVERSAL_FACTORS / MID_LONG_MOMENTUM_FACTORS 内容验证
  - __init__.py 导出验证
"""
from __future__ import annotations

import importlib
import sys

import numpy as np
import pandas as pd
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_df(**kwargs) -> pd.DataFrame:
    """构造含给定列的候选 DataFrame（5 只股票）。"""
    n = 5
    base = {col: np.array(vals, dtype=float) for col, vals in kwargs.items()}
    return pd.DataFrame(base)


def _compute(df: pd.DataFrame) -> pd.Series:
    """调用 AnalysisSystem._compute_momentum_score（静态方法，不需要实例）。"""
    from scripts.run_30day_sim import AnalysisSystem  # noqa: PLC0415
    return AnalysisSystem._compute_momentum_score(df)


# ──────────────────────────────────────────────────────────────────────────────
# 1. 输出范围和格式
# ──────────────────────────────────────────────────────────────────────────────

def test_output_range_0_to_100():
    """输出值必须在 [0, 100] 内（与 _factor_score 格式一致）。"""
    df = _make_df(
        reversal_1w=[0.01, -0.02, 0.005, 0.03, -0.01],
        momentum_1m=[0.05, -0.03, 0.02, 0.08, -0.05],
        momentum_6m=[0.10, 0.05, -0.02, 0.15, 0.08],
        momentum_12m_skip_1m=[0.20, 0.10, -0.05, 0.25, 0.12],
    )
    result = _compute(df)
    assert result.min() >= 0.0
    assert result.max() <= 100.0


def test_output_is_series_same_index():
    """输出是 pd.Series，与输入 DataFrame 同 index。"""
    df = _make_df(momentum_1m=[0.1, -0.1, 0.0, 0.05, -0.05])
    result = _compute(df)
    assert isinstance(result, pd.Series)
    assert list(result.index) == list(df.index)


def test_output_no_nan_with_all_columns():
    """所有列均有效时，输出不含 NaN。"""
    df = _make_df(
        reversal_1w=[0.01, -0.02, 0.005, 0.03, -0.01],
        momentum_1m=[0.05, -0.03, 0.02, 0.08, -0.05],
        momentum_6m=[0.10, 0.05, -0.02, 0.15, 0.08],
        momentum_12m_skip_1m=[0.20, 0.10, -0.05, 0.25, 0.12],
    )
    result = _compute(df)
    assert result.isna().sum() == 0


# ──────────────────────────────────────────────────────────────────────────────
# 2. 短期反转：负权重行为
# ──────────────────────────────────────────────────────────────────────────────

def test_short_reversal_high_momentum1m_gets_low_score():
    """momentum_1m 最高的股票应获得最低分（A股均值回归，负权重）。"""
    df = _make_df(
        momentum_1m=[0.20, 0.05, 0.01, -0.05, -0.15],  # 从高到低排序
    )
    result = _compute(df)
    # 最高 momentum_1m（index 0）应得最低分
    assert result.iloc[0] == result.min()


def test_short_reversal_low_momentum1m_gets_high_score():
    """momentum_1m 最低（最大幅度回调）的股票应获得最高分（负权重）。"""
    df = _make_df(
        momentum_1m=[0.20, 0.05, 0.01, -0.05, -0.15],
    )
    result = _compute(df)
    # 最低 momentum_1m（index 4）应得最高分
    assert result.iloc[4] == result.max()


def test_reversal_1w_negative_weight():
    """reversal_1w 最高的股票在其他条件相同时应得最低分。"""
    df = _make_df(
        reversal_1w=[0.05, 0.02, 0.00, -0.02, -0.05],  # 从高到低
        momentum_1m=[0.0, 0.0, 0.0, 0.0, 0.0],          # 中性（不影响排序）
    )
    result = _compute(df)
    assert result.iloc[0] < result.iloc[4]


# ──────────────────────────────────────────────────────────────────────────────
# 3. 中长期惯性：正权重行为
# ──────────────────────────────────────────────────────────────────────────────

def test_long_momentum_high_gets_high_score():
    """momentum_12m_skip_1m 最高的股票应获得最高分（正权重）。"""
    df = _make_df(
        momentum_12m_skip_1m=[0.40, 0.20, 0.05, -0.10, -0.30],
    )
    result = _compute(df)
    assert result.iloc[0] == result.max()


def test_momentum_6m_positive_weight():
    """momentum_6m 最高的股票在其他条件相同时应得最高分。"""
    df = _make_df(
        momentum_6m=[0.30, 0.15, 0.00, -0.10, -0.25],
        momentum_1m=[0.0, 0.0, 0.0, 0.0, 0.0],
    )
    result = _compute(df)
    assert result.iloc[0] == result.max()


# ──────────────────────────────────────────────────────────────────────────────
# 4. 缺失列降级逻辑
# ──────────────────────────────────────────────────────────────────────────────

def test_fallback_to_momentum_12m_when_skip_absent():
    """momentum_12m_skip_1m 缺失时，应降级使用 momentum_12m 且正权重方向一致。"""
    df_skip = _make_df(momentum_12m_skip_1m=[0.30, 0.10, 0.00, -0.10, -0.30])
    df_fallback = _make_df(momentum_12m=[0.30, 0.10, 0.00, -0.10, -0.30])

    res_skip = _compute(df_skip)
    res_fallback = _compute(df_fallback)

    # 两者排序应完全一致（正权重，同一数据）
    assert list(res_skip.rank()) == list(res_fallback.rank())


def test_no_columns_returns_homogeneous():
    """无任何动量相关列时，所有股票得相同分（score=0 → rank ties）。"""
    df = _make_df(unrelated=[1.0, 2.0, 3.0, 4.0, 5.0])
    result = _compute(df)
    # 全零 score → rank(pct=True) 所有 tie → 均等（pandas average rank）
    assert result.nunique() == 1, "全 tie 时所有股票应得相同分"
    assert result.min() >= 0.0 and result.max() <= 100.0


def test_partial_columns_still_produces_valid_output():
    """只有 momentum_1m 一列也能正常运行（仅负权重分量）。"""
    df = _make_df(momentum_1m=[0.10, 0.05, 0.00, -0.05, -0.10])
    result = _compute(df)
    assert result.isna().sum() == 0
    assert result.min() >= 0.0
    assert result.max() <= 100.0


# ──────────────────────────────────────────────────────────────────────────────
# 5. 鲁棒性
# ──────────────────────────────────────────────────────────────────────────────

def test_single_row_no_crash():
    """单只股票输入不应崩溃（rank pct 边界情况）。"""
    df = _make_df(momentum_1m=[0.05], momentum_12m_skip_1m=[0.20])
    result = _compute(df)
    assert len(result) == 1
    assert not np.isnan(result.iloc[0])


def test_all_nan_column_graceful():
    """某列全为 NaN 时，不应导致整体输出崩溃（zscore → 0，不传播 NaN）。"""
    df = pd.DataFrame({
        "momentum_1m": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "momentum_12m_skip_1m": [0.20, 0.10, 0.00, -0.10, -0.20],
    })
    result = _compute(df)
    # 全 NaN 列的贡献被归零，非 NaN 列正常贡献 → 输出不含 NaN
    assert result.isna().sum() == 0
    # 最高 momentum_12m_skip_1m 应得最高分（正权重）
    assert result.iloc[0] == result.max()


# ──────────────────────────────────────────────────────────────────────────────
# 6. expr_factors 新条目注册
# ──────────────────────────────────────────────────────────────────────────────

def test_new_expr_factors_registered():
    """4 个新因子应存在于 EXPR_FACTORS 字典中。"""
    from quantmind.features.expr_factors import EXPR_FACTORS  # noqa: PLC0415
    for name in ["reversal_1m_raw", "reversal_5d_raw", "momentum_6m_expr", "momentum_12m_skip_1m_expr"]:
        assert name in EXPR_FACTORS, f"{name!r} not found in EXPR_FACTORS"


def test_short_reversal_dict_keys():
    """SHORT_REVERSAL_FACTORS 含 reversal_1w 和 reversal_1m 键。"""
    from quantmind.features.expr_factors import SHORT_REVERSAL_FACTORS  # noqa: PLC0415
    assert "reversal_1w" in SHORT_REVERSAL_FACTORS
    assert "reversal_1m" in SHORT_REVERSAL_FACTORS


def test_mid_long_momentum_dict_keys():
    """MID_LONG_MOMENTUM_FACTORS 含 momentum_6m 和 momentum_12m_skip_1m 键。"""
    from quantmind.features.expr_factors import MID_LONG_MOMENTUM_FACTORS  # noqa: PLC0415
    assert "momentum_6m" in MID_LONG_MOMENTUM_FACTORS
    assert "momentum_12m_skip_1m" in MID_LONG_MOMENTUM_FACTORS


def test_momentum_factors_raw_length():
    """MOMENTUM_FACTORS_RAW 应包含 4 个条目（2 短 + 2 长）。"""
    from quantmind.features.expr_factors import MOMENTUM_FACTORS_RAW  # noqa: PLC0415
    assert len(MOMENTUM_FACTORS_RAW) == 4


# ──────────────────────────────────────────────────────────────────────────────
# 7. __init__.py 导出
# ──────────────────────────────────────────────────────────────────────────────

def test_init_exports_momentum_raw():
    """quantmind.features 应能直接导入 MOMENTUM_FACTORS_RAW 等新符号。"""
    import quantmind.features as qf  # noqa: PLC0415
    assert hasattr(qf, "MOMENTUM_FACTORS_RAW")
    assert hasattr(qf, "SHORT_REVERSAL_FACTORS")
    assert hasattr(qf, "MID_LONG_MOMENTUM_FACTORS")


# ──────────────────────────────────────────────────────────────────────────────
# 8. 综合：IC 方向一致性（构造已知因子验证符号）
# ──────────────────────────────────────────────────────────────────────────────

def test_combined_score_ic_direction():
    """构造理想数据（低短期动量 + 高长期动量 → 高评分），验证方向正确。

    设计：
      股票 A: momentum_1m=-0.10（近期回调），momentum_12m_skip_1m=+0.30（长趋势强）
      股票 E: momentum_1m=+0.10（近期上涨），momentum_12m_skip_1m=-0.30（长趋势弱）
    预期：score(A) > score(E)
    """
    df = pd.DataFrame({
        "momentum_1m":          [-0.10, -0.05, 0.00, +0.05, +0.10],
        "momentum_12m_skip_1m": [+0.30, +0.15, 0.00, -0.15, -0.30],
        "reversal_1w":          [-0.02, -0.01, 0.00, +0.01, +0.02],
        "momentum_6m":          [+0.15, +0.08, 0.00, -0.08, -0.15],
    })
    result = _compute(df)
    assert result.iloc[0] > result.iloc[4], (
        f"Expected score(A)={result.iloc[0]:.1f} > score(E)={result.iloc[4]:.1f}"
    )
