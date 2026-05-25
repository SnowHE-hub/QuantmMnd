"""tests/test_irm_sentiment.py — disclosure_surprise 因子单元测试。

覆盖范围：
  - IRM_SENTIMENT_FACTORS 常量
  - score_forecast_records: type_score / flag_weight / magnitude_scale 映射
  - score_forecast_records: 边界情况（空 df, 未知 type, 缺失 p_change）
  - build_disclosure_factor: MultiIndex 格式，滚动窗口，缓存 mock
  - build_disclosure_factor: 缺失股票填 0（不在因子中）
  - compute_disclosure_ic: 接口一致性，空因子，样本量不足
  - compute_correlation_with_ann_contrarian: 高相关/低相关 / 缓存缺失
  - _TYPE_SCORE 常量覆盖所有已知 type
  - _FLAG_WEIGHT 默认值
  - run_30day_sim 集成模式（列名正确）
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from quantmind.features.irm_sentiment import (
    IRM_SENTIMENT_FACTORS,
    _FLAG_WEIGHT,
    _FLAG_WEIGHT_DEFAULT,
    _MAGNITUDE_TYPES,
    _TYPE_SCORE,
    build_disclosure_factor,
    compute_correlation_with_ann_contrarian,
    compute_disclosure_ic,
    score_forecast_records,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_forecast_df(n: int = 10) -> pd.DataFrame:
    """创建模拟 forecast DataFrame."""
    types_all = ["预增", "续亏", "首亏", "扭亏", "略增", "略减", "预减", "续盈", "减亏", "不确定"]
    flags_all  = ["0", "0", "1", "0", "0", "1", "0", "2", "0", "0"]
    pmin_all   = [10.0, np.nan, np.nan, np.nan, 5.0, -5.0, -20.0, np.nan, np.nan, np.nan]
    pmax_all   = [30.0, np.nan, np.nan, np.nan, 8.0, -8.0, -25.0, np.nan, np.nan, np.nan]
    return pd.DataFrame({
        "ts_code":        [f"{str(i).zfill(6)}.SZ" for i in range(1, n + 1)],
        "ann_date":       pd.date_range("2025-07-15", periods=n, freq="D"),
        "end_date":       ["20250630"] * n,
        "type":           types_all[:n],
        "p_change_min":   pmin_all[:n],
        "p_change_max":   pmax_all[:n],
        "net_profit_min":  [None] * n,
        "net_profit_max":  [None] * n,
        "last_parent_net": [None] * n,
        "first_ann_date":  ["20250715"] * n,
        "summary":         ["预计..."] * n,
        "change_reason":   [None] * n,
        "update_flag":     flags_all[:n],
    })


def _make_returns_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "ts_code":   [f"{str(i).zfill(6)}.SZ" for i in range(1, n + 1)],
        "return_3m": rng.normal(0.05, 0.15, n),
    })


# ── 1. 常量 ───────────────────────────────────────────────────────────────────

def test_irm_sentiment_factors_list():
    """IRM_SENTIMENT_FACTORS 包含 disclosure_contrarian_30d."""
    assert "disclosure_contrarian_30d" in IRM_SENTIMENT_FACTORS


def test_type_score_all_known_types():
    """_TYPE_SCORE 覆盖所有已知的业绩预告 type 值."""
    known_types = {"扭亏", "预增", "减亏", "略增", "续盈", "不确定",
                   "略减", "预减", "续亏", "首亏"}
    assert known_types == set(_TYPE_SCORE.keys())


def test_type_score_polarity():
    """正面类型 > 0，负面类型 < 0，中性 = 0."""
    assert _TYPE_SCORE["扭亏"] > 0
    assert _TYPE_SCORE["预增"] > 0
    assert _TYPE_SCORE["首亏"] < 0
    assert _TYPE_SCORE["续亏"] < 0
    assert _TYPE_SCORE["不确定"] == 0.0


def test_type_score_magnitude_ordering():
    """扭亏 > 预增 > 略增；首亏 < 续亏 < 略减（极端信号 > 温和信号）."""
    assert _TYPE_SCORE["扭亏"] > _TYPE_SCORE["预增"] > _TYPE_SCORE["略增"]
    assert _TYPE_SCORE["首亏"] < _TYPE_SCORE["续亏"] < _TYPE_SCORE["略减"]


def test_flag_weight_ordering():
    """首次(0) > 修正(1) > 默认（0.3），默认代表补充."""
    assert _FLAG_WEIGHT["0"] > _FLAG_WEIGHT["1"] > _FLAG_WEIGHT_DEFAULT


# ── 2. score_forecast_records ─────────────────────────────────────────────────

def test_score_forecast_empty_df():
    """空 DataFrame 返回带 disclosure_score 列的空 DataFrame."""
    empty = pd.DataFrame(columns=["ts_code", "ann_date", "type", "update_flag"])
    result = score_forecast_records(empty)
    assert "disclosure_score" in result.columns
    assert len(result) == 0


def test_score_forecast_type_score_correct():
    """预增 + flag=0 → type_score=1.0 × flag=1.0 = 1.0（无幅度调整时）."""
    df = pd.DataFrame({
        "ts_code":     ["000001.SZ"],
        "ann_date":    [pd.Timestamp("2025-07-15")],
        "type":        ["预增"],
        "update_flag": ["0"],
        "p_change_max": [np.nan],   # 无幅度调整
    })
    result = score_forecast_records(df)
    # type_score=1.0 × flag=1.0 × magnitude=1.0 = 1.0
    assert result["disclosure_score"].iloc[0] == pytest.approx(1.0)


def test_score_forecast_negative_type():
    """首亏 + flag=1 → -1.5 × 0.7 = -1.05."""
    df = pd.DataFrame({
        "ts_code":     ["000002.SZ"],
        "ann_date":    [pd.Timestamp("2025-07-15")],
        "type":        ["首亏"],
        "update_flag": ["1"],
        "p_change_max": [np.nan],
    })
    result = score_forecast_records(df)
    assert result["disclosure_score"].iloc[0] == pytest.approx(-1.5 * 0.7)


def test_score_forecast_unknown_type_defaults_zero():
    """未知 type → type_score=0.0 → disclosure_score=0.0."""
    df = pd.DataFrame({
        "ts_code":     ["000003.SZ"],
        "ann_date":    [pd.Timestamp("2025-07-15")],
        "type":        ["其他未知"],
        "update_flag": ["0"],
        "p_change_max": [np.nan],
    })
    result = score_forecast_records(df)
    assert result["disclosure_score"].iloc[0] == pytest.approx(0.0)


def test_score_forecast_magnitude_scale_applied():
    """预增 + p_change_max=30% → magnitude_scale ≈ 1.0（基准）."""
    import math
    df = pd.DataFrame({
        "ts_code":     ["000004.SZ"],
        "ann_date":    [pd.Timestamp("2025-07-15")],
        "type":        ["预增"],
        "update_flag": ["0"],
        "p_change_max": [30.0],   # 30% → scale = log1p(1)/log1p(3) ≈ 0.631
    })
    result = score_forecast_records(df)
    # 30%: scale = log1p(30/30) / log1p(3) = log(2)/log(4) ≈ 0.5
    expected_scale = np.log1p(30.0 / 30.0) / np.log1p(3.0)
    expected_score = 1.0 * 1.0 * expected_scale
    assert result["disclosure_score"].iloc[0] == pytest.approx(expected_score, rel=1e-4)


def test_score_forecast_magnitude_not_applied_for_neutral_types():
    """续亏/续盈（非 MAGNITUDE_TYPES）即使有 p_change 也不应用幅度调整."""
    df = pd.DataFrame({
        "ts_code":     ["000005.SZ"],
        "ann_date":    [pd.Timestamp("2025-07-15")],
        "type":        ["续亏"],
        "update_flag": ["0"],
        "p_change_max": [100.0],   # 不应触发 magnitude 调整
    })
    result = score_forecast_records(df)
    # 续亏 type_score=-1.0 × flag=1.0 × magnitude=1.0
    assert result["disclosure_score"].iloc[0] == pytest.approx(-1.0)


def test_score_forecast_default_flag_weight():
    """update_flag='5'（未知）使用默认权重 0.3."""
    df = pd.DataFrame({
        "ts_code":     ["000006.SZ"],
        "ann_date":    [pd.Timestamp("2025-07-15")],
        "type":        ["预增"],
        "update_flag": ["5"],
        "p_change_max": [np.nan],
    })
    result = score_forecast_records(df)
    assert result["disclosure_score"].iloc[0] == pytest.approx(1.0 * 0.3 * 1.0)


# ── 3. build_disclosure_factor 格式 ───────────────────────────────────────────

def test_build_disclosure_factor_multiindex():
    """输出为 MultiIndex(ts_code, ann_date)，name='disclosure_contrarian_30d'."""
    raw = _make_forecast_df(n=5)
    scored = score_forecast_records(raw)

    with patch("quantmind.features.irm_sentiment.fetch_forecast_data", return_value=raw):
        factor = build_disclosure_factor(
            start_date="2025-07-01",
            end_date="2025-08-01",
            use_cache=False,
        )

    assert isinstance(factor, pd.Series)
    assert factor.name == "disclosure_contrarian_30d"
    assert factor.index.names == ["ts_code", "ann_date"]


def test_build_disclosure_factor_empty_on_no_data():
    """fetch_forecast_data 返回空时，因子为空 Series."""
    with patch("quantmind.features.irm_sentiment.fetch_forecast_data",
               return_value=pd.DataFrame()):
        factor = build_disclosure_factor(use_cache=False)

    assert factor.empty
    assert factor.name == "disclosure_contrarian_30d"


def test_build_disclosure_factor_rolling_window():
    """同一股票多次公告 → 30日窗口内均值."""
    # 同一股 4条记录，间隔 7 天，全在窗口内
    raw = pd.DataFrame({
        "ts_code":     ["000001.SZ"] * 4,
        "ann_date":    pd.date_range("2025-07-01", periods=4, freq="7D"),
        "end_date":    ["20250630"] * 4,
        "type":        ["预增", "预增", "预增", "预增"],
        "update_flag": ["0"] * 4,
        "p_change_max": [np.nan] * 4,
    })
    with patch("quantmind.features.irm_sentiment.fetch_forecast_data", return_value=raw):
        factor = build_disclosure_factor(
            start_date="2025-07-01",
            end_date="2025-08-01",
            use_cache=False,
            window_days=30,
        )

    # 因子已取反（IC=-0.154 均值回归），预增 type_score=+1.0 取反后应为 -1.0
    vals = factor[factor.index.get_level_values("ts_code") == "000001.SZ"].values
    np.testing.assert_allclose(vals, -1.0, atol=1e-6)


# ── 4. compute_disclosure_ic ─────────────────────────────────────────────────

def test_compute_disclosure_ic_empty():
    """空因子返回 nan, valid=False."""
    result = compute_disclosure_ic(pd.Series(dtype=float, name="disclosure_contrarian_30d"))
    assert np.isnan(result["ic"])
    assert result["valid"] is False
    assert result["n"] == 0


def test_compute_disclosure_ic_with_returns():
    """传入 returns DataFrame 时正常计算 IC."""
    rng = np.random.default_rng(1)
    n = 60
    codes = [f"{str(i).zfill(6)}.SZ" for i in range(1, n + 1)]
    dates = [pd.Timestamp("2025-07-15")] * n
    factor = pd.Series(
        rng.normal(0, 1, n),
        index=pd.MultiIndex.from_arrays([codes, dates], names=["ts_code", "ann_date"]),
        name="disclosure_contrarian_30d",
    )
    returns = pd.DataFrame({
        "ts_code":   codes,
        "return_3m": rng.normal(0.05, 0.1, n),
    })
    result = compute_disclosure_ic(factor, returns=returns)
    assert result["n"] == n
    assert -1.0 <= result["ic"] <= 1.0
    assert "valid" in result


def test_compute_disclosure_ic_small_sample():
    """n < 30 时 valid=False."""
    rng = np.random.default_rng(2)
    n = 10
    codes = [f"{str(i).zfill(6)}.SZ" for i in range(1, n + 1)]
    factor = pd.Series(
        rng.normal(0, 1, n),
        index=pd.MultiIndex.from_arrays(
            [codes, [pd.Timestamp("2025-07-15")] * n],
            names=["ts_code", "ann_date"],
        ),
        name="disclosure_contrarian_30d",
    )
    returns = pd.DataFrame({
        "ts_code":   codes[:5],
        "return_3m": rng.normal(0, 0.1, 5),
    })
    result = compute_disclosure_ic(factor, returns=returns)
    assert result["valid"] is False


# ── 5. compute_correlation_with_ann_contrarian ────────────────────────────────

def test_correlation_no_cache_returns_nan(tmp_path):
    """ann_contrarian 缓存不存在时，返回 nan correlation（不崩溃）."""
    rng = np.random.default_rng(3)
    n = 20
    codes = [f"{str(i).zfill(6)}.SZ" for i in range(1, n + 1)]
    disc_factor = pd.Series(
        rng.normal(0, 1, n),
        index=pd.MultiIndex.from_arrays(
            [codes, [pd.Timestamp("2025-07-15")] * n],
            names=["ts_code", "ann_date"],
        ),
    )
    # 使用 tmp_path，不存在缓存
    with patch("quantmind.features.irm_sentiment._TEXT_DIR", tmp_path):
        result = compute_correlation_with_ann_contrarian(
            disc_factor, ann_contrarian_factor=None
        )
    assert np.isnan(result["correlation"])
    assert result["independent"] is True  # 默认独立


def test_correlation_low_correlation_independent():
    """低相关（< 0.5）→ independent=True."""
    rng = np.random.default_rng(4)
    n = 60
    codes = [f"{str(i).zfill(6)}.SZ" for i in range(1, n + 1)]
    disc = pd.Series(
        rng.normal(0, 1, n),
        index=pd.MultiIndex.from_arrays(
            [codes, [pd.Timestamp("2025-07-15")] * n], names=["ts_code", "ann_date"]
        ),
    )
    ann = pd.Series(
        rng.normal(0, 1, n),  # 独立随机 → 低相关
        index=pd.MultiIndex.from_arrays(
            [codes, [pd.Timestamp("2025-07-15")] * n], names=["ts_code", "trade_date"]
        ),
        name="ann_contrarian_5d",
    )
    result = compute_correlation_with_ann_contrarian(disc, ann_contrarian_factor=ann)
    assert result["independent"] is True
    assert result["n"] == n


def test_correlation_high_correlation_not_independent():
    """高相关（≥ 0.5）→ independent=False."""
    rng = np.random.default_rng(5)
    n = 60
    codes = [f"{str(i).zfill(6)}.SZ" for i in range(1, n + 1)]
    base = rng.normal(0, 1, n)
    disc = pd.Series(
        base,
        index=pd.MultiIndex.from_arrays(
            [codes, [pd.Timestamp("2025-07-15")] * n], names=["ts_code", "ann_date"]
        ),
    )
    ann = pd.Series(
        base + rng.normal(0, 0.05, n),  # 几乎相同 → 高相关
        index=pd.MultiIndex.from_arrays(
            [codes, [pd.Timestamp("2025-07-15")] * n], names=["ts_code", "trade_date"]
        ),
        name="ann_contrarian_5d",
    )
    result = compute_correlation_with_ann_contrarian(disc, ann_contrarian_factor=ann)
    assert result["independent"] is False
    assert abs(result["correlation"]) >= 0.5


# ── 6. 与 run_30day_sim 的接口兼容性 ─────────────────────────────────────────

def test_disclosure_factor_column_name_matches_pipeline():
    """因子 Series.name 等于 pipeline 中检查的列名 'disclosure_contrarian_30d'."""
    raw = _make_forecast_df(n=3)
    with patch("quantmind.features.irm_sentiment.fetch_forecast_data", return_value=raw):
        factor = build_disclosure_factor(
            start_date="2025-07-01",
            end_date="2025-08-01",
            use_cache=False,
        )
    # 因子 name 必须与 run_30day_sim.py 中 "disclosure_contrarian_30d" in df.columns 匹配
    assert factor.name == "disclosure_contrarian_30d"
