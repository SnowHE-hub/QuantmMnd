"""tests/test_features_standardize.py — 标准化模块单元测试."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantmind.features import (
    cross_section_rank,
    cross_section_zscore,
    fillna_cross_section,
    information_coefficient,
    neutralize,
    standardize,
    winsorize,
)

# ============================================================================
# Winsorize
# ============================================================================


class TestWinsorize:
    def test_sigma_method_clips_outliers(self) -> None:
        # 大量正常值 + 2 个极端 outlier；clip 后极端值必显著收缩
        rng = np.random.default_rng(0)
        normal = rng.normal(loc=0, scale=1, size=98).tolist()
        df = pd.DataFrame({"x": normal + [1000.0, -1000.0]})
        out = winsorize(df, method="sigma", sigma=3.0)
        # outliers 已被截断（不再是 ±1000；3σ 约 ±450）
        assert out["x"].max() < 600
        assert out["x"].min() > -600
        # 正常值（绝对值 < 5）保持不变
        normal_count = ((out["x"].abs() < 5)).sum()
        assert normal_count >= 95  # 几乎所有正常值都未受影响

    def test_pct_method_clips_quantiles(self) -> None:
        df = pd.DataFrame({"x": list(range(100))})
        out = winsorize(df, method="pct", pct=0.05)
        # 5%~95% 分位
        assert out["x"].min() >= 4.95
        assert out["x"].max() <= 94.05

    def test_constant_column_unchanged(self) -> None:
        df = pd.DataFrame({"x": [5.0, 5.0, 5.0]})
        out = winsorize(df)
        pd.testing.assert_series_equal(out["x"], df["x"])


# ============================================================================
# fillna
# ============================================================================


class TestFillna:
    def test_median_strategy(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 3.0]})
        out = fillna_cross_section(df, strategy="median")
        assert out["x"].iloc[2] == 2.0  # median(1,2,3) = 2

    def test_industry_grouped_median(self) -> None:
        df = pd.DataFrame(
            {"x": [1.0, np.nan, 5.0, np.nan, 10.0, 12.0]},
            index=["a", "b", "c", "d", "e", "f"],
        )
        ind = pd.Series(["g1", "g1", "g1", "g2", "g2", "g2"], index=df.index)
        out = fillna_cross_section(df, industry=ind)
        # g1 median = 3 (from 1, 5)
        # g2 median = 11 (from 10, 12)
        assert out["x"].loc["b"] == 3.0
        assert out["x"].loc["d"] == 11.0


# ============================================================================
# Z-score
# ============================================================================


class TestZScore:
    def test_basic_zscore(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        out = cross_section_zscore(df)
        assert out["x"].mean() == pytest.approx(0.0, abs=1e-9)
        assert out["x"].std() == pytest.approx(1.0, abs=1e-9)

    def test_constant_column_returns_zero(self) -> None:
        df = pd.DataFrame({"x": [5.0, 5.0, 5.0]})
        out = cross_section_zscore(df)
        assert (out["x"] == 0.0).all()

    def test_near_constant_column_returns_zero(self) -> None:
        # 浮点噪声 < 1e-9 也应被识别为常数
        df = pd.DataFrame({"x": [5.0, 5.0 + 1e-12, 5.0 - 1e-12]})
        out = cross_section_zscore(df)
        assert (out["x"] == 0.0).all()


class TestRank:
    def test_pct_rank(self) -> None:
        df = pd.DataFrame({"x": [10, 20, 30, 40]})
        out = cross_section_rank(df, pct=True)
        # 排名分位：10→0.25, 20→0.5, 30→0.75, 40→1.0
        assert out["x"].tolist() == [0.25, 0.5, 0.75, 1.0]


# ============================================================================
# Neutralize
# ============================================================================


class TestNeutralize:
    def test_residual_orthogonal_to_x(self) -> None:
        # 构造 y = 2 * x + noise，残差应与 x 不相关
        rng = np.random.default_rng(0)
        idx = pd.Index([f"t{i}" for i in range(50)])
        x = pd.Series(rng.normal(size=50), index=idx)
        noise = pd.Series(rng.normal(size=50), index=idx)
        y = 2 * x + noise

        df = pd.DataFrame({"y": y})
        out = neutralize(df, log_market_cap=x)
        # 残差与 x 的相关系数接近 0
        corr = out["y"].corr(x)
        assert abs(corr) < 0.05

    def test_no_op_when_no_predictors(self) -> None:
        df = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
        out = neutralize(df)
        pd.testing.assert_frame_equal(out, df)


# ============================================================================
# Standardize end-to-end
# ============================================================================


class TestStandardizePipeline:
    def test_output_is_zscore_per_column(self) -> None:
        rng = np.random.default_rng(1)
        df = pd.DataFrame(
            {
                "factor_a": rng.normal(loc=10, scale=2, size=30),
                "factor_b": rng.normal(loc=-5, scale=3, size=30),
                "log_market_cap": rng.normal(loc=20, scale=1, size=30),
            }
        )
        out = standardize(df, log_market_cap=df["log_market_cap"])
        for col in out.columns:
            mu = out[col].mean()
            sd = out[col].std()
            # 容忍 log_market_cap 跳过中性化但仍 zscore
            assert abs(mu) < 1e-6, f"{col} mean={mu}"
            assert abs(sd - 1.0) < 1e-6 or sd == 0, f"{col} std={sd}"

    def test_constant_factor_handled_gracefully(self) -> None:
        df = pd.DataFrame({"f": [3.14] * 20, "g": list(range(20))})
        out = standardize(df, do_neutralize=False)
        # constant factor → 0
        assert (out["f"] == 0).all()
        # g should be standardized
        assert abs(out["g"].mean()) < 1e-9
        assert abs(out["g"].std() - 1.0) < 1e-9


# ============================================================================
# IC
# ============================================================================


class TestIC:
    def test_perfect_correlation_ic_one(self) -> None:
        idx = pd.Index([f"t{i}" for i in range(50)])
        f = pd.Series(range(50), index=idx, dtype="float64")
        r = pd.Series(range(50), index=idx, dtype="float64")
        ic = information_coefficient(f, r, method="spearman")
        assert ic == pytest.approx(1.0)

    def test_anti_correlation_ic_neg_one(self) -> None:
        idx = pd.Index([f"t{i}" for i in range(50)])
        f = pd.Series(range(50), index=idx, dtype="float64")
        r = pd.Series(range(49, -1, -1), index=idx, dtype="float64")
        ic = information_coefficient(f, r)
        assert ic == pytest.approx(-1.0)

    def test_independent_low_ic(self) -> None:
        rng = np.random.default_rng(0)
        idx = pd.Index([f"t{i}" for i in range(200)])
        f = pd.Series(rng.normal(size=200), index=idx)
        r = pd.Series(rng.normal(size=200), index=idx)
        ic = information_coefficient(f, r)
        assert abs(ic) < 0.2  # 独立时 IC 应接近 0

    def test_too_few_samples_returns_nan(self) -> None:
        f = pd.Series([1.0, 2.0])
        r = pd.Series([1.0, 2.0])
        ic = information_coefficient(f, r)
        assert pd.isna(ic)
