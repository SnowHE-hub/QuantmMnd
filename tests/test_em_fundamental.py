"""tests/test_em_fundamental.py

EM 隐变量基本面质量因子单元测试（≥ 15 个）。

覆盖范围
--------
  1   GaussianMixtureEM 实例化：属性默认值
  2   GaussianMixtureEM fit：不报错，设置 converged_
  3   GaussianMixtureEM predict_proba：形状 (N, K)
  4   GaussianMixtureEM predict_proba：每行和为 1
  5   GaussianMixtureEM predict_proba：值域 [0, 1]
  6   GaussianMixtureEM predict：argmax 与 predict_proba 一致
  7   GaussianMixtureEM 对可分离 1D 数据：成分均值单调有序
  8   GaussianMixtureEM sort_components_by_dim：排序后 μ 单调
  9   GaussianMixtureEM 未 fit 时 predict_proba 报 RuntimeError
  10  _cross_section_standardize：NaN 被填充为 0
  11  _cross_section_standardize：均值 ≈ 0，std ≈ 1
  12  _cross_section_standardize：全相同值列 → 全 0
  13  _cross_section_standardize：输入输出形状相同
  14  build_em_fundamental_factor：输出为 pd.Series
  15  build_em_fundamental_factor：值域 [0, 1]
  16  build_em_fundamental_factor：MultiIndex 包含 (as_of, ticker)
  17  build_em_fundamental_factor：股票数不足 min_samples 时跳过该期
  18  build_em_fundamental_factor：obs_col 全缺失返回空 Series
  19  build_em_fundamental_factor：自定义 ic_weights 可正常运行
  20  blend_quality_score：em_weight=0 → 等于 original_quality
  21  blend_quality_score：em_weight=1 → 近似等于 em_quality（同量纲后）
  22  blend_quality_score：em_quality 为空 → 返回 original_quality
  23  EM_IC_WEIGHTS 归一化（和 ≈ 1）
  24  EM_OBS_COLS 长度为 4
  25  EM_FUNDAMENTAL_FACTORS 包含 fundamental_quality_score
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantmind.features.em_fundamental import (
    EM_FUNDAMENTAL_FACTORS,
    EM_IC_WEIGHTS,
    EM_N_COMPONENTS,
    EM_OBS_COLS,
    GaussianMixtureEM,
    _cross_section_standardize,
    blend_quality_score,
    build_em_fundamental_factor,
)


# ─── 工具函数 ──────────────────────────────────────────────────────────────────

def _make_mini_panel(
    n_quarters: int = 3,
    n_stocks: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """生成包含所有 EM_OBS_COLS + forward_return_63d 的微型 panel。"""
    rng = np.random.default_rng(seed)
    quarters = pd.date_range("2022-03-31", periods=n_quarters, freq="QE")
    tickers  = [f"{i:06d}.SZ" for i in range(n_stocks)]

    idx = pd.MultiIndex.from_product(
        [quarters, tickers], names=["as_of", "ticker"]
    )
    n_total = len(idx)
    data: dict[str, np.ndarray] = {}
    for col in EM_OBS_COLS:
        data[col] = rng.normal(0, 1, n_total).astype(np.float32)
    data["forward_return_63d"] = rng.normal(0, 0.1, n_total).astype(np.float32)

    df = pd.DataFrame(data, index=idx)
    # 插入少量 NaN
    df.iloc[:5, 0] = np.nan
    return df


def _well_separated_1d(seed: int = 0) -> np.ndarray:
    """K=3 明显分离的 1D 数据，各 50 点。"""
    rng = np.random.default_rng(seed)
    return np.concatenate([
        rng.normal(-5, 0.5, 50),
        rng.normal( 0, 0.5, 50),
        rng.normal( 5, 0.5, 50),
    ]).reshape(-1, 1)


# ─── 测试 1-9：GaussianMixtureEM ──────────────────────────────────────────────

class TestGaussianMixtureEM:

    def test_instantiation_defaults(self):
        """1. 实例化后属性默认值正确。"""
        gmm = GaussianMixtureEM()
        assert gmm.n_components == EM_N_COMPONENTS
        assert gmm.max_iter == 200
        assert gmm.tol == pytest.approx(1e-4)
        assert gmm.mu_ is None
        assert not gmm.converged_

    def test_fit_no_error(self):
        """2. fit 可在正常数据上运行（不报错，n_iter_ > 0）。"""
        X = _well_separated_1d()
        gmm = GaussianMixtureEM(n_components=3, max_iter=50)
        gmm.fit(X)
        assert gmm.n_iter_ > 0
        assert gmm.mu_ is not None

    def test_predict_proba_shape(self):
        """3. predict_proba 输出形状为 (N, K)。"""
        X = _well_separated_1d()
        gmm = GaussianMixtureEM(n_components=3).fit(X)
        proba = gmm.predict_proba(X)
        assert proba.shape == (len(X), 3)

    def test_predict_proba_sums_to_one(self):
        """4. predict_proba 每行和为 1（概率归一化）。"""
        X = _well_separated_1d()
        gmm = GaussianMixtureEM(n_components=3).fit(X)
        proba = gmm.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_range_zero_one(self):
        """5. predict_proba 值域 [0, 1]。"""
        X = _well_separated_1d()
        gmm = GaussianMixtureEM(n_components=3).fit(X)
        proba = gmm.predict_proba(X)
        assert proba.min() >= 0.0 - 1e-9
        assert proba.max() <= 1.0 + 1e-9

    def test_predict_argmax_consistent(self):
        """6. predict 等于 predict_proba.argmax。"""
        X = _well_separated_1d()
        gmm = GaussianMixtureEM(n_components=3).fit(X)
        labels = gmm.predict(X)
        expected = gmm.predict_proba(X).argmax(axis=1)
        np.testing.assert_array_equal(labels, expected)

    def test_well_separated_means_monotone(self):
        """7. 对可分离 1D 数据，sort 后 μ 应严格单调递增。"""
        X = _well_separated_1d()
        gmm = GaussianMixtureEM(n_components=3).fit(X)
        gmm.sort_components_by_dim(dim=0)
        mus = gmm.mu_[:, 0]
        assert mus[0] < mus[1] < mus[2], f"μ 未单调递增: {mus}"

    def test_sort_components_by_dim(self):
        """8. sort_components_by_dim 后指定维度 μ 严格单调。"""
        rng = np.random.default_rng(123)
        X   = rng.normal(0, 1, (200, 2))
        gmm = GaussianMixtureEM(n_components=3).fit(X)
        # 按 dim=1 排序
        gmm.sort_components_by_dim(dim=1)
        mus_dim1 = gmm.mu_[:, 1]
        assert mus_dim1[0] <= mus_dim1[1] <= mus_dim1[2]

    def test_predict_proba_raises_before_fit(self):
        """9. 未 fit 时调用 predict_proba 应抛 RuntimeError。"""
        gmm = GaussianMixtureEM()
        with pytest.raises(RuntimeError, match="fit"):
            gmm.predict_proba(np.array([[1.0, 2.0]]))


# ─── 测试 10-13：_cross_section_standardize ────────────────────────────────────

class TestCrossSectionStandardize:

    def test_nan_filled_with_zero(self):
        """10. NaN 被填充为截面中性值 0。"""
        rng = np.random.default_rng(0)
        X   = rng.normal(0, 1, (100, 4))
        X[0, 0] = np.nan
        X[5, 2] = np.nan
        out = _cross_section_standardize(X)
        assert not np.isnan(out).any()

    def test_mean_zero_std_one(self):
        """11. 标准化后（非 NaN 列）均值 ≈ 0，std ≈ 1。"""
        rng = np.random.default_rng(1)
        X   = rng.normal(10, 3, (300, 4))
        out = _cross_section_standardize(X)
        for j in range(out.shape[1]):
            assert abs(out[:, j].mean()) < 0.05, f"col {j} 均值偏差"
            assert abs(out[:, j].std() - 1.0) < 0.05, f"col {j} std 偏差"

    def test_constant_column_returns_zeros(self):
        """12. 全相同值列 → 全 0（不崩溃）。"""
        X = np.ones((50, 3))
        X[:, 1] = np.random.default_rng(2).normal(0, 1, 50)
        out = _cross_section_standardize(X)
        np.testing.assert_array_equal(out[:, 0], 0.0)

    def test_shape_preserved(self):
        """13. 输入输出形状相同。"""
        X   = np.random.default_rng(3).normal(0, 1, (120, 4))
        out = _cross_section_standardize(X)
        assert out.shape == X.shape


# ─── 测试 14-19：build_em_fundamental_factor ──────────────────────────────────

class TestBuildEmFundamentalFactor:

    def test_output_is_series(self):
        """14. 输出类型为 pd.Series。"""
        panel = _make_mini_panel()
        result = build_em_fundamental_factor(panel)
        assert isinstance(result, pd.Series)

    def test_output_range_zero_one(self):
        """15. 输出值域 [0, 1]。"""
        panel  = _make_mini_panel()
        result = build_em_fundamental_factor(panel)
        assert result.min() >= -1e-9
        assert result.max() <= 1.0 + 1e-9

    def test_multiindex_names(self):
        """16. 输出 MultiIndex 包含 (as_of, ticker)。"""
        panel  = _make_mini_panel()
        result = build_em_fundamental_factor(panel)
        assert result.index.names == ["as_of", "ticker"]

    def test_skips_small_quarters(self):
        """17. 股票数不足 min_samples 时该期被跳过。"""
        panel = _make_mini_panel(n_stocks=15, n_quarters=2)
        # min_samples=20 → 15 < 20 → 全部跳过 → 空 Series
        result = build_em_fundamental_factor(panel, min_samples=20)
        assert len(result) == 0

    def test_missing_obs_col_returns_empty(self):
        """18. 所有 obs_col 均不存在时返回空 Series。"""
        panel  = _make_mini_panel()
        result = build_em_fundamental_factor(
            panel, obs_cols=["nonexistent_col"]
        )
        assert len(result) == 0

    def test_custom_ic_weights(self):
        """19. 自定义 ic_weights 可正常运行，输出值域 [0, 1]。"""
        panel  = _make_mini_panel()
        custom = np.array([0.5, 0.3, 0.2, 0.0])
        result = build_em_fundamental_factor(panel, ic_weights=custom)
        assert isinstance(result, pd.Series)
        assert result.min() >= -1e-9
        assert result.max() <= 1.0 + 1e-9


# ─── 测试 20-22：blend_quality_score ─────────────────────────────────────────

class TestBlendQualityScore:

    def _make_pair(self, n: int = 100, seed: int = 0):
        rng  = np.random.default_rng(seed)
        idx  = pd.Index([f"{i:06d}.SZ" for i in range(n)], name="ticker")
        orig = pd.Series(rng.normal(0, 1, n), index=idx, name="quality_score")
        em   = pd.Series(rng.uniform(0, 1, n), index=idx,
                         name="fundamental_quality_score")
        return orig, em

    def test_weight_zero_returns_original(self):
        """20. em_weight=0 → 与 original_quality 完全一致（共同 index）。"""
        orig, em = self._make_pair()
        result   = blend_quality_score(orig, em, em_weight=0.0)
        pd.testing.assert_series_equal(result, orig.rename("quality_score_blended"),
                                       check_names=False)

    def test_weight_one_approx_em(self):
        """21. em_weight=1 → 结果与 em_quality 秩相关 ≈ 1。"""
        orig, em = self._make_pair()
        result   = blend_quality_score(orig, em, em_weight=1.0)
        corr = result.corr(em, method="spearman")
        assert corr > 0.99, f"秩相关 {corr:.4f} < 0.99"

    def test_empty_em_returns_original(self):
        """22. em_quality 为空 → 返回 original_quality（不报错）。"""
        orig = pd.Series([1.0, 2.0, 3.0], name="quality_score")
        em   = pd.Series(dtype=float)
        result = blend_quality_score(orig, em)
        pd.testing.assert_series_equal(result, orig)


# ─── 测试 23-25：常量校验 ──────────────────────────────────────────────────────

def test_em_ic_weights_normalized():
    """23. EM_IC_WEIGHTS 归一化（和 ≈ 1）。"""
    assert abs(EM_IC_WEIGHTS.sum() - 1.0) < 1e-6, (
        f"EM_IC_WEIGHTS.sum() = {EM_IC_WEIGHTS.sum()}"
    )


def test_em_obs_cols_length():
    """24. EM_OBS_COLS 包含 4 个列名。"""
    assert len(EM_OBS_COLS) == 4


def test_em_fundamental_factors_name():
    """25. EM_FUNDAMENTAL_FACTORS 包含 'fundamental_quality_score'。"""
    assert "fundamental_quality_score" in EM_FUNDAMENTAL_FACTORS
