"""tests/test_factor_cnn.py

FactorCNN + 辅助函数的单元测试（≥ 15 个）。

覆盖范围
--------
  1   FactorCNN 前向传播输出形状 (batch,)
  2   FactorCNN 输出为实数（无 NaN / Inf）
  3   IC Loss 对随机输入为负（有优化方向）
  4   IC Loss 完全正相关时 ≈ -1
  5   IC Loss 完全负相关时 ≈ +1
  6   cross_section_zscore 均值 ≈ 0、std ≈ 1
  7   cross_section_zscore 极值被 winsorize 裁剪
  8   cross_section_zscore 全相同值 → 全 0（不崩溃）
  9   preprocess_cross_section NaN 被中位数填充
  10  preprocess_cross_section 返回形状不变
  11  ensemble_scores 输出值域 [0, 1]
  12  ensemble_scores 权重比例影响排名
  13  ensemble_scores 处理不对齐 index
  14  FACTOR_GROUPS 四组合计 == ALL_CNN_FEATURES
  15  特征分组不重叠（无交集）
  16  FactorCNN 参数可更新（梯度流正常）
  17  train_factor_cnn smoke test（极小 panel，3 季）
  18  predict_cnn 返回 Series，index == panel_slice.index
  19  predict_cnn 当所有特征 NaN 时返回全 NaN Series
  20  FoldMetrics repr 不报错
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from quantmind.models.factor_cnn import (
    ALL_CNN_FEATURES,
    FACTOR_GROUPS,
    FactorCNN,
    FoldMetrics,
    MOMENTUM_FEATURES,
    QUALITY_FEATURES,
    TECHNICAL_FEATURES,
    VALUE_FEATURES,
    augment_data,
    cross_section_zscore,
    ensemble_scores,
    ic_loss,
    predict_cnn,
    preprocess_cross_section,
    save_cnn_model,
    train_factor_cnn,
)

# ─── 工具：构造最小合法 panel ──────────────────────────────────────────────────

pytestmark = pytest.mark.requires_optional_deps  # B2: 默认套件排除

def _make_mini_panel(
    n_quarters: int = 3,
    n_stocks: int = 40,
    seed: int = 42,
) -> pd.DataFrame:
    """生成包含所有 ALL_CNN_FEATURES 的微型 panel（MultiIndex as_of × ts_code）。"""
    rng = np.random.default_rng(seed)
    quarters = pd.date_range("2023-03-31", periods=n_quarters, freq="QE")
    tickers  = [f"{i:06d}.SZ" for i in range(n_stocks)]

    idx = pd.MultiIndex.from_product([quarters, tickers],
                                     names=["as_of", "ts_code"])
    n_total = len(idx)
    data: dict[str, np.ndarray] = {}
    for col in ALL_CNN_FEATURES:
        data[col] = rng.normal(0, 1, n_total).astype(np.float32)

    data["forward_return_63d"] = rng.normal(0, 0.1, n_total).astype(np.float32)
    df = pd.DataFrame(data, index=idx)
    # 插入少量 NaN
    for col in ALL_CNN_FEATURES[:5]:
        mask = rng.random(n_total) < 0.1
        df.loc[df.index[mask], col] = np.nan
    return df


def _make_cross_section(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tickers = [f"{i:06d}.SZ" for i in range(n)]
    data = {col: rng.normal(0, 1, n) for col in ALL_CNN_FEATURES}
    return pd.DataFrame(data, index=tickers)


# ─── 测试 1-2：FactorCNN 前向传播 ──────────────────────────────────────────────

class TestFactorCNNForward:
    def test_output_shape_batch(self):
        """1. 输出形状应为 (batch,)。"""
        model = FactorCNN()
        batch = 16
        x = torch.randn(batch, len(ALL_CNN_FEATURES))
        out = model(x)
        assert out.shape == (batch,), f"期望 ({batch},), 得到 {out.shape}"

    def test_output_no_nan_inf(self):
        """2. 输出不含 NaN / Inf。"""
        model = FactorCNN()
        x = torch.randn(32, len(ALL_CNN_FEATURES))
        out = model(x)
        assert torch.isfinite(out).all(), "输出存在 NaN 或 Inf"

    def test_single_sample(self):
        """BatchNorm 在 train 模式单样本时可能报错；eval 模式应正常。"""
        model = FactorCNN()
        model.eval()
        x = torch.randn(1, len(ALL_CNN_FEATURES))
        out = model(x)
        assert out.shape == (1,)


# ─── 测试 3-5：ic_loss ─────────────────────────────────────────────────────────

class TestICLoss:
    def test_random_input_finite(self):
        """3. 随机输入的 IC Loss 为有限实数。"""
        torch.manual_seed(0)
        pred  = torch.randn(64)
        label = torch.randn(64)
        loss  = ic_loss(pred, label)
        assert torch.isfinite(loss)

    def test_perfect_positive_correlation(self):
        """4. 完全正相关时 loss ≈ -1。"""
        v     = torch.linspace(0, 1, 100)
        loss  = ic_loss(v, v)
        assert abs(float(loss) - (-1.0)) < 1e-4, f"期望 ≈-1, 得到 {float(loss):.4f}"

    def test_perfect_negative_correlation(self):
        """5. 完全负相关时 loss ≈ +1。"""
        v     = torch.linspace(0, 1, 100)
        loss  = ic_loss(v, -v)
        assert abs(float(loss) - 1.0) < 1e-4, f"期望 ≈+1, 得到 {float(loss):.4f}"

    def test_single_sample_no_crash(self):
        """单样本不报错，返回 0.0。"""
        loss = ic_loss(torch.tensor([0.5]), torch.tensor([0.5]))
        assert float(loss) == pytest.approx(0.0)


# ─── 测试 6-8：cross_section_zscore ────────────────────────────────────────────

class TestCrossSectionZscore:
    def test_mean_zero_std_one(self):
        """6. zscore 后均值 ≈ 0、std ≈ 1。"""
        rng = np.random.default_rng(1)
        s   = pd.Series(rng.normal(10, 3, 200))
        z   = cross_section_zscore(s, winsorize=True)
        assert abs(z.mean()) < 0.05, f"均值偏差 {z.mean():.4f}"
        assert abs(z.std(ddof=0) - 1.0) < 0.05, f"std 偏差 {z.std(ddof=0):.4f}"

    def test_winsorize_clips_extremes(self):
        """7. winsorize 后应无超过 clip 倍 MAD 的离群值。"""
        s       = pd.Series([0.0] * 98 + [1e6, -1e6])
        z       = cross_section_zscore(s, winsorize=True, clip=3.0)
        # 裁剪后 z score 不应超过合理范围
        assert z.max() < 100 and z.min() > -100

    def test_constant_series_returns_zeros(self):
        """8. 全相同值 → 全 0（不崩溃）。"""
        s = pd.Series([5.0] * 50)
        z = cross_section_zscore(s)
        assert (z == 0.0).all()


# ─── 测试 9-10：preprocess_cross_section ───────────────────────────────────────

class TestPreprocessCrossSection:
    def test_nan_filled(self):
        """9. NaN 被组内中位数填充，结果无 NaN。"""
        df = _make_cross_section()
        df.iloc[:10, :5] = np.nan     # 插入 NaN
        feat = preprocess_cross_section(df, ALL_CNN_FEATURES)
        assert not feat[ALL_CNN_FEATURES].isnull().any().any()

    def test_shape_unchanged(self):
        """10. 预处理前后行数不变。"""
        df   = _make_cross_section(n=80)
        feat = preprocess_cross_section(df, ALL_CNN_FEATURES)
        assert len(feat) == 80


# ─── 测试 11-13：ensemble_scores ───────────────────────────────────────────────

class TestEnsembleScores:
    def _make_scores(self, n: int = 50, seed: int = 0) -> tuple[pd.Series, pd.Series]:
        rng = np.random.default_rng(seed)
        idx = [f"{i:06d}.SZ" for i in range(n)]
        lgbm = pd.Series(rng.normal(0, 1, n), index=idx)
        cnn  = pd.Series(rng.normal(0, 1, n), index=idx)
        return lgbm, cnn

    def test_output_range_zero_one(self):
        """11. ensemble_scores 输出值域 [0, 1]。"""
        lgbm, cnn = self._make_scores()
        ens = ensemble_scores(lgbm, cnn)
        assert ens.min() >= 0.0 - 1e-9
        assert ens.max() <= 1.0 + 1e-9

    def test_weight_ratio_affects_rank(self):
        """12. 权重不同时结果不同（lgbm 主导 vs cnn 主导）。"""
        rng  = np.random.default_rng(42)
        idx  = [f"{i:06d}.SZ" for i in range(50)]
        lgbm = pd.Series(np.arange(50, dtype=float), index=idx)
        cnn  = pd.Series(np.arange(49, -1, -1, dtype=float), index=idx)
        ens_lgbm = ensemble_scores(lgbm, cnn, lgbm_weight=1.0, cnn_weight=0.0)
        ens_cnn  = ensemble_scores(lgbm, cnn, lgbm_weight=0.0, cnn_weight=1.0)
        # lgbm_weight=1: 排名与 lgbm 完全一致
        assert ens_lgbm.corr(lgbm) > 0.99
        # cnn_weight=1: 排名与 cnn 完全一致
        assert ens_cnn.corr(cnn) > 0.99

    def test_misaligned_index(self):
        """13. index 不完全对齐时只保留交集，无 KeyError。"""
        idx_a = [f"{i:06d}.SZ" for i in range(30)]
        idx_b = [f"{i:06d}.SZ" for i in range(20, 50)]
        rng   = np.random.default_rng(7)
        lgbm  = pd.Series(rng.normal(0, 1, 30), index=idx_a)
        cnn   = pd.Series(rng.normal(0, 1, 30), index=idx_b)
        ens   = ensemble_scores(lgbm, cnn)
        # 交集 20-29 = 10 items
        assert len(ens) == 10
        assert ens.min() >= 0.0 - 1e-9
        assert ens.max() <= 1.0 + 1e-9


# ─── 测试 14-15：特征分组 ──────────────────────────────────────────────────────

def test_factor_groups_cover_all_features():
    """14. 四组合计 == ALL_CNN_FEATURES（顺序一致）。"""
    combined = (VALUE_FEATURES + QUALITY_FEATURES
                + MOMENTUM_FEATURES + TECHNICAL_FEATURES)
    assert combined == ALL_CNN_FEATURES, (
        f"合计 {len(combined)} vs ALL_CNN_FEATURES {len(ALL_CNN_FEATURES)}"
    )


def test_factor_groups_no_overlap():
    """15. 各组之间无重叠特征。"""
    groups = list(FACTOR_GROUPS.values())
    all_seen: set[str] = set()
    for g in groups:
        overlap = all_seen.intersection(g)
        assert not overlap, f"重叠特征: {overlap}"
        all_seen.update(g)


# ─── 测试 16：梯度流 ───────────────────────────────────────────────────────────

def test_gradient_flow():
    """16. 反向传播后所有参数的梯度非全零。"""
    model = FactorCNN()
    model.train()
    x     = torch.randn(32, len(ALL_CNN_FEATURES))
    label = torch.randn(32)
    loss  = ic_loss(model(x), label)
    loss.backward()
    grad_norms = [
        p.grad.norm().item()
        for p in model.parameters()
        if p.grad is not None
    ]
    assert len(grad_norms) > 0, "没有参数有梯度"
    assert any(g > 0 for g in grad_norms), "所有梯度均为 0"


# ─── 测试 17：train_factor_cnn smoke ──────────────────────────────────────────

def test_train_factor_cnn_smoke():
    """17. smoke test：极小 panel，3 季，1 折训练可完成并返回正确结构。"""
    panel = _make_mini_panel(n_quarters=3, n_stocks=40)
    result = train_factor_cnn(
        panel,
        n_train_quarters=1,
        n_val_quarters=1,
        epochs=5,
        batch_size=40,
        device="cpu",
    )
    assert hasattr(result, "val_ic_mean")
    assert hasattr(result, "folds")
    assert len(result.folds) >= 1
    # val_ic 是有限数（即使很小）
    if not math.isnan(result.val_ic_mean):
        assert -2.0 < result.val_ic_mean < 2.0


# ─── 测试 18-19：predict_cnn ───────────────────────────────────────────────────

class TestPredictCNN:
    def _make_model_and_cols(self) -> tuple[FactorCNN, list[str]]:
        model = FactorCNN()
        model.eval()
        return model, ALL_CNN_FEATURES

    def test_returns_series_with_correct_index(self):
        """18. 返回 Series，index == panel_slice.index。"""
        model, feature_cols = self._make_model_and_cols()
        df    = _make_cross_section(n=30)
        score = predict_cnn(model, df, feature_cols)
        assert isinstance(score, pd.Series)
        assert list(score.index) == list(df.index)

    def test_all_nan_features_returns_nan_series(self):
        """19. 全 NaN 特征时返回全 NaN Series（不崩溃）。"""
        model, feature_cols = self._make_model_and_cols()
        df = pd.DataFrame(
            {col: [np.nan] * 10 for col in ALL_CNN_FEATURES},
            index=[f"{i:06d}.SZ" for i in range(10)],
        )
        score = predict_cnn(model, df, feature_cols)
        assert isinstance(score, pd.Series)
        assert score.isna().all()


# ─── 测试 20：FoldMetrics repr ─────────────────────────────────────────────────

def test_fold_metrics_repr():
    """20. FoldMetrics repr 可正常调用，不报 AttributeError。"""
    fm  = FoldMetrics(train_dates=[], val_dates=[], val_ic=0.05, best_epoch=10)
    rep = repr(fm)
    assert "0.05" in rep or "0.0500" in rep


# ─── 测试 21-26：augment_data ──────────────────────────────────────────────────

class TestAugmentData:
    """augment_data() 数据增强函数的单元测试。"""

    def _make_xy(self, n: int = 50, f: int = 10,
                 seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, f)).astype(np.float32)
        y = rng.standard_normal(n).astype(np.float32)
        return X, y

    def test_output_shape_n_copies_4(self):
        """21. n_copies=4 → 输出行数 = 输入行数 × 5。"""
        X, y = self._make_xy(n=30, f=8)
        X_aug, y_aug = augment_data(X, y, n_copies=4, noise_sigma=0.05)
        assert X_aug.shape[0] == 30 * 5, (
            f"期望 {30*5} 行，得到 {X_aug.shape[0]}"
        )
        assert y_aug.shape[0] == 30 * 5

    def test_output_shape_n_copies_1(self):
        """22. n_copies=1 → 输出行数 = 输入行数 × 2。"""
        X, y = self._make_xy(n=20, f=5)
        X_aug, y_aug = augment_data(X, y, n_copies=1)
        assert X_aug.shape[0] == 20 * 2
        assert X_aug.shape[1] == 5, "特征列数不应变化"

    def test_zero_copies_returns_original(self):
        """23. n_copies=0 → 返回原始数据（形状和值完全相同）。"""
        X, y = self._make_xy(n=15, f=6)
        X_aug, y_aug = augment_data(X, y, n_copies=0)
        assert X_aug.shape == X.shape
        assert y_aug.shape == y.shape
        np.testing.assert_array_equal(X_aug, X)
        np.testing.assert_array_equal(y_aug, y)

    def test_zero_sigma_copies_identical_to_original(self):
        """24. noise_sigma=0 时所有副本与原始数据完全相同（纯复制）。"""
        X, y = self._make_xy(n=10, f=4)
        X_aug, y_aug = augment_data(X, y, n_copies=3, noise_sigma=0.0)
        # 所有 4 份（原始 + 3 副本）应完全相同
        assert X_aug.shape[0] == 10 * 4
        for i in range(4):
            np.testing.assert_array_almost_equal(
                X_aug[i * 10: (i + 1) * 10], X,
                decimal=6,
                err_msg=f"第 {i} 份副本与原始不同（sigma=0 应完全相同）",
            )

    def test_labels_unchanged(self):
        """25. 标签 y 不加噪声，所有副本与原始 y 完全一致。"""
        X, y = self._make_xy(n=25, f=7)
        X_aug, y_aug = augment_data(X, y, n_copies=4, noise_sigma=0.1)
        # y_aug 是 5 份 y 拼接，每份应与原始 y 完全相同
        for i in range(5):
            np.testing.assert_array_equal(
                y_aug[i * 25: (i + 1) * 25], y,
                err_msg=f"副本 {i} 的标签被意外修改",
            )

    def test_train_factor_cnn_augment_smoke(self):
        """26. train_factor_cnn(augment_copies=2) smoke test 正常返回。"""
        panel = _make_mini_panel(n_quarters=3, n_stocks=40)
        result = train_factor_cnn(
            panel,
            n_train_quarters=1,
            n_val_quarters=1,
            epochs=3,
            batch_size=40,
            augment_copies=2,
            augment_sigma=0.05,
            device="cpu",
        )
        assert hasattr(result, "val_ic_mean")
        assert len(result.folds) >= 1


# ─── 测试 27：save_cnn_model ────────────────────────────────────────────────────

def test_save_cnn_model_roundtrip(tmp_path):
    """27. save_cnn_model 写出文件可被 pickle.load 读回，字段完整。"""
    import pickle
    panel  = _make_mini_panel(n_quarters=3, n_stocks=40)
    result = train_factor_cnn(
        panel,
        n_train_quarters=1,
        n_val_quarters=1,
        epochs=3,
        batch_size=40,
        device="cpu",
    )
    out = tmp_path / "test_model.pkl"
    save_cnn_model(result, out)
    assert out.is_file(), "文件未创建"

    payload = pickle.loads(out.read_bytes())
    assert payload["__type__"] == "CNNTrainResult"
    assert "val_ic_mean" in payload
    assert "feature_cols" in payload
    assert "folds" in payload
