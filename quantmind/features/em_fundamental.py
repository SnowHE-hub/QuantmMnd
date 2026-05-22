"""quantmind/features/em_fundamental.py

EM 隐变量基本面质量因子。

背景
----
A 股小盘股信息不对称严重，单一财务指标噪声大。
把"真实基本面质量"看作离散隐变量 z ∈ {0:差, 1:中, 2:好}，
用 4 个观测指标通过 EM/GMM 联合估计 P(z|x)。

观测变量（均已在 alpha_panel_v4 中，截面 zscore 标准化后输入）
  x1 = roe_ttm         净资产收益率
  x2 = gross_margin    毛利率
  x3 = revenue_yoy     营收增速
  x4 = earnings_accel_q 盈利加速度

输出
----
fundamental_quality_score = P(z=2 | x) ∈ [0, 1]
  → 越高表示模型认为真实基本面质量越好

训练方式
--------
每季度截面独立训练 GMM（不跨期，避免前视偏差）；
成分排序按 μ_k 的 roe_ttm 维度升序，保证 k=2 始终对应"高质量"。
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ─── 常量 ─────────────────────────────────────────────────────────────────────

EM_OBS_COLS: list[str] = [
    "roe_ttm",
    "gross_margin",
    "revenue_yoy",
    "earnings_accel_q",
]

EM_N_COMPONENTS: int  = 3    # 隐状态数：0=差，1=中，2=好
EM_FUNDAMENTAL_FACTORS: list[str] = ["fundamental_quality_score"]

# IC 方向先验权重（由历史数据估计，roe_ttm / gross_margin / revenue_yoy 权重正，
# earnings_accel_q 预测力为负被置 0；归一化后和为 1）
# 用于将 4D 特征投影到 1D 质量方向，再对 1D 运行 GMM
EM_IC_WEIGHTS: np.ndarray = np.array([0.306, 0.220, 0.473, 0.001], dtype=float)
EM_IC_WEIGHTS = EM_IC_WEIGHTS / EM_IC_WEIGHTS.sum()  # 归一化


# ─── GaussianMixtureEM ─────────────────────────────────────────────────────────

class GaussianMixtureEM:
    """4 维 Gaussian Mixture Model，K 个成分，纯 NumPy EM 实现。

    Parameters
    ----------
    n_components : int   混合成分数，默认 3
    max_iter     : int   最大 EM 迭代步数，默认 200
    tol          : float 对数似然收敛阈值，默认 1e-4
    reg          : float Σ 正则化项（加到对角线防止奇异），默认 1e-4
    random_state : int   随机种子

    Attributes（训练后）
    ----------
    pi_   : (K,)        混合权重 π_k
    mu_   : (K, D)      各成分均值 μ_k
    sigma_: (K, D, D)   各成分协方差 Σ_k
    log_likelihood_history : list[float]  每步对数似然
    n_iter_ : int        实际迭代次数
    converged_ : bool    是否收敛
    """

    def __init__(
        self,
        n_components: int  = EM_N_COMPONENTS,
        max_iter:     int  = 200,
        tol:          float = 1e-4,
        reg:          float = 1e-4,
        random_state: int  = 42,
    ) -> None:
        self.n_components = n_components
        self.max_iter     = max_iter
        self.tol          = tol
        self.reg          = reg
        self.random_state = random_state

        self.pi_:    Optional[np.ndarray] = None
        self.mu_:    Optional[np.ndarray] = None
        self.sigma_: Optional[np.ndarray] = None
        self.log_likelihood_history: list[float] = []
        self.n_iter_:     int  = 0
        self.converged_:  bool = False

    # ── 初始化（K-Means++ 风格）────────────────────────────────────────────────
    def _init_params(self, X: np.ndarray, rng: np.random.Generator) -> None:
        N, D = X.shape
        K = self.n_components

        # K-Means++ 选初始中心
        centers: list[np.ndarray] = []
        idx = int(rng.integers(N))
        centers.append(X[idx])

        for _ in range(1, K):
            dists = np.array([
                min(np.sum((x - c) ** 2) for c in centers)
                for x in X
            ])
            probs = dists / (dists.sum() + 1e-300)
            idx   = int(rng.choice(N, p=probs))
            centers.append(X[idx])

        self.mu_ = np.array(centers)               # (K, D)
        self.pi_ = np.full(K, 1.0 / K)             # 均匀初始化
        self.sigma_ = np.array([np.eye(D)] * K)    # (K, D, D)

    # ── 多元高斯 PDF（对数域计算，数值稳定）────────────────────────────────────
    @staticmethod
    def _log_gaussian(X: np.ndarray, mu: np.ndarray,
                      sigma: np.ndarray) -> np.ndarray:
        """log N(X; mu, sigma)，形状 (N,)"""
        D = mu.shape[0]
        diff = X - mu                               # (N, D)

        # 用 Cholesky 分解求逆和行列式（更稳定）
        try:
            L     = np.linalg.cholesky(sigma)
            alpha = np.linalg.solve(L, diff.T)      # (D, N)
            maha  = np.sum(alpha ** 2, axis=0)      # (N,)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
        except np.linalg.LinAlgError:
            # Cholesky 失败（非正定）：用 pinv 作退化处理
            inv_sigma = np.linalg.pinv(sigma)
            maha      = np.einsum("nd,dd,nd->n", diff, inv_sigma, diff)
            sign, log_det = np.linalg.slogdet(sigma)
            log_det = log_det if sign > 0 else 1e10

        return -0.5 * (D * np.log(2 * np.pi) + log_det + maha)

    # ── E步：计算后验 γ_ik = P(z=k | x_i) ───────────────────────────────────
    def _e_step(self, X: np.ndarray) -> np.ndarray:
        K = self.n_components
        N = X.shape[0]
        log_gamma = np.zeros((N, K))

        for k in range(K):
            log_gamma[:, k] = (np.log(self.pi_[k] + 1e-300)
                               + self._log_gaussian(X, self.mu_[k], self.sigma_[k]))

        # log-sum-exp 数值稳定
        log_gamma_max = log_gamma.max(axis=1, keepdims=True)
        log_gamma_shifted = log_gamma - log_gamma_max
        log_sum = np.log(np.exp(log_gamma_shifted).sum(axis=1, keepdims=True) + 1e-300)

        gamma = np.exp(log_gamma_shifted - log_sum)  # (N, K)
        gamma = np.clip(gamma, 1e-300, None)
        gamma /= gamma.sum(axis=1, keepdims=True)
        return gamma

    # ── M步：更新 π, μ, Σ ──────────────────────────────────────────────────────
    def _m_step(self, X: np.ndarray, gamma: np.ndarray) -> None:
        N, D = X.shape
        K = self.n_components
        Nk = gamma.sum(axis=0) + 1e-300             # (K,)

        self.pi_ = Nk / N
        self.mu_ = (gamma.T @ X) / Nk[:, None]     # (K, D)

        for k in range(K):
            diff = X - self.mu_[k]                  # (N, D)
            w    = gamma[:, k]                      # (N,)
            sigma_k = (w[:, None] * diff).T @ diff  # (D, D)
            sigma_k /= Nk[k]
            sigma_k += self.reg * np.eye(D)         # 正则化
            self.sigma_[k] = sigma_k

    # ── 对数似然 ───────────────────────────────────────────────────────────────
    def _log_likelihood(self, X: np.ndarray) -> float:
        K = self.n_components
        N = X.shape[0]
        log_lik = np.zeros((N, K))

        for k in range(K):
            log_lik[:, k] = (np.log(self.pi_[k] + 1e-300)
                             + self._log_gaussian(X, self.mu_[k], self.sigma_[k]))

        log_lik_max = log_lik.max(axis=1, keepdims=True)
        log_sum = (np.log(np.exp(log_lik - log_lik_max).sum(axis=1))
                   + log_lik_max.squeeze())
        return float(log_sum.mean())

    # ── 主训练入口 ─────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray) -> "GaussianMixtureEM":
        """EM 训练。

        Parameters
        ----------
        X : (N, D)  训练样本（建议截面 zscore 标准化后传入）

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        N, D = X.shape

        if N < self.n_components:
            raise ValueError(
                f"样本数 {N} < n_components {self.n_components}"
            )

        rng = np.random.default_rng(self.random_state)
        self._init_params(X, rng)
        self.log_likelihood_history = []

        prev_ll = -np.inf
        for i in range(self.max_iter):
            gamma = self._e_step(X)
            self._m_step(X, gamma)
            ll    = self._log_likelihood(X)
            self.log_likelihood_history.append(ll)

            if abs(ll - prev_ll) < self.tol:
                self.converged_ = True
                self.n_iter_    = i + 1
                break
            prev_ll = ll
        else:
            self.n_iter_ = self.max_iter

        return self

    # ── 推理 ──────────────────────────────────────────────────────────────────
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """后验概率 P(z=k | x)。

        Returns
        -------
        (N, K) array，每行和为 1
        """
        if self.mu_ is None:
            raise RuntimeError("请先调用 fit()")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self._e_step(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """最大后验成分标签 argmax P(z=k | x)。

        Returns
        -------
        (N,) int array
        """
        return self.predict_proba(X).argmax(axis=1)

    # ── 成分排序（保证 k=2 对应"高质量"）──────────────────────────────────────
    def sort_components_by_dim(self, dim: int = 0) -> None:
        """按指定维度的 μ 升序排列成分（0=差, 1=中, 2=好）。"""
        if self.mu_ is None:
            return
        order = np.argsort(self.mu_[:, dim])
        self.mu_    = self.mu_[order]
        self.sigma_ = self.sigma_[order]
        self.pi_    = self.pi_[order]


# ─── 截面预处理工具 ────────────────────────────────────────────────────────────

def _cross_section_standardize(arr: np.ndarray, clip: float = 3.0) -> np.ndarray:
    """截面标准化：MAD winsorize(±3σ_MAD) → zscore。"""
    out = arr.copy().astype(float)
    for j in range(out.shape[1]):
        col = out[:, j]
        nan_mask = np.isnan(col)
        valid    = col[~nan_mask]
        if len(valid) == 0:
            continue
        med = np.median(valid)
        mad = np.median(np.abs(valid - med)) + 1e-12
        col = np.clip(col, med - clip * mad, med + clip * mad)
        mu  = np.nanmean(col)
        std = np.nanstd(col) + 1e-12
        col = (col - mu) / std
        col[nan_mask] = 0.0          # NaN 填充为截面中性值 0
        out[:, j] = col
    return out


# ─── 主因子构造函数 ────────────────────────────────────────────────────────────

def build_em_fundamental_factor(
    panel: pd.DataFrame,
    obs_cols: list[str] = None,
    n_components: int = EM_N_COMPONENTS,
    ic_weights: Optional[np.ndarray] = None,
    max_iter: int = 200,
    tol: float = 1e-4,
    reg: float = 1e-4,
    random_state: int = 42,
    min_samples: int = 30,
) -> pd.Series:
    """按季度截面训练 GMM，输出 fundamental_quality_score ∈ [0, 1]。

    算法
    ----
    1. 截面 MAD-winsorize + zscore 标准化 4D 特征
    2. 用 ic_weights 将 4D 投影到 1D 质量方向（降噪 + 方向校准）
    3. 在 1D 复合得分上训练 K=3 GMM（避免高维 GMM 退化）
    4. 按 μ_k 排序成分，k=2 = "好质量"
    5. 输出 P(z=2 | x) ∈ [0, 1]

    Parameters
    ----------
    panel        : alpha_panel_v4 MultiIndex(as_of, ticker)
    obs_cols     : 观测变量列名，默认 EM_OBS_COLS
    n_components : GMM 成分数，默认 3
    ic_weights   : 4D→1D 投影权重，默认 EM_IC_WEIGHTS
                   （对应 roe_ttm/gross_margin/revenue_yoy/earnings_accel_q
                     的历史 IC 先验权重，归一化后和为 1）
    max_iter     : EM 最大迭代步数
    tol          : 收敛阈值
    reg          : Σ 正则化系数
    random_state : 随机种子
    min_samples  : 每季度最小股票数（不足则跳过）

    Returns
    -------
    pd.Series  MultiIndex(as_of, ticker) → [0, 1]
               值越高 → 被模型认为基本面质量越好
    """
    if obs_cols is None:
        obs_cols = EM_OBS_COLS

    if ic_weights is None:
        ic_weights = EM_IC_WEIGHTS

    ic_weights = np.asarray(ic_weights, dtype=float)
    # 确保权重非负且归一化
    ic_weights = np.clip(ic_weights, 0.0, None)
    w_sum = ic_weights.sum()
    if w_sum < 1e-9:
        ic_weights = np.ones(len(obs_cols)) / len(obs_cols)
    else:
        ic_weights = ic_weights / w_sum

    # 确认列存在
    avail = [c for c in obs_cols if c in panel.columns]
    if not avail:
        log.error("没有可用的观测变量列: %s", obs_cols)
        return pd.Series(dtype=float, name="fundamental_quality_score")

    if len(avail) < len(obs_cols):
        log.warning("部分观测变量缺失，只用 %d/%d 列", len(avail), len(obs_cols))
        # 对齐 ic_weights 到 avail
        avail_idx = [obs_cols.index(c) for c in avail]
        ic_weights = ic_weights[avail_idx]
        w_sum = ic_weights.sum()
        ic_weights = ic_weights / w_sum if w_sum > 1e-9 else np.ones(len(avail)) / len(avail)

    dates  = panel.index.get_level_values(0).unique().sort_values()
    chunks: list[pd.Series] = []

    for dt in dates:
        slice_df = panel.loc[dt][avail].copy().astype(float)
        tickers  = slice_df.index

        if len(tickers) < min_samples:
            log.warning("  %s: 股票数 %d < %d，跳过", dt, len(tickers), min_samples)
            continue

        # 步骤 1：截面标准化（MAD winsorize + zscore，NaN→0）
        X = _cross_section_standardize(slice_df.values)   # (N, D)

        # 步骤 2：投影到 1D 质量复合得分
        composite = X @ ic_weights[:X.shape[1]]            # (N,)
        composite = composite.reshape(-1, 1)               # (N, 1)

        # 步骤 3：在 1D 复合得分上训练 GMM
        gmm = GaussianMixtureEM(
            n_components=n_components,
            max_iter=max_iter,
            tol=tol,
            reg=reg,
            random_state=random_state,
        )

        try:
            gmm.fit(composite)
        except Exception as exc:
            log.warning("  %s: GMM fit 失败 (%s)，跳过", dt, exc)
            continue

        # 步骤 4：按 μ 排序 → k=2="好质量"
        gmm.sort_components_by_dim(dim=0)

        # 步骤 5：P(z=2 | x)
        proba    = gmm.predict_proba(composite)            # (N, K)
        good_prob = proba[:, n_components - 1]             # P(z=好)

        multi_idx = pd.MultiIndex.from_arrays(
            [[dt] * len(tickers), tickers],
            names=["as_of", "ticker"],
        )
        chunks.append(pd.Series(good_prob, index=multi_idx,
                                name="fundamental_quality_score"))

        log.debug("  %s: n=%d  conv=%s  iters=%d  μ=%s  π=%s",
                  dt, len(tickers), gmm.converged_, gmm.n_iter_,
                  np.round(gmm.mu_[:, 0], 3), np.round(gmm.pi_, 3))

    if not chunks:
        log.warning("build_em_fundamental_factor: 所有季度均跳过")
        return pd.Series(dtype=float, name="fundamental_quality_score")

    result = pd.concat(chunks)
    log.info(
        "fundamental_quality_score: %d 记录  "
        "min=%.3f  max=%.3f  mean=%.3f",
        len(result), result.min(), result.max(), result.mean(),
    )
    return result


# ─── AnalysisSystem 集成辅助 ──────────────────────────────────────────────────

def blend_quality_score(
    original_quality: pd.Series,
    em_quality:       pd.Series,
    em_weight:        float = 0.2,
) -> pd.Series:
    """将 EM 质量因子混入原始 quality_score。

    公式：
        quality_final = (1 - em_weight) * quality_original
                       + em_weight * em_quality

    Parameters
    ----------
    original_quality : 原始 quality_score（任意量纲）
    em_quality       : fundamental_quality_score ∈ [0, 1]
    em_weight        : EM 因子权重，默认 0.2（对应 strategy_config_v2 配置）

    Returns
    -------
    pd.Series  与 original_quality 同 index，混合后的质量分
    """
    if em_quality.empty:
        log.warning("em_quality 为空，返回原始 quality_score")
        return original_quality

    common = original_quality.index.intersection(em_quality.index)
    q_orig = original_quality.reindex(common)
    q_em   = em_quality.reindex(common)

    # 对齐量纲：把 em_quality [0,1] 缩放到与 original_quality 同量纲
    orig_std = q_orig.std()
    em_std   = q_em.std()
    if em_std > 1e-9 and orig_std > 1e-9:
        q_em_scaled = (q_em - q_em.mean()) / em_std * orig_std + q_orig.mean()
    else:
        q_em_scaled = q_em

    result = (1.0 - em_weight) * q_orig + em_weight * q_em_scaled
    return result.rename("quality_score_blended")


__all__ = [
    "EM_FUNDAMENTAL_FACTORS",
    "EM_IC_WEIGHTS",
    "EM_N_COMPONENTS",
    "EM_OBS_COLS",
    "GaussianMixtureEM",
    "blend_quality_score",
    "build_em_fundamental_factor",
]
