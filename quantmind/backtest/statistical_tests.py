"""quantmind.backtest.statistical_tests — 回测统计显著性检验.

参考文献：
  [1] Bailey, D.H. & López de Prado, M. (2014).
      "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
       Overfitting, and Non-Normality." Journal of Portfolio Management.

  [2] Bailey, D.H. & López de Prado, M. (2012).
      "The Sharpe Ratio Efficient Frontier." Journal of Risk.

  [3] Newey, W.K. & West, K.D. (1987).
      "A Simple, Positive Semi-Definite, Heteroskedasticity and
       Autocorrelation Consistent Covariance Matrix." Econometrica.

  [4] Politis, D.N. & Romano, J.P. (1994).
      "The Stationary Bootstrap." JASA.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "deflated_sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "bootstrap_ci",
    "ic_ttest_newey_west",
]

_TRADING_DAYS = 252
_RF_ANNUAL = 0.025


# ============================================================================
# 1. Deflated Sharpe Ratio（DSR）
# ============================================================================

def deflated_sharpe_ratio(
    strategies_sharpe_list: list[float],
    candidate_sharpe: float,
    n_obs: int = 252,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> dict[str, Any]:
    """修正多次试验 data snooping bias 的 Deflated Sharpe Ratio.

    数学原理（Bailey & López de Prado, 2014）：
    -----------------------------------------
    当策略研究员尝试 N 个策略时，最优 Sharpe 存在向上偏差。
    DSR 通过以下方式修正：

        SR* = E[max(SR_1, ..., SR_N)]

    对于 iid Normal：
        SR* ≈ σ_SR × [(1 - γ) × Φ^{-1}(1 - 1/N) + γ × Φ^{-1}(1 - 1/(N × e))]

    其中 γ = 欧拉–马斯刻罗尼常数 ≈ 0.5772

    PSR(SR*) = Φ{ [(SR_obs - SR*) × sqrt(n-1)] / sqrt(1 - γ_3·SR_obs + (γ_4-1)/4·SR_obs²) }

    Args:
        strategies_sharpe_list: 所有尝试过的策略 Sharpe 列表（含候选）
        candidate_sharpe:       候选策略的年化 Sharpe
        n_obs:                  观测数（交易日数），默认 252（1年）
        skew:                   收益偏度，默认 0（正态假设）
        kurt:                   收益峰度，默认 3（正态假设）

    Returns:
        dict: {dsr, sharpe_star, p_value, is_significant, interpretation}
    """
    n_trials = len(strategies_sharpe_list)
    if n_trials == 0:
        return {"dsr": np.nan, "p_value": np.nan, "is_significant": False,
                "interpretation": "无策略列表", "sharpe_star": np.nan}

    # SR 分布的标准差（考虑非正态修正）
    # Var(SR) ≈ (1 - skew·SR + (kurt-1)/4·SR²) / (n-1)
    sr = candidate_sharpe
    sr_variance = (1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2) / max(n_obs - 1, 1)
    sigma_sr = np.sqrt(sr_variance)

    if sigma_sr <= 0:
        return {"dsr": np.nan, "p_value": np.nan, "is_significant": False,
                "interpretation": "SR 方差为零", "sharpe_star": np.nan}

    # 期望最大 SR（Euler-Mascheroni 常数修正）
    EULER_GAMMA = 0.5772156649

    # Expected maximum of N standard normals（近似）
    e_max = (
        (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1.0 / n_trials)
        + EULER_GAMMA * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    ) if n_trials > 1 else 0.0

    # SR* = σ_SR × E[max]
    sharpe_star = sigma_sr * e_max

    # PSR(SR*) = P(SR_true > SR*) = Φ(z)
    # z = (SR_obs - SR*) × sqrt(n-1) / sqrt(Var_factor)
    var_factor = max(1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2, 1e-10)
    z = (sr - sharpe_star) * np.sqrt(max(n_obs - 1, 1)) / np.sqrt(var_factor)
    p_value = float(stats.norm.cdf(z))  # P(SR_true > SR*)

    is_significant = p_value >= 0.95  # 单侧 5% 显著性

    return {
        "dsr": float(sr - sharpe_star),
        "sharpe_star": float(sharpe_star),
        "p_value": float(p_value),
        "is_significant": is_significant,
        "n_trials": n_trials,
        "interpretation": (
            f"DSR={sr - sharpe_star:.3f}，p={p_value:.3f}，"
            f"{'显著' if is_significant else '不显著'}（{n_trials}次策略尝试，数据窥探修正后）"
        ),
    }


# ============================================================================
# 2. Probabilistic Sharpe Ratio（PSR）
# ============================================================================

def probabilistic_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    benchmark_sharpe: float = 0.0,
) -> dict[str, Any]:
    """考虑偏度和峰度修正的 Probabilistic Sharpe Ratio.

    数学原理（Bailey & López de Prado, 2012）：
    -----------------------------------------
    标准 Sharpe 假设正态分布，忽略高阶矩。

    PSR(SR*) = Φ{ [(SR_hat - SR*) × sqrt(n-1)] / sqrt(1 - γ_3·SR_hat + (γ_4-1)/4·SR_hat²) }

    其中：
      SR_hat = 样本 Sharpe（日频年化）
      γ_3    = 收益偏度
      γ_4    = 收益峰度
      SR*    = 基准 Sharpe（默认 0）

    Args:
        returns:          日收益率序列
        benchmark_sharpe: 基准 Sharpe（默认 0，即验证 SR > 0）

    Returns:
        dict: {psr, p_value, sharpe_hat, skew, kurtosis, interpretation}
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)

    if n < 4:
        return {"psr": np.nan, "p_value": np.nan, "sharpe_hat": np.nan,
                "interpretation": "样本量不足（<4）"}

    # 年化 Sharpe
    rf_daily = (1 + _RF_ANNUAL) ** (1 / _TRADING_DAYS) - 1
    excess = r - rf_daily
    sr_hat = float(np.mean(excess) / (np.std(excess, ddof=1) + 1e-10) * np.sqrt(_TRADING_DAYS))

    # 偏度和峰度
    skewness = float(stats.skew(r))
    kurtosis = float(stats.kurtosis(r, fisher=False))  # 总峰度（正态=3）

    # PSR 公式
    var_factor = max(
        1 - skewness * sr_hat + (kurtosis - 1) / 4.0 * sr_hat ** 2,
        1e-10,
    )
    z = (sr_hat - benchmark_sharpe) * np.sqrt(n - 1) / np.sqrt(var_factor)
    psr = float(stats.norm.cdf(z))

    return {
        "psr": psr,
        "p_value": psr,
        "sharpe_hat": sr_hat,
        "skew": skewness,
        "kurtosis": kurtosis,
        "n_obs": n,
        "is_significant": psr >= 0.95,
        "interpretation": (
            f"PSR={psr:.3f}（修正偏度={skewness:.2f}/峰度={kurtosis:.2f}），"
            f"年化 Sharpe={sr_hat:.3f}，"
            f"{'显著优于0' if psr >= 0.95 else '不显著'}"
        ),
    }


# ============================================================================
# 3. Block Bootstrap 置信区间
# ============================================================================

def bootstrap_ci(
    metric_func: callable,
    returns: pd.Series | np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    block_size: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    """Block Bootstrap 置信区间（保留时序自相关）.

    数学原理（Politis & Romano, 1994）：
    ------------------------------------
    标准 bootstrap 独立同分布假设在时间序列中不成立（自相关）。
    Stationary Block Bootstrap 通过随机长度的连续块重采样，
    在保留局部自相关结构的同时实现渐近有效。

    步骤：
      1. 随机选取起点 i ~ Uniform[0, n-1]
      2. 选取块长度 L ~ Geometric(1/block_size)
      3. 取 r[i:i+L]（循环边界）
      4. 重复直到填满 n 个样本
      5. 计算目标统计量 θ_b = metric_func(bootstrap_sample)
      6. CI = [θ_2.5%, θ_97.5%]

    Args:
        metric_func:  统计量函数，接受 np.ndarray 返回 float
        returns:      日收益率序列
        n_bootstrap:  bootstrap 次数（默认 1000）
        confidence:   置信水平（默认 0.95）
        block_size:   块大小（默认 20，约 1个月）
        seed:         随机种子

    Returns:
        dict: {estimate, ci_lower, ci_upper, std_error, interpretation}
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)

    if n < block_size * 2:
        return {"estimate": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
                "interpretation": "样本量不足"}

    rng = np.random.default_rng(seed)
    original_estimate = metric_func(r)

    bootstrap_stats: list[float] = []
    for _ in range(n_bootstrap):
        sample = _block_resample(r, n, block_size, rng)
        try:
            stat = metric_func(sample)
            if np.isfinite(stat):
                bootstrap_stats.append(stat)
        except Exception:
            pass

    if not bootstrap_stats:
        return {"estimate": original_estimate, "ci_lower": np.nan, "ci_upper": np.nan,
                "interpretation": "bootstrap 采样失败"}

    bootstrap_arr = np.array(bootstrap_stats)
    alpha = 1 - confidence
    ci_lower = float(np.percentile(bootstrap_arr, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_arr, 100 * (1 - alpha / 2)))
    std_error = float(np.std(bootstrap_arr))

    return {
        "estimate": float(original_estimate),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_error": std_error,
        "n_bootstrap": len(bootstrap_stats),
        "confidence": confidence,
        "block_size": block_size,
        "interpretation": (
            f"估计={original_estimate:.4f}，"
            f"{int(confidence*100)}% CI=[{ci_lower:.4f}, {ci_upper:.4f}]，"
            f"SE={std_error:.4f}（Block Bootstrap，块大小={block_size}）"
        ),
    }


def _block_resample(r: np.ndarray, n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """循环 Block Bootstrap 重采样."""
    sample = np.empty(n)
    idx = 0
    while idx < n:
        start = rng.integers(0, n)
        length = min(rng.geometric(1.0 / block_size), n - idx)
        for j in range(length):
            sample[idx] = r[(start + j) % n]
            idx += 1
            if idx >= n:
                break
    return sample


# ============================================================================
# 4. IC t-test（Newey-West 修正）
# ============================================================================

def ic_ttest_newey_west(
    ic_series: pd.Series | np.ndarray,
    lags: int = 4,
) -> dict[str, Any]:
    """因子 IC 显著性检验，Newey-West HAC 方差修正.

    数学原理（Newey & West, 1987）：
    --------------------------------
    因子 IC 序列存在自相关（今日 IC 与昨日 IC 相关），
    标准 t 检验低估方差，导致假阳性。

    Newey-West HAC 方差估计：
        Ω_NW = Σ_{h=0}^{lags} w_h × (γ_h + γ_h')

    其中：
      γ_h = 样本自协方差矩阵（lag h）
      w_h = Bartlett 核权重 = 1 - h/(lags+1)

    t 统计量：
        t = IC_bar × sqrt(T) / sqrt(Ω_NW)

    Args:
        ic_series: IC 序列（Spearman 相关系数，monthly/quarterly）
        lags:      Newey-West 截断阶数，推荐 T^(1/4)（默认 4）

    Returns:
        dict: {ic_mean, ic_std, ic_ir, t_stat, p_value, is_significant, interpretation}
    """
    ic = np.asarray(ic_series, dtype=float)
    ic = ic[~np.isnan(ic)]
    n = len(ic)

    if n < lags + 2:
        return {"ic_mean": np.nan, "t_stat": np.nan, "p_value": np.nan,
                "is_significant": False, "interpretation": "样本量不足"}

    ic_mean = float(np.mean(ic))
    ic_std = float(np.std(ic, ddof=1))
    ic_ir = float(ic_mean / (ic_std + 1e-10))

    # Newey-West HAC 方差估计
    # γ_0 = 样本方差
    gamma_0 = float(np.var(ic, ddof=1))
    nw_var = gamma_0

    # 加入高阶自协方差（Bartlett 核加权）
    for h in range(1, lags + 1):
        w = 1.0 - h / (lags + 1.0)  # Bartlett 权重
        if h < n:
            cov_h = float(np.cov(ic[h:], ic[:-h])[0, 1])
            nw_var += 2 * w * cov_h

    nw_var = max(nw_var, 1e-10)

    # t 统计量
    t_stat = ic_mean * np.sqrt(n) / np.sqrt(nw_var)
    # 双侧 p-value，自由度 = n - 1
    p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1)))

    is_significant = p_value < 0.05

    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "t_stat": float(t_stat),
        "p_value": p_value,
        "n_obs": n,
        "is_significant": is_significant,
        "nw_variance": nw_var,
        "lags": lags,
        "interpretation": (
            f"IC均值={ic_mean:.4f}，IR={ic_ir:.3f}，"
            f"t={t_stat:.3f}，p={p_value:.4f}（Newey-West lags={lags}），"
            f"{'因子显著' if is_significant else '因子不显著'}（5%水平）"
        ),
    }
