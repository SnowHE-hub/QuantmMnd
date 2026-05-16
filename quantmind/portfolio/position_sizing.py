"""HRP（分层风险平价）和 Kelly 准则仓位优化。

用法:
    from quantmind.portfolio.position_sizing import hrp_weights, kelly_weights, blend_weights

    w_hrp = hrp_weights(returns_df)          # returns: DataFrame(date × ticker)
    w_kelly = kelly_weights(mu, cov, frac=0.5)
    w_final = blend_weights(w_hrp, w_kelly, alpha=0.5)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Sequence


# ─────────────────────────────────────────────
# HRP
# ─────────────────────────────────────────────

def _corr_to_dist(corr: np.ndarray) -> np.ndarray:
    """将相关系数矩阵转为距离矩阵（0~1）."""
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0, 1))
    return dist


def _quasi_diag(link: np.ndarray) -> list[int]:
    """将 linkage 矩阵转换为准对角化的叶节点顺序（递归展开）."""
    n = int(link[-1, 3])          # 总叶节点数

    def expand(node_id: int) -> list[int]:
        if node_id < n:
            return [int(node_id)]
        k = int(node_id) - n
        left = int(link[k, 0])
        right = int(link[k, 1])
        return expand(left) + expand(right)

    root = n + len(link) - 1   # 最后合并的聚类为根节点
    return expand(root)


def _recursive_bisect(cov: np.ndarray, sorted_items: list[int]) -> np.ndarray:
    """递归二分风险平价（HRP 核心）.

    sorted_items: 准对角顺序的叶节点（ticker）索引列表。
    返回: 与 sorted_items 长度相同的权重数组（未归一化，最后归一化）。
    """
    n = len(sorted_items)
    weights = np.ones(n)   # weights[i] 对应 sorted_items[i]

    def cluster_var(ticker_indices: list[int]) -> float:
        """用逆方差加权计算子组合的方差贡献。"""
        c = cov[np.ix_(ticker_indices, ticker_indices)]
        inv_var = 1.0 / (np.diag(c) + 1e-12)
        inv_var /= inv_var.sum()
        return float(inv_var @ c @ inv_var)

    def bisect(pos_range: list[int]):
        """pos_range: 当前聚类在 weights 数组中的位置索引列表。"""
        if len(pos_range) <= 1:
            return
        mid = len(pos_range) // 2
        left_pos = pos_range[:mid]
        right_pos = pos_range[mid:]

        left_tickers = [sorted_items[p] for p in left_pos]
        right_tickers = [sorted_items[p] for p in right_pos]

        cv_l = cluster_var(left_tickers)
        cv_r = cluster_var(right_tickers)

        # 方差小的集群分配更多权重（风险平价）
        alloc_left = 1.0 - cv_l / (cv_l + cv_r + 1e-12)
        alloc_right = 1.0 - alloc_left

        for p in left_pos:
            weights[p] *= alloc_left
        for p in right_pos:
            weights[p] *= alloc_right

        bisect(left_pos)
        bisect(right_pos)

    bisect(list(range(n)))
    return weights / weights.sum()


def hrp_weights(
    returns: pd.DataFrame,
    min_periods: int = 63,
    max_weight: float = 0.20,
) -> pd.Series:
    """分层风险平价（HRP）仓位权重。

    Args:
        returns: 收益率 DataFrame，行=日期，列=标的，至少 min_periods 行。
        min_periods: 有效历史最低天数；不足时退化为等权。
        max_weight: 单标的权重上限。

    Returns:
        pd.Series，index 为 tickers，values 为权重（sum≈1）。
    """
    from scipy.cluster.hierarchy import linkage

    tickers = returns.columns.tolist()
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)
    if len(returns) < min_periods or n == 1:
        return pd.Series(1.0 / n, index=tickers)

    ret_arr = returns.dropna(axis=0, how="all").ffill().fillna(0).values  # (T, n)
    cov = np.cov(ret_arr.T) * 252     # 年化
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1, 1)
    np.fill_diagonal(corr, 1.0)

    dist = _corr_to_dist(corr)
    link = linkage(dist[np.triu_indices(n, k=1)], method="ward")
    sorted_idx = _quasi_diag(link)

    raw_w = _recursive_bisect(cov, sorted_idx)
    # 权重上限约束（迭代裁剪至收敛）
    for _ in range(30):
        clip = np.minimum(raw_w, max_weight)
        excess = (raw_w - clip).sum()
        if excess < 1e-9:
            break
        uncapped = clip < max_weight - 1e-9
        if not uncapped.any():
            break
        clip[uncapped] += excess * clip[uncapped] / clip[uncapped].sum()
        raw_w = clip

    raw_w /= raw_w.sum()
    return pd.Series(raw_w, index=tickers)


# ─────────────────────────────────────────────
# Kelly
# ─────────────────────────────────────────────

def kelly_weights(
    mu: np.ndarray | pd.Series,
    cov: np.ndarray,
    fraction: float = 0.5,
    max_weight: float = 0.20,
    allow_short: bool = False,
) -> pd.Series:
    """分数 Kelly 准则仓位。

    Kelly = f * Σ^{-1} μ  （多头）
    fraction 为 Kelly 分数（建议 0.25~0.5 防止过度杠杆）。

    Args:
        mu: 各标的预期超额收益（年化）。
        cov: 协方差矩阵（年化）。
        fraction: Kelly 分数。
        max_weight: 单标的权重上限。
        allow_short: 是否允许做空。

    Returns:
        pd.Series，归一化后权重。
    """
    if isinstance(mu, pd.Series):
        tickers = mu.index.tolist()
        mu_arr = mu.values.astype(float)
    else:
        tickers = list(range(len(mu)))
        mu_arr = np.array(mu, dtype=float)

    n = len(tickers)
    cov_arr = np.array(cov, dtype=float)

    # 正则化避免奇异
    cov_reg = cov_arr + np.eye(n) * 1e-6
    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        return pd.Series(1.0 / n, index=tickers)

    raw_w = fraction * (cov_inv @ mu_arr)

    if not allow_short:
        raw_w = np.maximum(raw_w, 0)

    if raw_w.sum() < 1e-9:
        return pd.Series(1.0 / n, index=tickers)

    raw_w /= raw_w.sum()

    # 权重上限裁剪
    for _ in range(30):
        over = raw_w > max_weight
        if not over.any():
            break
        excess = (raw_w[over] - max_weight).sum()
        raw_w[over] = max_weight
        room = (max_weight - raw_w[~over])
        if room.sum() < 1e-9:
            break
        raw_w[~over] += excess * room / room.sum()
    raw_w /= raw_w.sum()

    return pd.Series(raw_w, index=tickers)


# ─────────────────────────────────────────────
# 混合 HRP + Kelly
# ─────────────────────────────────────────────

def blend_weights(
    w_hrp: pd.Series,
    w_kelly: pd.Series,
    alpha: float = 0.5,
) -> pd.Series:
    """线性混合 HRP 和 Kelly 权重。

    Args:
        alpha: HRP 权重占比（0=全 Kelly，1=全 HRP）。

    Returns:
        pd.Series，归一化权重。
    """
    tickers = sorted(set(w_hrp.index) | set(w_kelly.index))
    h = w_hrp.reindex(tickers, fill_value=0.0)
    k = w_kelly.reindex(tickers, fill_value=0.0)
    w = alpha * h + (1 - alpha) * k
    if w.sum() > 1e-9:
        w /= w.sum()
    return w


# ─────────────────────────────────────────────
# 投资组合绩效辅助
# ─────────────────────────────────────────────

def portfolio_metrics(
    nav: pd.Series,
    freq: str = "D",
    rf: float = 0.02,
) -> dict:
    """从净值序列计算常用绩效指标。

    Args:
        nav: 净值时间序列（以 1.0 开始）。
        freq: 采样频率 'D'=日，'W'=周，'M'=月。
        rf: 无风险利率（年化）。
    """
    per_year = {"D": 252, "W": 52, "M": 12}.get(freq, 252)
    rets = nav.pct_change().dropna()
    total_ret = float(nav.iloc[-1] / nav.iloc[0] - 1)
    ann_ret = float((1 + total_ret) ** (per_year / max(len(rets), 1)) - 1)
    ann_vol = float(rets.std() * np.sqrt(per_year))
    sharpe = (ann_ret - rf) / (ann_vol + 1e-9)
    running_max = nav.cummax()
    dd = (nav / running_max - 1)
    max_dd = float(dd.min())
    calmar = ann_ret / (-max_dd + 1e-9) if max_dd < 0 else float("inf")
    return {
        "total_return": round(total_ret, 4),
        "ann_return": round(ann_ret, 4),
        "ann_volatility": round(ann_vol, 4),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": round(max_dd, 4),
        "calmar_ratio": round(calmar, 3),
        "n_periods": len(rets),
    }
