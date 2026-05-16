"""quantmind.risk.position_sizing — 多种仓位管理方法.

参考文献：
  [1] Markowitz, H. (1952). "Portfolio Selection." Journal of Finance.
      均值-方差优化（最小方差组合）
  [2] Maillard, S., Roncalli, T., & Teïletche, J. (2010).
      "The Properties of Equally Weighted Risk Contribution Portfolios."
      Journal of Portfolio Management. （风险平价）
  [3] López de Prado, M. (2016).
      "Building Diversified Portfolios that Outperform Out-of-Sample."
      Journal of Portfolio Management. （层次风险平价 HRP）
  [4] Kelly, J.L. (1956).
      "A New Interpretation of Information Rate." Bell System Tech. Journal.
      （Kelly 公式，实用中取半 Kelly）
  [5] Choueifaty, Y. & Coignard, Y. (2008).
      "Towards Maximum Diversification." Journal of Portfolio Management.
      （逆波动率加权）
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from loguru import logger

__all__ = ["PositionSizer"]

_MIN_WEIGHT = 1e-6   # 最小权重（避免数值零）
_MAX_ITER   = 1000   # 风险平价最大迭代次数


class PositionSizer:
    """多种仓位管理方法.

    所有方法返回 Dict[ticker, weight]，权重之和 = 1，均为多头（>= 0）。
    """

    # ── 1. 等权 ───────────────────────────────────────────────────────────────

    @staticmethod
    def equal_weight(tickers: list[str]) -> dict[str, float]:
        """等权组合.

        每只股票权重 = 1/N。
        最简单的多元化方法，无需参数估计。

        Args:
            tickers: 股票代码列表

        Returns:
            {ticker: 1/N}
        """
        n = len(tickers)
        if n == 0:
            return {}
        w = 1.0 / n
        return {t: w for t in tickers}

    # ── 2. 逆波动率 ──────────────────────────────────────────────────────────

    @staticmethod
    def inverse_volatility(
        returns_df: pd.DataFrame,
        min_periods: int = 20,
    ) -> dict[str, float]:
        """逆波动率加权.

        参考文献: Choueifaty & Coignard (2008).

        权重 ∝ 1/σ_i（高波动低权重，低波动高权重）
        归一化: w_i = (1/σ_i) / Σ(1/σ_j)

        Args:
            returns_df:  日收益 DataFrame（index=date, columns=ticker）
            min_periods: 最少观测数

        Returns:
            {ticker: weight}
        """
        vols = returns_df.std(ddof=1, skipna=True)
        # 过滤无效波动率
        vols = vols.replace(0, np.nan).dropna()
        valid = vols[vols > 0]

        if valid.empty:
            return PositionSizer.equal_weight(list(returns_df.columns))

        inv_vol = 1.0 / valid
        total = inv_vol.sum()
        weights = (inv_vol / total).to_dict()
        return weights

    # ── 3. 最小方差 ──────────────────────────────────────────────────────────

    @staticmethod
    def minimum_variance(
        cov_matrix: pd.DataFrame,
        weight_bounds: tuple[float, float] = (0.0, 1.0),
    ) -> dict[str, float]:
        """最小方差组合.

        参考文献: Markowitz (1952).

        最小化: w' Σ w
        约束:   Σw_i = 1, w_i ∈ [lb, ub]

        Args:
            cov_matrix:    协方差矩阵（N×N）
            weight_bounds: 每只股票权重上下界（默认 [0, 1]）

        Returns:
            {ticker: weight}
        """
        tickers = list(cov_matrix.columns)
        n = len(tickers)
        if n == 0:
            return {}
        if n == 1:
            return {tickers[0]: 1.0}

        Σ = cov_matrix.values.astype(float)
        # 正则化（避免奇异）
        Σ += np.eye(n) * 1e-8

        def portfolio_variance(w: np.ndarray) -> float:
            return float(w @ Σ @ w)

        def portfolio_variance_grad(w: np.ndarray) -> np.ndarray:
            return 2 * Σ @ w

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [weight_bounds] * n
        w0 = np.ones(n) / n  # 等权初始值

        result = minimize(
            portfolio_variance,
            w0,
            jac=portfolio_variance_grad,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 500},
        )

        if not result.success:
            logger.warning(f"[MinVar] 优化未收敛: {result.message}，退回等权")
            return PositionSizer.equal_weight(tickers)

        w = np.clip(result.x, 0, 1)
        w /= w.sum()
        return dict(zip(tickers, w.tolist()))

    # ── 4. 风险平价 ──────────────────────────────────────────────────────────

    @staticmethod
    def risk_parity(
        cov_matrix: pd.DataFrame,
        tol: float = 1e-8,
        max_iter: int = _MAX_ITER,
    ) -> dict[str, float]:
        """风险平价组合（等风险贡献）.

        参考文献: Maillard, Roncalli & Teïletche (2010).

        最小化: Σ_{i,j} (RC_i - RC_j)²
        其中风险贡献 RC_i = w_i × (∂σ_p/∂w_i) = w_i × (Σw)_i / σ_p

        等风险时 RC_i = σ_p / N（各股贡献相等）

        用 Cyclical Coordinate Descent (CCD) 迭代求解：
          w_i ← w_i × σ_target / (Σw)_i  （每次更新一个权重）

        Args:
            cov_matrix: 协方差矩阵（N×N）
            tol:        收敛容差
            max_iter:   最大迭代次数

        Returns:
            {ticker: weight}
        """
        tickers = list(cov_matrix.columns)
        n = len(tickers)
        if n == 0:
            return {}
        if n == 1:
            return {tickers[0]: 1.0}

        Σ = cov_matrix.values.astype(float)
        Σ += np.eye(n) * 1e-8

        # CCD 迭代（Cyclical Coordinate Descent）
        w = np.ones(n) / n
        for iteration in range(max_iter):
            w_old = w.copy()
            for i in range(n):
                # 固定其他权重，求解 w_i 使 RC_i 相等
                # 近似: w_i = sqrt(w_i / (Σw)_i) × σ_target
                Σw = Σ @ w
                sigma_p = np.sqrt(max(w @ Σw, 1e-10))
                rc_i = w[i] * Σw[i] / sigma_p
                target_rc = sigma_p / n
                # 牛顿步长
                w[i] = max(w[i] * target_rc / rc_i, _MIN_WEIGHT)

            # 归一化
            w = np.clip(w, _MIN_WEIGHT, None)
            w /= w.sum()

            # 检查收敛
            if np.max(np.abs(w - w_old)) < tol:
                logger.debug(f"[RiskParity] 在第 {iteration+1} 次迭代收敛")
                break

        return dict(zip(tickers, w.tolist()))

    # ── 5. 层次风险平价（HRP）────────────────────────────────────────────────

    @staticmethod
    def hierarchical_risk_parity(
        returns_df: pd.DataFrame,
        linkage_method: str = "ward",
    ) -> dict[str, float]:
        """层次风险平价（Hierarchical Risk Parity, HRP）.

        参考文献: López de Prado (2016).

        三步算法：
        Step 1 - Tree Clustering（树形聚类）:
            基于相关系数距离 d = sqrt(0.5 × (1 - ρ_ij)) 进行层次聚类

        Step 2 - Quasi-Diagonalization（准对角化）:
            按聚类结果重排协方差矩阵，使相关资产相邻

        Step 3 - Recursive Bisection（递归二分分配）:
            在每个子集内，按逆方差比例分配权重：
            α_1 = 1 - V_1/(V_1+V_2)，递归处理子集

        相比最小方差，HRP 无需矩阵求逆，对估计误差更稳健。

        Args:
            returns_df:     日收益 DataFrame（index=date, columns=ticker）
            linkage_method: scipy 聚类方法（ward/average/single/complete）

        Returns:
            {ticker: weight}
        """
        tickers = list(returns_df.columns)
        n = len(tickers)
        if n == 0:
            return {}
        if n == 1:
            return {tickers[0]: 1.0}

        # ── Step 1: 树形聚类 ──────────────────────────────────────────────────
        cov = returns_df.cov()
        corr = returns_df.corr()

        # 相关系数距离矩阵
        dist = np.sqrt(0.5 * (1 - corr.values.clip(-1, 1)))
        np.fill_diagonal(dist, 0)

        # 层次聚类
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method=linkage_method)

        # ── Step 2: 准对角化（按聚类叶节点顺序重排）─────────────────────────
        sorted_idx = PositionSizer._hrp_get_quasi_diag(Z, n)
        sorted_tickers = [tickers[i] for i in sorted_idx]

        # ── Step 3: 递归二分分配 ─────────────────────────────────────────────
        weights = PositionSizer._hrp_recursive_bisect(
            cov, sorted_tickers
        )
        return weights

    @staticmethod
    def _hrp_get_quasi_diag(Z: np.ndarray, n: int) -> list[int]:
        """从 linkage 矩阵提取叶节点排列（准对角化顺序）."""
        # 构建叶节点顺序
        def get_order(node_id: int, z: np.ndarray, n: int) -> list[int]:
            """递归获取叶节点顺序."""
            if node_id < n:
                return [int(node_id)]
            idx = int(node_id) - n
            left = get_order(int(z[idx, 0]), z, n)
            right = get_order(int(z[idx, 1]), z, n)
            return left + right

        root = 2 * n - 2  # 根节点 ID
        return get_order(root, Z, n)

    @staticmethod
    def _hrp_recursive_bisect(
        cov: pd.DataFrame,
        sorted_tickers: list[str],
    ) -> dict[str, float]:
        """递归二分分配权重."""
        weights = {t: 1.0 for t in sorted_tickers}
        clusters = [sorted_tickers]  # 待处理的子集列表

        while clusters:
            # 对每个子集进行二分
            new_clusters: list[list[str]] = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue
                # 二分
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                # 计算每个子集的逆方差（近似：用子集内资产方差的调和平均）
                def cluster_var(subcluster: list[str]) -> float:
                    """子集的方差（逆方差加权组合的方差）."""
                    n_sub = len(subcluster)
                    if n_sub == 1:
                        return float(cov.loc[subcluster[0], subcluster[0]])
                    # 子集内等权逆方差
                    sub_cov = cov.loc[subcluster, subcluster]
                    sub_var = np.diag(sub_cov.values)
                    sub_var = np.clip(sub_var, 1e-10, None)
                    inv_var = 1.0 / sub_var
                    w_sub = inv_var / inv_var.sum()
                    return float(w_sub @ sub_cov.values @ w_sub)

                var_left = cluster_var(left)
                var_right = cluster_var(right)

                # 分配比例：逆方差比
                alpha_left = 1.0 - var_left / (var_left + var_right + 1e-10)
                alpha_right = 1.0 - alpha_left

                # 按比例缩放子集内权重
                for t in left:
                    weights[t] *= alpha_left
                for t in right:
                    weights[t] *= alpha_right

                if len(left) > 1:
                    new_clusters.append(left)
                if len(right) > 1:
                    new_clusters.append(right)

            clusters = new_clusters

        # 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {t: v / total for t, v in weights.items()}
        return weights

    # ── 6. Kelly 公式（半 Kelly）─────────────────────────────────────────────

    @staticmethod
    def kelly_criterion(
        expected_returns: pd.Series | dict[str, float],
        cov_matrix: pd.DataFrame,
        fraction: float = 0.5,
        weight_bounds: tuple[float, float] = (0.0, 0.25),
    ) -> dict[str, float]:
        """半 Kelly 仓位公式.

        参考文献: Kelly (1956); Thorp (1997) "The Kelly Criterion in Blackjack..."

        完整 Kelly: w* = Σ^{-1} μ
        半 Kelly:   w_half = fraction × w*  （fraction=0.5 降低过度集中风险）

        Args:
            expected_returns: 预期年化收益 {ticker: return}
            cov_matrix:       协方差矩阵
            fraction:         Kelly 分数（0.5 = 半 Kelly，降低波动）
            weight_bounds:    单只权重上下界（默认 0~25%，防止过度集中）

        Returns:
            {ticker: weight}
        """
        if isinstance(expected_returns, dict):
            expected_returns = pd.Series(expected_returns)

        tickers = list(cov_matrix.columns)
        n = len(tickers)
        if n == 0:
            return {}

        # 对齐 expected_returns 和 cov_matrix
        common = [t for t in tickers if t in expected_returns.index]
        if not common:
            return PositionSizer.equal_weight(tickers)

        μ = expected_returns.loc[common].values.astype(float)
        Σ = cov_matrix.loc[common, common].values.astype(float)
        Σ += np.eye(len(common)) * 1e-8

        try:
            # Kelly: w* = Σ^{-1} μ
            w_kelly = np.linalg.solve(Σ, μ)
        except np.linalg.LinAlgError:
            w_kelly = np.ones(len(common)) / len(common)

        # 半 Kelly
        w = fraction * w_kelly

        # 只保留多头并裁剪
        w = np.clip(w, weight_bounds[0], weight_bounds[1])

        # 归一化（若全为 0 则等权）
        total = w.sum()
        if total <= 0:
            return PositionSizer.equal_weight(common)

        w /= total
        return dict(zip(common, w.tolist()))

    # ── 便捷方法：自动选择仓位方法 ────────────────────────────────────────────

    @staticmethod
    def compute(
        method: str,
        tickers: list[str] | None = None,
        returns_df: pd.DataFrame | None = None,
        cov_matrix: pd.DataFrame | None = None,
        expected_returns: pd.Series | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """统一入口，按 method 名称调用对应方法.

        Args:
            method:           方法名（equal_weight/inverse_volatility/minimum_variance/
                               risk_parity/hrp/kelly）
            tickers:          股票列表（equal_weight 用）
            returns_df:       收益 DataFrame
            cov_matrix:       协方差矩阵
            expected_returns: 预期收益（kelly 用）
            **kwargs:         传给具体方法的额外参数

        Returns:
            {ticker: weight}
        """
        method = method.lower().replace("-", "_").replace(" ", "_")

        if method == "equal_weight":
            t = tickers or (list(returns_df.columns) if returns_df is not None else [])
            return PositionSizer.equal_weight(t)

        elif method == "inverse_volatility":
            if returns_df is None:
                raise ValueError("inverse_volatility 需要 returns_df")
            return PositionSizer.inverse_volatility(returns_df, **kwargs)

        elif method == "minimum_variance":
            if cov_matrix is None and returns_df is not None:
                cov_matrix = returns_df.cov()
            if cov_matrix is None:
                raise ValueError("minimum_variance 需要 cov_matrix 或 returns_df")
            return PositionSizer.minimum_variance(cov_matrix, **kwargs)

        elif method in ("risk_parity", "equal_risk_contribution"):
            if cov_matrix is None and returns_df is not None:
                cov_matrix = returns_df.cov()
            if cov_matrix is None:
                raise ValueError("risk_parity 需要 cov_matrix 或 returns_df")
            return PositionSizer.risk_parity(cov_matrix, **kwargs)

        elif method in ("hrp", "hierarchical_risk_parity"):
            if returns_df is None:
                raise ValueError("hrp 需要 returns_df")
            return PositionSizer.hierarchical_risk_parity(returns_df, **kwargs)

        elif method == "kelly":
            if expected_returns is None or cov_matrix is None:
                raise ValueError("kelly 需要 expected_returns 和 cov_matrix")
            return PositionSizer.kelly_criterion(expected_returns, cov_matrix, **kwargs)

        else:
            raise ValueError(f"未知仓位方法: {method}，支持: equal_weight/inverse_volatility/"
                             "minimum_variance/risk_parity/hrp/kelly")
