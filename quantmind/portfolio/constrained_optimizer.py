"""quantmind/portfolio/constrained_optimizer.py

Barra 因子约束的组合优化器。

背景
----
Kelly / HRP 原始权重不控制因子暴露，可能在行业或风格维度产生意外集中，
例如持仓全部偏向「高动量 + 低流动性」。本模块在 Kelly 目标之上叠加：

1. **行业约束**：每个行业因子载荷的主动暴露绝对值 ≤ ``industry_limit``
2. **风格约束**：size/value/momentum/quality/volatility/beta/liquidity
   各载荷绝对值 ≤ ``style_limit``
3. **换手率约束**：L1 距离 ||w − w_prev||₁ ≤ ``turnover_limit``
4. **基础约束**：w ≥ 0，sum(w) = 1

优化问题
--------
  max  μᵀw − λ · wᵀΣw          (Kelly 均值-方差目标)
  s.t. |B_ind^T w| ≤ industry_limit   (行业约束，元素级)
       |B_style^T w| ≤ style_limit     (风格约束，元素级)
       ‖w − w_prev‖₁ ≤ turnover_limit (换手约束)
       w ≥ 0,  1ᵀw = 1

用 cvxpy 求解（CLARABEL 求解器，无需商业 license）。
求解失败时自动回退等权，并打印 WARNING 日志。

用法
----
>>> from quantmind.portfolio.constrained_optimizer import ConstrainedPortfolioOptimizer
>>> opt = ConstrainedPortfolioOptimizer(industry_limit=0.3, style_limit=0.5,
...                                     turnover_limit=0.6, risk_aversion=1.0)
>>> weights = opt.optimize(alpha_scores, factor_exposures, cov_matrix, prev_weights)
>>> exposure = opt.compute_active_exposure(weights, factor_exposures)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────────────────────────────────

_DEFAULT_INDUSTRY_LIMIT = 0.3   # 行业因子主动暴露上限（相对市值加权基准）
_DEFAULT_STYLE_LIMIT    = 0.5   # 风格因子暴露上限
_DEFAULT_TURNOVER_LIMIT = 0.6   # 换手率 L1 距离上限
_DEFAULT_RISK_AVERSION  = 1.0   # 风险厌恶系数 λ
_WEIGHT_CAP             = 0.25  # 单股最大仓位（防止集中度过高）


class ConstrainedPortfolioOptimizer:
    """在 Kelly/HRP 基础上叠加 Barra 因子约束的组合优化器。

    Parameters
    ----------
    industry_limit : float
        行业因子主动暴露（portfolio weight × industry dummy）绝对值上限。
        默认 0.3（允许相对基准 30% 的行业偏离）。
    style_limit : float
        风格因子暴露（size / value / momentum / quality / volatility / beta /
        liquidity）绝对值上限。默认 0.5。
    turnover_limit : float
        相邻两期 L1 换手率上限（0=不换手，2=完全换手）。默认 0.6。
    risk_aversion : float
        均值-方差目标中的风险厌恶系数 λ。λ 越大越保守。默认 1.0。
    max_weight : float
        单股权重上限。默认 0.25。
    solver : str
        cvxpy 求解器名称，默认 'CLARABEL'（开源，无需商业 license）。
        备选：'OSQP'（速度稍快但精度略低）。

    Examples
    --------
    >>> opt = ConstrainedPortfolioOptimizer(industry_limit=0.3, style_limit=0.5)
    >>> w = opt.optimize(alpha_scores, factor_exp, cov_mat)
    >>> exp = opt.compute_active_exposure(w, factor_exp)
    """

    def __init__(
        self,
        industry_limit: float = _DEFAULT_INDUSTRY_LIMIT,
        style_limit:    float = _DEFAULT_STYLE_LIMIT,
        turnover_limit: float = _DEFAULT_TURNOVER_LIMIT,
        risk_aversion:  float = _DEFAULT_RISK_AVERSION,
        max_weight:     float = _WEIGHT_CAP,
        solver:         str   = "CLARABEL",
    ) -> None:
        self.industry_limit = float(industry_limit)
        self.style_limit    = float(style_limit)
        self.turnover_limit = float(turnover_limit)
        self.risk_aversion  = float(risk_aversion)
        self.max_weight     = float(max_weight)
        self.solver         = solver

    # ── 主要接口 ──────────────────────────────────────────────────────────────

    def optimize(
        self,
        alpha_scores:     pd.Series,
        factor_exposures: Optional[pd.DataFrame],
        cov_matrix:       pd.DataFrame,
        prev_weights:     Optional[pd.Series] = None,
    ) -> pd.Series:
        """求解 Barra 约束优化，返回满足所有约束的最优权重向量。

        Parameters
        ----------
        alpha_scores : pd.Series
            index=ticker，values=预期超额收益或 alpha score（截面百分位 / Z-score）。
        factor_exposures : pd.DataFrame or None
            行=ticker，列=因子（风格 + 行业哑变量），由
            ``build_factor_exposures()`` 生成。若为 None 则跳过因子约束。
        cov_matrix : pd.DataFrame
            年化协方差矩阵（ticker × ticker）。
        prev_weights : pd.Series or None
            上期权重（index=ticker）；None 时跳过换手约束。

        Returns
        -------
        pd.Series
            index=ticker，values=最优权重（sum=1，≥ 0）。
            若求解失败自动回退到等权并打印 WARNING。
        """
        import cvxpy as cp

        tickers = alpha_scores.index.tolist()
        n = len(tickers)
        if n == 0:
            return pd.Series(dtype=float)
        if n == 1:
            return pd.Series([1.0], index=tickers)

        # ── 准备矩阵 ───────────────────────────────────────────────────────────
        mu = alpha_scores.reindex(tickers).fillna(0.0).values.astype(float)

        # 协方差：对齐 tickers，正则化
        cov_aligned = cov_matrix.reindex(index=tickers, columns=tickers).fillna(0.0)
        Sigma = cov_aligned.values.astype(float)
        Sigma += np.eye(n) * 1e-6   # 正则化避免奇异

        # ── cvxpy 变量与目标 ────────────────────────────────────────────────────
        w = cp.Variable(n, nonneg=True)

        objective = cp.Maximize(
            mu @ w - self.risk_aversion * cp.quad_form(w, Sigma)
        )

        constraints: list = [
            cp.sum(w) == 1,
            w <= self.max_weight,
        ]

        # ── 因子暴露约束 ─────────────────────────────────────────────────────────
        if factor_exposures is not None and not factor_exposures.empty:
            B = (
                factor_exposures
                .reindex(index=tickers)
                .fillna(0.0)
                .values
                .astype(float)
            )  # shape (n, k)

            col_names: List[str] = factor_exposures.columns.tolist()
            ind_idx   = [i for i, c in enumerate(col_names) if c.startswith("ind_")]
            style_idx = [i for i, c in enumerate(col_names) if not c.startswith("ind_")]

            if ind_idx:
                B_ind = B[:, ind_idx]               # (n, k_ind)
                # 等权基准行业暴露（候选池内等权时各行业占比）
                b_bench_ind = B_ind.mean(axis=0)    # (k_ind,) — 常数向量
                # 主动暴露 = 持仓暴露 − 等权基准暴露
                constraints.append(
                    cp.abs(B_ind.T @ w - b_bench_ind) <= self.industry_limit
                )

            if style_idx:
                B_style = B[:, style_idx]           # (n, k_style)
                # 风格因子已截面 Z-score，全市场均值≈0；候选池内仍用主动暴露
                b_bench_style = B_style.mean(axis=0)
                constraints.append(
                    cp.abs(B_style.T @ w - b_bench_style) <= self.style_limit
                )

        # ── 换手率约束 ────────────────────────────────────────────────────────────
        if prev_weights is not None:
            w0 = prev_weights.reindex(tickers).fillna(0.0).values.astype(float)
            w0 = w0 / (w0.sum() + 1e-12)           # 归一化
            constraints.append(cp.norm1(w - w0) <= self.turnover_limit)

        # ── 求解 ──────────────────────────────────────────────────────────────────
        prob = cp.Problem(objective, constraints)
        _solve_ok = True
        try:
            prob.solve(solver=self.solver, verbose=False)
        except Exception as exc:
            log.warning("ConstrainedPortfolioOptimizer: solver error — %s", exc)
            _solve_ok = False

        if _solve_ok and prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and w.value is not None:
            result = np.maximum(w.value, 0.0)
            total = result.sum()
            if total > 1e-9:
                result /= total
                log.info(
                    "ConstrainedPortfolioOptimizer: 求解成功 status=%s "
                    "max_w=%.3f  HHI=%.4f",
                    prob.status,
                    float(result.max()),
                    float((result ** 2).sum()),
                )
                return pd.Series(result, index=tickers)

        # ── 回退等权 ──────────────────────────────────────────────────────────────
        log.warning(
            "ConstrainedPortfolioOptimizer: 求解失败（status=%s），回退等权",
            prob.status if _solve_ok else "solver_exception",
        )
        return pd.Series(1.0 / n, index=tickers)

    def compute_active_exposure(
        self,
        weights:          pd.Series,
        factor_exposures: pd.DataFrame,
    ) -> pd.DataFrame:
        """计算当前持仓的主动因子暴露。

        Parameters
        ----------
        weights : pd.Series
            index=ticker，归一化权重（sum=1）。
        factor_exposures : pd.DataFrame
            行=ticker，列=因子（由 ``build_factor_exposures()`` 生成）。

        Returns
        -------
        pd.DataFrame
            columns=['factor', 'portfolio_exposure', 'limit', 'breach']
        """
        common = weights.index.intersection(factor_exposures.index)
        if common.empty:
            return pd.DataFrame(columns=["factor", "portfolio_exposure", "limit", "breach"])

        w = weights.loc[common]
        B = factor_exposures.loc[common]

        port_exp = B.T @ w           # 持仓绝对暴露
        bench_exp = B.mean(axis=0)   # 等权基准暴露（候选池内）
        active_exp = port_exp - bench_exp   # 主动暴露

        rows = []
        for factor in B.columns:
            is_ind = str(factor).startswith("ind_")
            lim = self.industry_limit if is_ind else self.style_limit
            val = float(active_exp[factor])
            rows.append({
                "factor":              factor,
                "portfolio_exposure":  round(float(port_exp[factor]), 5),
                "benchmark_exposure":  round(float(bench_exp[factor]), 5),
                "active_exposure":     round(val, 5),
                "limit":               lim,
                "breach":              abs(val) > lim,
            })
        df = pd.DataFrame(rows).set_index("factor")
        return df[["portfolio_exposure", "benchmark_exposure", "active_exposure", "limit", "breach"]]

    # ── 辅助工厂方法 ──────────────────────────────────────────────────────────────

    @staticmethod
    def build_factor_exposures(
        panel_xs:      pd.DataFrame,
        style_factors: Optional[List[str]] = None,
        industry_col:  str = "exposure_industry",
    ) -> pd.DataFrame:
        """从单期截面面板构建因子暴露矩阵。

        包装 ``quantmind.risk.barra._build_exposure``，返回
        DataFrame（行=ticker，列=风格因子 + 行业哑变量）。

        Parameters
        ----------
        panel_xs : pd.DataFrame
            单期截面，index=ticker，columns ⊇ STYLE_MAP 值。
        style_factors : list[str], optional
            使用的风格因子名（STYLE_MAP 的键），默认全部 7 个。
        industry_col : str
            行业列名，默认 'exposure_industry'。

        Returns
        -------
        pd.DataFrame  index=ticker，columns=因子名
        """
        from quantmind.risk.barra import STYLE_MAP, _build_exposure

        factors = style_factors or list(STYLE_MAP.keys())
        X, factor_names = _build_exposure(panel_xs, factors, industry_col)
        return X  # index=ticker, columns=factor_names


# ── 便捷顶层函数 ───────────────────────────────────────────────────────────────

def run_constrained_optimization(
    alpha_scores:     pd.Series,
    panel_xs:         pd.DataFrame,
    returns_history:  pd.DataFrame,
    prev_weights:     Optional[pd.Series] = None,
    industry_limit:   float = _DEFAULT_INDUSTRY_LIMIT,
    style_limit:      float = _DEFAULT_STYLE_LIMIT,
    turnover_limit:   float = _DEFAULT_TURNOVER_LIMIT,
    risk_aversion:    float = _DEFAULT_RISK_AVERSION,
    max_weight:       float = _WEIGHT_CAP,
) -> Tuple[pd.Series, pd.DataFrame]:
    """一站式约束优化函数，封装完整流程。

    Parameters
    ----------
    alpha_scores : pd.Series
        index=ticker，alpha score（截面百分位或模型得分）。
    panel_xs : pd.DataFrame
        当期截面面板，index=ticker（用于构建因子暴露）。
    returns_history : pd.DataFrame
        历史收益率（日期×ticker），用于计算协方差矩阵（年化）。
    prev_weights : pd.Series, optional
        上期权重（换手约束）。
    其余参数见 ConstrainedPortfolioOptimizer。

    Returns
    -------
    (weights, exposure_report)
        weights : pd.Series — 最优权重
        exposure_report : pd.DataFrame — 因子暴露汇总
    """
    tickers = alpha_scores.index.tolist()

    # ── 协方差矩阵 ────────────────────────────────────────────────────────────
    hist = returns_history[
        [t for t in tickers if t in returns_history.columns]
    ].dropna(how="all")

    if len(hist) >= 21:
        cov_mat = (hist.pct_change().dropna(how="all").cov() * 252).reindex(
            index=tickers, columns=tickers
        ).fillna(0.0)
    else:
        # 历史不足：用对角协方差（方差=0.04，等价年化波动20%）
        cov_mat = pd.DataFrame(
            np.diag([0.04] * len(tickers)),
            index=tickers, columns=tickers,
        )
        log.warning("run_constrained_optimization: 历史数据不足，使用对角协方差矩阵")

    # ── 因子暴露 ──────────────────────────────────────────────────────────────
    factor_exp = ConstrainedPortfolioOptimizer.build_factor_exposures(
        panel_xs.reindex(tickers)
    )

    # ── 求解 ──────────────────────────────────────────────────────────────────
    opt = ConstrainedPortfolioOptimizer(
        industry_limit=industry_limit,
        style_limit=style_limit,
        turnover_limit=turnover_limit,
        risk_aversion=risk_aversion,
        max_weight=max_weight,
    )
    weights = opt.optimize(alpha_scores, factor_exp, cov_mat, prev_weights)
    exposure_report = opt.compute_active_exposure(weights, factor_exp)

    return weights, exposure_report


__all__ = [
    "ConstrainedPortfolioOptimizer",
    "run_constrained_optimization",
]
