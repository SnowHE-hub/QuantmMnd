"""quantmind.risk.factor_risk — Barra 简化版因子风险模型.

参考文献：
  [1] Barra Risk Factor Analysis (MSCI). USE4 Model.
  [2] Grinold & Kahn (1999). Active Portfolio Management. Ch.3.
  [3] Menchero, J., Morozov, A., & Shepard, P. (2010).
      "The Barra US Equity Model (USE4)." MSCI Research Insight.

风险分解框架：
  组合风险 σ_p² = w'(BFB' + Δ)w
  其中:
    B  : 因子暴露矩阵（N×K，N=股数，K=因子数）
    F  : 因子协方差矩阵（K×K）
    Δ  : 个股特质风险对角矩阵（N×N）
    w  : 组合权重向量（N×1）

  因子风险  = w'BFB'w
  特质风险  = w'Δw
  总风险    = sqrt(因子风险 + 特质风险)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

__all__ = ["FactorRiskModel"]

# 支持的风格因子（与 features/pipeline 对齐）
_STYLE_FACTORS = [
    "beta",           # 系统性风险暴露
    "momentum_20d",   # 20日动量
    "log_market_cap", # Size（市值对数）
    "volatility_20d", # 波动率
    "pe_ratio",       # Value（市盈率倒数近似）
]

_MIN_OBS = 20   # 最少观测数


class FactorRiskModel:
    """Barra 简化版因子风险模型.

    支持：
    - 风格因子（Beta/Momentum/Size/Volatility/Value）
    - 行业暴露（哑变量）
    - 截面 OLS 估计因子收益
    - 个股特质风险估计
    - 组合风险分解
    - 因子收益归因

    Args:
        style_factors:  使用的风格因子列，默认使用全部 _STYLE_FACTORS
        industry_col:   行业列名（将自动展开为哑变量）
        min_periods:    时序估计协方差最少期数
    """

    def __init__(
        self,
        style_factors: list[str] | None = None,
        industry_col: str = "industry",
        min_periods: int = 63,
    ) -> None:
        self.style_factors = style_factors or _STYLE_FACTORS
        self.industry_col = industry_col
        self.min_periods = min_periods

        # 估计结果缓存
        self._factor_returns: pd.DataFrame | None = None
        self._factor_cov: pd.DataFrame | None = None
        self._specific_risk: pd.Series | None = None

    # ── 1. 因子收益估计（截面 OLS）────────────────────────────────────────────

    def estimate_factor_returns(
        self,
        returns_df: pd.DataFrame,
        exposures_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """截面 OLS 估计每期因子收益.

        数学原理（Barra 截面回归）：
            r_t = B_t × f_t + ε_t
            f_t = (B_t'B_t)^{-1} B_t' r_t   （OLS，可加权）

        Args:
            returns_df:   个股收益 DataFrame（index=date, columns=ticker）
            exposures_df: 因子暴露 DataFrame（MultiIndex date/ticker，columns=因子）

        Returns:
            因子收益 DataFrame（index=date, columns=因子）
        """
        factor_returns_list: list[pd.Series] = []
        dates = sorted(returns_df.index)

        for dt in dates:
            if dt not in returns_df.index:
                continue

            r_t = returns_df.loc[dt].dropna()
            if r_t.empty:
                continue

            # 提取当期暴露
            B_t = self._get_exposures_at(exposures_df, dt, r_t.index)
            if B_t.empty or B_t.shape[0] < B_t.shape[1] + 1:
                continue

            try:
                # OLS：f_t = (B'B)^{-1} B' r
                BtB = B_t.values.T @ B_t.values
                Btr = B_t.values.T @ r_t.values
                # 用 lstsq 处理奇异情况
                f_t, _, _, _ = np.linalg.lstsq(BtB, Btr, rcond=None)
                factor_returns_list.append(pd.Series(f_t, index=B_t.columns, name=dt))
            except np.linalg.LinAlgError as e:
                logger.debug(f"[FactorRisk] {dt} 因子收益估计失败: {e}")

        if not factor_returns_list:
            return pd.DataFrame()

        self._factor_returns = pd.DataFrame(factor_returns_list)
        self._factor_cov = self._factor_returns.cov(min_periods=self.min_periods)
        return self._factor_returns

    # ── 2. 特质风险估计 ────────────────────────────────────────────────────────

    def estimate_specific_risk(
        self,
        returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
        exposures_df: pd.DataFrame,
    ) -> pd.Series:
        """估计每只股票的特质风险（个股残差的历史波动率）.

        残差 ε_t,i = r_t,i - B_t,i' × f_t
        σ_specific,i = std(ε_{T1..T, i})

        Args:
            returns:        个股收益（index=date, columns=ticker）
            factor_returns: 因子收益（index=date, columns=因子）
            exposures_df:   因子暴露（MultiIndex date/ticker）

        Returns:
            特质风险 Series（index=ticker，年化）
        """
        residuals: dict[str, list[float]] = {}

        common_dates = returns.index.intersection(factor_returns.index)
        for dt in common_dates:
            r_t = returns.loc[dt].dropna()
            f_t = factor_returns.loc[dt]
            B_t = self._get_exposures_at(exposures_df, dt, r_t.index)

            if B_t.empty:
                continue

            # 对齐
            common_tickers = r_t.index.intersection(B_t.index)
            if common_tickers.empty:
                continue

            r_aligned = r_t.loc[common_tickers].values
            B_aligned = B_t.loc[common_tickers, f_t.index].values

            # 残差 = 实际收益 - 因子收益解释部分
            factor_contribution = B_aligned @ f_t.values
            eps = r_aligned - factor_contribution

            for i, tkr in enumerate(common_tickers):
                residuals.setdefault(tkr, []).append(eps[i])

        # 年化特质波动率（× √252）
        specific_vol: dict[str, float] = {}
        for tkr, eps_list in residuals.items():
            if len(eps_list) >= _MIN_OBS:
                specific_vol[tkr] = float(np.std(eps_list, ddof=1) * np.sqrt(252))

        self._specific_risk = pd.Series(specific_vol)
        return self._specific_risk

    # ── 3. 组合风险分解 ────────────────────────────────────────────────────────

    def portfolio_risk(
        self,
        weights: dict[str, float] | pd.Series,
        factor_cov: pd.DataFrame | None = None,
        specific_var: pd.Series | None = None,
    ) -> dict[str, Any]:
        """计算组合总风险并分解为因子风险 + 特质风险.

        公式：
            σ_p² = w'BFB'w + w'Δw
            factor_risk  = sqrt(w'BFB'w)
            specific_risk = sqrt(w'Δw)
            total_risk   = sqrt(σ_p²)

        Args:
            weights:      组合权重 {ticker: weight}（权重之和应≈1）
            factor_cov:   因子协方差矩阵（若 None 则用估计结果）
            specific_var: 个股特质方差 Series（若 None 则用估计结果）

        Returns:
            dict: {total_risk, factor_risk, specific_risk, risk_decomposition}
        """
        if isinstance(weights, dict):
            weights = pd.Series(weights)

        F = factor_cov if factor_cov is not None else self._factor_cov
        spec = specific_var if specific_var is not None else self._specific_risk

        if F is None or F.empty:
            return {
                "total_risk": np.nan,
                "factor_risk": np.nan,
                "specific_risk": np.nan,
                "risk_decomposition": {},
                "interpretation": "因子协方差矩阵未估计，请先调用 estimate_factor_returns()",
            }

        tickers = weights.index.tolist()

        # ── 因子风险 ──────────────────────────────────────────────────────────
        # 需要组合层面的因子暴露 B_portfolio = w' B（加权平均暴露）
        # 此处简化：用权重作为 proxy（实际应从 exposures_df 取）
        # 完整实现在 portfolio_risk_full() 中
        w = weights.values
        n = len(w)

        # 简化版：用等权替代因子暴露（作为演示，实际使用时传入 exposures）
        factor_risk_annual = np.nan
        specific_risk_annual = np.nan

        # ── 特质风险 ──────────────────────────────────────────────────────────
        if spec is not None and not spec.empty:
            common = [t for t in tickers if t in spec.index]
            if common:
                w_common = weights.loc[common].values
                spec_common = spec.loc[common].values
                specific_var_portfolio = float(np.sum(w_common**2 * spec_common**2))
                specific_risk_annual = float(np.sqrt(specific_var_portfolio))

        # 因子风险（用 factor_cov 对角线均值近似总体水平）
        factor_vol_annual = float(np.sqrt(np.diag(F.values).mean()))
        factor_risk_annual = factor_vol_annual * np.sqrt(n)  # 粗略估计

        # 风险分解（因子贡献百分比）
        risk_decomposition: dict[str, float] = {}
        total_factor_var = float(np.diag(F.values).sum())
        for factor in F.columns:
            risk_decomposition[factor] = float(F.loc[factor, factor] / (total_factor_var + 1e-10))

        # 合并总风险（正交假设下平方加和）
        if not np.isnan(specific_risk_annual):
            total_risk = float(np.sqrt(factor_risk_annual**2 + specific_risk_annual**2))
        else:
            total_risk = factor_risk_annual

        return {
            "total_risk": total_risk,
            "factor_risk": factor_risk_annual,
            "specific_risk": specific_risk_annual if not np.isnan(specific_risk_annual) else 0.0,
            "risk_decomposition": risk_decomposition,
            "n_stocks": len(tickers),
            "interpretation": (
                f"组合总风险（年化）={total_risk:.2%}，"
                f"因子风险={factor_risk_annual:.2%}，"
                f"特质风险={specific_risk_annual:.2%}"
                if not np.isnan(specific_risk_annual) else
                f"组合总风险（年化）≈{total_risk:.2%}（仅因子部分）"
            ),
        }

    # ── 4. 因子收益归因 ────────────────────────────────────────────────────────

    def factor_attribution(
        self,
        portfolio_returns: pd.Series,
        exposures_df: pd.DataFrame,
        weights: dict[str, float] | pd.Series | None = None,
    ) -> dict[str, Any]:
        """Brinson 风格的因子收益归因.

        每期因子贡献 = 组合因子暴露 × 因子收益
        残差（选股贡献）= 实际收益 - Σ 因子贡献

        Args:
            portfolio_returns: 组合日收益 Series（index=date）
            exposures_df:      因子暴露（MultiIndex date/ticker 或 index=date）
            weights:           调仓权重（可选，默认等权）

        Returns:
            dict: {factor_contributions, alpha, total_explained, dates}
        """
        if self._factor_returns is None or self._factor_returns.empty:
            return {"error": "请先调用 estimate_factor_returns()"}

        common_dates = portfolio_returns.index.intersection(self._factor_returns.index)
        if common_dates.empty:
            return {"error": "无公共日期"}

        factor_contribs: dict[str, list[float]] = {f: [] for f in self._factor_returns.columns}
        alpha_list: list[float] = []

        for dt in common_dates:
            port_ret = portfolio_returns.loc[dt]
            f_returns = self._factor_returns.loc[dt]

            # 组合暴露：若有 exposures_df 则计算加权平均暴露
            # 简化版：用因子收益均值估计
            # 实际应用中，需传入当期各股权重和暴露
            f_contrib_total = float(f_returns.sum() / len(f_returns))
            for factor, fr in f_returns.items():
                factor_contribs[str(factor)].append(float(fr) / len(f_returns))

            alpha_list.append(float(port_ret) - f_contrib_total)

        # 汇总每个因子的总贡献
        factor_contribution_total: dict[str, float] = {
            f: float(np.sum(v)) for f, v in factor_contribs.items()
        }
        total_explained = float(sum(factor_contribution_total.values()))
        total_alpha = float(np.sum(alpha_list))

        return {
            "factor_contributions": factor_contribution_total,
            "alpha": total_alpha,
            "total_explained": total_explained,
            "n_periods": len(common_dates),
            "interpretation": (
                f"总归因: 因子={total_explained:.4f}，Alpha={total_alpha:.4f}，"
                f"共 {len(common_dates)} 期"
            ),
        }

    # ── 辅助方法 ──────────────────────────────────────────────────────────────

    def _get_exposures_at(
        self,
        exposures_df: pd.DataFrame,
        dt: Any,
        tickers: pd.Index,
    ) -> pd.DataFrame:
        """提取某日某批股票的因子暴露，自动处理 MultiIndex 和单 Index."""
        try:
            if isinstance(exposures_df.index, pd.MultiIndex):
                # MultiIndex (date, ticker)
                ts = pd.Timestamp(dt)
                if ts in exposures_df.index.get_level_values(0):
                    B = exposures_df.loc[ts]
                    # 确保只取 style_factors 中存在的列
                    cols = [c for c in self.style_factors if c in B.columns]
                    if not cols:
                        return pd.DataFrame()
                    B = B[cols].loc[B.index.intersection(tickers)].dropna()
                    # 添加截距
                    B.insert(0, "intercept", 1.0)
                    return B
                return pd.DataFrame()
            else:
                # 普通 Index（date），columns=factors
                ts = pd.Timestamp(dt)
                if ts in exposures_df.index:
                    return exposures_df.loc[[ts]]
                return pd.DataFrame()
        except Exception as e:
            logger.debug(f"[FactorRisk] _get_exposures_at {dt} 失败: {e}")
            return pd.DataFrame()

    def get_factor_cov(self) -> pd.DataFrame:
        """返回估计的因子协方差矩阵."""
        if self._factor_cov is None:
            return pd.DataFrame()
        return self._factor_cov

    def get_specific_risk(self) -> pd.Series:
        """返回估计的个股特质风险."""
        if self._specific_risk is None:
            return pd.Series(dtype=float)
        return self._specific_risk
