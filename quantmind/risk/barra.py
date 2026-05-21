"""quantmind.risk.barra — Barra 截面因子归因模型（CNE5 简化版）.

框架:
  r_t = X_t @ f_t + e_t          （截面 WLS，权重 = sqrt(流通市值)）
  X_t: 标准化风格因子 + 行业哑变量（约束：行业权重 = 市值加权）
  f_t: 当期因子收益（WLS 估计）
  e_t: 个股特质收益（纯 Alpha 来源）

风格因子（7个，直接从 alpha_panel_v4 取列）:
  size, value, momentum, quality, volatility, beta, liquidity

输出:
  BarraResult.factor_returns  : DataFrame (as_of × factor)
  BarraResult.factor_ic       : Series   (factor → IC vs forward_return)
  BarraResult.factor_icir     : Series   (factor → ICIR)
  BarraResult.residuals       : DataFrame (MultiIndex as_of/ticker → residual)
  BarraResult.residual_ic     : float     (模型得分 vs 残差 IC，= 纯 Alpha IC)
  BarraResult.industry_returns: DataFrame (as_of × industry)
  BarraResult.r_squared       : Series   (as_of → 截面 R²)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

__all__ = ["BarraAttributor", "BarraResult", "STYLE_MAP"]

# Panel 列名 → Barra 风格因子映射
STYLE_MAP: dict[str, str] = {
    "size":       "log_market_cap",
    "value":      "book_to_market",
    "momentum":   "momentum_6m",
    "quality":    "roe_ttm",
    "volatility": "volatility_3m",
    "beta":       "beta_252d",
    "liquidity":  "turnover_3m_avg",
}

_WEIGHT_COL  = "log_circ_market_cap"   # 截面回归权重（流通市值对数→指数→√）
_INDUSTRY_COL = "exposure_industry"
_MIN_STOCKS  = 50                        # 最少股票数，否则跳过该期


def _winsorize(s: pd.Series, pct: float = 0.01) -> pd.Series:
    lo, hi = s.quantile(pct), s.quantile(1 - pct)
    return s.clip(lo, hi)


def _zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    if std < 1e-9:
        return s - s.mean()
    return (s - s.mean()) / std


def _build_exposure(
    xs: pd.DataFrame,
    style_cols: list[str],
    industry_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    """构建单期因子暴露矩阵 X（标准化风格 + 行业哑变量）.

    返回 (X, factor_names)，行 = ticker，列 = 因子。
    行业哑变量施加市值加权约束（去掉一个行业，改为约束列）。
    """
    rows = xs.copy()

    # ── 风格因子：去极值 + 截面标准化 ──────────────────────────────────────
    style_data: dict[str, pd.Series] = {}
    for fname, col in zip(style_cols, [STYLE_MAP.get(f, f) for f in style_cols]):
        if col not in rows.columns:
            continue
        s = _winsorize(rows[col].dropna())
        if len(s) < _MIN_STOCKS:
            continue
        style_data[fname] = _zscore(s)

    style_df = pd.DataFrame(style_data).reindex(rows.index)

    # ── 行业哑变量 ──────────────────────────────────────────────────────────
    if industry_col in rows.columns:
        ind = rows[industry_col].fillna("Unknown")
        dummies = pd.get_dummies(ind, prefix="ind", dtype=float)
        # 去掉样本量最大的行业（作为基准，避免完全多重共线性）
        if dummies.shape[1] > 1:
            drop_col = dummies.sum().idxmax()
            dummies = dummies.drop(columns=[drop_col])
    else:
        dummies = pd.DataFrame(index=rows.index)

    # ── 合并 ────────────────────────────────────────────────────────────────
    X = pd.concat([style_df, dummies], axis=1).dropna(how="all")
    X = X.fillna(0.0)
    factor_names = list(X.columns)
    return X, factor_names


@dataclass
class BarraResult:
    """Barra 归因结果容器。"""
    factor_returns:   pd.DataFrame = field(default_factory=pd.DataFrame)  # as_of × factor
    factor_ic:        pd.Series    = field(default_factory=pd.Series)      # factor → IC
    factor_icir:      pd.Series    = field(default_factory=pd.Series)      # factor → ICIR
    residuals:        pd.DataFrame = field(default_factory=pd.DataFrame)   # as_of × ticker
    r_squared:        pd.Series    = field(default_factory=pd.Series)      # as_of → R²
    industry_returns: pd.DataFrame = field(default_factory=pd.DataFrame)  # as_of × industry
    # residual IC（须在外部传入 model scores 后填充）
    residual_ic:      float        = float("nan")
    residual_icir:    float        = float("nan")
    raw_ic:           float        = float("nan")
    raw_icir:         float        = float("nan")
    style_names:      list[str]    = field(default_factory=list)
    n_periods:        int          = 0
    meta:             dict[str, Any] = field(default_factory=dict)


class BarraAttributor:
    """Barra 截面因子归因。

    Args:
        style_factors: 使用的风格因子名列表（对应 STYLE_MAP 键）
        industry_col:  行业列名（panel 中已有，如 'exposure_industry'）
        label_col:     前瞻收益列名（'forward_return_63d'）
        weight_col:    截面回归权重列名（'log_circ_market_cap'）
    """

    def __init__(
        self,
        style_factors: list[str] | None = None,
        industry_col: str = _INDUSTRY_COL,
        label_col: str = "forward_return_63d",
        weight_col: str = _WEIGHT_COL,
    ) -> None:
        self.style_factors = style_factors or list(STYLE_MAP.keys())
        self.industry_col  = industry_col
        self.label_col     = label_col
        self.weight_col    = weight_col

    # ── 核心：截面 WLS 回归 ─────────────────────────────────────────────────

    def _cross_section_wls(
        self,
        X: pd.DataFrame,
        r: pd.Series,
        w: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """单期截面 WLS：r = X @ f + e，返回 (f, e, R²)."""
        common = X.index.intersection(r.index).intersection(w.index)
        if len(common) < _MIN_STOCKS:
            return np.array([]), np.array([]), float("nan")

        Xm = X.loc[common].values.astype(float)
        rm = r.loc[common].values.astype(float)
        wm = np.sqrt(np.exp(w.loc[common].values.clip(-10, 20).astype(float)))

        W = np.diag(wm)
        XtW = Xm.T @ W
        XtWX = XtW @ Xm
        XtWr = XtW @ rm

        try:
            f, _, _, _ = np.linalg.lstsq(XtWX, XtWr, rcond=None)
        except np.linalg.LinAlgError:
            return np.array([]), np.array([]), float("nan")

        e = rm - Xm @ f

        # 加权 R²
        r_bar = np.average(rm, weights=wm)
        ss_tot = np.sum(wm * (rm - r_bar) ** 2)
        ss_res = np.sum(wm * e ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

        return f, e, r2

    # ── 主入口 ──────────────────────────────────────────────────────────────

    def fit(
        self,
        panel: pd.DataFrame,
        scores: pd.Series | None = None,
    ) -> BarraResult:
        """运行 Barra 归因。

        Args:
            panel:  MultiIndex (as_of, ticker)，含风格/行业/标签列
            scores: 可选，MultiIndex (as_of, ticker) 模型得分，用于计算 Pure Alpha IC

        Returns:
            BarraResult
        """
        if self.label_col not in panel.columns:
            raise ValueError(f"panel 缺少标签列 '{self.label_col}'")

        periods = sorted(panel.index.get_level_values("as_of").unique())
        logger.info(f"[Barra] 开始归因：{len(periods)} 期，{len(self.style_factors)} 风格因子")

        factor_ret_rows: list[pd.Series]  = []
        resid_rows:      list[pd.Series]  = []
        r2_vals:         dict[Any, float] = {}

        for as_of in periods:
            xs = panel.xs(as_of, level="as_of")
            r  = xs[self.label_col].dropna()
            if len(r) < _MIN_STOCKS:
                continue

            w_raw = xs[self.weight_col] if self.weight_col in xs.columns else pd.Series(0.0, index=xs.index)
            X, factor_names = _build_exposure(xs, self.style_factors, self.industry_col)

            f, e, r2 = self._cross_section_wls(X, r, w_raw)
            if f.size == 0:
                continue

            factor_ret_rows.append(pd.Series(f, index=factor_names, name=as_of))

            # 残差 → resid_rows（只保留风格部分的残差）
            common = X.index.intersection(r.index)
            resid_rows.append(pd.Series(e, index=common, name=as_of))
            r2_vals[as_of] = r2

        result = BarraResult(style_names=self.style_factors, n_periods=len(factor_ret_rows))

        if not factor_ret_rows:
            logger.warning("[Barra] 无有效期次，返回空结果")
            return result

        # ── 因子收益矩阵 ────────────────────────────────────────────────────
        fr = pd.DataFrame(factor_ret_rows)
        style_cols = [c for c in fr.columns if not c.startswith("ind_")]
        ind_cols   = [c for c in fr.columns if c.startswith("ind_")]

        result.factor_returns   = fr[style_cols]
        result.industry_returns = fr[ind_cols] if ind_cols else pd.DataFrame()
        result.r_squared        = pd.Series(r2_vals, name="r_squared")

        # ── 因子 IC（因子收益 vs 下期截面均值，近似） ────────────────────────
        # 用因子收益本身的时序均值 / 标准差 衡量稳定性
        ic_vals   = result.factor_returns.mean()
        icir_vals = result.factor_returns.mean() / (result.factor_returns.std() + 1e-9)
        result.factor_ic   = ic_vals.rename("ic_mean")
        result.factor_icir = icir_vals.rename("icir")

        # ── 残差矩阵 ────────────────────────────────────────────────────────
        resid_df = pd.DataFrame({s.name: s for s in resid_rows}).T  # Timestamp index
        resid_df.index.name = "as_of"
        result.residuals = resid_df

        # ── Raw IC（模型得分 vs 前瞻收益） ──────────────────────────────────
        if scores is not None:
            raw_ics, resid_ics = [], []
            for as_of in periods:
                try:
                    r_t = panel.xs(as_of, level="as_of")[self.label_col].dropna()
                    sc_t = scores.xs(as_of, level="as_of").reindex(r_t.index).dropna()
                    common = r_t.index.intersection(sc_t.index)
                    if len(common) < 20:
                        continue
                    raw_ics.append(r_t.loc[common].corr(sc_t.loc[common], method="spearman"))

                    if as_of in resid_df.index:
                        e_t = resid_df.loc[as_of].reindex(common).dropna()
                        common2 = e_t.index.intersection(sc_t.index)
                        if len(common2) >= 20:
                            resid_ics.append(e_t.loc[common2].corr(sc_t.loc[common2], method="spearman"))
                except KeyError:
                    continue

            if raw_ics:
                result.raw_ic   = float(np.mean(raw_ics))
                result.raw_icir = float(np.mean(raw_ics) / (np.std(raw_ics) + 1e-9))
            if resid_ics:
                result.residual_ic   = float(np.mean(resid_ics))
                result.residual_icir = float(np.mean(resid_ics) / (np.std(resid_ics) + 1e-9))

        result.meta = {
            "label_col":     self.label_col,
            "style_factors": self.style_factors,
            "n_periods":     len(factor_ret_rows),
            "avg_r2":        float(result.r_squared.mean()),
        }
        logger.info(
            f"[Barra] 完成：{result.n_periods} 期 | "
            f"平均 R²={result.meta['avg_r2']:.3f} | "
            f"Raw IC={result.raw_ic:.4f} | "
            f"Pure Alpha IC={result.residual_ic:.4f}"
        )
        return result
