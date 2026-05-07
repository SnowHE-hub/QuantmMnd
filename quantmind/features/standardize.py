"""quantmind.features.standardize — 因子横截面预处理.

工业级量化标准流水线：

    raw → winsorize → fillna → neutralize (industry/size) → zscore (cross-section)

每一步都是 ``DataFrame -> DataFrame`` 形式，保持 (ticker × factor) 结构不变。

参考
====

- Asness, Frazzini, Pedersen (2014) "Quality minus junk" 的标准化流程
- 国内多因子模型常用流程（中信、申万、华泰金工）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# ============================================================================
# Winsorize（极值处理）
# ============================================================================


def winsorize(
    df: pd.DataFrame,
    *,
    method: str = "sigma",
    sigma: float = 3.0,
    pct: float = 0.01,
) -> pd.DataFrame:
    """剔除极端值，每列独立处理.

    Args:
        method: 'sigma' (mean ± sigma·std) 或 'pct' (上下截断分位数)
        sigma: sigma 法的倍数
        pct: 分位数法的截断比例（如 0.01 = 1%~99%）
    """
    df = df.copy()
    for col in df.columns:
        s = df[col].astype("float64")
        if s.dropna().empty:
            continue
        if method == "sigma":
            mu, sd = s.mean(), s.std()
            if sd == 0 or pd.isna(sd):
                continue
            lo, hi = mu - sigma * sd, mu + sigma * sd
        elif method == "pct":
            lo, hi = s.quantile(pct), s.quantile(1 - pct)
        else:
            raise ValueError(f"unknown method: {method!r}")
        df[col] = s.clip(lo, hi)
    return df


# ============================================================================
# 缺失值填充
# ============================================================================


def fillna_cross_section(
    df: pd.DataFrame,
    *,
    strategy: str = "median",
    industry: pd.Series | None = None,
) -> pd.DataFrame:
    """按横截面填缺失值。

    Args:
        strategy: 'median' / 'mean' / 'zero'
        industry: 同 index Series，按行业分组填中位数
    """
    df = df.copy()
    if industry is not None:
        # 行业内中位数
        industry = industry.reindex(df.index)
        for col in df.columns:
            grouped = df[col].groupby(industry)
            df[col] = grouped.transform(lambda s: s.fillna(s.median()))
        # 还有 NaN 的（行业内全空），用全市场中位数
        for col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    else:
        for col in df.columns:
            s = df[col]
            if strategy == "median":
                fill_v = s.median()
            elif strategy == "mean":
                fill_v = s.mean()
            elif strategy == "zero":
                fill_v = 0.0
            else:
                raise ValueError(f"unknown strategy: {strategy!r}")
            df[col] = s.fillna(fill_v)
    return df


# ============================================================================
# 中性化（行业、市值）—— 回归残差法
# ============================================================================


def _residualize(y: pd.Series, X: pd.DataFrame) -> pd.Series:
    """y 对 X 做 OLS，返回残差（不带常数项的 X 自动加常数）.

    NaN 行（任意一列 NaN）跳过；返回的残差与 ``y`` 同 index，
    跳过的位置保留 NaN。
    """
    common = y.dropna().index.intersection(X.dropna().index)
    if len(common) < X.shape[1] + 2:
        return pd.Series(np.nan, index=y.index)
    yy = y.loc[common].astype("float64").values
    XX = X.loc[common].astype("float64").values
    XX_const = np.column_stack([np.ones(len(XX)), XX])
    try:
        beta, *_ = np.linalg.lstsq(XX_const, yy, rcond=None)
    except np.linalg.LinAlgError:
        return pd.Series(np.nan, index=y.index)
    pred = XX_const @ beta
    out = pd.Series(np.nan, index=y.index, dtype="float64")
    out.loc[common] = yy - pred
    return out


def neutralize(
    df: pd.DataFrame,
    *,
    industry: pd.Series | None = None,
    log_market_cap: pd.Series | None = None,
    cols: list[str] | None = None,
) -> pd.DataFrame:
    """对每个因子做行业 + 市值中性化（回归残差法）.

    Args:
        industry: 同 index 的 Series，类别变量（行业代码或字符串）
        log_market_cap: 同 index 的 Series（已取对数）
        cols: 限定要中性化的列；None 表示全部数值列

    返回值与输入同 shape，列内容为残差（已去掉行业 + 市值的线性影响）。
    """
    if industry is None and log_market_cap is None:
        return df.copy()

    cols = cols if cols is not None else list(df.columns)
    out = df.copy()

    # 构造解释矩阵
    pieces = []
    if industry is not None:
        ind = industry.reindex(df.index).astype("category")
        ind_dummies = pd.get_dummies(ind, drop_first=True, dtype="float64")
        # 限制 dummies 列数防止过拟合
        if not ind_dummies.empty:
            pieces.append(ind_dummies)
    if log_market_cap is not None:
        pieces.append(log_market_cap.reindex(df.index).rename("log_mv").to_frame())

    if not pieces:
        return out
    X = pd.concat(pieces, axis=1)

    for col in cols:
        if col not in df.columns:
            continue
        out[col] = _residualize(df[col].astype("float64"), X)
    return out


# ============================================================================
# Z-score 标准化（横截面）
# ============================================================================


_ZSCORE_STD_EPS = 1e-9


def cross_section_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """每列减均值除标准差（横截面标准化）；常数 / 近常数列置 0."""
    df = df.copy()
    for col in df.columns:
        s = df[col].astype("float64")
        mu, sd = s.mean(), s.std()
        if pd.isna(sd) or sd < _ZSCORE_STD_EPS:
            # 常数列或近常数列（含数值噪声），统一置 0
            df[col] = 0.0
        else:
            df[col] = (s - mu) / sd
    return df


def cross_section_rank(df: pd.DataFrame, *, pct: bool = True) -> pd.DataFrame:
    """每列横截面排名（pct=True 输出 [0,1] 分位）."""
    return df.rank(pct=pct, method="average")


# ============================================================================
# 全流程
# ============================================================================


_DEFAULT_NEUTRALIZE_EXCLUDE = {"log_market_cap", "log_circ_market_cap"}


def standardize(
    df: pd.DataFrame,
    *,
    industry: pd.Series | None = None,
    log_market_cap: pd.Series | None = None,
    winsorize_method: str = "sigma",
    winsorize_sigma: float = 3.0,
    do_neutralize: bool = True,
    neutralize_exclude: set[str] | None = None,
) -> pd.DataFrame:
    """一站式：winsorize → 初始 zscore（填 0）→ 中性化 → 最终 zscore.

    - ``log_market_cap`` 不应被自己中性化（残差恒为 0），默认从中性化列中排除
    - 常数因子列（如市场级情绪）在 zscore 后归 0，不会污染最终值
    """
    out = winsorize(df, method=winsorize_method, sigma=winsorize_sigma)
    out = cross_section_zscore(out).fillna(0.0)

    if do_neutralize:
        exclude = neutralize_exclude or _DEFAULT_NEUTRALIZE_EXCLUDE
        cols_to_neutralize = [c for c in out.columns if c not in exclude]
        out = neutralize(
            out,
            industry=industry,
            log_market_cap=log_market_cap,
            cols=cols_to_neutralize,
        )
        out = cross_section_zscore(out).fillna(0.0)
    return out


# ============================================================================
# 信息系数（IC）—— 因子分析常用
# ============================================================================


def information_coefficient(
    factor: pd.Series, forward_return: pd.Series, *, method: str = "spearman"
) -> float:
    """单期 IC：因子值与未来收益的横截面相关系数."""
    common = factor.dropna().index.intersection(forward_return.dropna().index)
    if len(common) < 5:
        return float("nan")
    f = factor.loc[common]
    r = forward_return.loc[common]
    if method == "spearman":
        rho, _ = stats.spearmanr(f, r)
    elif method == "pearson":
        rho, _ = stats.pearsonr(f, r)
    else:
        raise ValueError(f"unknown method: {method!r}")
    return float(rho) if not np.isnan(rho) else float("nan")


__all__ = [
    "cross_section_rank",
    "cross_section_zscore",
    "fillna_cross_section",
    "information_coefficient",
    "neutralize",
    "standardize",
    "winsorize",
]
