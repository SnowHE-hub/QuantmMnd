"""quantmind.features.neutralize — 截面行业中性化（可选市值残差）."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["neutralize_cross_section"]


def neutralize_cross_section(
    df: pd.DataFrame,
    *,
    industry_col: str = "industry",
    factors: list[str] | None = None,
    neutralize_mcap: bool = False,
    log_mktcap_col: str = "log_market_cap",
) -> pd.DataFrame:
    """对每个截面行集合，数值因子减去行业内均值。

    可选：在每列上对 ``log_mktcap_col`` 做一元 OLS，取残差（仍保留 NaN）。

    Args:
        df:             单行截面（index=ticker）或任意行集合；须含 ``industry_col``
        industry_col:   行业分组列（字符串）
        factors:        要处理的数值列；默认自动选取 float 列（排除 industry）
        neutralize_mcap: 是否对 log 市值回归取残差
        log_mktcap_col:  市值列名

    Returns:
        与 ``df`` 同索引；仅 ``factors`` 列被替换，其余列拷贝。
    """
    out = df.copy()
    if industry_col not in out.columns:
        return out

    if factors is None:
        factors = [
            c
            for c in out.columns
            if c != industry_col and pd.api.types.is_numeric_dtype(out[c])
        ]

    ind = out[industry_col].astype("string")

    for col in factors:
        if col not in out.columns:
            continue
        y = pd.to_numeric(out[col], errors="coerce")
        gm = y.groupby(ind).transform("mean")
        out[col] = y - gm

    if neutralize_mcap and log_mktcap_col in out.columns:
        x = pd.to_numeric(out[log_mktcap_col], errors="coerce").to_numpy(dtype=float)
        ok_x = np.isfinite(x)
        for col in factors:
            if col not in out.columns or col == log_mktcap_col:
                continue
            y = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
            mask = ok_x & np.isfinite(y)
            if int(mask.sum()) < 8:
                continue
            x_m = x[mask]
            y_m = y[mask]
            xc = x_m - x_m.mean()
            beta = float(np.dot(xc, y_m - y_m.mean()) / (np.dot(xc, xc) + 1e-12))
            pred = beta * (x - x_m.mean())
            resid = y - pred
            resid[~mask] = np.nan
            out[col] = resid

    return out
