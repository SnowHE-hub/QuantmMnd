"""Score transforms that preserve model ordering while removing degenerate ties."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def order_preserving_pct_rank(
    scores: pd.Series | np.ndarray | Sequence[float],
    *,
    higher_is_better: bool = True,
) -> pd.Series:
    """Map scores to unique percentiles in ``(0, 1]``, strictly monotone in raw score order.

    LambdaRank / LGBM 常输出大量并列分数；``rank(pct=True, method='first')`` 会按 Series 行顺序
    打破平局，引入任意噪声。本函数先按 raw score **稳定排序**，再用均匀间距赋分位，保证：
    - 分数更高（若 ``higher_is_better``）的股票 pct 更高；
    - 全截面 ``len(unique)==len(valid)``。
    """
    if isinstance(scores, pd.Series):
        s = scores.astype(np.float64)
    else:
        s = pd.Series(np.asarray(scores, dtype=np.float64).reshape(-1))
    out = pd.Series(np.nan, index=s.index, dtype=np.float64)
    mask = s.notna()
    sub = s.loc[mask]
    if sub.empty:
        return out
    ascending = not higher_is_better
    order = sub.sort_values(ascending=ascending, kind="mergesort").index
    n = len(order)
    pct_vals = np.linspace(1.0 / n, 1.0, n, dtype=np.float64)
    out.loc[order] = pct_vals
    return out
