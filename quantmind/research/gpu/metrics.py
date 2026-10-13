"""评估：逐日截面 RankIC、移动块自助法区间、pinball / 覆盖率."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rank_ic_by_date(score: pd.Series, y: pd.Series, min_n: int = 30) -> pd.Series:
    """score / y 的 index 均为 (as_of, ticker)；返回每个 as_of 的 Spearman IC（两者都非缺失的股票 ≥ min_n）。"""
    d = pd.DataFrame({"s": score, "y": y}).dropna()
    g = d.groupby(level="as_of")
    ic = g.apply(lambda x: x["s"].rank().corr(x["y"].rank()) if len(x) >= min_n else np.nan)
    return ic.dropna().sort_index()


def daily_rank_ic(pred: np.ndarray, y: np.ndarray, day: np.ndarray, mask: np.ndarray | None = None,
                  min_n: int = 30) -> np.ndarray:
    """行按 day 升序排列时的逐日 Spearman IC（pred、y 都有限且 mask 为真的行 ≥ min_n 的日子），返回长度 = 有效日数。"""
    from scipy.stats import rankdata

    from quantmind.research.gpu.transforms import day_bounds
    ok = np.isfinite(pred) & np.isfinite(y)
    if mask is not None:
        ok &= mask
    b = day_bounds(day)
    out = []
    for s, e in zip(b[:-1], b[1:], strict=True):
        o = ok[s:e]
        if o.sum() < min_n:
            continue
        rp, ry = rankdata(pred[s:e][o]), rankdata(y[s:e][o])
        out.append(np.corrcoef(rp, ry)[0, 1])
    return np.asarray(out)


def moving_block_bootstrap(x: np.ndarray, block: int, n_boot: int = 10_000, seed: int = 0) -> np.ndarray:
    """移动块自助法的均值分布（与 T-028 eval_event_aug.boot 逐字相同的抽样方式）."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    T = len(x)
    k = int(np.ceil(T / block))
    idx = (rng.integers(0, T - block + 1, size=(n_boot, k))[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :T]
    return x[idx].mean(axis=1)


def two_sided_ci(boots: np.ndarray, level: float) -> list[float]:
    a = (1 - level) / 2 * 100
    return [float(np.percentile(boots, a)), float(np.percentile(boots, 100 - a))]


def pinball(y: np.ndarray, q_pred: np.ndarray, q: float) -> float:
    d = y - q_pred
    return float(np.nanmean(np.maximum(q * d, (q - 1) * d)))
