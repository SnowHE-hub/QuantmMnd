"""当日截面变换（只用当日可见股票，无跨日信息）与训练窗 z-score（只在该折训练数据上拟合）."""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata


def day_bounds(day_codes: np.ndarray) -> np.ndarray:
    """行已按日期排好序时，每个交易日的 [start, end) 边界（长度 n_days + 1）。"""
    if len(day_codes) and np.any(np.diff(day_codes) < 0):
        raise ValueError("rows must be sorted by day")
    change = np.flatnonzero(np.diff(day_codes)) + 1
    return np.concatenate([[0], change, [len(day_codes)]])


def gauss_rank_block(x: np.ndarray) -> np.ndarray:
    """单日截面：每列平均秩 → Φ⁻¹((r − 0.5)/n)；NaN 保留为 NaN，n = 该列非缺失数。"""
    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    out = np.full(x.shape, np.nan)
    for j in range(x.shape[1]):
        col = x[:, j]
        ok = np.isfinite(col)
        n = int(ok.sum())
        if n == 0:
            continue
        out[ok, j] = ndtri((rankdata(col[ok]) - 0.5) / n)
    return out[:, 0] if squeeze else out


def gauss_rank_by_day(x: np.ndarray, day_codes: np.ndarray) -> np.ndarray:
    b = day_bounds(day_codes)
    out = np.empty(x.shape, dtype=np.float64)
    for s, e in zip(b[:-1], b[1:], strict=True):
        out[s:e] = gauss_rank_block(x[s:e])
    return out


class TrainZScore:
    """市场级（当日截面上为常数）的特征：用该折训练窗的均值 / 标准差标准化，再截断到 ±5."""

    def fit(self, x: np.ndarray) -> TrainZScore:
        self.mu = np.nanmean(x, axis=0)
        sd = np.nanstd(x, axis=0)
        self.sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mu) / self.sd, -5, 5)
