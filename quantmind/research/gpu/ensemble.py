"""种子平均与集成：各分数先做当日截面 gauss-rank，再等权平均（尺度无关）."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantmind.research.gpu.transforms import gauss_rank_by_day


def rank_average(scores: list[np.ndarray], day: np.ndarray) -> np.ndarray:
    """行按 day 升序；每个分数当日 gauss-rank 后取均值（某成员缺失的行按其余成员平均）。"""
    ranks = np.stack([gauss_rank_by_day(np.asarray(s, dtype=np.float64), day) for s in scores])
    with np.errstate(invalid="ignore"):
        return np.nanmean(ranks, axis=0)


def rank_average_frames(frames: dict[str, pd.Series]) -> pd.Series:
    """frames：成员名 → index=(as_of, ticker) 的分数。只在所有成员都有分数的行上平均。"""
    df = pd.DataFrame(frames).dropna().sort_index()
    day = pd.factorize(df.index.get_level_values("as_of"), sort=True)[0]
    return pd.Series(rank_average([df[c].to_numpy() for c in df.columns], day), index=df.index, name="score")
