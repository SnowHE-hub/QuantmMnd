"""按交易日分组的 batch：同一天的全部股票总在同一个 batch 里."""
from __future__ import annotations

import numpy as np

from quantmind.research.gpu.transforms import day_bounds


def day_batches(day_codes: np.ndarray, days_per_batch: int, rng: np.random.Generator | None = None) -> list[np.ndarray]:
    """行已按日期排序。返回行号数组的列表；rng 给出时打乱日期顺序（不拆分任何一天）。"""
    b = day_bounds(day_codes)
    order = np.arange(len(b) - 1)
    if rng is not None:
        rng.shuffle(order)
    out = []
    for i in range(0, len(order), days_per_batch):
        sel = order[i:i + days_per_batch]
        out.append(np.concatenate([np.arange(b[j], b[j + 1]) for j in sel]))
    return out
