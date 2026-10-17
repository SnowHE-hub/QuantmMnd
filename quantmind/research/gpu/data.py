"""按交易日集合读取训练矩阵（data/gpu_v1/model/v1，由 T-030 build_model_ready.py 生成）.

返回的行按 (as_of, ticker) 排序；标签在读取时做当日 gauss-rank（只用当日标签非缺失的面板行）。
市场级特征按行展开时只做拼接，标准化由调用方用该折训练日的统计量完成（TrainZScore）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quantmind.research.gpu.transforms import gauss_rank_by_day

LABEL_COL = {"20d": "y20", "63d": "y63"}
_LOOKAHEAD_MARKERS = ("canary", "lookahead", "fwd", "forward_return")


def assert_official_feature_names(names: list[str]) -> None:
    """正式特征名不得带未授权的前视列。诊断金丝雀只能存在于独立副本，不能进这个清单。"""
    bad = [n for n in names if any(m in n.lower() for m in _LOOKAHEAD_MARKERS)]
    if bad:
        raise ValueError(f"unauthorized lookahead column: {bad}")


@dataclass
class Block:
    X: np.ndarray                 # float16 [n, F_stock]，当日 gauss-rank，NaN = 缺失
    meta: pd.DataFrame            # as_of, ticker, y20, y63, top1500, sw_l1, seq_idx, day（0..n_days-1，本块内）
    days: pd.DatetimeIndex        # 本块的交易日（升序）

    def label_rank(self, horizon: str) -> np.ndarray:
        """当日 gauss-rank 标签；原始标签缺失 → NaN。"""
        y = self.meta[LABEL_COL[horizon]].to_numpy(dtype=np.float64)
        return gauss_rank_by_day(y, self.meta["day"].to_numpy())

    def subset(self, mask: np.ndarray) -> Block:
        meta = self.meta.loc[mask].reset_index(drop=True)
        days = pd.DatetimeIndex(sorted(meta["as_of"].unique()))
        meta["day"] = days.get_indexer(meta["as_of"])
        return Block(self.X[mask], meta, days)


class ModelData:
    def __init__(self, root: Path):
        self.dir = Path(root) / "data/gpu_v1/model/v1"
        self.features = json.loads((self.dir / "features.json").read_text())
        self.stock_features: list[str] = self.features["stock_features"]
        self.market_features: list[str] = self.features["market_features"]
        assert_official_feature_names(self.stock_features + self.market_features)
        self.market = pd.read_parquet(self.dir / "market.parquet")[self.market_features]
        self.months = sorted(p.name[:7] for p in self.dir.glob("*.meta.parquet"))

    def group_columns(self, groups: list[str]) -> list[int]:
        names = [c for g in groups for c in self.features["groups"][g]]
        pos = {c: i for i, c in enumerate(self.stock_features)}
        return [pos[c] for c in names]

    def load(self, days, columns: list[int] | None = None) -> Block:
        days = pd.DatetimeIndex(sorted(pd.to_datetime(list(days))))
        want = set(days)
        sels, metas = [], []
        for m in sorted({d.strftime("%Y-%m") for d in days}):
            meta = pd.read_parquet(self.dir / f"{m}.meta.parquet")
            meta["as_of"] = pd.to_datetime(meta["as_of"])
            keep = meta["as_of"].isin(want).to_numpy()
            if not keep.any():
                continue
            sels.append((m, np.flatnonzero(keep)))
            metas.append(meta.loc[keep])
        meta = pd.concat(metas, ignore_index=True)
        n_cols = len(self.stock_features) if columns is None else len(columns)
        X = np.empty((len(meta), n_cols), dtype=np.float16)     # 预分配：避免逐月数组 + 拼接的双份峰值
        o = 0
        for m, sel in sels:
            Xm = np.load(self.dir / f"{m}.X.npy", mmap_mode="r")
            X[o:o + len(sel)] = Xm[sel] if columns is None else Xm[sel][:, columns]
            o += len(sel)
        got = pd.DatetimeIndex(sorted(meta["as_of"].unique()))
        meta["day"] = got.get_indexer(meta["as_of"])
        return Block(X, meta, got)

    def market_rows(self, block: Block) -> np.ndarray:
        """市场级特征按行展开（float32，未标准化）。"""
        return self.market.reindex(block.meta["as_of"]).to_numpy(dtype=np.float32)
