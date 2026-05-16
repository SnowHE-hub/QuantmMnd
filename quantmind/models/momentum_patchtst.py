"""PatchTST — Patch Time-Series Transformer for MomentumAgent v4.

论文: Nie et al. 2022 "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
本实现针对 A 股二分类（涨跌）适配，channel-mixing 模式（5 特征联合处理）。

序列设计：
  seq_len=64  patch_len=16  stride=8 → n_patches=7
  d_model=128 n_heads=4  n_layers=3  dropout=0.1
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ── Patch 工具 ────────────────────────────────────────────────────────────────

def make_patches(x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    """
    x: (B, T, C)  →  patches: (B, n_patches, patch_len * C)
    最后不足 patch_len 的余量用零补齐。
    """
    B, T, C = x.shape
    n_patches = math.ceil((T - patch_len) / stride) + 1
    # pad if needed
    pad_len = (n_patches - 1) * stride + patch_len - T
    if pad_len > 0:
        x = torch.cat([x, x.new_zeros(B, pad_len, C)], dim=1)
    patches = []
    for i in range(n_patches):
        p = x[:, i * stride: i * stride + patch_len, :]  # (B, patch_len, C)
        patches.append(p.reshape(B, 1, patch_len * C))
    return torch.cat(patches, dim=1)   # (B, n_patches, patch_len*C)


# ── PatchTST 模型 ─────────────────────────────────────────────────────────────

class PatchTST(nn.Module):
    """
    PatchTST binary classifier：输出 sigmoid 概率（上涨 = 1）。

    参数：
      n_feats    : 输入特征维数（5）
      seq_len    : 序列长度（64）
      patch_len  : 每个 patch 的时间步数（16）
      stride     : patch 步长（8）
      d_model    : Transformer 隐层宽度（128）
      n_heads    : 多头注意力头数（4）
      n_layers   : Encoder 层数（3）
      dropout    : Dropout 概率（0.1）
    """

    def __init__(
        self,
        n_feats: int = 5,
        seq_len: int = 64,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = math.ceil((seq_len - patch_len) / stride) + 1
        patch_dim = patch_len * n_feats

        # patch projection
        self.patch_embed = nn.Linear(patch_dim, d_model)

        # learnable positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN（更稳定）
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → prob: (B, 1)."""
        patches = make_patches(x, self.patch_len, self.stride)  # (B, n, patch_dim)
        z = self.patch_embed(patches) + self.pos_embed           # (B, n, d_model)
        z = self.encoder(z)                                      # (B, n, d_model)
        z = self.norm(z)
        z = z.mean(dim=1)                                        # (B, d_model)
        return self.head(z)                                      # (B, 1)


# ── Dataset ───────────────────────────────────────────────────────────────────

class PatchTSTDataset(Dataset):
    """Wraps (N, seq_len, n_feats) arrays 与 (N,) labels."""

    def __init__(self, xs: np.ndarray, ys: np.ndarray) -> None:
        self.xs = torch.from_numpy(xs.astype(np.float32))
        self.ys = torch.from_numpy(ys.astype(np.float32))

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, i: int):
        return self.xs[i], self.ys[i]


# ── 特征构建（沿用 momentum_lstm 的 5 特征）────────────────────────────────────

from quantmind.models.momentum_lstm import (  # noqa: E402
    _zscore_window,
    build_feature_matrix_for_ticker,
    _load_ohlcv_pivots,
)


def build_patchtst_arrays(
    price_panel: pd.DataFrame,
    ohlcv_path: str | None,
    *,
    label_horizon: int = 5,
    seq_len: int = 64,
    train_end: str = "2022-12-31",
    val_end: str = "2023-12-31",
    tickers: Optional[list[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    与 build_lstm_arrays 逻辑相同，但 seq_len 默认 64（PatchTST 设计值）。
    若 ohlcv_path 为 None 或文件缺失，则跳过该 ticker（不生成样本）。
    返回 (X_train,y_train,X_val,y_val,X_test,y_test)。
    """
    import os

    px = price_panel.copy()
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    if px.index.has_duplicates:
        px = px[~px.index.duplicated(keep="last")]
    cal = px.index
    tickers = tickers or [str(c) for c in px.columns]

    pivots: dict[str, pd.DataFrame] = {}
    if ohlcv_path and os.path.isfile(ohlcv_path):
        pivots = _load_ohlcv_pivots(ohlcv_path, tickers, cal)

    te = pd.Timestamp(train_end)
    ve = pd.Timestamp(val_end)

    tr_x, tr_y = [], []
    va_x, va_y = [], []
    te_x, te_y = [], []

    for tkr in tickers:
        if tkr not in px.columns:
            continue
        close = px[tkr].dropna()
        if len(close) < seq_len + label_horizon + 5:
            continue
        ohl = pivots.get(str(tkr))
        feats_full = build_feature_matrix_for_ticker(close, ohl, cal=cal)
        if feats_full is None:
            continue
        prices = px[tkr].reindex(cal).values.astype(np.float64)

        for t_idx in range(seq_len - 1, len(cal) - label_horizon):
            if np.isnan(prices[t_idx]) or np.isnan(prices[t_idx + label_horizon]):
                continue
            win = feats_full[t_idx - seq_len + 1 : t_idx + 1]
            if np.isnan(win).any():
                continue
            win_z = _zscore_window(win.astype(np.float64)).astype(np.float32)
            y = 1.0 if prices[t_idx + label_horizon] > prices[t_idx] else 0.0
            day = cal[t_idx]

            if day <= te:
                tr_x.append(win_z)
                tr_y.append(y)
            elif day <= ve:
                va_x.append(win_z)
                va_y.append(y)
            else:
                te_x.append(win_z)
                te_y.append(y)

    def _stack(ax, ay):
        if not ax:
            return np.zeros((0, seq_len, 5), np.float32), np.zeros((0,), np.float32)
        return np.stack(ax), np.array(ay, np.float32)

    return (*_stack(tr_x, tr_y), *_stack(va_x, va_y), *_stack(te_x, te_y))
