"""MomentumAgent LSTM：60×5 序列输入，二分类涨跌概率."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


class MomentumLSTM(nn.Module):
    """输入 (batch, seq_len=60, feat=5)，输出 sigmoid 概率."""

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        return self.sigmoid(self.fc(out[:, -1, :]))


def _zscore_window(x: np.ndarray) -> np.ndarray:
    """Per-sample 标准化：沿时间轴 (axis=0) 对每个特征 z-score."""
    m = x.mean(axis=0, keepdims=True)
    s = x.std(axis=0, keepdims=True)
    s = np.where(s < 1e-8, 1.0, s)
    return (x - m) / s


def _load_ohlcv_pivots(
    ohlcv_path: str | pd.PathLike,
    tickers: list[str],
    cal: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    """返回每个 ticker 对齐 cal 的 OHLCV DataFrame（index=日期）."""
    df = pd.read_parquet(ohlcv_path)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    code_col = "ts_code" if "ts_code" in df.columns else "ticker"
    fields = ["open", "high", "low", "close", "volume", "pre_close"]
    out: dict[str, pd.DataFrame] = {}
    for col in fields:
        if col not in df.columns:
            df[col] = np.nan
    for tkr in tickers:
        sub = df[df[code_col] == tkr]
        if sub.empty:
            continue
        sub = sub.set_index("trade_date").sort_index()
        w = pd.DataFrame(index=cal)
        for f in fields:
            w[f] = sub[f].reindex(cal)
        out[str(tkr)] = w
    return out


class LSTMSequenceDataset(Dataset):
    def __init__(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
    ) -> None:
        """
        xs: (N, seq_len, n_feat)
        ys: (N,) 0/1
        """
        self.xs = xs.astype(np.float32)
        self.ys = ys.astype(np.float32)

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.xs[i])
        y = torch.tensor(self.ys[i], dtype=torch.float32)
        return x, y


def build_feature_matrix_for_ticker(
    close: pd.Series,
    ohlc: pd.DataFrame | None,
    *,
    cal: pd.DatetimeIndex,
) -> np.ndarray | None:
    """对齐 cal 计算每日 5 特征列: ret, vol_ratio, ma_ratio, high_low, gap."""
    c = close.reindex(cal).astype(float)
    ret = c.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    ma20 = c.rolling(20, min_periods=10).mean()
    ma_ratio = (c / ma20).replace([np.inf, -np.inf], np.nan).fillna(1.0).values

    if ohlc is None or ohlc.empty:
        return None

    vol = ohlc["volume"].astype(float).values
    vol_ma = pd.Series(vol, index=cal).rolling(20, min_periods=10).mean().values
    vol_ratio = np.where(vol_ma > 1e-12, vol / vol_ma, 1.0)

    hi = ohlc["high"].astype(float).values
    lo = ohlc["low"].astype(float).values
    cl = ohlc["close"].astype(float).values
    op = ohlc["open"].astype(float).values
    pre = ohlc["pre_close"].astype(float).values
    prev_close = np.roll(cl, 1)
    prev_close[0] = np.nan
    denom_gap = np.where(pre > 1e-12, pre, prev_close)
    gap_ratio = np.where(denom_gap > 1e-12, op / denom_gap - 1.0, 0.0)
    gap_ratio = np.nan_to_num(gap_ratio, nan=0.0)

    hl_denom = np.where(np.abs(cl) > 1e-12, cl, np.nan)
    high_low = (hi - lo) / hl_denom
    high_low = np.nan_to_num(high_low, nan=0.0, posinf=0.0, neginf=0.0)

    feats = np.column_stack([ret, vol_ratio, ma_ratio, high_low, gap_ratio]).astype(np.float32)
    return feats


def build_lstm_arrays(
    price_panel: pd.DataFrame,
    ohlcv_path: str | pd.PathLike,
    *,
    label_horizon: int = 5,
    seq_len: int = 60,
    train_end: str = "2022-12-31",
    val_end: str = "2023-12-31",
    tickers: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    构造 train/val/test 的 X,y（已按样本 z-score）。
    返回 (X_train,y_train,X_val,y_val,X_test,y_test)
    """
    px = price_panel.copy()
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    if px.index.has_duplicates:
        px = px[~px.index.duplicated(keep="last")]
    cal = px.index
    tickers = tickers or [str(c) for c in px.columns]

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

    def _stack(ax: list, ay: list) -> tuple[np.ndarray, np.ndarray]:
        if not ax:
            return np.zeros((0, seq_len, 5), np.float32), np.zeros((0,), np.float32)
        return np.stack(ax, axis=0), np.array(ay, dtype=np.float32)

    return (
        *_stack(tr_x, tr_y),
        *_stack(va_x, va_y),
        *_stack(te_x, te_y),
    )
