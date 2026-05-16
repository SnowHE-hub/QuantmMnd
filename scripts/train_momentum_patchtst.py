#!/usr/bin/env python3
"""训练 MomentumAgent PatchTST v4（LSTM v3 → Transformer 升级）.

数据源：data/raw/alpha_prices_panel.parquet（长表，含完整 OHLCV）

架构要点：
  - PatchTST：64日序列 → 7个16日Patch → TransformerEncoder(3层) → 二分类
  - 5 特征：日收益率 / 量比 / 价格/20日均线 / 振幅 / 跳空比
  - 标签：5日后涨跌（binary）
  - 数据切分：train≤2022-12-31，val=2023，test=2024+
  - 每 --stride-days 天采样一次（减少 2M+ 样本 → ~440K）

运行：
  python scripts/train_momentum_patchtst.py \\
    --panel data/raw/alpha_prices_panel.parquet \\
    --out   models/agents/momentum_patchtst_v4.pt \\
    --epochs 20 --batch 512 --stride-days 5
"""
from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent


def set_seed(s: int = 42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ── 特征提取 ───────────────────────────────────────────────────────────────────

def _build_features(df_ticker: "pd.DataFrame") -> "np.ndarray | None":
    """
    输入：单只股票按 trade_date 排好序的 DataFrame
    输出：(T, 5) float32 特征矩阵
      [0] ret         : 日涨跌幅
      [1] vol_ratio   : 当日量 / 20日均量
      [2] ma_ratio    : 收盘 / 20日均价
      [3] high_low    : (high-low) / close  振幅
      [4] gap_ratio   : (open / pre_close) - 1  跳空比
    """
    import pandas as pd

    d = df_ticker.copy()
    c = d["adj_close"].astype(float)

    ret = c.pct_change(fill_method=None).fillna(0.0).values

    ma20 = c.rolling(20, min_periods=10).mean()
    ma_ratio = (c / ma20.replace(0, np.nan)).fillna(1.0).values

    vol = pd.to_numeric(d["vol"], errors="coerce").fillna(0.0)
    vol_ma = vol.rolling(20, min_periods=10).mean().fillna(1.0)
    vol_ratio = (vol / vol_ma.replace(0, 1.0)).fillna(1.0).values

    hi  = pd.to_numeric(d["high"], errors="coerce").fillna(0.0).values
    lo  = pd.to_numeric(d["low"],  errors="coerce").fillna(0.0).values
    cl  = pd.to_numeric(d["close"], errors="coerce").fillna(1.0).values
    op  = pd.to_numeric(d["open"], errors="coerce").fillna(0.0).values
    pre = pd.to_numeric(d["pre_close"], errors="coerce").fillna(0.0).values

    hl_denom = np.where(np.abs(cl) > 1e-8, cl, 1.0)
    high_low = (hi - lo) / hl_denom
    high_low = np.nan_to_num(high_low, nan=0.0, posinf=0.0)

    gap_denom = np.where(pre > 1e-8, pre, 1.0)
    gap = op / gap_denom - 1.0
    gap = np.nan_to_num(gap, nan=0.0, posinf=0.0, neginf=0.0)

    feats = np.column_stack([ret, vol_ratio, ma_ratio, high_low, gap]).astype(np.float32)
    return feats


def _zscore_window(w: "np.ndarray") -> "np.ndarray":
    """Per-window z-score（沿时间轴每个特征独立）."""
    m = w.mean(axis=0, keepdims=True)
    s = w.std(axis=0, keepdims=True)
    s = np.where(s < 1e-8, 1.0, s)
    return (w - m) / s


def build_samples(
    df: "pd.DataFrame",
    *,
    seq_len: int = 64,
    label_horizon: int = 5,
    train_end: str = "2022-12-31",
    val_end: str = "2023-12-31",
    stride_days: int = 5,
    min_days: int = 300,
) -> tuple["np.ndarray", ...]:
    """
    从长表构建 train/val/test 样本集。
    stride_days：每隔几日采样一个窗口（减少样本量，默认每周1个）。
    """
    import pandas as pd

    te_ts = pd.Timestamp(train_end)
    ve_ts = pd.Timestamp(val_end)

    tr_x, tr_y, va_x, va_y, te_x, te_y = [], [], [], [], [], []

    tickers = df["ts_code"].unique()
    logger.info(f"处理 {len(tickers)} 只股票…")

    for i, tkr in enumerate(tickers):
        sub = df[df["ts_code"] == tkr].sort_values("trade_date")
        sub = sub.drop_duplicates("trade_date")

        if len(sub) < min_days:
            continue

        feats = _build_features(sub)
        if feats is None or feats.shape[0] < seq_len + label_horizon:
            continue

        prices = sub["adj_close"].astype(float).values
        dates  = sub["trade_date"].values  # numpy datetime64

        T = len(dates)
        # 从 seq_len-1 开始，每 stride_days 采一个窗口
        for t in range(seq_len - 1, T - label_horizon, stride_days):
            if np.isnan(prices[t]) or np.isnan(prices[t + label_horizon]):
                continue
            win = feats[t - seq_len + 1 : t + 1]
            if np.isnan(win).any():
                continue
            win_z = _zscore_window(win)
            y = 1.0 if prices[t + label_horizon] > prices[t] else 0.0
            day = pd.Timestamp(dates[t])

            if day <= te_ts:
                tr_x.append(win_z)
                tr_y.append(y)
            elif day <= ve_ts:
                va_x.append(win_z)
                va_y.append(y)
            else:
                te_x.append(win_z)
                te_y.append(y)

        if (i + 1) % 200 == 0:
            logger.info(
                f"  进度 {i+1}/{len(tickers)}: "
                f"train={len(tr_x)} val={len(va_x)} test={len(te_x)}"
            )

    def _arr(xs, ys):
        if not xs:
            return np.zeros((0, seq_len, 5), np.float32), np.zeros((0,), np.float32)
        return np.stack(xs).astype(np.float32), np.array(ys, np.float32)

    return (*_arr(tr_x, tr_y), *_arr(va_x, va_y), *_arr(te_x, te_y))


# ── 训练主流程 ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel",        default="data/raw/alpha_prices_panel.parquet")
    parser.add_argument("--out",          default="models/agents/momentum_patchtst_v4.pt")
    parser.add_argument("--seq-len",      type=int,   default=64)
    parser.add_argument("--patch-len",    type=int,   default=16)
    parser.add_argument("--stride",       type=int,   default=8)
    parser.add_argument("--d-model",      type=int,   default=64,
                        help="Transformer 隐层宽度（减小以减少过拟合）")
    parser.add_argument("--n-heads",      type=int,   default=4)
    parser.add_argument("--n-layers",     type=int,   default=2,
                        help="Encoder 层数（2层 vs 3层，减少过拟合）")
    parser.add_argument("--dropout",      type=float, default=0.2,
                        help="Dropout（增大以减少过拟合）")
    parser.add_argument("--epochs",       type=int,   default=15)
    parser.add_argument("--batch",        type=int,   default=512)
    parser.add_argument("--lr",           type=float, default=1e-4,
                        help="较小学习率更稳定")
    parser.add_argument("--label-horizon",type=int,   default=5)
    parser.add_argument("--stride-days",  type=int,   default=5,
                        help="每隔几日采样一个训练窗口（减少样本量）")
    parser.add_argument("--train-end",    default="2022-12-31")
    parser.add_argument("--val-end",      default="2023-12-31")
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    import pandas as pd
    from quantmind.models.momentum_patchtst import PatchTST, PatchTSTDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    panel_path = ROOT / args.panel
    out_path   = ROOT / args.out

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    logger.info(f"加载价格面板: {panel_path}")
    df = pd.read_parquet(panel_path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    logger.info(f"  {len(df):,} 行 | {df['ts_code'].nunique()} 只 | "
                f"{df['trade_date'].min().date()} → {df['trade_date'].max().date()}")

    # ── 构建样本 ──────────────────────────────────────────────────────────────
    logger.info(f"构建样本（stride_days={args.stride_days}，约需 3-8 分钟）…")
    X_tr, y_tr, X_va, y_va, X_te, y_te = build_samples(
        df,
        seq_len=args.seq_len,
        label_horizon=args.label_horizon,
        train_end=args.train_end,
        val_end=args.val_end,
        stride_days=args.stride_days,
    )
    logger.info(
        f"样本数: train={len(X_tr):,} val={len(X_va):,} test={len(X_te):,}"
    )
    logger.info(
        f"标签分布: train 涨={y_tr.mean():.3f}  "
        f"val 涨={y_va.mean():.3f}  "
        f"test 涨={y_te.mean():.3f}"
    )

    if len(X_tr) == 0:
        raise SystemExit("训练集为空，检查数据路径和过滤条件")

    # ── 数据加载器 ────────────────────────────────────────────────────────────
    tr_ds = PatchTSTDataset(X_tr, y_tr)
    va_ds = PatchTSTDataset(X_va, y_va)
    n_workers = min(4, 4)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                           num_workers=n_workers, pin_memory=(device.type == "cuda"))
    va_loader = DataLoader(va_ds, batch_size=args.batch, shuffle=False,
                           num_workers=n_workers, pin_memory=(device.type == "cuda"))

    # ── 模型 ──────────────────────────────────────────────────────────────────
    model = PatchTST(
        n_feats=5,
        seq_len=args.seq_len,
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"PatchTST 参数量: {n_params:,}")

    # Label smoothing: 将 0/1 → 0.05/0.95，减少过拟合
    label_smooth = 0.05

    def smooth_bce(prob: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_s = y * (1 - 2 * label_smooth) + label_smooth
        return nn.functional.binary_cross_entropy(prob, y_s)

    criterion  = smooth_bce
    optimizer  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )

    best_val_acc = 0.0
    best_state   = None

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            prob = model(xb).squeeze(1)
            loss = criterion(prob, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(tr_ds)

        model.eval()
        correct = total = 0
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                prob = model(xb).squeeze(1)
                va_loss += criterion(prob, yb).item() * len(xb)
                correct += ((prob > 0.5).float() == yb).sum().item()
                total   += len(yb)
        va_loss /= len(va_ds)
        val_acc = correct / total if total > 0 else 0.5

        scheduler.step()

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        logger.info(
            f"Epoch {epoch:02d}/{args.epochs}  "
            f"tr_loss={tr_loss:.4f}  va_loss={va_loss:.4f}  "
            f"val_acc={val_acc:.4f}{'  ★' if improved else ''}"
        )

    # ── 测试集评估 ─────────────────────────────────────────────────────────────
    test_acc = float("nan")
    if len(X_te) > 0 and best_state is not None:
        model.load_state_dict(best_state)
        model.eval()
        te_ds = PatchTSTDataset(X_te, y_te)
        te_loader = DataLoader(te_ds, batch_size=args.batch, shuffle=False, num_workers=n_workers)
        correct = total = 0
        with torch.no_grad():
            for xb, yb in te_loader:
                xb, yb = xb.to(device), yb.to(device)
                prob = model(xb).squeeze(1)
                correct += ((prob > 0.5).float() == yb).sum().item()
                total   += len(yb)
        test_acc = correct / total if total > 0 else float("nan")
        logger.info(f"[TEST] accuracy={test_acc:.4f}  (LSTM v3 基线: 0.5428)")
    else:
        logger.warning("[TEST] 测试集为空，跳过")

    # ── 保存 ──────────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "state_dict": best_state,
        "config": {
            "n_feats":        5,
            "seq_len":        args.seq_len,
            "patch_len":      args.patch_len,
            "stride":         args.stride,
            "d_model":        args.d_model,
            "n_heads":        args.n_heads,
            "n_layers":       args.n_layers,
            "dropout":        args.dropout,
            "label_horizon":  args.label_horizon,
            "prob_threshold": 0.5,
        },
        "version":    "patchtst_v4",
        "kind":       "momentum_patchtst_v4",
        "created_at": datetime.now().isoformat(),
        "metrics": {
            "best_val_acc": round(best_val_acc, 5),
            "test_acc":     round(test_acc, 5) if np.isfinite(test_acc) else None,
            "n_train":      int(len(X_tr)),
            "n_val":        int(len(X_va)),
            "n_test":       int(len(X_te)),
            "train_label_mean": round(float(y_tr.mean()), 4),
            "lstm_v3_baseline_val_acc": 0.5428,
        },
        "data_config": {
            "train_end":     args.train_end,
            "val_end":       args.val_end,
            "stride_days":   args.stride_days,
            "label_horizon": args.label_horizon,
        },
    }
    torch.save(bundle, out_path)
    logger.info(
        f"保存: {out_path} | "
        f"best_val_acc={best_val_acc:.4f}  test_acc={test_acc:.4f}"
    )


if __name__ == "__main__":
    main()
