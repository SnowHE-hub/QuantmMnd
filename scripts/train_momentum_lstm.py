#!/usr/bin/env python3
"""训练 MomentumAgent 使用的 LSTM v3（PyTorch，GPU 可选）."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from quantmind.core.logger import get_logger, setup_logger
from quantmind.models.momentum_lstm import (
    LSTMSequenceDataset,
    MomentumLSTM,
    build_lstm_arrays,
)

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--price-panel", type=Path, default=_ROOT / "data/prices/csi300_daily_adj_close.parquet")
    p.add_argument("--ohlcv", type=Path, default=_ROOT / "data/prices/csi300_daily_ohlcv.parquet")
    p.add_argument("--label-horizon", type=int, default=5)
    p.add_argument("--seq-len", type=int, default=60)
    p.add_argument("--epochs", type=int, default=None, help="默认 GPU=200 / CPU=50")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--early-stopping", type=int, default=10)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--output", type=Path, default=_ROOT / "models/agents/momentum_lstm_v3.pt")
    p.add_argument("--metrics-json", type=Path, default=_ROOT / "reports/model_training/momentum_lstm_metrics.json")
    return p.parse_args()


def main() -> int:
    setup_logger()
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = args.epochs if args.epochs is not None else (200 if device.type == "cuda" else 50)
    log.info("device={} epochs={}", device, epochs)

    px = pd_read_panel(args.price_panel)
    X_tr, y_tr, X_va, y_va, X_te, y_te = build_lstm_arrays(
        px,
        args.ohlcv,
        label_horizon=args.label_horizon,
        seq_len=args.seq_len,
        train_end="2022-12-31",
        val_end="2023-12-31",
    )
    log.info(
        "samples train={} val={} test={}",
        len(X_tr),
        len(X_va),
        len(X_te),
    )
    if len(X_tr) < 1000 or len(X_va) < 100:
        log.error("样本过少，请检查价格/OHLCV 面板")
        return 1

    train_ds = LSTMSequenceDataset(X_tr, y_tr)
    val_ds = LSTMSequenceDataset(X_va, y_va)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = MomentumLSTM(
        input_size=5,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=1,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss()

    best_val = float("inf")
    best_state = None
    stall = 0
    epochs_trained = 0

    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).unsqueeze(1)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * len(xb)
            tr_correct += int(((pred.squeeze() > 0.5).float() == yb.squeeze()).sum())
            tr_n += len(xb)
        tr_loss /= max(tr_n, 1)
        train_acc = tr_correct / max(tr_n, 1)

        model.eval()
        va_loss = 0.0
        va_correct = 0
        va_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device).unsqueeze(1)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                va_loss += float(loss.item()) * len(xb)
                va_correct += int(((pred.squeeze() > 0.5).float() == yb.squeeze()).sum())
                va_n += len(xb)
        va_loss /= max(va_n, 1)
        val_acc = va_correct / max(va_n, 1)

        epochs_trained = ep + 1
        if ep % max(1, epochs // 20) == 0 or ep == epochs - 1:
            log.info(
                "epoch {:>3} train_loss={:.4f} train_acc={:.4f} val_loss={:.4f} val_acc={:.4f}",
                ep + 1,
                tr_loss,
                train_acc,
                va_loss,
                val_acc,
            )

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= args.early_stopping:
                log.info("early stopping at epoch {}", ep + 1)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    bundle = {
        "state_dict": best_state or model.state_dict(),
        "config": {
            "input_size": 5,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "seq_len": args.seq_len,
            "label_horizon": args.label_horizon,
            "prob_threshold": 0.5,
        },
        "feature_layout": ["ret", "vol_ratio", "ma_ratio", "high_low", "gap_ratio"],
    }

    # final metrics on best weights + 验证集概率阈值扫描（缓解类不均衡导致的 0.5 阈值失真）
    model.eval()

    def _acc(loader):
        correct = 0
        n = 0
        tot_loss = 0.0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device).unsqueeze(1)
                pred = model(xb)
                tot_loss += float(loss_fn(pred, yb).item()) * len(xb)
                correct += int(((pred.squeeze() > 0.5).float() == yb.squeeze()).sum())
                n += len(xb)
        return tot_loss / max(n, 1), correct / max(n, 1)

    def _collect_probs(loader):
        probs_l: list[np.ndarray] = []
        ys_l: list[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                pr = model(xb).squeeze(-1).detach().cpu().numpy()
                probs_l.append(pr.astype(np.float64))
                ys_l.append(yb.numpy().astype(np.float64).ravel())
        return np.concatenate(probs_l), np.concatenate(ys_l)

    def _scan_threshold(probs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
        best_acc = -1.0
        best_t = 0.5
        for t in np.linspace(0.25, 0.75, 51):
            pred = (probs >= t).astype(np.float64)
            acc = float((pred == ys).mean())
            if acc > best_acc:
                best_acc = acc
                best_t = float(t)
        return best_t, best_acc

    tr_loader_full = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    final_train_loss, final_train_acc = _acc(tr_loader_full)
    final_val_loss, final_val_acc = _acc(val_loader)

    v_probs, v_y = _collect_probs(val_loader)
    t_star, val_acc_tuned = _scan_threshold(v_probs, v_y)
    bundle["config"]["prob_threshold"] = float(t_star)

    torch.save(bundle, args.output)
    log.info("saved {} (prob_threshold={:.4f})", args.output.resolve(), t_star)

    metrics = {
        "train_acc": round(float(final_train_acc), 6),
        "val_acc": round(float(val_acc_tuned), 6),
        "val_acc_at_0.5": round(float(final_val_acc), 6),
        "val_prob_threshold": round(float(t_star), 6),
        "train_loss": round(float(final_train_loss), 6),
        "val_loss": round(float(final_val_loss), 6),
        "epochs_trained": int(epochs_trained),
        "device": str(device),
        "model_path": str(args.output.resolve()),
    }
    args.metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("metrics → {}", args.metrics_json)
    return 0


def pd_read_panel(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
        date_col = "trade_date" if "trade_date" in df.columns else "date"
        code_col = "ts_code" if "ts_code" in df.columns else "ticker"
        val_col = "adj_close" if "adj_close" in df.columns else "close"
        df = df.pivot(index=date_col, columns=code_col, values=val_col)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


if __name__ == "__main__":
    raise SystemExit(main())
