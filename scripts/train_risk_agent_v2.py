#!/usr/bin/env python
"""训练 RiskAgent GARCH(1,1) v2：逐票估计波动并序列化参数 / EWMA 兜底."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _pick_price_frame() -> tuple[pd.DataFrame, str]:
    candidates = [
        _ROOT / "data" / "raw" / "daily_prices_panel.parquet",
        _ROOT / "data" / "prices" / "csi300_daily_adj_close.parquet",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_parquet(p)
            # 宽表：index 为日期，列为 ticker
            if isinstance(df.index, pd.DatetimeIndex) or "trade_date" not in df.columns:
                return df, str(p)
            # 长表尝试透视
            if {"ticker", "close"}.issubset(df.columns) and "trade_date" in df.columns:
                wide = df.pivot_table(index="trade_date", columns="ticker", values="close")
                wide.index = pd.to_datetime(wide.index)
                wide.sort_index(inplace=True)
                return wide, str(p)
    raise FileNotFoundError(f"未找到价格 parquet，尝试过: {[str(c) for c in candidates]}")


def _annualize_from_daily_variance(var_daily: float) -> float:
    return float(np.sqrt(max(var_daily, 1e-16)) * np.sqrt(252.0))


def _fit_one_series(prices: pd.Series) -> dict:
    """返回单票估计结果；失败则 EWMA 兜底."""
    s = prices.dropna().astype(float)
    if len(s) < 252:
        raise ValueError("too_short")
    ret = np.log(s / s.shift(1)).dropna()
    ret_pct = ret.values * 100.0

    # ── GARCH(1,1) ──────────────────────────────────────────────────────────
    try:
        from arch import arch_model

        am = arch_model(ret_pct, mean="Constant", vol="Garch", p=1, q=1)
        res = am.fit(disp="off")
        params = res.params
        omega = float(params.get("omega", np.nan))
        alpha = float(params.get("alpha[1]", params.get("alpha1", np.nan)))
        beta = float(params.get("beta[1]", params.get("beta1", np.nan)))

        fcst = res.forecast(horizon=21, reindex=False)
        var_mat = getattr(fcst, "variance", None)
        if var_mat is None:
            raise RuntimeError("no variance forecast")
        var21 = float(np.asarray(var_mat.iloc[-1]).ravel()[-1])

        # ret_pct 方差单位：(百分之一)^2；换回小数收益近似：/100
        ann_vol_21 = float(np.sqrt(max(var21, 1e-12)) / 100.0 * np.sqrt(252.0))

        persistence = alpha + beta
        if persistence >= 0.999:
            lr_var_daily = np.var(ret.values)
        else:
            lr_var_daily = (omega / max(1e-12, (1.0 - persistence))) / 10000.0
        long_run_vol = _annualize_from_daily_variance(lr_var_daily)

        last_price = float(s.iloc[-1])
        return {
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "long_run_vol": float(long_run_vol),
            "last_price": last_price,
            "ann_vol_forecast_21d": ann_vol_21,
            "fallback": False,
        }
    except Exception as exc:
        logger.debug(f"[risk_v2] GARCH 失败，EWMA 兜底: {exc}")
        lam = 0.94
        r = ret.values.astype(float)
        r2 = r * r
        var = np.zeros_like(r2)
        var[0] = np.mean(r2[: min(20, len(r2))])
        for i in range(1, len(r2)):
            var[i] = lam * var[i - 1] + (1.0 - lam) * r2[i - 1]
        ann_vol = _annualize_from_daily_variance(float(var[-1]))
        last_price = float(s.iloc[-1])
        return {
            "omega": float("nan"),
            "alpha": float("nan"),
            "beta": float("nan"),
            "long_run_vol": ann_vol,
            "last_price": last_price,
            "ann_vol_forecast_21d": ann_vol,
            "fallback": True,
        }


def train_risk_garch_v2(output_model: Path, metrics_path: Path, max_tickers: int | None) -> None:
    df, src = _pick_price_frame()
    logger.info(f"[risk_v2] 价格来源: {src}, shape={df.shape}")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()

    vol_by_ticker: dict[str, dict] = {}
    n_failed = 0
    fallback_count = 0
    cols = list(df.columns)
    if max_tickers is not None:
        cols = cols[:max_tickers]

    for col in cols:
        ticker = str(col)
        try:
            vol_by_ticker[ticker] = _fit_one_series(df[col])
            if vol_by_ticker[ticker].get("fallback"):
                fallback_count += 1
        except Exception as e:
            n_failed += 1
            logger.debug(f"[risk_v2] skip {ticker}: {e}")

    bundle = {"kind": "risk_garch_v2", "vol_by_ticker": vol_by_ticker}

    output_model.parent.mkdir(parents=True, exist_ok=True)
    with open(output_model, "wb") as f:
        pickle.dump(bundle, f)

    vals = [v["long_run_vol"] for v in vol_by_ticker.values() if np.isfinite(v["long_run_vol"])]
    metrics = {
        "n_fitted": int(len(vol_by_ticker)),
        "n_failed": int(n_failed),
        "mean_long_run_vol": float(np.mean(vals)) if vals else 0.0,
        "fallback_count": int(fallback_count),
        "price_source": src,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[risk_v2] 完成: {metrics}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train RiskAgent garch_v2 bundle")
    ap.add_argument(
        "--output-model",
        type=Path,
        default=_ROOT / "models" / "agents" / "risk_garch_v2.pkl",
    )
    ap.add_argument(
        "--metrics-json",
        type=Path,
        default=_ROOT / "reports" / "model_training" / "risk_garch_v2_metrics.json",
    )
    ap.add_argument("--max-tickers", type=int, default=None, help="调试：仅拟合前 N 只股票")
    args = ap.parse_args()
    train_risk_garch_v2(args.output_model, args.metrics_json, args.max_tickers)


if __name__ == "__main__":
    main()
