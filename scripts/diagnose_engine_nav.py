#!/usr/bin/env python3
"""诊断 LGBM 引擎 NAV：打印首尾 NAV / 日收益序列波动是否过低."""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quantmind.backtest.engine import BacktestConfig


def load_ohlcv_for_engine(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).reset_index()
    rename_map = {"trade_date": "date", "ts_code": "ticker"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "volume" not in df.columns and "vol" in df.columns:
        df["volume"] = pd.to_numeric(df["vol"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["date", "ticker"]).sort_index()
    return df


def load_benchmark_from_index_parquet(path: Path, code: str = "000300.SH") -> pd.Series:
    df = pd.read_parquet(path).reset_index()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    sub = df.loc[df["ts_code"] == code].set_index("trade_date")["close"].sort_index()
    return sub


def load_csi300_universe_tickers() -> list[str]:
    univ = _ROOT / "data/snapshots/2024-12-31/universe.parquet"
    if not univ.is_file():
        return []
    u = pd.read_parquet(univ)
    col = "ticker" if "ticker" in u.columns else "ts_code"
    return u[col].astype(str).tolist()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--model", type=Path, default=_ROOT / "models/lgbm_v1_final.pkl")
    p.add_argument("--panel-dir", type=Path, default=_ROOT / "data/panel")
    p.add_argument("--price-path", type=Path, default=_ROOT / "data/prices/csi300_daily_ohlcv.parquet")
    p.add_argument("--rebalance-freq", choices=["M", "Q"], default="Q")
    args = p.parse_args()

    from quantmind.backtest import BacktestEngine  # noqa: E402
    from quantmind.backtest.lgbm_strategy import LGBMStrategy  # noqa: E402

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    prices_df = load_ohlcv_for_engine(args.price_path)
    bench_path = args.price_path.parent / "index_daily.parquet"
    benchmark_series = None
    if bench_path.is_file():
        benchmark_series = load_benchmark_from_index_parquet(bench_path)

    universe_lgb = load_csi300_universe_tickers()
    avail = set(prices_df.index.get_level_values("ticker"))
    universe_lgb = [t for t in universe_lgb if t in avail]

    adj_parquet = _ROOT / "data/prices/csi300_daily_adj_close.parquet"
    adj_existing = adj_parquet if adj_parquet.is_file() else None
    reb = "quarterly" if args.rebalance_freq == "Q" else "monthly"

    config = BacktestConfig()
    strategy = LGBMStrategy(
        model_path=args.model,
        panel_dir=args.panel_dir,
        top_n=50,
        long_n=10,
        rebalance=reb,
        price_path=args.price_path,
        adj_close_path=adj_existing,
        config=config,
    )
    engine = BacktestEngine(config=config, prices_df=prices_df, benchmark_df=benchmark_series)
    result = engine.run(strategy, start, end, universe_lgb)

    nav = result.nav_series.sort_index()
    r = nav.pct_change().dropna()

    print("=== NAV head (10) ===")
    print(nav.head(10).to_string())
    print("\n=== NAV tail (10) ===")
    print(nav.tail(10).to_string())

    sd = float(r.std(ddof=1)) if len(r) > 1 else float("nan")
    mu = float(r.mean()) if len(r) else float("nan")
    vol_ann = sd * math.sqrt(252.0) if sd == sd else float("nan")

    print("\n=== Daily returns stats ===")
    print(f"  mean_daily≈{mu:.6f}")
    print(f"  std_daily≈{sd:.6f}")
    print(f"  annual_vol_approx≈{vol_ann:.2%}")

    if sd == sd and sd < 0.002:
        print("\n[WARN] daily std < 0.002 → NAV may be frozen between rebalance days (inspect MTM).")

    m = result.metrics or {}
    print("\n=== PerformanceMetrics snapshot ===")
    print(f"  sharpe_ratio={m.get('sharpe_ratio')}")
    print(f"  max_drawdown={m.get('max_drawdown')}")
    print(f"  volatility={m.get('volatility')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
