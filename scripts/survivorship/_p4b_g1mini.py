"""G1-mini：v6 Alpha158 重提取正确性抽查（universe 变了 handler 输出仍对）。

抽 a158_MA5 (=Mean($close,5)/$close) 与 a158_ROC5 (=Ref($close,5)/$close)，
对 3 只活跃票，用 alpha_prices_panel_v6 的 adj_close 在每个 as_of 手算，要求 corr>0.99。
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
import pandas as pd

a = pd.read_parquet(REPO / "data/bakeoff/alpha158_asof_v6.parquet")
cols = [c for c in a.columns if c in ("a158_MA5", "a158_ROC5")]
print("available check cols:", cols)
px = pd.read_parquet(REPO / "data/raw/alpha_prices_panel_v6.parquet",
                     columns=["ts_code", "trade_date", "adj_close"])
px["trade_date"] = pd.to_datetime(px["trade_date"])

TICKERS = ["600000.SH", "600519.SH", "000001.SZ"]
for col in cols:
    rows_m, rows_x = [], []
    for ts in TICKERS:
        s = (px[px.ts_code == ts].drop_duplicates("trade_date")
             .set_index("trade_date")["adj_close"].sort_index())
        if col == "a158_MA5":
            manual = s.rolling(5).mean() / s          # Mean(close,5)/close
        else:  # a158_ROC5
            manual = s.shift(5) / s                   # Ref(close,5)/close
        try:
            xs = a.xs(ts, level="ticker")[col]
        except KeyError:
            continue
        man_at = manual.reindex(xs.index.get_level_values(0) if xs.index.nlevels > 1
                                else xs.index)
        man_at.index = xs.index
        d = pd.concat([man_at.rename("m"), xs.rename("x")], axis=1).dropna()
        rows_m.append(d["m"]); rows_x.append(d["x"])
    if rows_m:
        M = pd.concat(rows_m); X = pd.concat(rows_x)
        corr = float(np.corrcoef(M.values, X.values)[0, 1])
        print(f"  {col}: n={len(M)} corr={corr:.5f} {'PASS' if corr > 0.99 else 'FAIL'}")
