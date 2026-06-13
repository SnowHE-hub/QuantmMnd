"""退市标签专项抽查（要求 A 验证）：v6 面板里退市票最后几个在市 as_of 的
forward_return_12d 是否 = 到末交易日的真实收益（手算复核），且行确实进了面板。"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
import pandas as pd

panel = pd.read_parquet(REPO / "data/panel/alpha_panel_weekly_v6.parquet",
                        columns=["forward_return_12d", "forward_return_63d"])
px = pd.read_parquet(REPO / "data/raw/alpha_prices_panel_v6.parquet",
                     columns=["ts_code", "trade_date", "adj_close"])
px["trade_date"] = pd.to_datetime(px["trade_date"])

CASES = {
    "000005.SZ": ("2024-04-26", "2024-03-05"),
    "000018.SZ": ("2020-01-07", "2020-01-06"),
    "000023.SZ": ("2024-09-02", "2024-07-24"),
}

for ts, (delist, last_trade) in CASES.items():
    print(f"\n=== {ts}  delist={delist}  last_trade={last_trade} ===")
    s = (px[px.ts_code == ts].drop_duplicates("trade_date")
         .set_index("trade_date")["adj_close"].sort_index())
    last_px = float(s.iloc[-1]); last_dt = s.index[-1]
    print(f"  v6 价格序列末日={last_dt.date()} adj_close={last_px:.3f}")
    sub = panel.xs(ts, level="ticker", drop_level=False)
    in_market = sub[sub["forward_return_12d"].notna()]
    print(f"  面板内行数={len(sub)}  forward_12d 非NaN行数={len(in_market)}")
    # 末尾 3 个在市 as_of
    for as_of, row in in_market.tail(3).iterrows():
        a = as_of[0] if isinstance(as_of, tuple) else as_of
        fr12 = row["forward_return_12d"]
        # 手算：as_of 后第 12 个交易日；若已超末交易日 → 用末交易价（退市真实退出）
        fut = s.index[s.index > a]
        base = float(s.loc[s.index[s.index <= a][-1]])
        if len(fut) >= 12:
            exit_px = float(s.loc[fut[11]])
            tag = f"t+12={fut[11].date()}"
        else:
            exit_px = last_px
            tag = f"退市退出(末交易日 {last_dt.date()})"
        manual = exit_px / base - 1.0
        ok = "OK" if abs(manual - fr12) < 1e-6 else "MISMATCH"
        print(f"    as_of={a.date()} 面板fr12={fr12:+.4f} 手算={manual:+.4f} [{tag}] {ok}")
