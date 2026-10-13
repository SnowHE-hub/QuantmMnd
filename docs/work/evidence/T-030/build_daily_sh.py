"""T-030 P0：16 个短周期 survivor 因子在日频网格上的取值（= `p2_build_v6.py --sh` 的逻辑，只换取样网格）.

宽表因子本来就是逐日计算的（只用 ≤ 当日的数据）；v6_1 只在周频 as_of 上取样。这里取完整交易日历的每一天，不 reindex 到面板
（训练时再与 base 的 (as_of, ticker) 对齐，值不变）。输入与 T-017 相同：价格 alpha_prices_panel_v6_20261009、
daily_basic circ_mv、stock_basic_full_merged_20261009 的行业（缺行业的票显式剔除出残差回归）。
输出 <data_root>/data/gpu_v1/panel/sh/<YYYY-MM>.parquet；整组已存在则拒绝运行。
用法：PYTHONPATH=. python build_daily_sh.py <data_root>
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.survivorship.p2_build_v6 as p2
from quantmind.features.short_horizon_factors import (
    build_short_horizon_factors,
    build_wide_frames,
    sample_at_asof,
)


def main() -> None:
    root = Path(sys.argv[1])
    out_dir = root / "data/gpu_v1/panel/sh"
    if out_dir.exists() and any(out_dir.glob("*.parquet")):
        raise SystemExit(f"refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    p2.configure(data_root=root, prices_file=Path("data/raw/alpha_prices_panel_v6_20261009.parquet"),
                 calendar_file=Path("data/lake/trade_cal_sse.parquet"),
                 stock_basic=Path("data/lake/stock_basic_full_merged_20261009.parquet"),
                 sh_out=root / "data/gpu_v1/_never_written.parquet")
    cal = p2.trading_calendar()
    surv = json.loads((root / "data/loss_signals_v4/short_horizon_survivors.json").read_text())["survivors"]
    F = build_wide_frames(pd.read_parquet(p2.V6_PRICES))
    tickers = list(F["adj_close"].columns)
    db = pd.read_parquet(root / "data/lake/daily_basic.parquet", columns=["ts_code", "trade_date", "circ_mv"])
    db["trade_date"] = pd.to_datetime(db["trade_date"]).dt.normalize()
    db = db[(db.circ_mv > 0) & db.ts_code.isin(set(tickers))]
    mc = (db.drop_duplicates(["trade_date", "ts_code"], keep="last")
          .pivot(index="trade_date", columns="ts_code", values="circ_mv").sort_index())
    lc = np.log(mc).reindex(index=F["adj_close"].index.normalize(), columns=tickers)
    lc.index = F["adj_close"].index
    sb = pd.read_parquet(p2.SB_FULL)
    ind_map = sb.set_index(sb["ts_code"].astype(str))["industry"]
    rt, n_dropped = p2._residual_returns_v6(F["returns"], lc, ind_map)
    facs = {k: v for k, v in build_short_horizon_factors(F, rt).items() if k in surv}
    del F, rt, db, mc, lc
    print(f"[sh] {len(facs)} factors; missing-industry dropped {n_dropped} ({time.time() - t0:.0f}s)", flush=True)
    for ym in sorted({d.strftime("%Y-%m") for d in cal}):
        days = [d for d in cal if d.strftime("%Y-%m") == ym]
        part = sample_at_asof(facs, days, tickers=tickers)
        tmp = out_dir / f"{ym}.tmp"
        part.to_parquet(tmp)
        os.replace(tmp, out_dir / f"{ym}.parquet")
    print(f"[sh] done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
