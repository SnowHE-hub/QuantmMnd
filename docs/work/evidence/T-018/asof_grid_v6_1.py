"""T-018：v6_1 的 as_of 网格（= p2_build_v6 --calendar-file 时的 build_asof_grid(完整交易日历)），存成 JSON 供 qlib 环境使用。"""
import json
import sys
from pathlib import Path

import pandas as pd

from quantmind.features.weekly_panel import build_asof_grid

root, out = Path(sys.argv[1]), Path(sys.argv[2])
px = pd.read_parquet(root / "data/raw/alpha_prices_panel_v6_20261009.parquet", columns=["trade_date"])
d = pd.to_datetime(px["trade_date"]).drop_duplicates()
cal = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(root / "data/lake/trade_cal_sse.parquet")["cal_date"])).sort_values()
cal = cal[(cal >= d.min()) & (cal <= d.max())]
assert set(cal) == set(d), "price dates must equal the trade calendar"
grid = [x.strftime("%Y-%m-%d") for x in build_asof_grid(cal)]
out.write_text(json.dumps({"n": len(grid), "first": grid[0], "last": grid[-1], "as_of": grid}, indent=0))
print(len(grid), grid[:3], grid[-3:])
