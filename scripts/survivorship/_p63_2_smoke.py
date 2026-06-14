"""P63-2 smoke：单 as_of 构 snapshot → compute_all_fundamental_factors → 报各因子非空覆盖.

验证 lake 数据（daily_basic + fina_indicator）能喂通现有因子函数。
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
import pandas as pd
from quantmind.features.fundamental import compute_all_fundamental_factors, FUNDAMENTAL_FACTORS

AS_OF = pd.Timestamp("2024-06-03")
LAKE = REPO / "data/lake"

# daily_basic：as_of 当日或之前最后一个交易日
db = pd.read_parquet(LAKE / "daily_basic.parquet")
db["trade_date"] = pd.to_datetime(db["trade_date"])
last_td = db.loc[db.trade_date <= AS_OF, "trade_date"].max()
db_asof = db[db.trade_date == last_td].copy()
if "ticker" not in db_asof.columns:
    db_asof["ticker"] = db_asof["ts_code"].astype(str)
print(f"daily_basic @ {last_td.date()}: {len(db_asof)} 票")

# fina_indicator：ann_date <= as_of（PIT），加 ticker + report_date
fi = pd.read_parquet(LAKE / "fina_indicator.parquet")
asof_str = AS_OF.strftime("%Y%m%d")
fi_pit = fi[fi["ann_date"] <= asof_str].copy()
fi_pit["ticker"] = fi_pit["ts_code"].astype(str)
fi_pit["report_date"] = fi_pit["end_date"]
print(f"fina_indicator PIT(ann<= {asof_str}): {len(fi_pit)} 行, {fi_pit.ticker.nunique()} 票")

snapshot = {"daily_basic": db_asof, "financial_indicators": fi_pit}
factors = compute_all_fundamental_factors(snapshot, AS_OF.date())
print(f"\n因子面板 shape={factors.shape}")
print(f"{'factor':24} {'非空票数':>8} {'均值':>10} {'中位':>10}")
live = []
for name, _ in FUNDAMENTAL_FACTORS:
    s = factors[name].dropna()
    if len(s):
        live.append(name)
        print(f"{name:24} {len(s):>8} {s.mean():>10.3f} {s.median():>10.3f}")
    else:
        print(f"{name:24} {0:>8}   (空)")
print(f"\n=> 可用因子 {len(live)}/{len(FUNDAMENTAL_FACTORS)}: {live}")
