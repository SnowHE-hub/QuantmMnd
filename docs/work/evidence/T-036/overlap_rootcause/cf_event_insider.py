"""event 反事实（只算 insider_net_ratio_90d，按冻结构建器同一函数）：
A = 冻结的增减持表 + 冻结 base 网格 → 应逐值等于冻结面板；B = 新增减持表 + 影子 base 网格 → 应逐值等于影子面板。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/home/lenovo/projects/qm-t035/docs/work/evidence/T-030")
import build_daily_events as B
REAL = Path("/home/lenovo/projects/quantmind"); SH = Path("/home/lenovo/qm_fwd_root/20261020/main")
W0, W1 = pd.Timestamp("2026-09-01"), pd.Timestamp("2026-10-09")
E = REAL / "data/lake/events"
def ht(tag):
    h = pd.read_parquet(E / f"stk_holdertrade_{tag}.parquet")
    sign = h["in_de"].astype(str).map({"IN": 1.0, "DE": -1.0})
    return h.assign(ticker=h["ts_code"].astype(str), date=B.ymd(h["ann_date"]), v=sign * h["change_ratio"]).dropna(subset=["v"])
for label, tag, root in (("A_frozen", "20261009", REAL), ("B_new", "20261020", SH)):
    H = ht(tag)
    for m in ("2026-09", "2026-10"):
        idx = pd.read_parquet(root / "data/gpu_v1/panel/base" / f"{m}.parquet", columns=["momentum_1m"]).index
        grid = idx.to_frame(index=False); grid["as_of"] = pd.to_datetime(grid["as_of"])
        s = B.winsorize(B.window_sum(grid, H, "v", 90).reindex(idx))
        ref = pd.read_parquet(root / "data/gpu_v1/panel/event" / f"{m}.parquet", columns=["insider_net_ratio_90d"])["insider_net_ratio_90d"]
        d = ref.index.get_level_values("as_of"); ref = ref[(d >= W0) & (d <= W1)]
        x, y = ref.to_numpy(float), s.reindex(ref.index).to_numpy(float)
        mm = ~((x == y) | (np.isnan(x) & np.isnan(y)))
        print(label, m, "rows", len(ref), "cells_diff", int(mm.sum()), flush=True)
# 被改变的行是否都在"迟到增减持记录涉及的股票"或同日缩尾边界上
o = ht("20261009"); n = ht("20261020")
k = lambda d: set(map(tuple, d[["ticker", "ann_date", "in_de", "change_ratio"]].astype(str).to_numpy()))
late = {t for t, a, *_ in (k(n) - k(o)) if a <= "20261009"}
print("late holdertrade tickers (ann<=1009):", len(late))
