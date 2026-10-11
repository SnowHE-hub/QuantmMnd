"""T-017：合并两份 stock_basic → 新文件（不覆盖原文件）.

stock_basic_full.parquet（2026-06 拉取，含 industry/area 等 15 列）+ stock_basic_full_20261009.parquet（T-015 刷新，6 列）：
- 以旧表为底（保留 industry、area 等字段）；
- 用新表更新同代码的 list_status / delist_date / name（6 月之后退市或改名的票）；
- 追加新表里才有的代码（6 月之后上市；industry 等为空——它们在 120 交易日 seasoning 之前不会进入 v6 票池）。
用法：python merge_stock_basic.py <data_root> → <data_root>/data/lake/stock_basic_full_merged_20261009.parquet
"""
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
lake = root / "data" / "lake"
out = lake / "stock_basic_full_merged_20261009.parquet"
if out.exists():
    raise SystemExit(f"refusing to overwrite {out}")
old = pd.read_parquet(lake / "stock_basic_full.parquet")
new = pd.read_parquet(lake / "stock_basic_full_20261009.parquet")
old["ts_code"], new["ts_code"] = old["ts_code"].astype(str), new["ts_code"].astype(str)
upd = new.set_index("ts_code")[["list_status", "delist_date", "name"]]
m = old.set_index("ts_code")
common = m.index.intersection(upd.index)
changed = {c: int((m.loc[common, c].astype(object).fillna("") != upd.loc[common, c].astype(object).fillna("")).sum())
           for c in ("list_status", "delist_date", "name")}
for c in ("list_status", "delist_date", "name"):
    m.loc[common, c] = upd.loc[common, c].values
add = new[~new["ts_code"].isin(m.index)].copy()
add["ticker"] = add["ts_code"]
merged = pd.concat([m.reset_index(), add], ignore_index=True)[old.columns]
merged.to_parquet(out, index=False)
print(f"old {len(old)} + new-only {len(add)} = {len(merged)}; updated among {len(common)} common: {changed}; "
      f"old-only (not in refreshed list) {len(m.index.difference(upd.index))}; -> {out}")
