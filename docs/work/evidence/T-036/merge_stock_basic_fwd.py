"""T-036：股票基础表的前瞻合并版（规则与 T-017 merge_stock_basic.py 逐行相同，只把新表和输出改成参数）.

stock_basic_full.parquet（2026-06 拉取，含 industry/area 等 15 列）+ stock_basic_full_<NEW>.parquet（较新的刷新，6 列）：
- 以旧表为底（保留 industry、area 等字段）；
- 用新表更新同代码的 list_status / delist_date / name；
- 追加新表里才有的代码（industry 等为空——它们在 120 交易日 seasoning 之前不会进入 v6 票池）。
用法：python merge_stock_basic_fwd.py <data_root> <NEW_YYYYMMDD> <OUT_YYYYMMDD>
      → <data_root>/data/lake/stock_basic_full_merged_<OUT>.parquet（不覆盖已有文件）
"""
import sys
from pathlib import Path

import pandas as pd

root, new_tag, out_tag = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
lake = root / "data" / "lake"
out = lake / f"stock_basic_full_merged_{out_tag}.parquet"
if out.exists():
    raise SystemExit(f"refusing to overwrite {out}")
old = pd.read_parquet(lake / "stock_basic_full.parquet")
new = pd.read_parquet(lake / f"stock_basic_full_{new_tag}.parquet")
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
