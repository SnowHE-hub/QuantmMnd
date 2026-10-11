"""T-022：按时点（PIT）的申万 2021 行业与行业特征（只读输入，写新文件）.

行业：as_of 时点的成分 = in_date <= as_of 且 (out_date 为空 或 out_date > as_of)；一级 / 二级 / 三级各一列。
  限制：申万 2021 体系 2021-07 发布，此前的日期是把 2021 体系往回套（分类口径本身有轻微前视，但成员进出日期是时点的）。
行业特征（一级行业，选股池内等权，as_of 当时可得的数据）：
  ind_mom_1m / 3m / 6m        行业内 momentum_1m / 3m / 6m 的均值
  rel_mom_3m / 6m             个股动量 − 行业动量
  ind_dm_roe_ttm / pb / pe_ttm / operating_profit_yoy   基本面因子 − 行业中位数
输出 data/panel/sector_features_v6_1.parquet（index = (as_of, ticker)），拒绝覆盖。
用法：python build_sector_features.py <data_root> <out.json>
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

root, out_json = Path(sys.argv[1]), Path(sys.argv[2])
D = root / "data"
out = D / "panel/sector_features_v6_1.parquet"
if out.exists():
    raise SystemExit(f"refusing to overwrite {out}")
t0 = time.time()
mom = pd.read_parquet(D / "panel/alpha_panel_weekly_v6_1.parquet", columns=["momentum_1m", "momentum_3m", "momentum_6m"])
fnd = pd.read_parquet(D / "panel/fundamental_factors_v6_1.parquet",
                      columns=["roe_ttm", "pb", "pe_ttm", "operating_profit_yoy"]).reindex(mom.index)
sw = pd.read_parquet(D / "lake/events/index_member_all_20261009.parquet")
sw["in_date"] = pd.to_datetime(sw["in_date"], errors="coerce")
sw["out_date"] = pd.to_datetime(sw["out_date"], errors="coerce")
idx = mom.index.to_frame(index=False)
m = idx.merge(sw[["ts_code", "l1_name", "l2_name", "l3_name", "in_date", "out_date"]],
              left_on="ticker", right_on="ts_code", how="left")
ok = (m["in_date"] <= m["as_of"]) & (m["out_date"].isna() | (m["out_date"] > m["as_of"]))
ind = (m[ok].drop_duplicates(["as_of", "ticker"]).set_index(["as_of", "ticker"])[["l1_name", "l2_name", "l3_name"]]
       .rename(columns={"l1_name": "sw_l1", "l2_name": "sw_l2", "l3_name": "sw_l3"}).reindex(mom.index))
df = pd.concat([mom, fnd, ind], axis=1)
key = [df.index.get_level_values("as_of"), df["sw_l1"]]
feat = pd.DataFrame(index=df.index)
for c, name in (("momentum_1m", "ind_mom_1m"), ("momentum_3m", "ind_mom_3m"), ("momentum_6m", "ind_mom_6m")):
    feat[name] = df.groupby(key)[c].transform("mean")
feat["rel_mom_3m"] = df["momentum_3m"] - feat["ind_mom_3m"]
feat["rel_mom_6m"] = df["momentum_6m"] - feat["ind_mom_6m"]
for c in ("roe_ttm", "pb", "pe_ttm", "operating_profit_yoy"):
    feat[f"ind_dm_{c}"] = df[c] - df.groupby(key)[c].transform("median")
feat[df["sw_l1"].isna()] = float("nan")                 # 无行业的票：行业特征缺失（训练时标准化后填 0）
feat = pd.concat([ind, feat], axis=1)
feat.to_parquet(out)
res = {"rows": int(len(feat)), "coverage_l1": round(float(feat["sw_l1"].notna().mean()), 4),
       "n_l1": int(feat["sw_l1"].nunique()), "nonnull": {c: round(float(feat[c].notna().mean()), 4)
                                                        for c in feat.columns if c.startswith(("ind_", "rel_"))},
       "min_l1_size_median": int(feat.groupby([feat.index.get_level_values("as_of"), "sw_l1"]).size()
                                 .groupby(level=1).median().min()),
       "seconds": round(time.time() - t0, 1), "output": str(out)}
out_json.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
