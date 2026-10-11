"""T-022：申万 2021 行业成分（index_member_all）对 v6_1 选股池的时点覆盖（只读）.

规则：as_of 时点的行业 = in_date <= as_of 且 (out_date 为空 或 out_date > as_of) 的那条；一只票同一时点应只有一个一级行业。
报告：按年覆盖率、同一时点多个一级行业的冲突数、无行业的票的特征（多为早年退市）。
用法：python sw_coverage.py <data_root> <out.json>
"""
import json
import sys
from pathlib import Path

import pandas as pd

root, out = Path(sys.argv[1]), Path(sys.argv[2])
D = root / "data"
sw = pd.read_parquet(D / "lake/events/index_member_all_20261009.parquet")
sw["in_date"] = pd.to_datetime(sw["in_date"], errors="coerce")
sw["out_date"] = pd.to_datetime(sw["out_date"], errors="coerce")
uni = pd.read_parquet(D / "panel/alpha_panel_weekly_v6_1.parquet", columns=["momentum_1m"]).index.to_frame(index=False)
res = {"sw_rows": int(len(sw)), "sw_tickers": int(sw["ts_code"].nunique()), "l1": int(sw["l1_code"].nunique()),
       "l2": int(sw["l2_code"].nunique()), "l3": int(sw["l3_code"].nunique()),
       "in_date_min": str(sw["in_date"].min().date()), "is_new_counts": sw["is_new"].value_counts().to_dict()}
m = uni.merge(sw[["ts_code", "l1_code", "in_date", "out_date"]], left_on="ticker", right_on="ts_code", how="left")
valid = (m["in_date"] <= m["as_of"]) & (m["out_date"].isna() | (m["out_date"] > m["as_of"]))
v = m[valid]
n_l1 = v.groupby(["as_of", "ticker"])["l1_code"].nunique()
res["multi_l1_conflicts"] = int((n_l1 > 1).sum())
cov = uni.merge(n_l1.rename("n").reset_index(), on=["as_of", "ticker"], how="left")
cov["has"] = cov["n"].fillna(0) > 0
res["coverage_by_year"] = cov.groupby(cov["as_of"].dt.year)["has"].mean().round(4).to_dict()
res["coverage_overall"] = round(float(cov["has"].mean()), 4)
# 用"不看时点"的当前成分（is_new=Y）兜底时的覆盖（只作对照）
cur = set(sw.loc[sw["is_new"] == "Y", "ts_code"])
res["coverage_if_current_only"] = round(float(uni["ticker"].isin(cur).mean()), 4)
miss = cov.loc[~cov["has"], "ticker"].unique()
res["n_tickers_never_mapped"] = int(len(set(miss) - set(cov.loc[cov["has"], "ticker"])))
res["examples_missing"] = sorted(miss)[:10]
out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
