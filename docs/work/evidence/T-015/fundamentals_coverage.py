"""T-015 财报补齐的合理性检查（只读）：新增行（相对 _backup_20261010）的报告期 / 公告月分布；
2026-06-30 前上市且 2026-09-01 仍在市的公司中，有 2026 半年报（end_date=20260630）的比例（半年报法定截止 8 月 31 日）。
用法：python fundamentals_coverage.py <data_root> <out.json>
"""
import json
import sys
from pathlib import Path

import pandas as pd

root, out = Path(sys.argv[1]), Path(sys.argv[2])
L = root / "data/lake"
KEY = ["ts_code", "end_date", "ann_date"]
sb = pd.read_parquet(L / "stock_basic_full_20261009.parquet")
daily = pd.read_parquet(L / "daily.parquet", columns=["ts_code", "trade_date"])
daily["trade_date"] = pd.to_datetime(daily["trade_date"])
trading_sep = set(daily.loc[daily["trade_date"] >= "2026-09-01", "ts_code"])
eligible = set(sb.loc[(sb["list_date"].astype(str) < "20260630") & (sb["list_status"] == "L"), "ts_code"]) & trading_sep
res = {"eligible_listed_trading_in_sep": len(eligible)}
for t in ("income", "balancesheet", "cashflow", "fina_indicator_full"):
    cur, old = pd.read_parquet(L / f"{t}.parquet"), pd.read_parquet(L / f"_backup_20261010/{t}.parquet")
    k = lambda d: d[KEY].astype(object).where(d[KEY].notna(), "<null>").astype(str).agg("|".join, axis=1)  # noqa: E731
    new = cur[~k(cur).isin(set(k(old)))]
    have_h1 = set(cur.loc[cur["end_date"] == "20260630", "ts_code"])
    missing = sorted(eligible - have_h1)
    res[t] = {"new_rows": int(len(new)),
              "new_by_end_date": new["end_date"].value_counts().head(6).to_dict(),
              "new_by_ann_month": new["ann_date"].astype(str).str[:6].value_counts().sort_index().to_dict(),
              "h1_2026_coverage_of_eligible": round(len(eligible & have_h1) / len(eligible), 4),
              "eligible_without_h1": len(missing), "examples_without_h1": missing[:10]}
out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
