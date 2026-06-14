"""P63-1 验收：ann_date PIT 正确性（A 股基本面最易错处）.

抽 3 家公司 ≥5 报告期，核对 ann_date（实际公告日）严格晚于 end_date（报告期末），
且滞后在合理区间（季报~1mo / 半年报~2mo / 年报~2-4mo）。
全市场层面：统计 ann_date<=end_date 的违规行数（应为 0）。
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
import pandas as pd

fi = pd.read_parquet(REPO / "data/lake/fina_indicator.parquet")
fi = fi.dropna(subset=["ann_date", "end_date"]).copy()
fi["ann"] = pd.to_datetime(fi["ann_date"], format="%Y%m%d", errors="coerce")
fi["end"] = pd.to_datetime(fi["end_date"], format="%Y%m%d", errors="coerce")
fi = fi.dropna(subset=["ann", "end"])
fi["lag_days"] = (fi["ann"] - fi["end"]).dt.days

print(f"=== fina_indicator: {len(fi):,} 行(有效日期), {fi.ts_code.nunique()} 票 ===")
# 全市场违规统计
viol = fi[fi["lag_days"] <= 0]
print(f"全市场 ann_date <= end_date 违规行: {len(viol):,} ({len(viol)/len(fi):.4%})")
print(f"滞后天数分位: p5={fi.lag_days.quantile(.05):.0f} p50={fi.lag_days.quantile(.5):.0f} "
      f"p95={fi.lag_days.quantile(.95):.0f} max={fi.lag_days.max():.0f}")

print("\n=== 抽查 3 家公司最近报告期 (end_date, ann_date, lag, roe) ===")
all_ok = len(viol) == 0
for ts in ["600000.SH", "000001.SZ", "600519.SH"]:
    sub = fi[fi.ts_code == ts].sort_values("end").tail(6)
    print(f"\n{ts}:")
    for _, r in sub.iterrows():
        flag = "OK" if r["lag_days"] > 0 else "❌违规"
        print(f"  end={r['end_date']} ann={r['ann_date']} lag={int(r['lag_days']):>3}d roe={r['roe']} {flag}")
    if (sub["lag_days"] <= 0).any():
        all_ok = False

print(f"\n=== ann_date 验收: {'PASS ✅' if all_ok else 'FAIL ❌'} ===")
