import json, re, numpy as np, pandas as pd
root = "/home/lenovo/projects/quantmind"
src = open("/home/lenovo/projects/qm-t035/quantmind/features/weekly_panel.py", encoding="utf-8").read()
for m in re.finditer(r"north_bound_30d_net_inflow", src):
    print("weekly_panel.py:", src[max(0, m.start()-300):m.start()+120].replace("\n", " | ")[:420]); break
f = json.load(open(f"{root}/data/gpu_v1/model/v1/features.json"))
print("in stock_features:", "north_bound_30d_net_inflow" in f["stock_features"], " in market_features:", "north_bound_30d_net_inflow" in f["market_features"])
p = pd.read_parquet(f"{root}/data/panel/alpha_panel_weekly_v6_1.parquet", columns=["north_bound_30d_net_inflow"])
s = p["north_bound_30d_net_inflow"]
d = s.index.get_level_values(0) if isinstance(s.index, pd.MultiIndex) else None
if d is not None:
    per_date_nuniq = s.groupby(level=0).nunique()
    print("weekly panel: distinct values per date (median):", per_date_nuniq.median())
    by = s.groupby(pd.to_datetime(d).year).agg(["mean", "std"]).round(0)
    print(by.to_string())
import xgboost as xgb
b = xgb.Booster(); b.load_model(f"{root}/data/gpu_v1/forward_models/20d_fold18/xgb/k18_s0.ubj")
gain = b.get_score(importance_type="total_gain")
names = f["stock_features"] + f["market_features"]
tot = sum(gain.values())
key = None
for k in gain:
    idx = int(k[1:]) if re.fullmatch(r"f\d+", k) else None
    nm = names[idx] if idx is not None and idx < len(names) else k
    if nm == "north_bound_30d_net_inflow": key = (k, gain[k])
rank = sorted(gain.values(), reverse=True)
print("fold18 xgb s0: features with gain:", len(gain), " north gain share:", round(key[1]/tot, 4) if key else 0,
      " rank:", (rank.index(key[1]) + 1) if key else None)
