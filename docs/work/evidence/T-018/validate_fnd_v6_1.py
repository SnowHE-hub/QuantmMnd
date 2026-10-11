"""T-018：v6_1 基本面因子与旧 v6 的对比（只读；只取 16 个入选因子）.

在两版都有的 (as_of, ticker) 上逐列比较（相对误差 1e-9）。v6_1 与 v6 的差异来源预期有两类：
  1 网格相位不同 → 共同 as_of 只有一部分（T-017 §3.1）
  2 T-015 补齐财报（新增 2026 年报告 + 2022–2026 的更正补发）、价格补齐（市值类因子）
另报：v6_1 在 2026-04-20 之后的新截面覆盖率。
用法：python validate_fnd_v6_1.py <data_root> <out.json>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root, out = Path(sys.argv[1]), Path(sys.argv[2])
D = root / "data"
surv = json.loads((D / "loss_signals_v4/fundamental_survivors_v6.json").read_text())["survivors"]
new = pd.read_parquet(D / "panel/fundamental_factors_v6_1.parquet", columns=surv)
old = pd.read_parquet(D / "panel/fundamental_factors_v6.parquet", columns=surv)
res = {"survivors": surv, "shape_new": list(new.shape), "shape_old": list(old.shape)}
common = new.index.intersection(old.index)
res["common_rows"] = int(len(common))
res["common_as_of"] = int(common.get_level_values("as_of").nunique())
a, b = new.loc[common], old.loc[common]
per = {}
for c in surv:
    x, y = a[c].to_numpy(float), b[c].to_numpy(float)
    both_nan = np.isnan(x) & np.isnan(y)
    same = np.isclose(x, y, rtol=1e-9, atol=1e-12) | both_nan
    per[c] = {"n": int(len(x)), "diff_share": round(float((~same).mean()), 5),
              "nan_new_only": int((np.isnan(x) & ~np.isnan(y)).sum()), "nan_old_only": int((~np.isnan(x) & np.isnan(y)).sum())}
res["per_factor"] = per
asof = pd.Series(new.index.get_level_values("as_of"))
recent = new[new.index.get_level_values("as_of") > pd.Timestamp("2026-04-20")]
res["new_as_of_after_2026_04_20"] = int(recent.index.get_level_values("as_of").nunique())
res["coverage_after_2026_04_20"] = {c: round(float(recent[c].notna().mean()), 4) for c in surv}
res["coverage_overall_new"] = {c: round(float(new[c].notna().mean()), 4) for c in surv}
out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k not in ("coverage_overall_new",)}, ensure_ascii=False, indent=1))
