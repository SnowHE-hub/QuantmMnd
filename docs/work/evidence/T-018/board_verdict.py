"""T-018 板块模型判读（只读）：P（按板块分别训练）− C（统一训练）在同板块、同截面的 RankIC 配对差.

预登记规则（卡片 9f58c03）：某板块 P−C 配对差均值 ≤ 0 或其 95% 区间含 0 → 该板块"板块专用模型无增益"；
三个板块都无增益 → 建议停用主线 A 的板块模型及 BoardModelRouter 调用方。
IC 计算与区间方法与 eval_line_b_v6_1.py 相同（同截面、同板块 ≥30 只；独立自助法 10,000 次，种子 0）。
输入：同一 --inf-to-nan 口径下的 P 与 C（出结果前的修正，见卡片）。
用法：PYTHONPATH=. python board_verdict.py <data_root> <P.parquet> <C.parquet> <out.json>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quantmind.backtest.wf_costs import board_of

root, fp, fc, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
y = pd.read_parquet(root / "data/panel/alpha_panel_weekly_v6_1.parquet", columns=["forward_return_63d"])["forward_return_63d"]


def ic_table(preds: pd.DataFrame) -> pd.DataFrame:
    p = preds.copy()
    p["as_of"] = pd.to_datetime(p["as_of"])
    p = p.join(y.rename("y"), on=["as_of", "ticker"]).dropna(subset=["y", "score"])
    p["board"] = p["ticker"].map(board_of)
    rows = []
    for a, g in p.groupby("as_of"):
        for b, gb in [("ALL", g)] + [(k, v) for k, v in g.groupby("board") if k in ("MAIN", "CHINEXT", "STAR")]:
            if len(gb) >= 30:
                rows.append({"as_of": a, "board": b, "ic": gb["score"].rank().corr(gb["y"].rank())})
    return pd.DataFrame(rows)


tp, tc = ic_table(pd.read_parquet(fp)), ic_table(pd.read_parquet(fc))
m = tp.merge(tc, on=["as_of", "board"], suffixes=("_p", "_c"))
rng = np.random.default_rng(0)
res = {"inputs": {"P": fp.name, "C": fc.name}, "boards": {}}
for b, g in m.groupby("board"):
    d = (g.ic_p - g.ic_c).to_numpy()
    boots = d[rng.integers(0, len(d), size=(10_000, len(d)))].mean(axis=1)
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    rec = {"n_as_of": int(len(d)), "ic_P": float(g.ic_p.mean()), "ic_C": float(g.ic_c.mean()),
           "mean_diff": float(d.mean()), "ci95": ci, "share_p_better": float((d > 0).mean()),
           "lag1_autocorr": float(pd.Series(d).autocorr()) if len(d) > 2 else None}
    if b != "ALL":
        rec["board_model_gain"] = bool(d.mean() > 0 and ci[0] > 0)
    res["boards"][b] = rec
per = [res["boards"][b]["board_model_gain"] for b in ("MAIN", "CHINEXT", "STAR") if b in res["boards"]]
res["verdict"] = ("三个板块均无增益 → 按预登记规则建议停用主线 A 的板块模型及 BoardModelRouter 调用方" if not any(per)
                  else "至少一个板块 P 显著优于 C：见各板块明细，不建议整体停用")
res["note"] = "相邻截面的 63d 标签重叠，配对差有自相关（见 lag1_autocorr）；独立自助法区间偏窄，判读时需考虑"
out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
