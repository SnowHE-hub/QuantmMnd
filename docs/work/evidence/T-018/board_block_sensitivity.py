"""T-018 板块判读的时间依赖敏感性（Codex 第二轮驳回后补；事后分析，不替换预登记主检验）.

与 board_verdict.py 相同的 IC 表（同截面、同板块 ≥30 只），对 P−C 逐截面配对差做移动块自助法：
块长 1（= 原独立自助法）、13、26 个周频截面（63d 标签约跨 13 周）；10,000 次，种子 0。
用法：PYTHONPATH=. python board_block_sensitivity.py <data_root> <P.parquet> <C.parquet> <out.json>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quantmind.backtest.wf_costs import board_of

root, fp, fc, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
y = pd.read_parquet(root / "data/panel/alpha_panel_weekly_v6_1.parquet", columns=["forward_return_63d"])["forward_return_63d"]
BLOCKS, N_BOOT = (1, 13, 26), 10_000


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


def block_ci(d: np.ndarray, block: int) -> list[float]:
    rng = np.random.default_rng(0)
    T = len(d)
    k = int(np.ceil(T / block))
    idx = (rng.integers(0, T - block + 1, size=(N_BOOT, k))[:, :, None] + np.arange(block)).reshape(N_BOOT, -1)[:, :T]
    m = d[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


tp, tc = ic_table(pd.read_parquet(fp)), ic_table(pd.read_parquet(fc))
m = tp.merge(tc, on=["as_of", "board"], suffixes=("_p", "_c")).sort_values("as_of")
res = {"inputs": {"P": fp.name, "C": fc.name}, "blocks_in_weekly_cross_sections": list(BLOCKS), "boards": {}}
for b, g in m.groupby("board"):
    d = (g.ic_p - g.ic_c).to_numpy()
    res["boards"][b] = {"n_as_of": int(len(d)), "mean_diff": float(d.mean()),
                        "lag1_autocorr": float(pd.Series(d).autocorr()),
                        "ci95_by_block": {str(k): block_ci(d, k) for k in BLOCKS}}
out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
