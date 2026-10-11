"""T-016 第二轮修订（Codex REJECT 后）：A 与 B 在逐期相同的共同候选池上重算主比较.

原主比较的问题：变体 1（A_own）用 A 当期全部有分数的票，变体 2（B_on_A_candidates）是 A 与 B 都有分数的交集，
两边可选域不同（例 2022-09-30 A 可执行候选 748、B 725）。本脚本让两边都只在交集上选：
  1c A_own_common       A 的分数，限于当期 A、B 都有分数的票
  2  B_on_A_candidates  B 的分数，同一交集（与原变体 2 完全相同）
引擎、对齐、Top 30、选股池等其余设置与 eval_a_with_b_scores.py 相同。
输出每期两边的候选数量与股票列表 sha256（应逐期相同）、两边净值、日净收益差的年化均值与移动块自助法区间（块长 63 个交易日，10,000 次，种子 0）。
这是回顾性分析（第一次结果出来之后才做），不是预登记检验。
用法（t/016-eval 工作区根目录）：
  PYTHONPATH=. python docs/work/evidence/T-016/eval_common_pool.py --data-root ~/projects/quantmind \
      --a-scores docs/work/evidence/T-010/run_v4_1/oos_scores.csv --out docs/work/evidence/T-016/result_common_pool.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.run_line_a_oos as lao

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("t016_eval", _here / "eval_a_with_b_scores.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

BLOCK, N_BOOT = 63, 10_000


def pool_sha(tickers) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, tickers))).encode()).hexdigest()[:16]


def block_ci(d: np.ndarray) -> list[float]:
    rng = np.random.default_rng(0)
    T = len(d)
    k = int(np.ceil(T / BLOCK))
    idx = (rng.integers(0, T - BLOCK + 1, size=(N_BOOT, k))[:, :, None] + np.arange(BLOCK)).reshape(N_BOOT, -1)[:, :T]
    m = d[idx].mean(axis=1) * 252
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--a-scores", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    root = a.data_root
    lao.V6_PRICES_REL = base.V6_PRICES

    a_sc = pd.read_csv(a.a_scores, parse_dates=["as_of"])
    b = pd.read_parquet(root / base.B_PREDS)
    b["as_of"] = pd.to_datetime(b["as_of"])
    b_sc, align_log = base.align_b(b, sorted(a_sc["as_of"].unique()))
    dates = sorted(b_sc["as_of"].unique())
    a_sub = a_sc[a_sc["as_of"].isin(dates)]
    common = a_sub[["as_of", "ticker"]].merge(b_sc[["as_of", "ticker"]], on=["as_of", "ticker"], how="inner")
    variants = {
        "A_own_common": a_sub.merge(common, on=["as_of", "ticker"])[["as_of", "ticker", "score"]],
        "B_on_A_candidates": b_sc.merge(common, on=["as_of", "ticker"])[["as_of", "ticker", "score"]],
        "A_own_original": a_sub[["as_of", "ticker", "score"]],          # 原变体 1，对照
    }
    pools = []
    for d in dates:
        row = {"as_of": str(pd.Timestamp(d).date())}
        for k, v in variants.items():
            t = v.loc[v["as_of"] == d, "ticker"]
            row[f"n_{k}"] = int(len(t))
            row[f"sha_{k}"] = pool_sha(t)
        row["same_pool_A_common_vs_B"] = row["sha_A_own_common"] == row["sha_B_on_A_candidates"]
        row["n_dropped_from_A_original"] = row["n_A_own_original"] - row["n_A_own_common"]
        pools.append(row)
    out = {"inputs": {"a_scores": str(a.a_scores), "a_scores_sha256": base.sha(a.a_scores),
                      "b_preds": base.B_PREDS, "b_preds_sha256": base.sha(root / base.B_PREDS)},
           "nature": "回顾性分析：第一次结果之后才做；不是预登记检验",
           "pools_per_rebalance": pools, "results": {}}
    navs = {}
    for name, sc in variants.items():
        res = lao.executable_nav(sc, 30, 1500, root)
        out["results"][name] = base.summarize(res)
        navs[name] = res["nav_daily"].set_index("date")["nav_net"]
        r = out["results"][name]
        print(f"{name:20s} excess {r['excess']:+.4f} IR {r['ir']:+.3f} MDD {r['mdd']:+.3f} yearly {r['yearly_excess']}", flush=True)
    ra = navs["B_on_A_candidates"].pct_change().dropna()
    rb = navs["A_own_common"].pct_change().dropna()
    ra, rb = ra.align(rb, join="inner")
    d = (ra - rb).to_numpy()
    out["B_minus_A_common"] = {"n_days": int(len(d)), "ann_mean_diff": float(d.mean() * 252), "ci95_block63": block_ci(d)}
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out["B_minus_A_common"]), flush=True)
    print(pd.DataFrame(pools).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
