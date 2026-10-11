"""T-022 评估（只读）：行业专用模型（IND）与行业特征（SECF），基线 = T-018 的 C_infnan（同一口径）.

E1（行业专用模型）：同一截面、同一申万一级行业内（该行业有标签的票 ≥ 20 只）的 RankIC 配对差 IND − C_infnan；
  每个截面先对行业等权平均，得到按时间排列的序列；移动块自助法（块长 13 个周截面 ≈ 63 个交易日，10,000 次，种子 0）95% 区间。
  判读：均值 ≤ 0 或区间含 0 → "行业专用模型无增益"。另报各行业的均值差（描述性）。
E2（行业特征）：全市场 RankIC 配对差 SECF − C_infnan（按截面，块长 13 的移动块自助法）；
  判读：均值 > 0 且区间下界 > 0 → "行业特征有效"；否则"无可检测增益"。NAV（63d 配置）与日净收益差（块长 20）作描述。
用法：PYTHONPATH=. python eval_sector.py <data_root> <preds_dir> <out_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quantmind.execution.nav_engine import ExecutableNavEngine, NavConfig
from quantmind.execution.nav_gate import GateThresholds, evaluate_gate

root, preds_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)
D = root / "data"
panel = pd.read_parquet(D / "panel/alpha_panel_weekly_v6_1.parquet", columns=["forward_return_63d", "amihud_illiquidity"])
y = panel["forward_return_63d"]
l1 = pd.read_parquet(D / "panel/sector_features_v6_1.parquet", columns=["sw_l1"])["sw_l1"]
MIN_N = 20


def load(name):
    p = pd.read_parquet(preds_dir / f"ridge_63d_v6_1_{name}.parquet")
    p["as_of"] = pd.to_datetime(p["as_of"])
    return p.set_index(["as_of", "ticker"])["score"]


def block_ci(x: np.ndarray, block: int, scale: float = 1.0, n: int = 10_000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    T = len(x)
    k = int(np.ceil(T / block))
    idx = (rng.integers(0, T - block + 1, size=(n, k))[:, :, None] + np.arange(block)).reshape(n, -1)[:, :T]
    m = x[idx].mean(axis=1) * scale
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def rank_ic(df: pd.DataFrame, col: str) -> float:
    return df[col].rank().corr(df["y"].rank())


C = load("C_infnan")
res = {}
# ── E1 ──
IND = load("IND_infnan")
d = pd.DataFrame({"ind": IND, "c": C}).join(y.rename("y")).join(l1.rename("l1")).dropna()
rows = []
for (a, g), sub in d.groupby([d.index.get_level_values("as_of"), "l1"]):
    if len(sub) >= MIN_N:
        rows.append({"as_of": a, "l1": g, "n": len(sub), "ic_ind": rank_ic(sub, "ind"), "ic_c": rank_ic(sub, "c")})
e1 = pd.DataFrame(rows).dropna()
e1["diff"] = e1["ic_ind"] - e1["ic_c"]
series = e1.groupby("as_of")["diff"].mean().sort_index()
ci = block_ci(series.to_numpy(), 13)
per_ind = e1.groupby("l1").agg(n_as_of=("diff", "size"), ic_ind=("ic_ind", "mean"), ic_c=("ic_c", "mean"),
                               mean_diff=("diff", "mean")).sort_values("mean_diff")
res["E1_industry_models"] = {
    "n_as_of": int(len(series)), "n_industry_cross_sections": int(len(e1)), "n_industries": int(e1["l1"].nunique()),
    "ic_ind_mean": float(e1["ic_ind"].mean()), "ic_c_mean": float(e1["ic_c"].mean()),
    "mean_diff": float(series.mean()), "ci95": ci, "share_industries_better": float((per_ind["mean_diff"] > 0).mean()),
    "verdict": "行业专用模型有增益" if series.mean() > 0 and ci[0] > 0 else "行业专用模型无增益",
    "per_industry": per_ind.round(4).reset_index().to_dict("records")}
print("E1", {k: v for k, v in res["E1_industry_models"].items() if k != "per_industry"}, flush=True)
# ── E2 ──
SECF = load("SECF_infnan")
d2 = pd.DataFrame({"s": SECF, "c": C}).join(y.rename("y")).dropna()
ic = d2.groupby(level="as_of").apply(lambda g: pd.Series({"s": rank_ic(g, "s"), "c": rank_ic(g, "c")})).dropna()
diff = (ic["s"] - ic["c"]).sort_index()
ci2 = block_ci(diff.to_numpy(), 13)
res["E2_sector_features"] = {"n_as_of": int(len(ic)), "ic_secf": float(ic["s"].mean()), "ic_c": float(ic["c"].mean()),
                             "mean_diff": float(diff.mean()), "ci95": ci2,
                             "verdict": "行业特征有效" if diff.mean() > 0 and ci2[0] > 0 else "无可检测增益"}
COLS = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "adj_factor", "amount"]
prices = pd.read_parquet(D / "raw/alpha_prices_panel_v6_20261009.parquet", columns=COLS)
navd = {}
for name, s in (("C_infnan", C), ("SECF_infnan", SECF)):
    pr = s.reset_index()
    r = ExecutableNavEngine(pr, prices, NavConfig(horizon_td=63, rebalance_step=13, top_n=300, universe_size=1500),
                            amihud=panel[["amihud_illiquidity"]]).run()
    sm = r["summary"]
    nav = r["nav_daily"].set_index("date")
    navd[name] = nav["nav_net"].pct_change().dropna()
    ex = nav["nav_net"] / nav["nav_bench"]
    res[f"nav_{name}"] = {"ann_net_excess": sm["nav"]["ann_net_excess"], "ir": sm["nav"]["information_ratio"],
                          "relative_mdd": float((ex / ex.cummax() - 1).min()), "yearly": sm["nav"]["yearly_net_excess"],
                          "gate_pass": evaluate_gate(sm, GateThresholds(max_drawdown=0.12))["gate_pass"]}
a, b = navd["SECF_infnan"].align(navd["C_infnan"], join="inner")
dd = (a - b).to_numpy()
res["E2_nav_paired"] = {"n_days": int(len(dd)), "ann_mean_diff": float(dd.mean() * 252), "ci95": block_ci(dd, 20, 252)}
print("E2", res["E2_sector_features"], res["nav_C_infnan"], res["nav_SECF_infnan"], res["E2_nav_paired"], flush=True)
(out_dir / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
