"""T-022 第二轮修订的评估（只读）：PIT 行业数据下的 E1 / E2（规则在运行前写进卡片"第二轮修订"一节）.

E1 主比较：IND_pit − CPOST_pit（两者训练数据都限于体系发布后，只差"分行业 vs 统一"）。
  同一截面、同一申万一级行业内（有标签 ≥ 20 只）的 RankIC 配对差，每截面对行业等权平均；移动块自助法（块长 13，10,000 次，种子 0）。
  判读：均值 > 0 且区间下界 > 0 → "行业专用模型有增益"；否则"未检测到增益"。
E1 次要：IND_pit − C_infnan（实际基线），同一批截面。
E2 主比较：SECF_pit − C_infnan 的全市场 RankIC 配对差（全部截面，块长 13）；判读同上（"有效" / "未检测到增益"）。NAV 与日净收益差（块长 20）作描述。
另报 CPOST_pit − C_infnan（只用发布后数据训练的代价，描述性）。
用法：PYTHONPATH=. python eval_sector_pit.py <data_root> <preds_dir> <out_dir>
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
l1 = pd.read_parquet(D / "panel/sector_features_v6_1_pit.parquet", columns=["sw_l1"])["sw_l1"]
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


def within_industry(a: pd.Series, b: pd.Series) -> dict:
    d = pd.DataFrame({"a": a, "b": b}).join(y.rename("y")).join(l1.rename("l1")).dropna()
    rows = []
    for (dt, g), sub in d.groupby([d.index.get_level_values("as_of"), "l1"]):
        if len(sub) >= MIN_N:
            rows.append({"as_of": dt, "l1": g, "ic_a": rank_ic(sub, "a"), "ic_b": rank_ic(sub, "b")})
    e = pd.DataFrame(rows).dropna()
    e["diff"] = e["ic_a"] - e["ic_b"]
    s = e.groupby("as_of")["diff"].mean().sort_index()
    ci = block_ci(s.to_numpy(), 13)
    per = e.groupby("l1").agg(n_as_of=("diff", "size"), ic_a=("ic_a", "mean"), ic_b=("ic_b", "mean"),
                              mean_diff=("diff", "mean")).sort_values("mean_diff")
    return {"n_as_of": int(len(s)), "as_of_range": [str(s.index.min().date()), str(s.index.max().date())],
            "n_industries": int(e["l1"].nunique()), "ic_a_mean": float(e["ic_a"].mean()), "ic_b_mean": float(e["ic_b"].mean()),
            "mean_diff": float(s.mean()), "ci95": ci, "share_industries_better": float((per["mean_diff"] > 0).mean()),
            "verdict": "有增益" if s.mean() > 0 and ci[0] > 0 else "未检测到增益",
            "per_industry": per.round(4).reset_index().to_dict("records")}


def market_ic(a: pd.Series, b: pd.Series) -> dict:
    d = pd.DataFrame({"a": a, "b": b}).join(y.rename("y")).dropna()
    ic = d.groupby(level="as_of").apply(lambda g: pd.Series({"a": rank_ic(g, "a"), "b": rank_ic(g, "b")})).dropna()
    diff = (ic["a"] - ic["b"]).sort_index()
    ci = block_ci(diff.to_numpy(), 13)
    return {"n_as_of": int(len(ic)), "ic_a": float(ic["a"].mean()), "ic_b": float(ic["b"].mean()),
            "mean_diff": float(diff.mean()), "ci95": ci, "verdict": "有效" if diff.mean() > 0 and ci[0] > 0 else "未检测到增益"}


C, IND, CPOST, SECF = load("C_infnan"), load("IND_pit_infnan"), load("CPOST_pit_infnan"), load("SECF_pit_infnan")
dates = IND.index.get_level_values("as_of").unique().intersection(CPOST.index.get_level_values("as_of").unique())
on = lambda s: s[s.index.get_level_values("as_of").isin(dates)]  # noqa: E731
res = {"taxonomy_public": "2021-12-13", "E1_primary_IND_minus_CPOST": within_industry(on(IND), on(CPOST)),
       "E1_secondary_IND_minus_C": within_industry(on(IND), on(C)),
       "E2_primary_SECF_minus_C": market_ic(SECF, C),
       "E2_on_E1_dates": market_ic(on(SECF), on(C)),
       "CPOST_minus_C_market_ic": market_ic(on(CPOST), on(C))}
for k, v in res.items():
    if isinstance(v, dict):
        print(k, {kk: vv for kk, vv in v.items() if kk != "per_industry"}, flush=True)
COLS = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "adj_factor", "amount"]
prices = pd.read_parquet(D / "raw/alpha_prices_panel_v6_20261009.parquet", columns=COLS)
navd = {}
for name, s in (("C_infnan", C), ("SECF_pit_infnan", SECF)):
    r = ExecutableNavEngine(s.reset_index(), prices, NavConfig(horizon_td=63, rebalance_step=13, top_n=300, universe_size=1500),
                            amihud=panel[["amihud_illiquidity"]]).run()
    sm = r["summary"]
    nav = r["nav_daily"].set_index("date")
    navd[name] = nav["nav_net"].pct_change().dropna()
    ex = nav["nav_net"] / nav["nav_bench"]
    res[f"nav_{name}"] = {"ann_net_excess": sm["nav"]["ann_net_excess"], "ir": sm["nav"]["information_ratio"],
                          "relative_mdd": float((ex / ex.cummax() - 1).min()), "yearly": sm["nav"]["yearly_net_excess"],
                          "gate_pass": evaluate_gate(sm, GateThresholds(max_drawdown=0.12))["gate_pass"]}
a, b = navd["SECF_pit_infnan"].align(navd["C_infnan"], join="inner")
dd = (a - b).to_numpy()
res["E2_nav_paired"] = {"n_days": int(len(dd)), "ann_mean_diff": float(dd.mean() * 252), "ci95": block_ci(dd, 20, 252)}
print(res["nav_C_infnan"], res["nav_SECF_pit_infnan"], res["E2_nav_paired"], flush=True)
(out_dir / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
