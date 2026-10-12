"""T-027：主线 B 种子的风险分层选股（只读数据；结果写到 <out_dir>）.

方案（运行前写定，见 docs/work/tasks/T-027.md）：
  S0 基线        ExecutableNavEngine 原配置（Top 300 / 选股池 1500 / 等权）
  S1 行业        申万一级（时点成分）分层
  S2 市值        流通市值对数五分位分层
  S3 beta        beta_252d 五分位分层      ← 主比较（预先指定）
  S4 行业×市值   两者组合分层
主估计量：Sk 与 S0 的日净收益差的年化均值（×252）；区间 = 移动块自助法（块长 = 一个调仓间隔，10,000 次，种子 0），
  报 95% 与 Bonferroni 98.75%（4 个对比）。两者基准相同（选股池等权、无成本）。
风险指标：事后 beta（组合日净收益对基准日收益）、相对回撤（nav_net / nav_bench 的最大回撤）、绝对回撤、IR；
  另报 2026 年（1-01 至数据末）的同一组指标（描述性）。
主动暴露诊断：每个调仓日目标组合与候选池的 beta_252d 均值差、市值对数均值差、行业构成差（Σ|份额差|/2），取调仓日平均。
用法：PYTHONPATH=. python run_risk_strata.py <data_root> <preds.parquet> <seed: 63d|20d> <out_dir>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quantmind.execution.nav_engine import NavConfig
from quantmind.execution.nav_gate import GateThresholds, evaluate_gate
from quantmind.execution.nav_stratified import StratifiedNavEngine

SPEC = {"63d": (63, 13, 0.12), "20d": (20, 4, 0.15)}           # 与 T-018 / T-023 相同
VARIANTS = {
    "S0_baseline": (),
    "S1_industry": ("sw_l1",),
    "S2_size": ("log_circ_market_cap",),
    "S3_beta": ("beta_252d",),
    "S4_industry_size": ("sw_l1", "log_circ_market_cap"),
}
PRIMARY, BASE = "S3_beta", "S0_baseline"
N_BOOT, SEED = 10_000, 0
BONF = 0.05 / 4


def block_bootstrap_mean(d: np.ndarray, block: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    T = len(d)
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :T]
    return d[idx].mean(axis=1)


def risk_stats(nav: pd.DataFrame) -> dict:
    rp = nav["nav_net"].pct_change().dropna()
    rb = nav["nav_bench"].pct_change().dropna()
    rp, rb = rp.align(rb, join="inner")
    beta = float(np.cov(rp, rb)[0, 1] / np.var(rb, ddof=1)) if len(rb) > 2 else float("nan")
    rel = nav["nav_net"] / nav["nav_bench"]
    years = len(nav) / 252.0
    ex = rp - rb
    return {"n_days": int(len(nav)), "beta": beta,
            "rel_mdd": float((rel / rel.cummax() - 1).min()),
            "abs_mdd": float((nav["nav_net"] / nav["nav_net"].cummax() - 1).min()),
            "ann_net": float((nav["nav_net"].iloc[-1] / nav["nav_net"].iloc[0]) ** (1 / years) - 1) if years > 0 else None,
            "ann_bench": float((nav["nav_bench"].iloc[-1] / nav["nav_bench"].iloc[0]) ** (1 / years) - 1) if years > 0 else None,
            "cum_excess": float(rel.iloc[-1] / rel.iloc[0] - 1),
            "ir": float(ex.mean() / ex.std() * np.sqrt(252)) if ex.std() > 0 else 0.0,
            "vol_net": float(rp.std() * np.sqrt(252)), "vol_bench": float(rb.std() * np.sqrt(252))}


def active_exposure(targets: pd.DataFrame, cand: dict, expo: pd.DataFrame) -> dict:
    rows = []
    for _, r in targets.iterrows():
        a = pd.Timestamp(r["as_of"])
        c = cand.get(a)
        if c is None or a not in expo.index.get_level_values("as_of"):
            continue
        x = expo.xs(a, level="as_of")
        xt, xc = x.reindex(r["targets"]), x.reindex(c)
        st = xt["sw_l1"].fillna("NA").value_counts(normalize=True)
        sc = xc["sw_l1"].fillna("NA").value_counts(normalize=True)
        rows.append({"beta": xt["beta_252d"].mean() - xc["beta_252d"].mean(),
                     "size": xt["log_circ_market_cap"].mean() - xc["log_circ_market_cap"].mean(),
                     "industry_dev": float(st.sub(sc, fill_value=0).abs().sum() / 2)})
    df = pd.DataFrame(rows)
    return {k: float(df[k].mean()) for k in df.columns} | {"n_rebalances": int(len(df))}


def main() -> None:
    root, preds_path, seed, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    out_dir.mkdir(parents=True, exist_ok=True)
    D = root / "data"
    h, step, mdd = SPEC[seed]
    block = step * 5                                             # 周频网格：一个调仓间隔 ≈ step × 5 个交易日
    preds = pd.read_parquet(preds_path)
    preds["as_of"] = pd.to_datetime(preds["as_of"])
    cols = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "adj_factor", "amount"]
    prices = pd.read_parquet(D / "raw/alpha_prices_panel_v6_20261009.parquet", columns=cols)
    amihud = pd.read_parquet(D / "panel/alpha_panel_weekly_v6_1.parquet", columns=["amihud_illiquidity"])
    expo = pd.concat([
        pd.read_parquet(D / "panel/sector_features_v6_1.parquet", columns=["sw_l1"]),
        pd.read_parquet(D / "panel/fundamental_factors_v6_1.parquet", columns=["log_circ_market_cap"]),
        pd.read_parquet(D / "panel/alpha_panel_weekly_v6_1_base.parquet", columns=["beta_252d"]),
    ], axis=1)
    expo = expo.reset_index()
    expo["as_of"] = pd.to_datetime(expo["as_of"]).dt.normalize()
    expo = expo.set_index(["as_of", "ticker"]).sort_index()
    pdates = pd.DatetimeIndex(preds.as_of.unique())
    sub = expo[expo.index.get_level_values("as_of").isin(pdates)]
    res = {"inputs": {"preds": str(preds_path), "seed": seed, "horizon_td": h, "rebalance_step": step,
                      "gate_max_drawdown": mdd, "block_days": block, "n_as_of": int(len(pdates)),
                      "as_of_range": [str(pdates.min().date()), str(pdates.max().date())],
                      "exposure_date_coverage": float(pdates.isin(expo.index.get_level_values("as_of").unique()).mean()),
                      "exposure_nonnull_on_pred_dates": {c: float(sub[c].notna().mean()) for c in sub.columns}},
           "variants": {}}
    print(json.dumps(res["inputs"], ensure_ascii=False), flush=True)
    daily, cand_by_date = {}, {}
    for name, dims in VARIANTS.items():
        t0 = time.time()
        cfg = NavConfig(horizon_td=h, rebalance_step=step, top_n=300, universe_size=1500)
        eng = StratifiedNavEngine(preds, prices, cfg, amihud=amihud, exposures=expo, dims=dims)
        if name == BASE:                                          # 记录每个调仓日的候选（供主动暴露诊断）
            orig = eng._select

            def rec(ranked, as_of, _orig=orig):
                cand_by_date[pd.Timestamp(as_of)] = list(ranked)
                return _orig(ranked, as_of)
            eng._select = rec
        r = eng.run()
        s = r["summary"]
        n = s["nav"]
        nav = r["nav_daily"].set_index("date")
        nav[["nav_net", "nav_gross", "nav_bench", "cum_cost", "n_positions"]].to_parquet(out_dir / f"nav_{name}.parquet")
        daily[name] = nav["nav_net"].pct_change().dropna()
        g = evaluate_gate(s, GateThresholds(max_drawdown=mdd))
        y26 = nav[nav.index >= "2026-01-01"]
        sl = pd.DataFrame(eng.strata_log)
        res["variants"][name] = {
            "dims": list(dims), "period": s["period"],
            "ann_net_excess": n["ann_net_excess"], "information_ratio": n["information_ratio"],
            "max_drawdown_net": n["max_drawdown_net"], "yearly_net_excess": n["yearly_net_excess"],
            "cost_drag_ann": s["costs"]["cost_drag_ann"], "turnover_annualized_twoway": s["turnover"]["annualized_twoway"],
            "risk_full": risk_stats(nav), "risk_2026": risk_stats(y26) if len(y26) > 20 else None,
            "active_exposure": active_exposure(r["targets"], cand_by_date, expo),
            "mean_swapped_vs_plain_top300": float(sl["n_swapped_vs_plain_top"].mean()) if len(sl) else 0.0,
            "mean_na_candidates": float(sl["n_na_candidates"].mean()) if len(sl) else 0.0,
            "gate_pass_current": g["gate_pass"],
            "gate_checks": [{k: c[k] for k in ("criterion", "passed")} for c in g["checks"]],
            "seconds": round(time.time() - t0, 1),
        }
        v = res["variants"][name]
        print(name, json.dumps({"excess": v["ann_net_excess"], "ir": v["information_ratio"], "beta": v["risk_full"]["beta"],
                                "rel_mdd": v["risk_full"]["rel_mdd"], "abs_mdd": v["max_drawdown_net"],
                                "expo": v["active_exposure"], "sec": v["seconds"]}), flush=True)

    comp = {}
    b0 = res["variants"][BASE]
    for name in VARIANTS:
        if name == BASE:
            continue
        a, b = daily[name].align(daily[BASE], join="inner")
        d = (a - b).to_numpy()
        boots = block_bootstrap_mean(d, block) * 252
        v = res["variants"][name]
        d_beta = v["risk_full"]["beta"] - b0["risk_full"]["beta"]
        d_rel = v["risk_full"]["rel_mdd"] - b0["risk_full"]["rel_mdd"]
        d_ex = float(d.mean() * 252)
        d_ir = v["information_ratio"] - b0["information_ratio"]
        comp[f"{name}_minus_{BASE}"] = {
            "n_days": int(len(d)), "ann_mean_diff": d_ex,
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "ci_bonf": [float(np.percentile(boots, 100 * BONF / 2)), float(np.percentile(boots, 100 * (1 - BONF / 2)))],
            "d_beta": d_beta, "d_rel_mdd": d_rel, "d_ir": d_ir,
            "effective": bool(d_beta <= -0.10 and d_rel > 0 and d_ex >= -0.01 and d_ir >= -0.10),
            "return_improved": bool(np.percentile(boots, 100 * BONF / 2) > 0),
            "primary": name == PRIMARY}
    res["paired_vs_baseline"] = comp
    res["rule"] = ("effective = 事后 beta 至少降 0.10 且 相对回撤变浅 且 年化净收益差 ≥ −1% 且 IR 降幅 ≤ 0.10；"
                   "return_improved = Bonferroni 98.75% 区间下界 > 0；主比较 S3_beta")
    (out_dir / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(comp, indent=1), flush=True)


if __name__ == "__main__":
    main()
