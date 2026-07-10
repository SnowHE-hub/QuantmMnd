#!/usr/bin/env python
"""scripts/screen_short_horizon.py — R3：short_horizon 单因子中性化 IC 筛选.

口径（bake-off 决策记录 R3）：
- 市场项 = cap 加权截面均值（在 residual_returns 内）。
- 选因子用 **in-sample（as_of<2022-01）** neut IC；**OOS（>=2022-01）** 留无偏确认。
- 保留线：|in-sample neut ICIR| >= 0.2 且 翻转率<35% 且 max|corr|<0.7。
产物：data/loss_signals_v4/short_horizon_screen.csv（排序表）
      data/panel/short_horizon_factors.parquet（(as_of,ticker) 因子面板，供 merge_increment）
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from quantmind.features.weekly_panel import OUT_PATH, WHITELIST_35
from quantmind.features.short_horizon_factors import (
    build_wide_frames, residual_returns, build_short_horizon_factors, sample_at_asof,
)

PRICES = REPO / "data" / "raw" / "alpha_prices_panel.parquet"
DAILY_BASIC = REPO / "data" / "lake" / "daily_basic.parquet"
OUT_DIR = REPO / "data" / "loss_signals_v4"; OUT_DIR.mkdir(parents=True, exist_ok=True)
PANEL_OUT = REPO / "data" / "panel" / "short_horizon_factors.parquet"
LABEL = "forward_return_12d"
OOS_START = pd.Timestamp("2022-01-01")
NUM_BASE = [c for c in WHITELIST_35 if c not in ("exposure_industry", "exposure_area")]


def spearman(a, b):
    d = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(d) < 15:
        return np.nan
    return float(stats.spearmanr(d["a"], d["b"])[0])


def neutralize_xs(fac, label, ind, logcap):
    """单截面：fac 对 [1, 行业哑变量, z(logcap)] 残差，残差 vs label 的 spearman。返回 (raw_ic, neut_ic)。"""
    df = pd.DataFrame({"f": fac, "y": label, "ind": ind, "lc": logcap}).dropna()
    if df["ind"].nunique() < 2 or len(df) < 50:
        return np.nan, np.nan
    D = pd.get_dummies(df["ind"], drop_first=True).astype(float)
    z = (df["lc"] - df["lc"].mean()) / (df["lc"].std() + 1e-12)
    X = np.column_stack([np.ones(len(df)), z.values, D.values])
    beta, *_ = np.linalg.lstsq(X, df["f"].values.astype(float), rcond=None)
    resid = pd.Series(df["f"].values - X @ beta, index=df.index)
    raw = spearman(df["f"], df["y"]); neu = spearman(resid, df["y"])
    return raw, neu


def main():
    t0 = time.perf_counter()
    print("[load] prices"); pl = pd.read_parquet(PRICES)
    F = build_wide_frames(pl)
    dates = F["adj_close"].index; tickers = list(F["adj_close"].columns)
    print(f"[wide] dates={len(dates)} tickers={len(tickers)} ({time.perf_counter()-t0:.0f}s)")

    # log circ_mv（日频 wide）
    db = pd.read_parquet(DAILY_BASIC, columns=["ticker", "trade_date", "circ_mv"])
    db["trade_date"] = pd.to_datetime(db["trade_date"]).dt.normalize()
    db = db[db["ticker"].isin(set(tickers))]
    db = db[db["circ_mv"] > 0]
    mc = db.drop_duplicates(["trade_date", "ticker"], keep="last").pivot(
        index="trade_date", columns="ticker", values="circ_mv").sort_index()
    logcap_daily = np.log(mc).reindex(index=pd.to_datetime(dates).normalize(), columns=tickers)
    logcap_daily.index = dates
    print(f"[size] logcap daily {logcap_daily.shape} ({time.perf_counter()-t0:.0f}s)")

    # 行业 map（静态，from panel）
    panel = pd.read_parquet(OUT_PATH)
    ind_map = panel.groupby(level="ticker")["exposure_industry"].last().astype("object")
    print(f"[industry] {ind_map.nunique()} classes")

    # 残差日收益
    print("[resid] computing residual daily returns ...")
    rtilde = residual_returns(F["returns"], logcap_daily, ind_map)
    print(f"[resid] done ({time.perf_counter()-t0:.0f}s)")

    # 因子
    facs = build_short_horizon_factors(F, rtilde)
    print(f"[factors] {len(facs)} candidates")

    # 取到 as_of 网格
    as_of_list = sorted(panel.index.get_level_values("as_of").unique())
    sh_panel = sample_at_asof(facs, as_of_list, tickers=tickers)
    # 只保留与 panel 对齐的 (as_of,ticker)
    sh_panel = sh_panel.reindex(panel.index)
    sh_panel.to_parquet(PANEL_OUT)
    print(f"[save] {PANEL_OUT} shape={sh_panel.shape}")

    # 评估辅助：label / industry / logcap 于 (as_of,ticker)
    lab = panel[LABEL]
    ind_at = pd.Series(ind_map.reindex(panel.index.get_level_values("ticker")).values,
                       index=panel.index)
    # as_of 的 logcap
    logcap_at = []
    for a in as_of_list:
        a = pd.Timestamp(a).normalize()
        if a in set(pd.to_datetime(dates).normalize()):
            row = logcap_daily.loc[logcap_daily.index.normalize() == a]
            if len(row):
                s = row.iloc[0]; s.index = pd.MultiIndex.from_product([[pd.Timestamp(a)], s.index],
                                                                       names=["as_of", "ticker"])
                logcap_at.append(s)
    logcap_at = pd.concat(logcap_at) if logcap_at else pd.Series(dtype=float)
    logcap_at = logcap_at.reindex(panel.index)

    # 冗余：within-as_of rank（pooled Pearson ≈ 平均截面 Spearman）
    def asof_rank(s):
        return s.groupby(level="as_of").rank(pct=True)
    base_ranks = {c: asof_rank(panel[c]) for c in NUM_BASE}
    cand_ranks = {n: asof_rank(sh_panel[n]) for n in facs}

    rows = []
    names = list(facs.keys())
    for n in names:
        f = sh_panel[n]
        # 逐 as_of raw/neut IC
        recs = []
        for a in as_of_list:
            try:
                fa = f.xs(a, level="as_of"); ya = lab.xs(a, level="as_of")
                ia = ind_at.xs(a, level="as_of"); la = logcap_at.xs(a, level="as_of")
            except KeyError:
                continue
            raw, neu = neutralize_xs(fa, ya, ia, la)
            recs.append((pd.Timestamp(a), raw, neu))
        ic = pd.DataFrame(recs, columns=["as_of", "raw", "neu"]).set_index("as_of")
        is_mask = ic.index < OOS_START; oos_mask = ic.index >= OOS_START
        is_neu = ic.loc[is_mask, "neu"].dropna(); oos_neu = ic.loc[oos_mask, "neu"].dropna()
        is_raw = ic.loc[is_mask, "raw"].dropna()
        # 翻转率：in-sample 按季度 neu IC 符号 != 总体符号
        if len(is_neu):
            q = is_neu.groupby(is_neu.index.to_period("Q")).mean()
            dom = np.sign(is_neu.mean()) or 1
            flip = float((np.sign(q) != dom).mean()) if len(q) else np.nan
        else:
            flip = np.nan
        is_icir = float(is_neu.mean() / is_neu.std()) if is_neu.std() else np.nan
        oos_icir = float(oos_neu.mean() / oos_neu.std()) if oos_neu.std() else np.nan
        # 冗余 max|corr|
        cr = cand_ranks[n]
        mb = max((abs(cr.corr(base_ranks[c])) for c in NUM_BASE), default=np.nan)
        mc_ = max((abs(cr.corr(cand_ranks[m])) for m in names if m != n), default=np.nan)
        maxcorr = float(np.nanmax([mb, mc_]))
        keep = (abs(is_icir) >= 0.2) and (flip is not np.nan and flip < 0.35) and (maxcorr < 0.7)
        rows.append({
            "factor": n,
            "is_raw_ic": round(float(is_raw.mean()), 4) if len(is_raw) else np.nan,
            "is_neut_ic": round(float(is_neu.mean()), 4) if len(is_neu) else np.nan,
            "is_neut_icir": round(is_icir, 3) if is_icir == is_icir else np.nan,
            "is_flip": round(flip, 2) if flip == flip else np.nan,
            "oos_neut_ic": round(float(oos_neu.mean()), 4) if len(oos_neu) else np.nan,
            "oos_neut_icir": round(oos_icir, 3) if oos_icir == oos_icir else np.nan,
            "maxcorr_base": round(float(mb), 2) if mb == mb else np.nan,
            "maxcorr_cand": round(float(mc_), 2) if mc_ == mc_ else np.nan,
            "keep": bool(keep),
        })
        print(f"  {n:32s} is_neutIC={rows[-1]['is_neut_ic']} icir={rows[-1]['is_neut_icir']} "
              f"flip={rows[-1]['is_flip']} maxcorr={maxcorr:.2f} keep={keep}")

    tab = pd.DataFrame(rows).sort_values("is_neut_icir", key=lambda s: s.abs(), ascending=False)
    tab.to_csv(OUT_DIR / "short_horizon_screen.csv", index=False)
    surv = tab[tab["keep"]]["factor"].tolist()
    (OUT_DIR / "short_horizon_survivors.json").write_text(
        json.dumps({"survivors": surv, "n_candidates": len(names)}, ensure_ascii=False, indent=2))
    print("\n==== SURVIVORS ====")
    print(tab.to_string(index=False))
    print(f"\nsurvivors ({len(surv)}): {surv}")
    print(f"[done] {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
