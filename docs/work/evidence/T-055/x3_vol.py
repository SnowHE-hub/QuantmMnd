"""T-055 X3：波动预测（B1 历史 / B2 EWMA / B3 HAR / M LightGBM），DM 检验，以及剔除高预测波动的挑战臂 V（登记见 T-055.md）.

用法：PYTHONPATH=. python docs/work/evidence/T-055/x3_vol.py <data_root> <out_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs/work/evidence/T-030"))
sys.path.insert(0, str(REPO / "docs/work/evidence/T-042"))
sys.path.insert(0, str(REPO / "docs/work/evidence/T-043"))

from eval_primary import ENS_MEMBERS, NAV, load_family, nav_metrics  # noqa: E402
from x1_filters import wide_at  # noqa: E402
from x2_extended import N_BOOT, SEED, boot_idx, mrwd  # noqa: E402

from quantmind.execution.nav_engine import ExecutableNavEngine, NavConfig  # noqa: E402
from quantmind.research.gpu.ensemble import rank_average_frames  # noqa: E402
from quantmind.research.gpu.folds import calendar_and_grid, make_folds  # noqa: E402

A = np.sqrt(252)
LAST_DATE = pd.Timestamp("2026-09-24")
LGB = dict(n_estimators=500, learning_rate=0.03, num_leaves=63, min_child_samples=200, colsample_bytree=0.8,
           random_state=0, verbose=-1, n_jobs=8)
FEATS = ["lrv5", "lrv20", "lrv60", "lpark20", "maxdrop20", "lturn20", "lmv", "amihud", "lmkt_rv20", "beta60"]
HAR = ["lrv5", "lrv20", "lrv60"]


def nw_tstat(d: np.ndarray, lag: int = 4) -> tuple[float, float]:
    """均值的 Newey-West t 统计量与单侧 p（H1：均值 < 0）。"""
    from scipy import stats
    d = d[np.isfinite(d)]
    n = len(d)
    e = d - d.mean()
    s = (e @ e) / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    t = d.mean() / np.sqrt(s / n)
    return float(t), float(stats.t.cdf(t, df=n - 1))


def build(root: Path, grid: list[pd.Timestamp]) -> pd.DataFrame:
    D, L = root / "data", root / "data/lake"
    px = pd.read_parquet(D / "raw/alpha_prices_panel_v6_20261009.parquet",
                         columns=["ts_code", "trade_date", "high", "low", "close", "adj_factor", "amount"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px.drop_duplicates(["trade_date", "ts_code"], keep="last")

    def wide(c):
        return px.pivot(index="trade_date", columns="ts_code", values=c).sort_index()
    adjc = wide("close") * wide("adj_factor")
    lr = np.log(adjc).diff()
    lr = lr.where(lr.abs() < 0.5)                                   # 去掉复权缺口等异常
    hi, lo = wide("high"), wide("low")
    park = (np.log(hi / lo) ** 2 / (4 * np.log(2))).where(hi > 0)
    amt = wide("amount") * 1000.0
    f = {"lrv5": np.log(lr.rolling(5, min_periods=4).std() * A), "lrv20": np.log(lr.rolling(20, min_periods=15).std() * A),
         "lrv60": np.log(lr.rolling(60, min_periods=40).std() * A),
         "lpark20": np.log(np.sqrt(park.rolling(20, min_periods=15).mean()) * A),
         "maxdrop20": lr.rolling(20, min_periods=15).min(), "amt20": amt.rolling(20, min_periods=10).mean()}
    ew = (lr ** 2).ewm(alpha=0.06, min_periods=20).mean()
    f["lewma"] = np.log(np.sqrt(ew) * A)
    mkt = lr.mean(axis=1)
    f["lmkt_rv20"] = pd.DataFrame(np.repeat(np.log(mkt.rolling(20).std() * A).to_numpy()[:, None], lr.shape[1], 1),
                                  index=lr.index, columns=lr.columns)
    mx, mm = lr.mul(mkt, axis=0).rolling(60, min_periods=40).mean(), mkt.rolling(60, min_periods=40).mean()
    f["beta60"] = (mx - lr.rolling(60, min_periods=40).mean().mul(mm, axis=0)).div(mkt.rolling(60, min_periods=40).var(),
                                                                                  axis=0)
    f["y"] = np.log(lr.rolling(20, min_periods=15).std().shift(-20) * A)       # t+1 … t+20
    dates = pd.DatetimeIndex([d for d in grid if d in lr.index])
    rows = {k: v.reindex(dates).stack(future_stack=True) for k, v in f.items()}
    df = pd.DataFrame(rows)
    df.index.names = ["as_of", "ticker"]
    mv = wide_at(L / "daily_basic.parquet", "total_mv", dates, (dates[0] - pd.Timedelta(days=30)).strftime("%Y%m%d"))
    mvs = mv.stack()
    mvs.index.names = ["as_of", "ticker"]
    df["lmv"] = np.log(mvs.reindex(df.index) * 1e4)
    df["lturn20"] = np.log(df["amt20"] / np.exp(df["lmv"]))
    ami = pd.read_parquet(D / "panel/alpha_panel_weekly_v6_1.parquet", columns=["amihud_illiquidity"])
    df["amihud"] = ami["amihud_illiquidity"].reindex(df.index).to_numpy()
    meta = pd.concat([pd.read_parquet(p, columns=["as_of", "ticker", "top1500"])
                      for p in sorted((D / "gpu_v1/model/v1").glob("*.meta.parquet"))], ignore_index=True)
    meta["as_of"] = pd.to_datetime(meta["as_of"])
    pool = meta.set_index(["as_of", "ticker"])["top1500"]
    df["pool"] = pool.reindex(df.index).fillna(False).astype(bool).to_numpy()
    return df.replace([np.inf, -np.inf], np.nan)


def forecast(df: pd.DataFrame, test_years: list[int], cal: pd.DatetimeIndex) -> pd.DataFrame:
    import lightgbm as lgb
    out = []
    for y in test_years:
        start = pd.Timestamp(f"{y}-01-01")
        cut = cal[max(0, cal.searchsorted(start) - 30)]                   # 训练样本 t+20 已知，再留间隔
        tr = df[(df.index.get_level_values("as_of") < cut) & df["pool"]].dropna(subset=["y", *HAR])
        te = df[(df.index.get_level_values("as_of") >= start) & (df.index.get_level_values("as_of") < pd.Timestamp(f"{y + 1}-01-01"))
                & (df.index.get_level_values("as_of") <= LAST_DATE) & df["pool"]].dropna(subset=["y", *HAR])
        if te.empty:
            continue
        X = np.c_[np.ones(len(tr)), tr[HAR].to_numpy()]
        beta = np.linalg.lstsq(X, tr["y"].to_numpy(), rcond=None)[0]
        m = lgb.LGBMRegressor(**LGB).fit(tr[FEATS], tr["y"])
        o = te[["y"]].copy()
        o["B1"], o["B2"] = te["lrv20"], te["lewma"]
        o["B3"] = np.c_[np.ones(len(te)), te[HAR].to_numpy()] @ beta
        o["M"] = m.predict(te[FEATS])
        o["B2"] = o["B2"].fillna(o["B1"])
        o["year"] = y
        out.append(o)
        print(y, "train", len(tr), "test", len(te), flush=True)
    return pd.concat(out)


def losses(fc: pd.DataFrame) -> dict:
    res, dl = {}, {}
    for k in ("B1", "B2", "B3", "M"):
        mse = (fc[k] - fc["y"]) ** 2
        ratio = np.exp(2 * (fc["y"] - fc[k]))
        ql = ratio - np.log(ratio) - 1
        dl[k] = pd.DataFrame({"mse": mse, "qlike": ql}).groupby(level="as_of").mean()
        res[k] = {"mse": float(dl[k]["mse"].mean()), "qlike": float(dl[k]["qlike"].mean())}
    best = {L: min(("B1", "B2", "B3"), key=lambda k: res[k][L]) for L in ("mse", "qlike")}
    dm = {}
    for L in ("mse", "qlike"):
        d = (dl["M"][L] - dl[best[L]][L]).to_numpy()
        t, p = nw_tstat(d)
        dm[L] = {"vs": best[L], "mean_diff": float(d.mean()), "t": t, "p_one_sided": p}
    m_better = all(dm[L]["p_one_sided"] < 0.05 and dm[L]["mean_diff"] < 0 for L in dm)
    both_best = [k for k in ("B1", "B2", "B3") if best["mse"] == k and best["qlike"] == k]
    pick = "M" if m_better else (both_best[0] if both_best else "B3")
    return {"loss": res, "best_baseline": best, "dm": dm, "M_better": m_better, "picked": pick}


def portfolio(root: Path, fc: pd.DataFrame, pick: str, out: Path) -> dict:
    D = root / "data"
    cal, grid = calendar_and_grid(root)
    test = sorted({d for f in make_folds(cal, grid, "20d") for d in f.test_asof})
    meta = pd.concat([pd.read_parquet(p, columns=["as_of", "ticker", "top1500"])
                      for p in sorted((D / "gpu_v1/model/v1").glob("*.meta.parquet"))], ignore_index=True)
    meta["as_of"] = pd.to_datetime(meta["as_of"])
    meta = meta[meta["as_of"].isin(set(test))].set_index(["as_of", "ticker"])
    fams = {f: load_family(root, f, "20d") for f in ENS_MEMBERS}
    ens = rank_average_frames({m: s.reindex(meta.index) for m, s in fams.items()}).reindex(meta.index).dropna()
    hd, step, blk = NAV["20d"]
    cols = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "adj_factor", "amount"]
    prices = pd.read_parquet(D / "raw/alpha_prices_panel_v6_20261009.parquet", columns=cols)
    amihud = pd.read_parquet(D / "panel/alpha_panel_weekly_v6_1.parquet", columns=["amihud_illiquidity"])
    pv = fc[pick]
    q = pv.groupby(level="as_of").transform(lambda s: s.rank(pct=True))
    drop = set(q[q > 0.8].index)
    ens_v = ens[[i not in drop for i in ens.index]]
    navs, runs = {}, {}
    for name, s in (("R", ens), ("V", ens_v)):
        r = ExecutableNavEngine(s.rename("score").reset_index(), prices,
                                NavConfig(horizon_td=hd, rebalance_step=step, top_n=300, universe_size=1500),
                                amihud=amihud).run()
        runs[name], navs[name] = r, r["nav_daily"].set_index("date")[["nav_net", "nav_bench"]]
        navs[name].to_parquet(out / f"nav_{name}.parquet")
        print(name, "nav done", flush=True)
    bench = navs["R"]["nav_bench"]
    j = pd.DataFrame({"v": navs["V"]["nav_net"].pct_change(), "r": navs["R"]["nav_net"].pct_change(),
                      "b": bench.pct_change()}).dropna()
    # 只在有预测的区间比较（2022 起）
    j = j[j.index >= pd.Timestamp("2022-01-01")]
    rv, rr, rb = j["v"].to_numpy(), j["r"].to_numpy(), j["b"].to_numpy()
    bi = boot_idx(len(j), blk, N_BOOT, SEED)
    d = rv - rr
    bt = d[bi].mean(axis=1) * 252
    bdd = mrwd(rv[bi], rb[bi]) - mrwd(rr[bi], rb[bi])
    bvol = (rv[bi].std(axis=1) - rr[bi].std(axis=1)) * A
    res = {"n_days": int(len(j)), "dropped_rows": int(len(drop)), "ann_excess_diff": float(d.mean() * 252),
           "ci95": [float(np.quantile(bt, .025)), float(np.quantile(bt, .975))],
           "mrwd_diff": float(mrwd(rv[None], rb[None])[0] - mrwd(rr[None], rb[None])[0]),
           "mrwd_diff_ci95": [float(np.quantile(bdd, .025)), float(np.quantile(bdd, .975))],
           "vol_diff": float((rv.std() - rr.std()) * A), "vol_diff_ci95": [float(np.quantile(bvol, .025)),
                                                                          float(np.quantile(bvol, .975))],
           "R": nav_metrics(pd.DataFrame({"nav_net": navs["R"]["nav_net"], "nav_bench": bench}).dropna()),
           "V": nav_metrics(pd.DataFrame({"nav_net": navs["V"]["nav_net"], "nav_bench": bench.reindex(navs["V"].index)}).dropna())}
    res["noninferior"] = bool(res["ci95"][0] > -0.01 and res["mrwd_diff_ci95"][1] < 0.02)
    res["risk_improved"] = bool(res["vol_diff_ci95"][1] < 0)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cal, grid = calendar_and_grid(a.root)
    df = build(a.root, list(grid))
    print("panel", df.shape, flush=True)
    fc = forecast(df, [2022, 2023, 2024, 2025, 2026], pd.DatetimeIndex(cal))
    fc.to_parquet(a.out / "forecasts.parquet")
    lv = losses(fc)
    print(json.dumps(lv, ensure_ascii=False, indent=1), flush=True)
    pf = portfolio(a.root, fc, lv["picked"], a.out)
    res = {"forecast": lv, "portfolio_V_minus_R": pf, "n_forecast_rows": int(len(fc)),
           "by_year_mse": {int(y): {k: float(((g[k] - g["y"]) ** 2).mean()) for k in ("B1", "B2", "B3", "M")}
                           for y, g in fc.groupby("year")}}
    (a.out / "x3_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: pf[k] for k in ("ann_excess_diff", "ci95", "mrwd_diff_ci95", "vol_diff", "vol_diff_ci95",
                                         "noninferior", "risk_improved")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
