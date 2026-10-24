"""T-055 X3 v2（登记见 T-055.md "X3 v2 登记"；同一评估期上的第二次尝试）.

预测：M2 = HAR + LightGBM 学 HAR 残差（训练窗最后一年早停）；E = B2、B3、M2 的 log 空间平均。
组合（T-052 模拟器的研究口径，R / V10 / W 同一引擎）：R 原样；V10 剔除预测波动最高 10%；W 逆波动加权（0.5–2 倍等权截断）。
用法：PYTHONPATH=. python docs/work/evidence/T-055/x3_v2.py <data_root> <out_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[3]))
from x3_vol import FEATS, HAR, LAST_DATE, LGB, A, boot_idx, build, losses, mrwd, nw_tstat  # noqa: E402,F401

sys.path.insert(0, str(HERE.parents[3] / "docs/work/evidence/T-030"))
from eval_primary import ENS_MEMBERS, load_family  # noqa: E402

from quantmind.backtest.wf_costs import SlippageTiers  # noqa: E402
from quantmind.line_a.sim_account import AccountSimulator, Market, SimConfig  # noqa: E402
from quantmind.research.gpu.ensemble import rank_average_frames  # noqa: E402
from quantmind.research.gpu.folds import calendar_and_grid, make_folds  # noqa: E402


def forecast_v2(df: pd.DataFrame, fc1: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    import lightgbm as lgb
    out = []
    for y in sorted(fc1["year"].unique()):
        start = pd.Timestamp(f"{y}-01-01")
        cut = cal[max(0, cal.searchsorted(start) - 30)]
        tr = df[(df.index.get_level_values("as_of") < cut) & df["pool"]].dropna(subset=["y", *HAR])
        X = np.c_[np.ones(len(tr)), tr[HAR].to_numpy()]
        beta = np.linalg.lstsq(X, tr["y"].to_numpy(), rcond=None)[0]
        har_tr = X @ beta
        vcut = cut - pd.Timedelta(days=365)
        is_val = tr.index.get_level_values("as_of") >= vcut
        m = lgb.LGBMRegressor(**(LGB | {"n_estimators": 2000})).fit(
            tr.loc[~is_val, FEATS], tr["y"].to_numpy()[~is_val] - har_tr[~is_val],
            eval_set=[(tr.loc[is_val, FEATS], tr["y"].to_numpy()[is_val] - har_tr[is_val])],
            callbacks=[lgb.early_stopping(200, verbose=False)])
        g = fc1[fc1["year"] == y].copy()
        te = df.reindex(g.index)
        g["M2"] = g["B3"] + m.predict(te[FEATS])
        g["E"] = (g["B2"] + g["B3"] + g["M2"]) / 3
        out.append(g)
        print(y, "best_iter", m.best_iteration_, flush=True)
    return pd.concat(out)


def judge(fc: pd.DataFrame) -> dict:
    res, dl = {}, {}
    for k in ("B1", "B2", "B3", "M", "M2", "E"):
        mse = (fc[k] - fc["y"]) ** 2
        r = np.exp(2 * (fc["y"] - fc[k]))
        dl[k] = pd.DataFrame({"mse": mse, "qlike": r - np.log(r) - 1}).groupby(level="as_of").mean()
        res[k] = {"mse": float(dl[k]["mse"].mean()), "qlike": float(dl[k]["qlike"].mean())}
    best = {L: min(("B1", "B2", "B3"), key=lambda k: res[k][L]) for L in ("mse", "qlike")}
    dm = {}
    for cand in ("M2", "E"):
        dm[cand] = {}
        for L in ("mse", "qlike"):
            d = (dl[cand][L] - dl[best[L]][L]).to_numpy()
            t, p = nw_tstat(d)
            dm[cand][L] = {"vs": best[L], "mean_diff": float(d.mean()), "t": t, "p_one_sided": p}
    ok = [c for c in ("M2", "E") if all(dm[c][L]["p_one_sided"] < 0.05 and dm[c][L]["mean_diff"] < 0 for L in dm[c])]
    pick = min(ok, key=lambda c: res[c]["mse"]) if ok else "B3"
    return {"loss": res, "best_baseline": best, "dm": dm, "passed": ok, "picked": pick}


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
    start = pd.Timestamp("2022-01-01")
    ens = ens[ens.index.get_level_values("as_of") >= start]
    sig = np.exp(fc[pick])                                          # 预测的年化波动
    q = fc[pick].groupby(level="as_of").transform(lambda s: s.rank(pct=True))
    drop10 = set(q[q > 0.9].index)
    cols = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "vol", "amount", "adj_factor"]
    prices = pd.read_parquet(D / "raw/alpha_prices_panel_v6_20261009.parquet", columns=cols)
    amihud = pd.read_parquet(D / "panel/alpha_panel_weekly_v6_1.parquet", columns=["amihud_illiquidity"])
    end = pd.Timestamp("2026-10-09")
    mk = Market(prices, start, end)

    def inv_vol(d, target):
        s = sig.reindex(pd.MultiIndex.from_arrays([[d] * len(target), target])).to_numpy()
        s = np.where(np.isfinite(s), s, np.nanmedian(s) if np.isfinite(s).any() else 1.0)
        w = (1 / s) / np.sum(1 / s)
        eq = 1 / len(target)
        w = np.clip(w, 0.5 * eq, 2 * eq)
        return dict(zip(target, w / w.sum(), strict=True))
    base = dict(aum=1.0, top_n=300, rebalance_step=4, universe_size=1500, buffer_mult=None, drift_band=0.0,
                commission_rate=3e-4, min_commission=0.0, transfer_rate=2e-5, tiers=SlippageTiers(), participation=None,
                mode="adj", dividend_tax=False)
    arms = {"R": (ens, None), "V10": (ens[[i not in drop10 for i in ens.index]], None), "W": (ens, inv_vol)}
    navs = {}
    for name, (s, wf) in arms.items():
        r = AccountSimulator(s.rename("score").reset_index(), mk, SimConfig(**base, weight_fn=wf), start=start, end=end,
                             amihud=amihud).run()
        navs[name] = r["nav"]
        r["nav"].to_parquet(out / f"v2_nav_{name}.parquet")
        print(name, "done", round(float(r["nav"]["equity"].iloc[-1]), 4), flush=True)
    bench = navs["R"]["bench_pool_ew"]
    res = {}
    for k in ("V10", "W"):
        j = pd.DataFrame({"a": navs[k]["equity"].pct_change(), "r": navs["R"]["equity"].pct_change(),
                          "b": bench.pct_change()}).dropna()
        ra, rr, rb = j["a"].to_numpy(), j["r"].to_numpy(), j["b"].to_numpy()
        bi = boot_idx(len(j), 20, 10_000, 0)
        d = ra - rr
        bt = d[bi].mean(axis=1) * 252
        bdd = mrwd(ra[bi], rb[bi]) - mrwd(rr[bi], rb[bi])
        bvol = (ra[bi].std(axis=1) - rr[bi].std(axis=1)) * A
        o = {"n_days": int(len(j)), "ann_excess_diff": float(d.mean() * 252),
             "ci95": [float(np.quantile(bt, .025)), float(np.quantile(bt, .975))],
             "mrwd_diff_ci95": [float(np.quantile(bdd, .025)), float(np.quantile(bdd, .975))],
             "vol_diff": float((ra.std() - rr.std()) * A),
             "vol_diff_ci95": [float(np.quantile(bvol, .025)), float(np.quantile(bvol, .975))],
             "maxdd": {k: float((navs[k]["equity"] / navs[k]["equity"].cummax() - 1).min()),
                       "R": float((navs["R"]["equity"] / navs["R"]["equity"].cummax() - 1).min())}}
        o["noninferior"] = bool(o["ci95"][0] > -0.01 and o["mrwd_diff_ci95"][1] < 0.02)
        o["risk_improved"] = bool(o["vol_diff_ci95"][1] < 0)
        res[f"{k}-R"] = o
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    cal, grid = calendar_and_grid(a.root)
    fc1 = pd.read_parquet(a.out / "forecasts.parquet")
    df = build(a.root, list(grid))
    fc = forecast_v2(df, fc1, pd.DatetimeIndex(cal))
    fc.to_parquet(a.out / "forecasts_v2.parquet")
    j = judge(fc)
    print(json.dumps(j, ensure_ascii=False, indent=1), flush=True)
    pf = portfolio(a.root, fc, j["picked"], a.out)
    (a.out / "x3_v2_result.json").write_text(json.dumps({"forecast": j, "portfolio": pf}, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    print(json.dumps(pf, ensure_ascii=False, indent=1))
    _ = LAST_DATE


if __name__ == "__main__":
    main()
