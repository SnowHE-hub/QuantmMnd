"""P63-2b：基本面因子筛选（复用 short_horizon 同套口径，63d 标签，UNKNOWN 桶）.

口径：
- 选因子用 in-sample（as_of<2022-01）neut IC/ICIR；OOS（>=2022-01）留无偏确认。
- 中性化：fac 对 [1, z(logcap), 行业哑变量(含 UNKNOWN 桶)] 残差，残差 vs forward_return_63d 的 spearman。
- 保留线：|in-sample neut ICIR| >= 0.2 且 翻转率<35% 且 与已留因子 max|corr|<0.7（贪心去冗余）。
产物：data/loss_signals_v4/fundamental_screen_v6.csv + .../fundamental_survivors_v6.json
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))

FUND = REPO / "data/panel/fundamental_factors_v6.parquet"
V6_PANEL = REPO / "data/panel/alpha_panel_weekly_v6.parquet"
DB = REPO / "data/lake/daily_basic.parquet"
OUT_DIR = REPO / "data/loss_signals_v4"; OUT_DIR.mkdir(parents=True, exist_ok=True)
LABEL = "forward_return_63d"
OOS_START = pd.Timestamp("2022-01-01")
ICIR_MIN, FLIP_MAX, CORR_MAX = 0.2, 0.35, 0.7


def _spear(a, b):
    d = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    return float(stats.spearmanr(d["a"], d["b"])[0]) if len(d) >= 30 else np.nan


def _neut_ic_xs(fac, y, ind, lc):
    df = pd.DataFrame({"f": fac, "y": y, "ind": ind, "lc": lc}).copy()
    df["ind"] = df["ind"].astype("object").where(df["ind"].notna(), "UNKNOWN")
    df = df.dropna(subset=["f", "y", "lc"])
    if len(df) < 50 or df["ind"].nunique() < 2:
        return np.nan
    D = pd.get_dummies(df["ind"], drop_first=True).astype(float)
    z = (df["lc"] - df["lc"].mean()) / (df["lc"].std() + 1e-12)
    X = np.column_stack([np.ones(len(df)), z.values, D.values])
    beta, *_ = np.linalg.lstsq(X, df["f"].values.astype(float), rcond=None)
    resid = pd.Series(df["f"].values - X @ beta, index=df.index)
    return _spear(resid, df["y"])


def main():
    t0 = time.time()
    fund = pd.read_parquet(FUND)
    lab = pd.read_parquet(V6_PANEL, columns=[LABEL, "exposure_industry"])
    db = pd.read_parquet(DB, columns=["ts_code", "trade_date", "circ_mv"])
    db["trade_date"] = pd.to_datetime(db["trade_date"]).dt.normalize()
    db = db[db.circ_mv > 0]; db["lc"] = np.log(db.circ_mv.astype(float))
    lc_by = {pd.Timestamp(d): s.set_index(s.ts_code.astype(str))["lc"] for d, s in db.groupby("trade_date")}

    df = fund.join(lab, how="left")
    factors = list(fund.columns)
    as_ofs = sorted(df.index.get_level_values("as_of").unique())
    print(f"[screen] {len(factors)} 因子, {len(as_ofs)} as_of ({time.time()-t0:.0f}s)", flush=True)

    # 逐 as_of 逐因子 neut IC
    per = {f: [] for f in factors}
    for a in as_ofs:
        sub = df.xs(a, level="as_of")
        lc = lc_by.get(pd.Timestamp(a).normalize())
        if lc is None:
            continue
        lc_a = lc.reindex(sub.index)
        y = sub[LABEL]; ind = sub["exposure_industry"]
        for f in factors:
            per[f].append((a, _neut_ic_xs(sub[f], y, ind, lc_a)))

    rows = []
    series = {}
    for f in factors:
        s = pd.Series({a: v for a, v in per[f]}).dropna()
        series[f] = s
        ins = s[s.index < OOS_START]; oos = s[s.index >= OOS_START]
        ic_in = ins.mean(); icir_in = ins.mean() / ins.std() if ins.std() else np.nan
        flip = float((np.sign(s) != np.sign(s.shift())).mean())  # 翻转率
        rows.append({"factor": f, "n_in": len(ins), "neut_ic_in": ic_in,
                     "neut_icir_in": icir_in, "neut_ic_oos": oos.mean(),
                     "flip": flip, "abs_icir_in": abs(icir_in)})
    screen = pd.DataFrame(rows).sort_values("abs_icir_in", ascending=False)

    # 贪心去冗余：按 |ICIR| 降序，过 ICIR/flip 门槛，且与已留 max|corr|<0.7
    fac_panel = fund[factors]
    survivors = []
    for _, r in screen.iterrows():
        f = r["factor"]
        if not (r["abs_icir_in"] >= ICIR_MIN and r["flip"] < FLIP_MAX):
            continue
        ok = True
        for s in survivors:
            c = fac_panel[[f, s]].dropna()
            if len(c) > 100 and abs(c[f].corr(c[s])) >= CORR_MAX:
                ok = False; break
        if ok:
            survivors.append(f)
    screen["survivor"] = screen["factor"].isin(survivors)
    screen.to_csv(OUT_DIR / "fundamental_screen_v6.csv", index=False)
    (OUT_DIR / "fundamental_survivors_v6.json").write_text(json.dumps(
        {"survivors": survivors, "label": LABEL, "criteria": {
            "abs_neut_icir_in>=": ICIR_MIN, "flip<": FLIP_MAX, "max_corr<": CORR_MAX}},
        ensure_ascii=False, indent=2))
    print(f"\n=== P63-2 筛选结果（63d, in-sample neut ICIR）===")
    cols = ["factor", "neut_ic_in", "neut_icir_in", "neut_ic_oos", "flip", "survivor"]
    print(screen[cols].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(f"\n=> survivors ({len(survivors)}): {survivors}")
    print(f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
