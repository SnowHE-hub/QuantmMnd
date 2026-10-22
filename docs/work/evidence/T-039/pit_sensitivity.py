"""T-039：时点版 vs 冻结版基本面因子的敏感性分析（登记见 T-039「敏感性分析登记」，口径以那里为准）.

同一评估窗口：T-030 的可评估行（20d：frozen_eval_rows_20d，222 日；63d：frozen_eval_rows_63d，189 日），标签 y20 / y63。
冻结版 = data/gpu_v1/panel/fnd；时点版 = data/research/pit_audit/fnd_evaldates/pit（build_fnd_pit.py --mode pit）。
因子：F2 的 27 列；其中 16 列是 Ridge C 的基本面因子（fundamental_survivors_v6.json，冻结值与 F2 逐行相同）。
1. 逐因子 RankIC：每日在可评估行里、两版都有值的行上算 Spearman（≥30 行），d_t = IC_时点 − IC_冻结；
   移动块自助（块长 13、26，10,000 次，种子 0）。另报各版在自己有值的行上的 IC（次要）。
2. 合成因子：F2_27、RidgeC_16。方向 = 冻结版在本窗口的 IC 均值符号（两版共用）；每日在 meta 全部股票上
   逐因子 gauss-rank × 方向，再对有值的因子取平均（rank_average）。IC 同第 1 条。
3. 净值：ExecutableNavEngine，配置与 eval_primary.py 相同（20d：持有 20、步长 4；63d：63、13；top 300 / 1500 池），
   分数 = 合成因子（主）或单因子 × 方向（次）。配对日收益差 时点 − 冻结，年化；块自助块长 = (20, 40) / (65, 130) 个交易日。
4. 判定（每个比较）：
   差异明显：两个块长的 95% 区间都不含 0，且 |均值| ≥ 门槛；
   可视为相同：两个块长的 90% 区间都落在 (−门槛, +门槛) 内；
   其余：不确定。门槛：IC 0.005；净值年化 1.0 个百分点。
结果写 JSON，终端只打印完成与 sha256。

用法：PYTHONPATH=. python docs/work/evidence/T-039/pit_sensitivity.py <data_root> <20d|63d> <out.json> [--no-factor-nav]
      PYTHONPATH=. python docs/work/evidence/T-039/pit_sensitivity.py <data_root> 20d <out.json> --self-test
      （自测：两版都用冻结面板、标签按日打乱；只报 d_t 与净值差是否全为 0，不打印任何水平值）
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pds

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[4]))
from quantmind.execution.nav_engine import ExecutableNavEngine, NavConfig  # noqa: E402
from quantmind.research.gpu.ensemble import rank_average  # noqa: E402
from quantmind.research.gpu.metrics import moving_block_bootstrap, two_sided_ci  # noqa: E402

_spec = importlib.util.spec_from_file_location("eval_primary", HERE.parents[1] / "T-030/eval_primary.py")
EP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EP)

P6 = HERE.parents[1] / "T-030/p6"
H = {"20d": ("y20", "frozen_eval_rows_20d.parquet", (20, 4, 20)),
     "63d": ("y63", "frozen_eval_rows_63d.parquet", (63, 13, 65))}
IC_MARGIN, NAV_MARGIN = 0.005, 0.01
N_BOOT = 10_000


def panel(dirpath: Path, dates, cols=None) -> pd.DataFrame:
    t = pds.dataset(sorted(str(p) for p in dirpath.glob("*.parquet")), format="parquet").to_table(
        filter=pds.field("as_of").isin(pd.DatetimeIndex(dates).as_unit("ns").to_numpy()),
        columns=None if cols is None else ["as_of", "ticker", *cols]).to_pandas()
    if "as_of" not in t.columns:
        t = t.reset_index()
    return t.set_index(["as_of", "ticker"]).sort_index()


def verdict(d: np.ndarray, blocks: tuple[int, int], margin: float, scale: float = 1.0) -> dict:
    out = {"n": int(len(d)), "mean": float(d.mean() * scale)}
    ex0, inside = [], []
    for b in blocks:
        bt = moving_block_bootstrap(d, b, N_BOOT, 0) * scale
        c95, c90 = two_sided_ci(bt, 0.95), two_sided_ci(bt, 0.90)
        out[f"block{b}"] = {"ci95": c95, "ci90": c90}
        ex0.append(c95[0] > 0 or c95[1] < 0)
        inside.append(c90[0] > -margin and c90[1] < margin)
    out["verdict"] = ("差异明显" if all(ex0) and abs(out["mean"]) >= margin else
                      "可视为相同" if all(inside) else "不确定")
    return out


def compare_ic(fz: pd.Series, pt: pd.Series, y: pd.Series) -> dict:
    j = pd.DataFrame({"f": fz, "p": pt, "y": y}).dropna()
    a, b = EP.ic_by_date(j, "f"), EP.ic_by_date(j, "p")
    d = (b - a).dropna().sort_index()
    own_f = EP.ic_by_date(pd.DataFrame({"s": fz, "y": y}).dropna(), "s")
    own_p = EP.ic_by_date(pd.DataFrame({"s": pt, "y": y}).dropna(), "s")
    return {"rows_common": int(len(j)), "ic_frozen": float(a.reindex(d.index).mean()),
            "ic_pit": float(b.reindex(d.index).mean()), "ic_frozen_own_rows": float(own_f.mean()),
            "ic_pit_own_rows": float(own_p.mean()), "delta": verdict(d.to_numpy(), (13, 26), IC_MARGIN)}


def composite(df: pd.DataFrame, cols: list[str], sign: dict) -> pd.Series:
    df = df.sort_index()
    day = pd.factorize(df.index.get_level_values("as_of"), sort=True)[0]
    s = rank_average([df[c].to_numpy(dtype=np.float64) * sign[c] for c in cols], day)
    return pd.Series(s, index=df.index, name="score")


def run_nav(score: pd.Series, prices, amihud, nav_cfg) -> pd.DataFrame:
    hd, step, _ = nav_cfg
    s = score.dropna().rename("score").reset_index()
    r = ExecutableNavEngine(s, prices, NavConfig(horizon_td=hd, rebalance_step=step, top_n=300, universe_size=1500),
                            amihud=amihud).run()
    return r["nav_daily"].set_index("date")[["nav_net", "nav_bench"]]


def compare_nav(sf: pd.Series, sp: pd.Series, prices, amihud, nav_cfg) -> dict:
    nf, np_ = run_nav(sf, prices, amihud, nav_cfg), run_nav(sp, prices, amihud, nav_cfg)
    start, end = max(nf.index[0], np_.index[0]), min(nf.index[-1], np_.index[-1])
    nf, np_ = nf[(nf.index >= start) & (nf.index <= end)], np_[(np_.index >= start) & (np_.index <= end)]
    x, y = np_["nav_net"].pct_change().dropna().align(nf["nav_net"].pct_change().dropna(), join="inner")
    blk = nav_cfg[2]
    return {"window": [str(start.date()), str(end.date())], "frozen": EP.nav_metrics(nf), "pit": EP.nav_metrics(np_),
            "paired_ann_diff": verdict((x - y).to_numpy(), (blk, 2 * blk), NAV_MARGIN, scale=252.0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("horizon", choices=list(H))
    ap.add_argument("out", type=Path)
    ap.add_argument("--no-factor-nav", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--pit-subdir", default="pit", help="fnd_evaldates 下的时点版目录：pit（v1a，主）或 pit_v1（次要）")
    a = ap.parse_args()
    D = a.root / "data"
    ycol, ev_file, nav_cfg = H[a.horizon]
    ev = pd.read_parquet(P6 / ev_file).set_index(["as_of", "ticker"]).index
    dates = sorted(ev.get_level_values("as_of").unique())
    surv = json.loads((D / "loss_signals_v4/fundamental_survivors_v6.json").read_text())["survivors"]
    meta = pd.concat([pd.read_parquet(p, columns=["as_of", "ticker", ycol]) for p in
                      sorted((D / "gpu_v1/model/v1").glob("*.meta.parquet"))], ignore_index=True)
    meta["as_of"] = pd.to_datetime(meta["as_of"])
    meta = meta[meta.as_of.isin(set(dates))].set_index(["as_of", "ticker"]).sort_index()
    fz = panel(D / "gpu_v1/panel/fnd", dates)
    cols = list(fz.columns)
    pt = fz.copy() if a.self_test else panel(D / "research/pit_audit/fnd_evaldates" / a.pit_subdir, dates, cols)
    U = meta.index
    fz, pt = fz.reindex(U), pt.reindex(U)
    y = meta[ycol].reindex(ev)
    if a.self_test:   # 标签按日打乱：自测不接触真实 IC
        rng = np.random.default_rng(0)
        y = y.groupby(level="as_of").transform(lambda s: pd.Series(rng.permutation(s.to_numpy()), index=s.index))
    res = {"horizon": a.horizon, "self_test": a.self_test, "pit_subdir": a.pit_subdir, "n_dates": len(dates),
           "n_eval_rows": int(len(ev)),
           "n_meta_rows": int(len(U)), "factors": {}, "composites": {}}

    sign = {}
    for c in cols:
        ic = EP.ic_by_date(pd.DataFrame({"s": fz[c].reindex(ev), "y": y}).dropna(), "s")
        sign[c] = 1.0 if ic.mean() >= 0 else -1.0
    res["direction"] = sign
    for c in cols:
        f, p = fz[c].reindex(ev), pt[c].reindex(ev)
        both = f.notna() & p.notna()
        r = {"in_ridge_c": c in surv,
             "share_rows_value_differs": float((both & ~np.isclose(f, p, rtol=0, atol=0, equal_nan=True)).mean()),
             "share_rows_nan_status_differs": float((f.notna() != p.notna()).mean()),
             **compare_ic(f, p, y)}
        res["factors"][c] = r

    amihud = pd.read_parquet(D / "panel/alpha_panel_weekly_v6_1.parquet", columns=["amihud_illiquidity"])
    pcols = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "adj_factor", "amount"]
    prices = pd.read_parquet(D / "raw/alpha_prices_panel_v6_20261009.parquet", columns=pcols)
    for name, cs in (("F2_27", cols), ("RidgeC_16", [c for c in cols if c in surv])):
        cf, cp = composite(fz[cs], cs, sign), composite(pt[cs], cs, sign)
        res["composites"][name] = {"n_factors": len(cs), **compare_ic(cf.reindex(ev), cp.reindex(ev), y),
                                   "nav": compare_nav(cf, cp, prices, amihud, nav_cfg)}
        if a.self_test:
            break
    if not a.no_factor_nav and not a.self_test:
        for c in cols:
            res["factors"][c]["nav"] = compare_nav(fz[c] * sign[c], pt[c] * sign[c], prices, amihud, nav_cfg)

    prim = {k: {"ic": v["delta"]["verdict"], "nav": v["nav"]["paired_ann_diff"]["verdict"]}
            for k, v in res["composites"].items()}
    res["primary_verdicts"] = prim
    if a.self_test:   # 自测不写文件（里面会有冻结版的净值水平），只报差是否全为 0
        d_all = [v["delta"]["mean"] for v in res["factors"].values()] + \
                [v["delta"]["mean"] for v in res["composites"].values()]
        nav_d = [v["nav"]["paired_ann_diff"]["mean"] for v in res["composites"].values()]
        ok = all(x == 0 for x in d_all) and all(x == 0 for x in nav_d)
        print(f"[self-test] {'PASS' if ok else 'FAIL'}：{len(d_all)} 个 IC 比较、{len(nav_d)} 个净值比较的差全为 0")
        return 0 if ok else 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"[pit_sensitivity {a.horizon}] done {a.out} sha256 {hashlib.sha256(a.out.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
