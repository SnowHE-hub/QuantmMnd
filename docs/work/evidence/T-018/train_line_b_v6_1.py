"""T-018：主线 B 两颗 Ridge 种子在 v6_1（完整交易日历、价格到 2026-10-09）上重训（quantmind env；研究产物，不改生产）.

种子（模型与特征逐字沿用原脚本，只换数据）：
  12d  = scripts/survivorship/p4c_train_v6.py：feats = 35 + 16 short-horizon survivors + Alpha158；h=12, emb=20；季度切点
  63d  = scripts/survivorship/p63_3_train_v6.py 的 full_fnd：35 + 16 + 16 基本面 survivors + Alpha158；h=63, emb=63；半年切点（4/30、10/31）
  Ridge(α=10)、PurgedWalkForwardSplit(rolling 756, n_val=2)、oos_start=2022-01-01、方向只由验证集决定（decide_direction）、score *= d
切分用**完整交易日历**（原脚本用缺 18 天的 lake 日历）。

预先登记的方案（运行前写定）：
  R  复刻：原切点区间（12d 2022-01→2025-10；63d 2022-04→2025-10），测试窗口 (C_k+E, C_{k+1}]（最后一折到 OOS 末）
  X  延长：切点延长到最新（12d 到 2026-07-01；63d 到 2026-04-30），测试窗口同 R
  C  连续样本外：切点同 X，测试窗口 (C_k+E, C_{k+1}+E]——每个测试日距所用模型的切点仍 > E，但时间轴连续覆盖
  P  按板块分别训练（仅 63d，回答"板块模型"问题）：折同 C，主板 / 创业板 / 科创板各自训练 Ridge
输出 <out_dir>/ridge_<seed>_v6_1_<protocol>.parquet（as_of, ticker, score）+ .dirs.json；已存在则拒绝。
用法：PYTHONPATH=. python train_line_b_v6_1.py <data_root> <seed: 12d|63d> <protocol: R|X|C|P> <out_dir> [--inf-to-nan]
--inf-to-nan（2026-10-11 出 P 结果前增加，见卡片"出结果前的修正"）：特征里的 ±inf 视为缺失；输出文件名加 _infnan。
  原因：两个短周期因子各有几十个 ±inf；统一训练时它们让整列标准化变成 NaN、被填 0（该因子实际被弃用），
  按板块训练时某板块训练集没有 inf、测试截面有 inf，打分报错。板块判读用同一开关下的 P 与 C 比较。
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quantmind.backtest.wf_costs import board_of
from quantmind.backtest.wf_gate import decide_direction
from quantmind.backtest.wf_split import PurgedWalkForwardSplit
from quantmind.features.weekly_panel import WHITELIST_35, merge_increment
from scripts.bakeoff.p3b_tabular import RidgePredictor, make_quarterly_cutoffs
from scripts.survivorship.p63_3_train_v6 import make_semiannual_cutoffs

SEEDS = {
    "12d": {"label": "forward_return_12d", "H": 12, "E": 20, "fnd": False,
            "cuts": lambda cal, end: make_quarterly_cutoffs(cal, pd.Timestamp("2022-01-01"), pd.Timestamp(end)),
            "end_R": "2025-10-01", "end_X": "2026-07-01"},
    "63d": {"label": "forward_return_63d", "H": 63, "E": 63, "fnd": True,
            "cuts": lambda cal, end: make_semiannual_cutoffs(cal, "2022-04-01", end),
            "end_R": "2025-10-31", "end_X": "2026-04-30"},
}
OOS_START = pd.Timestamp("2022-01-01")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while b := fh.read(1 << 22):
            h.update(b)
    return h.hexdigest()


def build(root: Path, seed: str) -> tuple[pd.DataFrame, list[str], dict]:
    D = root / "data"
    paths = {"panel": D / "panel/alpha_panel_weekly_v6_1.parquet", "a158": D / "bakeoff/alpha158_asof_v6_1.parquet"}
    base = pd.read_parquet(paths["panel"])
    sh = [c for c in json.loads((D / "loss_signals_v4/short_horizon_survivors.json").read_text())["survivors"]
          if c in base.columns]
    feats = list(WHITELIST_35) + sh
    panel = base
    if SEEDS[seed]["fnd"]:
        paths["fnd"] = D / "panel/fundamental_factors_v6_1.parquet"
        surv = json.loads((D / "loss_signals_v4/fundamental_survivors_v6.json").read_text())["survivors"]
        fnd = pd.read_parquet(paths["fnd"])[surv].reindex(base.index)
        fnd.columns = [f"fnd_{c}" for c in fnd.columns]
        panel = merge_increment(panel, fnd)
        feats += list(fnd.columns)
    a = pd.read_parquet(paths["a158"]).reindex(base.index)
    a = a.loc[:, a.notna().any()]
    panel = merge_increment(panel, a)
    feats += list(a.columns)
    return panel, feats, {k: {"path": str(p), "sha256": sha(p)} for k, p in paths.items()}


def calendar(root: Path) -> pd.DatetimeIndex:
    D = root / "data"
    px = pd.to_datetime(pd.read_parquet(D / "raw/alpha_prices_panel_v6_20261009.parquet",
                                        columns=["trade_date"])["trade_date"]).drop_duplicates()
    cal = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(D / "lake/trade_cal_sse.parquet")["cal_date"])).sort_values()
    cal = cal[(cal >= px.min()) & (cal <= px.max())]
    assert set(cal) == set(px), "price dates must equal the trade calendar"
    return cal


def make_folds(sp: PurgedWalkForwardSplit, as_of: list, cuts: list, protocol: str) -> list:
    oos_end = as_of[-1]
    if protocol in ("R", "X"):
        return sp.make_folds(as_of, cuts, oos_start=OOS_START, oos_end=oos_end)
    folds = []
    for k, c in enumerate(cuts):
        if k + 1 < len(cuts):
            j = sp.idx(cuts[k + 1]) + sp.E
            hi = sp.cal[min(j, len(sp.cal) - 1)]
        else:
            hi = oos_end
        folds.append(sp.split(as_of, c, oos_start=OOS_START, oos_end=hi))
    return folds


def fit_score(panel, feats, label, fold) -> tuple[list[pd.DataFrame], int]:
    pred = RidgePredictor(alpha=10.0).fit(panel, fold.train_dates, fold.val_dates, feats, label)
    d = decide_direction(pred, panel, fold.val_dates, feats, label)
    rows = []
    for dt in fold.test_dates:
        dt = pd.Timestamp(dt)
        try:
            xs = panel.xs(dt, level="as_of")
        except KeyError:
            continue
        s = pred.score(panel, dt, feats) * d
        rows.append(pd.DataFrame({"as_of": dt, "ticker": xs.index, "score": s.values}).dropna(subset=["score"]))
    return rows, int(d)


def main() -> None:
    root, seed, protocol, out_dir = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
    inf_to_nan = "--inf-to-nan" in sys.argv[5:]
    cfg = SEEDS[seed]
    if protocol == "P" and seed != "63d":
        raise SystemExit("protocol P is only registered for the 63d seed")
    suffix = "_infnan" if inf_to_nan else ""
    out = out_dir / f"ridge_{seed}_v6_1_{protocol}{suffix}.parquet"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    t0 = time.time()
    panel, feats, inputs = build(root, seed)
    if inf_to_nan:
        num = [f for f in feats if f not in ("exposure_industry", "exposure_area")]
        n_inf = int(np.isinf(panel[num].to_numpy(dtype=float)).sum())
        panel[num] = panel[num].replace([np.inf, -np.inf], np.nan)
        inputs["inf_to_nan"] = {"n_values_replaced": n_inf}
    cal = calendar(root)
    as_of = sorted(panel.index.get_level_values("as_of").unique())
    sp = PurgedWalkForwardSplit(cal, horizon=cfg["H"], embargo=cfg["E"], mode="rolling",
                                rolling_lookback_td=756, n_val=2)
    cuts = cfg["cuts"](cal, cfg["end_R"] if protocol == "R" else cfg["end_X"])
    folds = make_folds(sp, as_of, cuts, protocol)
    print(f"[{seed}/{protocol}] feats={len(feats)} cutoffs={len(cuts)} {cuts[0].date()}→{cuts[-1].date()} "
          f"folds={len(folds)} panel={panel.shape}", flush=True)
    rows, dirs, fold_log = [], {}, []
    for i, fold in enumerate(folds):
        if not fold.test_dates or not fold.train_dates:
            continue
        PurgedWalkForwardSplit.assert_no_leakage(fold)
        if protocol == "P":
            boards = pd.Series([board_of(t) for t in panel.index.get_level_values("ticker")], index=panel.index)
            for b in ("MAIN", "CHINEXT", "STAR"):
                sub = panel[(boards == b).values]
                r, d = fit_score(sub, feats, cfg["label"], fold)
                rows += r
                dirs[f"{i}_{b}"] = d
        else:
            r, d = fit_score(panel, feats, cfg["label"], fold)
            rows += r
            dirs[str(i)] = d
        fold_log.append({"cutoff": str(fold.cutoff.date()), "train": [str(fold.train_dates[0].date()),
                         str(fold.train_dates[-1].date())], "test": [str(pd.Timestamp(fold.test_dates[0]).date()),
                         str(pd.Timestamp(fold.test_dates[-1]).date())], "n_test": len(fold.test_dates)})
        print(f"  fold {i} cut={fold.cutoff.date()} test {fold_log[-1]['test']} ({time.time() - t0:.0f}s)", flush=True)
    preds = pd.concat(rows, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out, index=False)
    out.with_suffix(".dirs.json").write_text(json.dumps(
        {"seed": seed, "protocol": protocol, "dirs": dirs, "n_flips": sum(1 for v in dirs.values() if v < 0),
         "n_feats": len(feats), "folds": fold_log, "inputs": inputs, "calendar": "trade_cal_sse (complete)",
         "train_seconds": round(time.time() - t0, 1)}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{seed}/{protocol}] saved {out} {preds.shape}; as_of {preds.as_of.min().date()}→{preds.as_of.max().date()} "
          f"({preds.as_of.nunique()} dates); flips {sum(1 for v in dirs.values() if v < 0)}/{len(dirs)}", flush=True)


if __name__ == "__main__":
    main()
