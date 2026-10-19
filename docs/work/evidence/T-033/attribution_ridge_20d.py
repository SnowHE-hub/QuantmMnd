"""T-033：20d 增益归因（探索性）。同一套 Ridge C 配方，一次只换一样东西：特征变换 / 特征集 / 训练行.

设计见 docs/work/tasks/T-033.md（运行前提交）。折、切点、方向判定逐项照抄 T-028 train_event_aug.py 的 20d BASE：
PurgedWalkForwardSplit(H=20, E=20, rolling 756, n_val=2)、季度切点 2022-01-04 → 2026-07-01、方案 C；
Ridge α=10：训练行（标签非缺失）上 z-score（ddof=1，sd=0 → 1；±inf 视为缺失），标准化后缺失填 0；方向 = 验证段 IC 均值的符号。
Ridge 用闭式解 (Z'Z + αI) w = Z'y（与 sklearn Ridge(fit_intercept=True) 相同的中心化），分块累加，承受日频行。
  A0   Ridge C 原 223 个数值特征，原始值，v6_1 周频行        → 必须逐值复现已登记的 ridge_20d_v6_1_BASE（≤ 1e-6）
  A0r  同 A0，每列先做当日截面 gauss-rank                     → 只换截面变换
  A1   T-030 特征集（250 股票列 gauss-rank + 26 市场列），v6_1 周频行 → 换特征集
  A2   同 A1，训练行为训练窗内全部交易日（gpu_v1 日频行，标签 y20）  → 线性模型多用日频数据
评估：T-030 冻结的 20d 可评估行（323,783 行、222 日），eval_primary.paired（块长 13 / 26，10,000 次，种子 0）。
输出：data/gpu_v1/attrib/ridge_20d_<臂>.parquet；docs/work/evidence/T-033/attribution_20d.json
用法：PYTHONPATH=. python docs/work/evidence/T-033/attribution_ridge_20d.py <data_root>
"""
from __future__ import annotations

import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EV = HERE.parent
ALPHA = 10.0
CHUNK = 200_000


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class Ridge:
    """RidgePredictor 的同口径实现：fit 接收训练行的 (X, y) 块迭代器工厂，三遍扫描（均值 → 标准差 → Z'Z）。"""

    def fit(self, chunks) -> Ridge:
        s = c = None
        for X, _y in chunks():
            X = np.where(np.isfinite(X), X, np.nan)
            if s is None:
                p = X.shape[1]
                s, c = np.zeros(p), np.zeros(p)
            s += np.nansum(X, axis=0)
            c += np.isfinite(X).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.mu = s / c                                    # 全缺失列 → NaN（与 pandas 相同），标准化后填 0
        ss = np.zeros_like(s)
        for X, _y in chunks():
            X = np.where(np.isfinite(X), X, np.nan)
            ss += np.nansum((X - self.mu) ** 2, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            sd = np.sqrt(ss / (c - 1))
        self.sd = np.where(sd == 0, 1.0, sd)                   # pandas .replace(0, 1.0)；NaN 保持 NaN
        p = len(s)
        ztz, zty, zs, ys, n = np.zeros((p, p)), np.zeros(p), np.zeros(p), 0.0, 0
        for X, y in chunks():
            Z = self._z(X)
            ztz += Z.T @ Z
            zty += Z.T @ y
            zs += Z.sum(axis=0)
            ys += float(y.sum())
            n += len(y)
        zm, ym = zs / n, ys / n                                # sklearn：先按样本均值中心化
        A = ztz - n * np.outer(zm, zm) + ALPHA * np.eye(p)
        self.w = np.linalg.solve(A, zty - n * zm * ym)
        self.b = ym - zm @ self.w
        return self

    def _z(self, X: np.ndarray) -> np.ndarray:
        X = np.where(np.isfinite(X), X, np.nan)
        with np.errstate(invalid="ignore"):
            Z = (X - self.mu) / self.sd
        return np.nan_to_num(Z, nan=0.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._z(X) @ self.w + self.b


def _direction(scores_by_date: list[tuple[np.ndarray, np.ndarray]]) -> int:
    from quantmind.backtest.wf_metrics import cross_sectional_ic
    ics = [cross_sectional_ic(pd.Series(s), pd.Series(y)) for s, y in scores_by_date]
    ics = [v for v in ics if v == v]
    return 1 if (not ics or float(np.mean(ics)) >= 0) else -1


def main() -> None:
    root = Path(sys.argv[1])
    t0 = time.time()
    log = lambda m: print(f"[t033 {time.time() - t0:.0f}s] {m}", flush=True)  # noqa: E731
    t18 = _load(EV / "T-018/train_line_b_v6_1.py", "t18train")
    ep = _load(EV / "T-030/eval_primary.py", "eval_primary")
    from quantmind.backtest.wf_split import PurgedWalkForwardSplit
    from quantmind.research.gpu.data import ModelData
    from quantmind.research.gpu.ensemble import rank_average_frames
    from quantmind.research.gpu.folds import calendar_and_grid, make_folds
    from quantmind.research.gpu.transforms import gauss_rank_by_day
    from scripts.bakeoff.p3b_tabular import make_quarterly_cutoffs

    out_dir = root / "data/gpu_v1/attrib"
    out_dir.mkdir(parents=True, exist_ok=True)
    # ---------- v6_1 周频面板与 Ridge C 的特征（与 T-028 20d BASE 相同） ----------
    panel, feats, _ = t18.build(root, "63d")
    lab = pd.read_parquet(root / "data/panel/labels_extra_v6_1.parquet").reindex(panel.index)["forward_return_20d"]
    num = [f for f in feats if f not in ("exposure_industry", "exposure_area")]
    panel = panel.sort_index(level=["as_of", "ticker"])
    idx = panel.index
    asof = pd.to_datetime(idx.get_level_values("as_of"))
    X0 = panel[num].to_numpy(dtype=np.float64)                 # 原始值保持 float64：与 RidgePredictor 同精度
    y = lab.reindex(idx).to_numpy(dtype=np.float64)
    del panel
    gc.collect()
    log(f"v6_1 panel rows={len(idx)} feats={len(num)}")
    cal = t18.calendar(root)
    sp = PurgedWalkForwardSplit(cal, horizon=20, embargo=20, mode="rolling", rolling_lookback_td=756, n_val=2)
    cuts = make_quarterly_cutoffs(cal, pd.Timestamp("2022-01-01"), pd.Timestamp("2026-07-01"))
    as_list = sorted(asof.unique())
    folds = [f for f in t18.make_folds(sp, as_list, cuts, "C") if f.test_dates and f.train_dates]
    day_codes = pd.factorize(asof, sort=True)[0]
    # ---------- A1 / A2 用的 T-030 特征（对齐到 v6_1 行） ----------
    md = ModelData(root)
    wk = pd.DatetimeIndex(sorted(asof.unique()))
    blk = md.load(wk)
    mkt = md.market_rows(blk)
    key = pd.MultiIndex.from_arrays([pd.to_datetime(blk.meta["as_of"]), blk.meta["ticker"]])
    pos = pd.Series(np.arange(len(key)), index=key).reindex(idx).to_numpy()
    ok_pos = np.isfinite(pos)
    X1 = np.full((len(idx), blk.X.shape[1] + mkt.shape[1]), np.nan, dtype=np.float32)
    X1[ok_pos, :blk.X.shape[1]] = blk.X[pos[ok_pos].astype(int)]
    X1[ok_pos, blk.X.shape[1]:] = mkt[pos[ok_pos].astype(int)]
    del blk, mkt
    gc.collect()
    log(f"T-030 features aligned: {X1.shape}, v6_1 rows missing in gpu_v1: {int((~ok_pos).sum())}")

    def weekly_arm(name: str, X: np.ndarray) -> pd.DataFrame:
        rows, dirs = [], []
        for f in folds:
            tr = np.isin(asof, pd.DatetimeIndex(f.train_dates)) & np.isfinite(y)
            Xtr, ytr = X[tr], y[tr]

            def chunks(Xtr=Xtr, ytr=ytr):
                for s in range(0, len(ytr), CHUNK):
                    yield Xtr[s:s + CHUNK].astype(np.float64), ytr[s:s + CHUNK]
            m = Ridge().fit(chunks)
            d = _direction([(m.predict(X[asof == v].astype(np.float64)), y[asof == v]) for v in f.val_dates])
            dirs.append(d)
            for v in f.test_dates:
                sel = asof == pd.Timestamp(v)
                if sel.any():
                    rows.append(pd.DataFrame({"as_of": pd.Timestamp(v), "ticker": idx.get_level_values("ticker")[sel],
                                              "score": m.predict(X[sel].astype(np.float64)) * d}))
        log(f"{name} done, direction flips {sum(1 for d in dirs if d < 0)}")
        return pd.concat(rows, ignore_index=True), dirs

    results = {"design": "docs/work/tasks/T-033.md", "n_folds": len(folds), "arms": {}}
    preds = {}
    for name, X in (("A0", X0), ("A0r", None), ("A1", X1)):
        if name == "A0r":                                      # 按列分块做当日 gauss-rank，避免整块 float64 副本
            X = np.empty(X0.shape, dtype=np.float32)
            for j in range(0, X0.shape[1], 16):
                xs = X0[:, j:j + 16]
                X[:, j:j + 16] = gauss_rank_by_day(np.where(np.isfinite(xs), xs, np.nan), day_codes)
        p, dirs = weekly_arm(name, X)
        p.to_parquet(out_dir / f"ridge_20d_{name}.parquet", index=False)
        preds[name] = p
        results["arms"][name] = {"directions": dirs, "n_rows": int(len(p))}
        if name == "A0r":
            del X
            gc.collect()
    # ---------- A0 必须复现已登记的 Ridge C ----------
    base = pd.read_parquet(root / "data/bakeoff/preds_v6_1_t028/ridge_20d_v6_1_BASE.parquet")
    base["as_of"] = pd.to_datetime(base["as_of"])
    m = preds["A0"].merge(base, on=["as_of", "ticker"], how="outer", suffixes=("_a0", "_base"))
    diff = float(np.nanmax(np.abs(m["score_a0"] - m["score_base"])))
    results["A0_reproduces_base"] = {"rows_a0": int(len(preds["A0"])), "rows_base": int(len(base)),
                                     "rows_unmatched": int(m[["score_a0", "score_base"]].isna().any(axis=1).sum()),
                                     "max_abs_diff": diff, "ok": bool(diff <= 1e-6)}
    log(f"A0 vs BASE: {results['A0_reproduces_base']}")
    if not results["A0_reproduces_base"]["ok"]:
        (HERE / "attribution_20d.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        raise SystemExit("A0 does not reproduce Ridge C: stop (pre-registered)")
    # ---------- A2：日频训练行 ----------
    rows, dirs = [], []
    for f in folds:
        tr_days = cal[(cal >= min(f.train_dates)) & (cal <= max(f.train_dates))]
        b = md.load(tr_days)
        mk = md.market_rows(b)
        yd = b.meta["y20"].to_numpy(dtype=np.float64)
        okr = np.flatnonzero(np.isfinite(yd))

        def chunks(b=b, mk=mk, yd=yd, okr=okr):
            for s in range(0, len(okr), CHUNK):
                r = okr[s:s + CHUNK]
                yield np.concatenate([b.X[r].astype(np.float64), mk[r].astype(np.float64)], axis=1), yd[r]
        mdl = Ridge().fit(chunks)
        d = _direction([(mdl.predict(X1[asof == v].astype(np.float64)), y[asof == v]) for v in f.val_dates])
        dirs.append(d)
        for v in f.test_dates:
            sel = asof == pd.Timestamp(v)
            if sel.any():
                rows.append(pd.DataFrame({"as_of": pd.Timestamp(v), "ticker": idx.get_level_values("ticker")[sel],
                                          "score": mdl.predict(X1[sel].astype(np.float64)) * d}))
        del b, mk
        gc.collect()
    preds["A2"] = pd.concat(rows, ignore_index=True)
    preds["A2"].to_parquet(out_dir / "ridge_20d_A2.parquet", index=False)
    results["arms"]["A2"] = {"directions": dirs, "n_rows": int(len(preds["A2"]))}
    log(f"A2 done, direction flips {sum(1 for d in dirs if d < 0)}")
    # ---------- 评估：冻结的 20d 可评估行 ----------
    cal2, grid = calendar_and_grid(root)
    test = sorted({d for f in make_folds(cal2, grid, "20d") for d in f.test_asof})
    meta = pd.concat([pd.read_parquet(p, columns=["as_of", "ticker", "y20", "top1500"])
                      for p in sorted((root / "data/gpu_v1/model/v1").glob("*.meta.parquet"))], ignore_index=True)
    meta["as_of"] = pd.to_datetime(meta["as_of"])
    meta = meta[meta["as_of"].isin(set(test))].set_index(["as_of", "ticker"]).rename(columns={"y20": "y"})
    fams = {mm: ep.load_family(root, mm, "20d").reindex(meta.index) for mm in ep.ENS_MEMBERS}
    ens = rank_average_frames(fams).reindex(meta.index)
    frozen = pd.read_parquet(EV / "T-030/p6/frozen_eval_rows_20d.parquet")
    frozen["as_of"] = pd.to_datetime(frozen["as_of"])
    fi = pd.MultiIndex.from_frame(frozen[["as_of", "ticker"]])
    ev = pd.DataFrame({"y": meta["y"].reindex(fi).to_numpy(), "ens": ens.reindex(fi).to_numpy(),
                       "ridgeC": base.set_index(["as_of", "ticker"])["score"].reindex(fi).to_numpy()}, index=fi)
    for name, p in preds.items():
        ev[name] = p.set_index(["as_of", "ticker"])["score"].reindex(fi).to_numpy()
    n_before = len(ev)
    ev = ev.dropna()
    results["eval_rows"] = {"frozen": n_before, "all_scorers_present": int(len(ev))}
    ic = {c: ep.ic_by_date(ev, c) for c in ev.columns if c != "y"}
    results["ic_mean"] = {c: float(v.mean()) for c, v in ic.items()}
    pairs = [("A0r", "A0"), ("A1", "A0"), ("A1", "A0r"), ("A2", "A1"), ("ens", "A1"), ("ens", "A2"),
             ("A0", "ridgeC"), ("A0r", "ridgeC"), ("A1", "ridgeC"), ("A2", "ridgeC"), ("ens", "ridgeC")]
    results["paired"] = {f"{a}-{b}": ep.paired(ic[a], ic[b]) for a, b in pairs}
    results["note"] = "探索性归因：不是确认性比较，不改变 T-030 的任何判定。"
    (HERE / "attribution_20d.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    log(json.dumps({k: (round(v["mean_diff"], 4), [round(x, 4) for x in v["block13"]["ci95"]], v["decision"])
                    for k, v in results["paired"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
