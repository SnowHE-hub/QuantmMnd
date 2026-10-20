"""quantmind.line_a.ridge_scoring — Ridge C 参考臂的前瞻打分（F3；03_final_plan_v2 第 5 节）.

- 训练：逐项照抄 T-018 / T-028 的方案 C（v6_1 周频面板、223 个数值特征、PurgedWalkForwardSplit rolling 756、
  n_val = 2、方向只由验证段决定），只重建**最后一折**；Ridge 用 T-033 已逐值复现 Ridge C 的分块闭式解。
  20d：季度切点 2022-01-01 → 2026-07-01；63d：半年切点 2022-04-01 → 2026-04-30（下一次重训切点 2026-10-31，需另行登记）。
- 前瞻特征：从 T-030 日频面板（真实或影子数据根）按冻结的特征名单逐列取出；T-030 P0 已核验日频构建器在
  374 个周频截面上与 v6_1 文件逐值一致（sh 组 3.1e-10）。
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
EV = REPO / "docs/work/evidence"
CATEGORICAL = ("exposure_industry", "exposure_area")
CHUNK = 200_000


def _load(path: Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class RidgeRef:
    horizon: str
    feats: list[str]
    model: object
    direction: int
    cutoff: pd.Timestamp
    train_dates: int
    frozen_test: pd.DataFrame          # 最后一折在研究期的测试预测（as_of, ticker, score），用于核对


def fit_reference(root: Path, horizon: str) -> RidgeRef:
    t18 = _load(EV / "T-018/train_line_b_v6_1.py", "t18train")
    t33 = _load(EV / "T-033/attribution_ridge_20d.py", "t033ridge")
    from quantmind.backtest.wf_split import PurgedWalkForwardSplit
    from scripts.bakeoff.p3b_tabular import make_quarterly_cutoffs
    from scripts.survivorship.p63_3_train_v6 import make_semiannual_cutoffs

    panel, feats, _ = t18.build(root, "63d")
    num = [f for f in feats if f not in CATEGORICAL]
    panel = panel.sort_index(level=["as_of", "ticker"])
    idx = panel.index
    asof = pd.to_datetime(idx.get_level_values("as_of"))
    if horizon == "20d":
        y = pd.read_parquet(root / "data/panel/labels_extra_v6_1.parquet").reindex(idx)["forward_return_20d"]
        H = E = 20
    else:
        y = panel["forward_return_63d"]
        H = E = 63
    y = y.to_numpy(dtype=np.float64)
    X = panel[num].to_numpy(dtype=np.float64)
    del panel
    cal = t18.calendar(root)
    sp = PurgedWalkForwardSplit(cal, horizon=H, embargo=E, mode="rolling", rolling_lookback_td=756, n_val=2)
    cuts = (make_quarterly_cutoffs(cal, pd.Timestamp("2022-01-01"), pd.Timestamp("2026-07-01")) if horizon == "20d"
            else make_semiannual_cutoffs(cal, "2022-04-01", "2026-04-30"))
    folds = [f for f in t18.make_folds(sp, sorted(asof.unique()), cuts, "C") if f.test_dates and f.train_dates]
    f = folds[-1]
    tr = np.isin(asof, pd.DatetimeIndex(f.train_dates)) & np.isfinite(y)
    Xtr, ytr = X[tr], y[tr]

    def chunks():
        for s in range(0, len(ytr), CHUNK):
            yield Xtr[s:s + CHUNK], ytr[s:s + CHUNK]
    m = t33.Ridge().fit(chunks)
    d = t33._direction([(m.predict(X[asof == v]), y[asof == v]) for v in f.val_dates])
    rows = []
    for v in f.test_dates:
        sel = asof == pd.Timestamp(v)
        if sel.any():
            rows.append(pd.DataFrame({"as_of": pd.Timestamp(v), "ticker": idx.get_level_values("ticker")[sel],
                                      "score": m.predict(X[sel]) * d}))
    cutoff = getattr(f, "cutoff", None)
    return RidgeRef(horizon, num, m, d, pd.Timestamp(cutoff) if cutoff is not None else pd.NaT, len(f.train_dates),
                    pd.concat(rows, ignore_index=True))


def features_from_daily(root: Path, dates, feats: list[str]) -> pd.DataFrame:
    """按冻结的特征名单，从日频面板拼出 Ridge C 的特征；行集合 = base 面板（PIT 入池）在这些日期上的行。"""
    import json
    P = Path(root) / "data/gpu_v1/panel"
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(list(dates))))
    months = sorted({d.strftime("%Y-%m") for d in dates})
    sh_names = set(json.loads((Path(root) / "data/loss_signals_v4/short_horizon_survivors.json").read_text())["survivors"])

    def grp(g: str, cols: list[str]) -> pd.DataFrame:
        parts = [pd.read_parquet(P / g / f"{m}.parquet", columns=cols) for m in months]
        df = pd.concat(parts)
        return df[df.index.get_level_values("as_of").isin(dates)]
    want = {"base": [], "sh": [], "fnd": [], "a158": []}
    for c in feats:
        if c.startswith("fnd_"):
            want["fnd"].append(c[4:])
        elif c.startswith("a158_"):
            want["a158"].append(c)
        elif c in sh_names:
            want["sh"].append(c)
        else:
            want["base"].append(c)
    base = grp("base", want["base"])
    out = [base]
    for g in ("sh", "fnd", "a158"):
        if want[g]:
            df = grp(g, want[g]).reindex(base.index)
            if g == "fnd":
                df.columns = [f"fnd_{c}" for c in df.columns]
            out.append(df)
    return pd.concat(out, axis=1)[feats].sort_index()


def score(ref: RidgeRef, X: pd.DataFrame) -> pd.Series:
    return pd.Series(ref.model.predict(X[ref.feats].to_numpy(dtype=np.float64)) * ref.direction, index=X.index,
                     name="score")
