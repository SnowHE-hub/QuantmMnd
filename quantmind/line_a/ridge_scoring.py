"""quantmind.line_a.ridge_scoring — Ridge C 参考臂的前瞻打分（F3；03_final_plan_v2 第 5 节）.

- 训练：逐项照抄 T-018 / T-028 的方案 C（v6_1 周频面板、223 个数值特征、PurgedWalkForwardSplit rolling 756、
  n_val = 2、方向只由验证段决定），只重建**最后一折**；Ridge 用 T-033 已逐值复现 Ridge C 的分块闭式解。
  20d：季度切点 2022-01-01 → 2026-07-01；63d：半年切点 2022-04-01 → 2026-04-30（下一次重训切点 2026-10-31，需另行登记）。
- 前瞻特征：从 T-030 日频面板（真实或影子数据根）按冻结的特征名单逐列取出；T-030 P0 已核验日频构建器在
  374 个周频截面上与 v6_1 文件逐值一致（sh 组 3.1e-10）。
"""
from __future__ import annotations

import importlib.util
import json
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


def build_lean(root: Path, dates) -> tuple[pd.DataFrame, list[str]]:
    """与 T-018 build(root, "63d") 相同的列与列序，但只读 dates 这些 as_of 的行（整表 build 需要 7 GB 以上内存）.

    唯一需要看整表的是 a158 的"非全空列"筛选（build 在全部面板行上做 a.notna().any()），这里逐列在全部面板行上判断，
    结果与 build 相同。"""
    import pyarrow.parquet as pq

    from quantmind.features.weekly_panel import WHITELIST_35
    D = Path(root) / "data"
    P = D / "panel/alpha_panel_weekly_v6_1.parquet"
    names = set(pq.read_schema(P).names)
    sh = [c for c in json.loads((D / "loss_signals_v4/short_horizon_survivors.json").read_text())["survivors"]
          if c in names]
    feats = list(WHITELIST_35) + sh
    filt = [("as_of", "in", list(pd.DatetimeIndex(sorted(dates))))]
    base = pd.read_parquet(P, columns=feats + ["forward_return_63d"], filters=filt)
    surv = json.loads((D / "loss_signals_v4/fundamental_survivors_v6.json").read_text())["survivors"]
    fnd = pd.read_parquet(D / "panel/fundamental_factors_v6_1.parquet", columns=surv, filters=filt).reindex(base.index)
    fnd.columns = [f"fnd_{c}" for c in fnd.columns]
    A = D / "bakeoff/alpha158_asof_v6_1.parquet"
    full_idx = pd.read_parquet(P, columns=["forward_return_63d"]).index
    a_cols = [c for c in pq.read_schema(A).names if c not in ("as_of", "ticker", "__index_level_0__")]
    a_idx = pd.read_parquet(A, columns=[a_cols[0]]).index
    in_base = a_idx.isin(full_idx)
    keep = [c for c in a_cols if pd.read_parquet(A, columns=[c])[c][in_base].notna().any()]   # 与 build 的 notna 口径相同
    a = pd.read_parquet(A, columns=keep, filters=filt).reindex(base.index)
    return base, [fnd, a], feats + list(fnd.columns) + keep


def _matrix(frames: list[pd.DataFrame], cols: list[str]) -> np.ndarray:
    """按 cols 的顺序逐列填进预先分配的 float64 矩阵；从 frames 里逐张取出，填完就释放
    （调用方不要再持有这些表的引用；不生成拼接后的整表，省一倍内存）。"""
    pos = {c: j for j, c in enumerate(cols)}
    X = np.empty((len(frames[0]), len(cols)), dtype=np.float64)
    while frames:
        fr = frames.pop(0)
        for c in fr.columns:
            if c in pos:
                X[:, pos[c]] = fr[c].to_numpy(dtype=np.float64)
        del fr
    return X


def fit_reference(root: Path, horizon: str) -> RidgeRef:
    t18 = _load(EV / "T-018/train_line_b_v6_1.py", "t18train")
    t33 = _load(EV / "T-033/attribution_ridge_20d.py", "t033ridge")
    from quantmind.backtest.wf_split import PurgedWalkForwardSplit
    from scripts.bakeoff.p3b_tabular import make_quarterly_cutoffs
    from scripts.survivorship.p63_3_train_v6 import make_semiannual_cutoffs

    P = root / "data/panel/alpha_panel_weekly_v6_1.parquet"
    all_asof = sorted(pd.to_datetime(pd.read_parquet(P, columns=["forward_return_63d"]).index
                                     .get_level_values("as_of")).unique())
    if horizon == "20d":
        H = E = 20
    else:
        H = E = 63
    cal = t18.calendar(root)
    sp = PurgedWalkForwardSplit(cal, horizon=H, embargo=E, mode="rolling", rolling_lookback_td=756, n_val=2)
    cuts = (make_quarterly_cutoffs(cal, pd.Timestamp("2022-01-01"), pd.Timestamp("2026-07-01")) if horizon == "20d"
            else make_semiannual_cutoffs(cal, "2022-04-01", "2026-04-30"))
    folds = [f for f in t18.make_folds(sp, all_asof, cuts, "C") if f.test_dates and f.train_dates]
    f = folds[-1]
    need = sorted(set(f.train_dates) | set(f.val_dates) | set(f.test_dates))
    base, extras, feats = build_lean(root, need)
    num = [c for c in feats if c not in CATEGORICAL]
    # 与"拼接后 sort_index"等价：先排基础表，其余表对齐到排好的索引
    base = base.sort_index(level=["as_of", "ticker"])
    extras = [e.reindex(base.index) for e in extras]
    idx = base.index
    asof = pd.to_datetime(idx.get_level_values("as_of"))
    if horizon == "20d":
        y = pd.read_parquet(root / "data/panel/labels_extra_v6_1.parquet", columns=["forward_return_20d"],
                            filters=[("as_of", "in", list(pd.DatetimeIndex(need)))]).reindex(idx)["forward_return_20d"]
    else:
        y = base["forward_return_63d"]
    y = y.to_numpy(dtype=np.float64, copy=True)      # 拷贝，不让 y 引用住基础表的内存块
    frames = [base, *extras]
    del base, extras
    X = _matrix(frames, num)
    tr = np.flatnonzero(np.isin(asof, pd.DatetimeIndex(f.train_dates)) & np.isfinite(y))

    def chunks():                       # 训练行按下标分块取，块边界与原来"先复制训练行再切块"相同
        for s in range(0, len(tr), CHUNK):
            sel = tr[s:s + CHUNK]
            yield X[sel], y[sel]
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


def save_ref(ref: RidgeRef, path: Path) -> str:
    """把重建的最后一折存成 npz（均值、标准差、系数、截距、方向、特征名），日常打分只读它，不再重训。返回 sha256。"""
    import io

    from quantmind.line_a.atomic import atomic_write_bytes
    m = ref.model
    buf = io.BytesIO()
    np.savez(buf, mu=m.mu, sd=m.sd, w=m.w, b=np.float64(m.b), direction=np.int64(ref.direction),
             feats=np.array(ref.feats), horizon=np.array(ref.horizon),
             cutoff=np.array(str(ref.cutoff.date()) if pd.notna(ref.cutoff) else ""),
             train_dates=np.int64(ref.train_dates))
    return atomic_write_bytes(Path(path), buf.getvalue())


def load_ref(path: Path) -> RidgeRef:
    t33 = _load(EV / "T-033/attribution_ridge_20d.py", "t033ridge")
    z = np.load(path, allow_pickle=False)
    m = t33.Ridge()
    m.mu, m.sd, m.w, m.b = z["mu"], z["sd"], z["w"], float(z["b"])
    cut = str(z["cutoff"])
    return RidgeRef(str(z["horizon"]), [str(f) for f in z["feats"]], m, int(z["direction"]),
                    pd.Timestamp(cut) if cut else pd.NaT, int(z["train_dates"]), pd.DataFrame())
