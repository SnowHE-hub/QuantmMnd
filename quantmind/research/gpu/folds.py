"""外层折：逐折复刻主线 B 方案 C（T-018 `train_line_b_v6_1.py::make_folds(protocol="C")`），再加日频训练窗与内层验证.

外层（与基线完全相同）：PurgedWalkForwardSplit(rolling 756, n_val=2)，切点同 T-018 / T-023，
测试窗 (C_k + E, C_{k+1} + E]（最后一折到周频网格末日），OOS 起点 2022-01-01。主比较只在这些周频测试 as_of 上做。

日频训练窗（本卡新增，交易日索引空间）：idx(C) − 756 ≤ idx(d) ≤ idx(C) − H − 1，
即每条训练样本的 label_end = idx(d) + H < idx(C)（比基线的 ≤ 严格一天）。
内层验证 = 训练窗最后 inner_val_td 个交易日；内层训练 = 其前、且 label_end < 内层验证首日的交易日。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from quantmind.backtest.wf_split import PurgedWalkForwardSplit

ROLLING_TD = 756
OOS_START = pd.Timestamp("2022-01-01")
HORIZONS = {
    "20d": {"label": "forward_return_20d", "H": 20, "E": 20, "cuts": "quarterly",
            "cut_start": "2022-01-01", "cut_end": "2026-07-01", "inner_val_td": 126},
    "63d": {"label": "forward_return_63d", "H": 63, "E": 63, "cuts": "semiannual",
            "cut_start": "2022-04-01", "cut_end": "2026-04-30", "inner_val_td": 252},
}


def quarterly_cutoffs(cal, start, end) -> list[pd.Timestamp]:
    """= scripts/bakeoff/p3b_tabular.make_quarterly_cutoffs."""
    cal_sorted = sorted(cal)
    out = []
    for qd in pd.date_range(start, end, freq="QS-JAN"):
        after = [d for d in cal_sorted if d >= qd]
        if after:
            out.append(after[0])
    return out


def semiannual_cutoffs(cal, start, end) -> list[pd.Timestamp]:
    """= scripts/survivorship/p63_3_train_v6.make_semiannual_cutoffs（4/30、10/31 snap 到首个 ≥ 的交易日）."""
    cal_sorted = sorted(cal)
    out = []
    for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        for mmdd in ("04-30", "10-31"):
            d = pd.Timestamp(f"{y}-{mmdd}")
            if d < pd.Timestamp(start) or d > pd.Timestamp(end):
                continue
            after = [c for c in cal_sorted if c >= d]
            if after:
                out.append(after[0])
    return sorted(set(out))


@dataclass
class DailyFold:
    k: int
    horizon: str
    cutoff: pd.Timestamp
    H: int
    E: int
    test_asof: list[pd.Timestamp]                       # 基线的周频样本外 as_of（主比较）
    weekly_train: list[pd.Timestamp]                    # 基线自己的周频训练日（周频训练消融用）
    train_days: list[pd.Timestamp] = field(default_factory=list)
    inner_train_days: list[pd.Timestamp] = field(default_factory=list)
    inner_val_days: list[pd.Timestamp] = field(default_factory=list)
    pred_days: list[pd.Timestamp] = field(default_factory=list)   # 该折模型负责的全部交易日（日频预测）

    def summary(self) -> dict:
        f = lambda xs: [str(xs[0].date()), str(xs[-1].date()), len(xs)] if xs else None  # noqa: E731
        return {"k": self.k, "cutoff": str(self.cutoff.date()), "train": f(self.train_days),
                "inner_train": f(self.inner_train_days), "inner_val": f(self.inner_val_days),
                "test_asof": f(self.test_asof), "pred_days": f(self.pred_days), "weekly_train": f(self.weekly_train)}


def make_folds(cal: pd.DatetimeIndex, weekly_grid: list[pd.Timestamp], horizon: str) -> list[DailyFold]:
    cfg = HORIZONS[horizon]
    H, E = cfg["H"], cfg["E"]
    cal = pd.DatetimeIndex(sorted(cal))
    sp = PurgedWalkForwardSplit(cal, horizon=H, embargo=E, mode="rolling", rolling_lookback_td=ROLLING_TD, n_val=2)
    cut_fn = quarterly_cutoffs if cfg["cuts"] == "quarterly" else semiannual_cutoffs
    cuts = cut_fn(cal, pd.Timestamp(cfg["cut_start"]), pd.Timestamp(cfg["cut_end"]))
    grid = sorted(pd.Timestamp(a) for a in weekly_grid)
    oos_end = grid[-1]
    folds: list[DailyFold] = []
    for k, c in enumerate(cuts):
        if k + 1 < len(cuts):
            hi = sp.cal[min(sp.idx(cuts[k + 1]) + E, len(sp.cal) - 1)]
            pred_hi_idx = sp.idx(cuts[k + 1]) + E
        else:
            hi = oos_end
            pred_hi_idx = len(cal) - 1
        wf = sp.split(grid, c, oos_start=OOS_START, oos_end=hi)
        if not wf.test_dates or not wf.train_dates:
            continue
        PurgedWalkForwardSplit.assert_no_leakage(wf)
        ci = sp.idx(c)
        train = [d for i, d in enumerate(cal) if ci - ROLLING_TD <= i <= ci - H - 1]
        n_val = cfg["inner_val_td"]
        val = train[-n_val:]
        v0 = sp.idx(val[0])
        inner = [d for d in train if sp.idx(d) + H < v0]
        pred = [d for i, d in enumerate(cal) if ci + E < i <= min(pred_hi_idx, len(cal) - 1) and d >= OOS_START]
        folds.append(DailyFold(k=len(folds), horizon=horizon, cutoff=pd.Timestamp(c), H=H, E=E,
                               test_asof=[pd.Timestamp(x) for x in wf.test_dates],
                               weekly_train=[pd.Timestamp(x) for x in wf.train_dates],
                               train_days=train, inner_train_days=inner, inner_val_days=val, pred_days=pred))
    return folds


def calendar_and_grid(root: Path) -> tuple[pd.DatetimeIndex, list[pd.Timestamp]]:
    """完整交易日历（= 价格面板日期）与 v6_1 周频网格（build_asof_grid），同 T-018 asof_grid_v6_1.py。"""
    from quantmind.features.weekly_panel import build_asof_grid
    D = Path(root) / "data"
    px = pd.to_datetime(pd.read_parquet(D / "raw/alpha_prices_panel_v6_20261009.parquet",
                                        columns=["trade_date"])["trade_date"]).drop_duplicates()
    cal = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(D / "lake/trade_cal_sse.parquet")["cal_date"])).sort_values()
    cal = cal[(cal >= px.min()) & (cal <= px.max())]
    assert set(cal) == set(px), "price dates must equal the trade calendar"
    return cal, [pd.Timestamp(x) for x in build_asof_grid(cal)]


def check_folds(folds: list[DailyFold], cal: pd.DatetimeIndex, dirs_json: Path | None = None) -> dict:
    """标签成熟 / 隔离 / 与基线 .dirs.json 的切点与测试数逐折一致；任何一条不满足即 AssertionError."""
    pos = {d: i for i, d in enumerate(pd.DatetimeIndex(sorted(cal)))}
    seen: set = set()
    for f in folds:
        ci = pos[f.cutoff]
        assert all(pos[d] + f.H < ci for d in f.train_days), "train label_end >= cutoff"
        assert all(pos[d] + f.H < pos[f.inner_val_days[0]] for d in f.inner_train_days), "inner train overlaps val"
        assert set(f.inner_train_days) <= set(f.train_days) and set(f.inner_val_days) <= set(f.train_days)
        assert all(pos[d] > ci + f.E for d in f.test_asof + f.pred_days), "test inside embargo"
        assert not (set(f.test_asof) & seen), "duplicate test as_of across folds"
        seen |= set(f.test_asof)
        assert set(f.test_asof) <= set(f.pred_days), "weekly test as_of not covered by daily pred days"
    out = {"n_folds": len(folds), "n_test_asof": len(seen)}
    if dirs_json is not None:
        ref = json.loads(Path(dirs_json).read_text())["folds"]
        assert len(ref) == len(folds), f"fold count {len(folds)} != baseline {len(ref)}"
        for f, r in zip(folds, ref, strict=True):
            assert str(f.cutoff.date()) == r["cutoff"], (f.cutoff, r["cutoff"])
            assert len(f.test_asof) == r["n_test"], (f.cutoff, len(f.test_asof), r["n_test"])
            if "test" in r:
                assert [str(f.test_asof[0].date()), str(f.test_asof[-1].date())] == r["test"]
            if "train" in r:
                assert [str(f.weekly_train[0].date()), str(f.weekly_train[-1].date())] == r["train"]
        out["matches_baseline_dirs_json"] = str(dirs_json)
    return out
