"""T-033：分块闭式 Ridge 与 RidgePredictor（sklearn Ridge）同口径——含 NaN、±inf、全缺失列、常数列、分块边界。"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_P = Path(__file__).resolve().parents[1] / "docs/work/evidence/T-033/attribution_ridge_20d.py"
_spec = importlib.util.spec_from_file_location("t033attr", _P)
mod = importlib.util.module_from_spec(_spec)
sys.modules["t033attr"] = mod
_spec.loader.exec_module(mod)


def _panel(n=3000, p=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)) * np.array([1, 5, 0.1, 1, 1, 1])
    X[rng.random((n, p)) < 0.1] = np.nan
    X[5, 0], X[7, 1] = np.inf, -np.inf
    X[:, 4] = np.nan                                          # 全缺失列
    X[:, 5] = 3.0                                             # 常数列（sd = 0 → 1）
    y = X[:, 0].copy()
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0) * 0.5 + rng.normal(size=n)
    y[rng.random(n) < 0.05] = np.nan
    dates = np.repeat(pd.date_range("2020-01-03", periods=30, freq="W-FRI"), n // 30)
    idx = pd.MultiIndex.from_arrays([dates, [f"T{i % 100:03d}" for i in range(n)]], names=["as_of", "ticker"])
    feats = [f"f{j}" for j in range(p)]
    return pd.DataFrame(X, index=idx, columns=feats).assign(label=y), feats


def test_chunked_ridge_matches_ridge_predictor():
    from scripts.bakeoff.p3b_tabular import RidgePredictor
    panel, feats = _panel()
    dates = sorted(panel.index.get_level_values("as_of").unique())
    train, test = dates[:20], dates[25]
    ref = RidgePredictor(alpha=10.0).fit(panel, train, [], feats, "label")
    tr = panel[panel.index.get_level_values("as_of").isin(set(train)) & panel["label"].notna()]
    X, y = tr[feats].to_numpy(dtype=float), tr["label"].to_numpy(dtype=float)

    def chunks():
        for s in range(0, len(y), 777):                       # 故意不整除的分块
            yield X[s:s + 777], y[s:s + 777]
    m = mod.Ridge().fit(chunks)
    xs = panel.xs(test, level="as_of")
    a = ref.score(panel, test, feats).to_numpy()
    b = m.predict(xs[feats].to_numpy(dtype=float))
    assert np.max(np.abs(a - b)) < 1e-10
