"""T-026 RidgePredictor：±inf 视为缺失，不再让整列标准化后被填 0（合成数据）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.bakeoff.p3b_tabular import RidgePredictor

DATES = pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-19"])
FEATS = ["f1", "f2"]


def panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.MultiIndex.from_product([DATES, [f"T{i:02d}" for i in range(50)]], names=["as_of", "ticker"])
    df = pd.DataFrame({"f1": rng.normal(size=len(idx)), "f2": rng.normal(size=len(idx))}, index=idx)
    df["y"] = 0.5 * df["f1"] + 0.5 * df["f2"] + rng.normal(scale=0.1, size=len(idx))
    return df


def test_inf_in_training_keeps_the_column():
    p = panel()
    p.iloc[3, p.columns.get_loc("f2")] = np.inf
    m = RidgePredictor().fit(p, DATES[:2], [], FEATS, "y")
    assert np.isfinite(m._mu["f2"]) and np.isfinite(m._sd["f2"])
    assert abs(m._m.coef_[1]) > 0.1                 # 修复前该列全为 0，系数为 0


def test_same_as_replacing_inf_with_nan_first():
    p = panel()
    col = p.columns.get_loc("f2")
    p.iloc[3, col] = np.inf                          # 训练期
    p.iloc[120, col] = -np.inf                       # 打分截面
    q = p.replace([np.inf, -np.inf], np.nan)
    a = RidgePredictor().fit(p, DATES[:2], [], FEATS, "y").score(p, DATES[2], FEATS)
    b = RidgePredictor().fit(q, DATES[:2], [], FEATS, "y").score(q, DATES[2], FEATS)
    pd.testing.assert_series_equal(a, b)
    assert np.isfinite(a).all()


def test_no_inf_unchanged():
    p = panel()
    a = RidgePredictor().fit(p, DATES[:2], [], FEATS, "y").score(p, DATES[2], FEATS)
    X = p[FEATS]
    tr = X.loc[DATES[:2]]
    z = ((X.xs(DATES[2], level="as_of") - tr.mean()) / tr.std()).fillna(0.0)
    from sklearn.linear_model import Ridge
    ref = Ridge(alpha=10.0).fit(((tr - tr.mean()) / tr.std()).values, p.loc[DATES[:2], "y"].values).predict(z.values)
    np.testing.assert_allclose(a.values, ref)
