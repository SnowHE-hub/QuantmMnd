"""tests/test_wf_gate.py — 闸门编排：H-A 方向只用 train/val、H-D 胜率、指标、判线.

合成/小样本，不碰真实全量面板，不跑 LGBM。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantmind.backtest.wf_split import PurgedWalkForwardSplit
from quantmind.backtest import wf_gate as g
from quantmind.backtest import wf_metrics as m


# ---------------------------------------------------------------------------
# 极简注入式 predictor：按面板里某特征列打分（不训练、确定性）
# ---------------------------------------------------------------------------
class _FeatPredictor:
    def __init__(self, col: str = "signal"):
        self.col = col

    def fit(self, panel, train_dates, val_dates, feature_cols, label_col):
        return self

    def score(self, panel, test_date, feature_cols):
        return panel.xs(pd.Timestamp(test_date), level="as_of")[self.col]


def _panel_with_signal(sign: float = 1.0, seed: int = 3):
    """构造 panel：在所有截面 signal 与 label 的关系由 sign 决定（+1 正相关）."""
    rng = np.random.default_rng(seed)
    cal = list(pd.bdate_range("2021-01-01", periods=300))
    grid = cal[::5]
    tickers = [f"T{i:02d}" for i in range(30)]
    rows = []
    for a in grid:
        sig = rng.standard_normal(len(tickers))
        lab = sign * sig + 0.3 * rng.standard_normal(len(tickers))  # 正/负相关 + 噪声
        for tk, s, l in zip(tickers, sig, lab):
            rows.append((pd.Timestamp(a), tk, float(s), float(l)))
    df = pd.DataFrame(rows, columns=["as_of", "ticker", "signal", "forward_return_12d"])
    return df.set_index(["as_of", "ticker"]), cal, grid


# ===========================================================================
# H-A：方向只用 train/val，绝不碰 OOS
# ===========================================================================
def test_direction_uses_only_val_not_oos():
    panel, cal, grid = _panel_with_signal(sign=1.0)
    sp = PurgedWalkForwardSplit(cal, horizon=12, embargo=12)
    fold = sp.split(grid, grid[len(grid) // 2], oos_start=grid[len(grid) // 2],
                    oos_end=grid[-1])
    pred = _FeatPredictor("signal")

    d_base = g.decide_direction(pred, panel, fold.val_dates, ["signal"], "forward_return_12d")
    assert d_base == 1  # val 正相关 → +1

    # 把【全部 OOS 截面】标签整体取反，方向必须不变（只用 val）
    panel2 = panel.copy()
    oos_mask = panel2.index.get_level_values("as_of").isin(fold.test_dates)
    panel2.loc[oos_mask, "forward_return_12d"] = -panel2.loc[oos_mask, "forward_return_12d"]
    d_after = g.decide_direction(pred, panel2, fold.val_dates, ["signal"], "forward_return_12d")
    assert d_after == d_base, "方向受 OOS 标签影响 = H-A 泄漏"


def test_direction_flips_with_val_sign():
    # val 段负相关 → 方向 -1（仅由 val 决定，符合先验/训练期口径）
    panel, cal, grid = _panel_with_signal(sign=-1.0)
    sp = PurgedWalkForwardSplit(cal, horizon=12, embargo=12)
    fold = sp.split(grid, grid[len(grid) // 2])
    d = g.decide_direction(_FeatPredictor("signal"), panel, fold.val_dates,
                           ["signal"], "forward_return_12d")
    assert d == -1


def test_evaluate_oos_ic_direction_applied():
    panel, cal, grid = _panel_with_signal(sign=1.0)
    sp = PurgedWalkForwardSplit(cal, horizon=12, embargo=12)
    fold = sp.split(grid, grid[len(grid) // 2], oos_start=grid[len(grid) // 2],
                    oos_end=grid[-1])
    res = g.evaluate_oos_ic(panel, fold, _FeatPredictor("signal"), ["signal"],
                            "forward_return_12d")
    assert res.direction == 1
    assert res.ic_summary["ic_mean"] > 0.3  # 正相关信号，OOS IC 正


# ===========================================================================
# H-D：每换仓期对基准胜率（非逐日正收益）
# ===========================================================================
def test_per_rebalance_win_rate():
    port = pd.Series([0.02, -0.01, 0.03, 0.00, 0.05])
    bench = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01])
    # 组合 > 基准 的期：#0(0.02>0.01)✓ #1(-0.01>0.01)✗ #2✓ #3(0>0.01)✗ #4✓ → 3/5
    assert m.per_rebalance_win_rate(port, bench) == 0.6


def test_win_rate_is_not_daily_positive():
    # 全为负收益但都跑赢（更小负）基准 → 胜率高（逐日正收益口径会给 0）
    port = pd.Series([-0.01, -0.02, -0.005])
    bench = pd.Series([-0.05, -0.06, -0.04])
    assert m.per_rebalance_win_rate(port, bench) == 1.0


# ===========================================================================
# 指标：分位单调性 / 净超额年化 / IC
# ===========================================================================
def test_quantile_monotonicity_strict():
    # 预测分与实现收益完全单调 → 严格单调、order_corr=1
    n = 100
    pred = pd.Series(np.arange(n, dtype=float))
    realized = pd.Series(np.arange(n, dtype=float) * 0.001)
    q = m.quantile_monotonicity(pred, realized, n_groups=5)
    assert q.strictly_monotone
    assert q.order_corr > 0.99


def test_net_excess_annualized_sign():
    port = pd.Series([0.02] * 10)
    bench = pd.Series([0.01] * 10)
    ex = m.net_excess_annualized(port, bench, periods_per_year=21)  # ~每12交易日
    assert ex > 0
    # 组合不如基准 → 负超额
    ex2 = m.net_excess_annualized(bench, port, periods_per_year=21)
    assert ex2 < 0


def test_cross_sectional_ic_perfect():
    pred = pd.Series({"A": 3, "B": 1, "C": 2})
    realized = pd.Series({"A": 30, "B": 10, "C": 20})
    assert m.cross_sectional_ic(pred, realized) > 0.99


# ===========================================================================
# 判线 + 幸存者标注
# ===========================================================================
def test_evaluate_gate_pass_fail():
    good = {"ic_mean": 0.04, "ic_ir": 0.5, "net_excess_annual": 0.08,
            "win_rate": 0.55, "max_drawdown": 0.18, "monotonicity": 0.5}
    assert g.evaluate_gate(good)["passed"]
    bad = dict(good, ic_mean=0.01)  # IC 不达线
    out = g.evaluate_gate(bad)
    assert not out["passed"] and not out["checks"]["ic_mean"]


def test_gate_meta_has_survivorship_caveat():
    meta = g.gate_meta()
    assert "optimistic upper bound" in meta["survivorship_caveat"] or "乐观上界" in meta["survivorship_caveat"]
    assert "label_not_nav" in meta and "direction_policy" in meta


# ===========================================================================
# Categorical 特征支持（修复 build_lgbm_arrays / LGBMPredictor）
# ===========================================================================
def _panel_with_categorical(seed: int = 7):
    """合成 panel：1 个 string 类别列 + 1 个数值列 + 标签."""
    rng = np.random.default_rng(seed)
    cal = list(pd.bdate_range("2022-01-01", periods=200))
    grid = cal[::5]
    tickers = [f"T{i:02d}" for i in range(40)]
    sectors = np.array(["金融", "科技", "医疗", "消费", "能源"])
    rows = []
    # 每只票固定行业（不随时间变化，符合现实）
    ticker_sector = {tk: sectors[rng.integers(0, len(sectors))] for tk in tickers}
    for a in grid:
        sig = rng.standard_normal(len(tickers))
        lab = sig + 0.3 * rng.standard_normal(len(tickers))
        for tk, s, l in zip(tickers, sig, lab):
            rows.append((pd.Timestamp(a), tk, float(s),
                         ticker_sector[tk], float(l)))
    df = pd.DataFrame(rows, columns=["as_of", "ticker", "signal",
                                      "exposure_industry", "forward_return_12d"])
    df["exposure_industry"] = df["exposure_industry"].astype("string")
    return df.set_index(["as_of", "ticker"]), cal, grid


def test_lgbm_predictor_handles_string_categorical():
    """含 string 列时 LGBMPredictor.fit/score 不崩，IC 正常."""
    from quantmind.backtest.wf_split import PurgedWalkForwardSplit
    panel, cal, grid = _panel_with_categorical()
    sp = PurgedWalkForwardSplit(cal, horizon=12, embargo=12)
    fold = sp.split(grid, grid[len(grid) // 2],
                    oos_start=grid[len(grid) // 2], oos_end=grid[-1])

    pred = g.LGBMPredictor(n_quantiles=5, n_estimators=50,
                            learning_rate=0.1, verbose=-1)
    # fit 不抛
    pred.fit(panel, fold.train_dates, fold.val_dates,
             ["signal", "exposure_industry"], "forward_return_12d")
    # cat 检测到（_cat_cols 非空）
    assert pred._cat_cols == ["exposure_industry"]
    # score 不抛、返回 Series
    s = pred.score(panel, fold.test_dates[0], ["signal", "exposure_industry"])
    assert isinstance(s, pd.Series)
    assert len(s) > 0
    assert s.notna().any()


def test_detect_and_encode_categorical_helpers():
    """factor_model.detect_categorical_cols / encode_categorical_codes 单元测试."""
    from quantmind.models.factor_model import (
        detect_categorical_cols, encode_categorical_codes,
    )
    df = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0],
        "str_col": pd.Series(["a", "b", "a", None], dtype="string"),
        "obj_col": ["x", "y", "x", "y"],
        "cat_col": pd.Categorical(["p", "q", "p", "p"]),
    })
    cat = detect_categorical_cols(df, ["num", "str_col", "obj_col", "cat_col"])
    assert set(cat) == {"str_col", "obj_col", "cat_col"}
    assert "num" not in cat

    enc = encode_categorical_codes(df, cat)
    # 数值列不变
    assert enc["num"].dtype == float
    # cat 列转 int32
    assert enc["str_col"].dtype == np.int32
    # NaN → -1
    assert enc["str_col"].iloc[3] == -1
    # 相同字符串得到相同 code
    assert enc["str_col"].iloc[0] == enc["str_col"].iloc[2]
