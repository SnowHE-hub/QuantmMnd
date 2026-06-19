"""tests/test_wf_split.py — Purged walk-forward 切分 + purge 消融反证（核心硬闸门）.

合成/小样本，不碰真实全量面板。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantmind.backtest.wf_split import PurgedWalkForwardSplit
from quantmind.backtest.wf_gate import purge_ablation, CarryForwardLabelProbe


# ---------------------------------------------------------------------------
# 合成交易日历 + as_of 网格
# ---------------------------------------------------------------------------
def _calendar(n: int = 400) -> list[pd.Timestamp]:
    return list(pd.bdate_range("2020-01-01", periods=n))


def _grid(cal, step: int = 5):
    return list(cal[::step])


# ===========================================================================
# A. 切分正确性（purge + embargo）
# ===========================================================================
def test_purge_embargo_boundaries_expanding():
    cal = _calendar(400)
    grid = _grid(cal)
    sp = PurgedWalkForwardSplit(cal, horizon=12, embargo=12, mode="expanding")
    cutoff = cal[200]
    C = sp.idx(cutoff)
    fold = sp.split(grid, cutoff)

    train_idx = [sp.idx(a) for a in fold.train_dates]
    test_idx = [sp.idx(a) for a in fold.test_dates]
    # purge：训练标签窗口不越过 C → idx ≤ C-H
    assert all(i <= C - 12 for i in train_idx)
    # embargo：测试 idx > C+E
    assert all(i > C + 12 for i in test_idx)
    # 零交集
    assert set(fold.train_dates).isdisjoint(set(fold.test_dates))
    # gap ≥ H+E
    assert min(test_idx) - max(train_idx) >= 12 + 12
    PurgedWalkForwardSplit.assert_no_leakage(fold)


def test_isolation_band_drops_samples():
    # 隔离带 (C-H, C+E] 内的 as_of 既不在 train 也不在 test
    cal = _calendar(400)
    grid = _grid(cal)
    sp = PurgedWalkForwardSplit(cal, horizon=12, embargo=12)
    cutoff = cal[200]
    C = sp.idx(cutoff)
    fold = sp.split(grid, cutoff)
    tr, te = set(fold.train_dates), set(fold.test_dates)
    band = [a for a in grid if C - 12 < sp.idx(a) <= C + 12]
    for a in band:
        assert a not in tr and a not in te


def test_rolling_window_lower_bound():
    cal = _calendar(600)
    grid = _grid(cal)
    sp = PurgedWalkForwardSplit(cal, horizon=12, embargo=12, mode="rolling",
                                rolling_lookback_td=200)
    cutoff = cal[400]
    C = sp.idx(cutoff)
    fold = sp.split(grid, cutoff)
    train_idx = [sp.idx(a) for a in fold.train_dates]
    assert all(C - 200 <= i <= C - 12 for i in train_idx)
    # 扩展窗口应纳入更早样本（更多训练点）
    sp_exp = PurgedWalkForwardSplit(cal, horizon=12, embargo=12, mode="expanding")
    fold_exp = sp_exp.split(grid, cutoff)
    assert len(fold_exp.train_dates) >= len(fold.train_dates)


def test_nopurge_mode_abuts():
    cal = _calendar(400)
    grid = _grid(cal)
    sp = PurgedWalkForwardSplit(cal, horizon=0, embargo=0)
    cutoff = cal[200]
    C = sp.idx(cutoff)
    fold = sp.split(grid, cutoff)
    assert all(sp.idx(a) <= C for a in fold.train_dates)
    assert all(sp.idx(a) > C for a in fold.test_dates)
    # 紧贴：训练末与测试首在网格上相邻（gap == step）
    gap = min(sp.idx(a) for a in fold.test_dates) - max(sp.idx(a) for a in fold.train_dates)
    assert gap <= 5


def test_embargo_must_cover_horizon():
    cal = _calendar(100)
    with pytest.raises(ValueError):
        PurgedWalkForwardSplit(cal, horizon=12, embargo=5)  # E < H


def test_val_within_purge():
    cal = _calendar(400)
    grid = _grid(cal)
    sp = PurgedWalkForwardSplit(cal, horizon=12, embargo=12, n_val=2)
    fold = sp.split(grid, cal[200])
    assert len(fold.val_dates) == 2
    # val ⊂ train，仍满足 purge
    assert set(fold.val_dates).issubset(set(fold.train_dates))
    C = sp.idx(cal[200])
    assert all(sp.idx(a) <= C - 12 for a in fold.val_dates)


# ===========================================================================
# 🔒 H-C：purge 消融反证（核心硬闸门）
# ===========================================================================
def _synthetic_overlap_panel(seed: int = 7, n_tickers: int = 150, n_days: int = 320, H: int = 12):
    """合成面板：日收益 iid（无持久横截面信号）→ 唯一的横截面相关来自标签窗口【重叠】。

    label y(i,a) = a 之后 H 个交易日的累计收益。相邻 as_of 标签窗口重叠 → 横截面排名相关；
    非重叠 as_of 互相独立。这样：
      - 无 purge：训练末标签与边界测试标签重叠 → leak-probe IC 虚高；
      - 有 purge：训练末与测试标签不重叠 → IC ~0。
    """
    rng = np.random.default_rng(seed)
    cal = list(pd.bdate_range("2020-01-01", periods=n_days))
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    # 日收益矩阵 (days × tickers)，标准正态，完全 iid（无持久 drift）
    R = rng.standard_normal((n_days, n_tickers))
    cum = np.cumsum(R, axis=0)

    grid = list(pd.bdate_range("2020-01-01", periods=n_days))[::5]
    rows = []
    pos = {d: i for i, d in enumerate(cal)}
    for a in grid:
        ia = pos[a]
        if ia + H >= n_days:
            continue
        # 前瞻 H 日累计收益 = cum[ia+H] - cum[ia]
        fwd = cum[ia + H] - cum[ia]
        for j, tk in enumerate(tickers):
            rows.append((pd.Timestamp(a), tk, float(fwd[j]), 0.0))
    df = pd.DataFrame(rows, columns=["as_of", "ticker", "forward_return_12d", "feat_dummy"])
    return df.set_index(["as_of", "ticker"]), cal, grid


def test_purge_ablation_counterproof():
    """关掉 purge 必见 IC 虚高；开/关相等 = purge 未生效 = FAIL。"""
    panel, cal, grid = _synthetic_overlap_panel()
    cutoff = grid[len(grid) // 2]
    # 把 OOS 限制在 cutoff 之后的短窗，让边界重叠点主导无 purge 的均值
    cutoff_pos = {d: i for i, d in enumerate(cal)}[cutoff]
    oos_end = cal[min(cutoff_pos + 25, len(cal) - 1)]

    res = purge_ablation(
        panel, cal, grid, cutoff,
        feature_cols=["feat_dummy"], label_col="forward_return_12d",
        predictor_factory=CarryForwardLabelProbe,
        horizon=12, embargo=12,
        oos_start=cutoff, oos_end=oos_end,
    )
    # 核心断言：无 purge 的 IC 明显高于 purge 后
    assert res.purge_effective, (
        f"purge 未生效：ic_purged={res.ic_purged:.3f} ic_nopurge={res.ic_nopurge:.3f}")
    assert res.ic_nopurge - res.ic_purged > 0.05, (
        f"IC 抬升不足：inflation={res.inflation:.3f} "
        f"(purged={res.ic_purged:.3f}, nopurge={res.ic_nopurge:.3f})")
    # purge 后 IC 应塌回近 0（重叠被剔除）
    assert abs(res.ic_purged) < 0.10, f"purged IC 未塌回近0: {res.ic_purged:.3f}"


def test_purge_ablation_purged_has_gap():
    # 反证的两个切分确实不同：purged 训练末远离 cutoff，nopurge 紧贴
    panel, cal, grid = _synthetic_overlap_panel()
    cutoff = grid[len(grid) // 2]
    sp_p = PurgedWalkForwardSplit(cal, horizon=12, embargo=12)
    sp_n = PurgedWalkForwardSplit(cal, horizon=0, embargo=0)
    fp = sp_p.split(grid, cutoff)
    fn = sp_n.split(grid, cutoff)
    C = sp_p.idx(cutoff)
    assert sp_p.idx(fp.train_dates[-1]) <= C - 12
    assert sp_n.idx(fn.train_dates[-1]) <= C
    assert sp_n.idx(fn.train_dates[-1]) > sp_p.idx(fp.train_dates[-1])
