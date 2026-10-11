"""T-010：主线 A 样本外切分的 PIT 性质."""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_line_a_oos import LABEL_TD, label_ends_before, oos_folds

QUARTERS = list(pd.date_range("2019-03-31", periods=30, freq="QE"))
# 模拟 A 股日历：工作日去掉春节与国庆各 5 个交易日（一个季度约 58–61 个交易日 < 63）
_B = pd.bdate_range("2019-01-01", "2027-12-31")
_HOLI = {d for y in range(2019, 2028) for m in (2, 10) for d in _B[(_B.year == y) & (_B.month == m)][:5]}
CAL = pd.DatetimeIndex([d for d in _B if d not in _HOLI])


def test_folds_are_strictly_before_test():
    for f in oos_folds(QUARTERS, gap=1, n_val=2, min_train=8):
        assert max(f.train) < min(f.val) < f.test
        assert len(f.val) == 2 and len(f.train) >= 8


def test_gap_leaves_one_quarter_between_val_and_test():
    folds = oos_folds(QUARTERS, gap=1, n_val=2, min_train=8)
    for f in folds:
        i = QUARTERS.index(f.test)
        assert QUARTERS.index(f.val[-1]) == i - 2


def test_every_training_label_ends_before_test():
    """63 个交易日的标签窗口：gap=1 时所有训练/验证样本的目标日不晚于测试截面的基准日."""
    for f in oos_folds(QUARTERS, gap=1, n_val=2, min_train=8):
        for d in f.train + f.val:
            assert label_ends_before(d, f.test, CAL, LABEL_TD), (d, f.test)


def test_gap_zero_would_leak():
    """反例：gap=0 时最后一个验证截面的标签窗口跨过测试截面."""
    leaked = [
        (f.val[-1], f.test)
        for f in oos_folds(QUARTERS, gap=0, n_val=2, min_train=8)
        if not label_ends_before(f.val[-1], f.test, CAL, LABEL_TD)
    ]
    assert leaked


def test_first_fold_and_count():
    folds = oos_folds(QUARTERS, gap=1, n_val=2, min_train=8)
    assert folds[0].test == QUARTERS[11]          # 8 训练 + 2 验证 + 1 gap
    assert len(folds) == len(QUARTERS) - 11


def test_invalid_args():
    with pytest.raises(ValueError):
        oos_folds(QUARTERS, gap=-1)
