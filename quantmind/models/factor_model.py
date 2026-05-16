"""quantmind.models.factor_model — 因子模型基类与训练工具.

包含：
  - FactorModel        抽象基类，定义 fit/predict/save/load 接口
  - WalkForwardSplit   扩展窗口（expanding window）时序分割器
  - CrossSectionalLabel  将截面连续收益率转为分位数整数标签
  - FactorSelector     从 top_factors.json 中按 IC_IR 阈值选取因子

数据流
======
::

    panel[(as_of, ticker)] = [41 factors] + [forward_return_21d, forward_return_63d]
                │
                ▼
    FactorSelector.select_relaxed(ir_threshold=0.2)  → ~11 features
                │
                ▼
    WalkForwardSplit.split(dates)
        → [(train_dates, val_dates, test_date), ...]     # expanding window
                │
                ▼
    CrossSectionalLabel.transform(panel, dates)
        → integer rank 0-4 (Q1=lowest → Q5=highest)
                │
                ▼
    LGBMRanker.fit(X_train, y_train, groups_train) + eval(X_val, y_val)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "CrossSectionalLabel",
    "FactorModel",
    "FactorSelector",
    "WalkForwardFold",
    "WalkForwardSplit",
    "build_lgbm_arrays",
]


# ============================================================================
# 抽象基类
# ============================================================================


class FactorModel(ABC):
    """所有因子模型的抽象基类.

    子类必须实现 fit / predict；save/load 提供基于 joblib 的默认实现。
    """

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        groups: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        groups_val: np.ndarray | None = None,
    ) -> "FactorModel":
        """拟合模型."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测评分（越高越好）."""

    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FactorModel":
        import joblib

        return joblib.load(path)

    @property
    def feature_names(self) -> list[str] | None:
        return getattr(self, "_feature_names", None)


# ============================================================================
# Walk-Forward 分割
# ============================================================================


@dataclass
class WalkForwardFold:
    """单个 Walk-Forward 折叠."""

    fold_idx: int
    train_dates: list[pd.Timestamp]
    val_dates: list[pd.Timestamp]      # 训练集末尾切出的验证集（用于 early stopping）
    test_date: pd.Timestamp


class WalkForwardSplit:
    """扩展窗口（Expanding Window）Walk-Forward 分割器.

    将时序截面日期列表切分为若干 (train, val, test) 三元组，确保严格时序。
    没有任何未来数据泄漏：test_date > val_dates > train_dates。

    参数
    ----
    min_train_periods:
        最少训练截面数（季频默认 10 ≈ 2.5 年）。
    n_val_periods:
        从 train 末尾切出的验证截面数（用于 early stopping），默认 2。

    示例
    ----
    22 个季度截面，min_train_periods=10，n_val_periods=2：
        Fold  0: train=[0..7], val=[8..9], test=[10]
        Fold  1: train=[0..8], val=[9..10], test=[11]
        ...
        Fold 11: train=[0..18], val=[19..20], test=[21]
    → 共 12 个测试折叠
    """

    def __init__(
        self,
        min_train_periods: int = 10,
        n_val_periods: int = 2,
    ) -> None:
        if min_train_periods <= n_val_periods:
            raise ValueError("min_train_periods must be > n_val_periods")
        self.min_train_periods = min_train_periods
        self.n_val_periods = n_val_periods

    def split(self, dates: list[pd.Timestamp]) -> list[WalkForwardFold]:
        """按 dates 生成所有折叠.

        Args:
            dates: 已排序的时间戳列表（每个元素是 panel 的 as_of 截面）

        Returns:
            WalkForwardFold 列表，按时间顺序
        """
        dates = sorted(set(dates))
        n = len(dates)
        if n < self.min_train_periods + 1:
            raise ValueError(
                f"Not enough dates ({n}) for min_train_periods={self.min_train_periods}. "
                "Decrease min_train_periods or provide more data."
            )

        folds: list[WalkForwardFold] = []
        for test_idx in range(self.min_train_periods, n):
            all_before = dates[:test_idx]            # [0, test_idx)
            val_dates = all_before[-self.n_val_periods :]
            train_dates = all_before[: -self.n_val_periods]
            if not train_dates:
                continue
            folds.append(
                WalkForwardFold(
                    fold_idx=len(folds),
                    train_dates=train_dates,
                    val_dates=val_dates,
                    test_date=dates[test_idx],
                )
            )
        return folds

    def __repr__(self) -> str:
        return (
            f"WalkForwardSplit(min_train_periods={self.min_train_periods}, "
            f"n_val_periods={self.n_val_periods})"
        )


# ============================================================================
# 截面标签（分位数整数 0-4）
# ============================================================================


class CrossSectionalLabel:
    """将截面连续收益率转换为分位数整数标签.

    LightGBM lambdarank 需要非负整数 label：值越大越相关（越好）。
    这里将同截面的 forward_return 按 n_quantiles 等分为 0..n_quantiles-1。

    Q0 = 最低 20%（空头端），Q4 = 最高 20%（多头端）。
    """

    def __init__(self, n_quantiles: int = 5) -> None:
        self.n_quantiles = n_quantiles

    def transform(
        self,
        panel: pd.DataFrame,
        dates: list[pd.Timestamp],
        label_col: str = "forward_return_21d",
    ) -> pd.Series:
        """为指定日期集返回整数标签 Series（MultiIndex[as_of, ticker]）.

        Args:
            panel: MultiIndex[as_of, ticker] DataFrame
            dates: 需要处理的截面日期列表
            label_col: 目标收益率列名

        Returns:
            pd.Series，index=MultiIndex[as_of, ticker]，values=int 0..n_quantiles-1
        """
        out: list[pd.Series] = []
        for d in sorted(dates):
            if d not in panel.index.get_level_values("as_of"):
                continue
            xs = panel.xs(d, level="as_of")[label_col].dropna()
            if len(xs) < self.n_quantiles * 3:
                continue
            try:
                labels = pd.qcut(
                    xs, self.n_quantiles, labels=False, duplicates="drop"
                ).astype(int)
            except ValueError:
                # 极端情况：所有值相同，退化为均匀 0
                labels = pd.Series(0, index=xs.index, dtype=int)

            labels.index = pd.MultiIndex.from_tuples(
                [(d, t) for t in labels.index], names=["as_of", "ticker"]
            )
            out.append(labels)
        if not out:
            return pd.Series(dtype=int)
        return pd.concat(out)


# ============================================================================
# 因子选择器
# ============================================================================


class FactorSelector:
    """从 top_factors.json 中选取因子.

    支持两种模式：
    - ``strict``：直接读取 ``selected_factors``（通过 Phase 2.2 双阈值的因子）
    - ``relaxed``：从 ``factor_stats`` 中筛选 |IC_IR| >= ir_threshold 的因子
    """

    def __init__(self, top_factors_path: str | Path) -> None:
        self._path = Path(top_factors_path)
        with self._path.open(encoding="utf-8") as f:
            self._data: dict[str, Any] = json.load(f)

    def select_strict(self) -> list[str]:
        """返回严格通过双阈值筛选的因子列表."""
        return list(self._data.get("selected_factors", []))

    def select_relaxed(
        self,
        ir_threshold: float = 0.2,
        exclude: list[str] | None = None,
    ) -> list[str]:
        """返回 |IC_IR| >= ir_threshold 的因子列表，按 |IC_IR| 降序.

        Args:
            ir_threshold: IC_IR 绝对值阈值，默认 0.2
            exclude: 强制排除的因子名列表（如全 NaN 的因子）
        """
        stats: dict[str, dict] = self._data.get("factor_stats", {})
        exclude_set = set(exclude or [])
        selected = []
        for fname, s in stats.items():
            if fname in exclude_set:
                continue
            ic_ir = s.get("ic_ir")
            if ic_ir is None or (isinstance(ic_ir, float) and np.isnan(ic_ir)):
                continue
            if abs(ic_ir) >= ir_threshold:
                selected.append((fname, abs(ic_ir)))

        # 按 |IC_IR| 降序
        selected.sort(key=lambda x: x[1], reverse=True)
        return [f for f, _ in selected]

    def get_stats(self, factors: list[str]) -> pd.DataFrame:
        """返回指定因子的 IC 统计表."""
        stats = self._data.get("factor_stats", {})
        rows = []
        for f in factors:
            if f in stats:
                rows.append({"factor": f, **stats[f]})
        return pd.DataFrame(rows).set_index("factor") if rows else pd.DataFrame()


# ============================================================================
# LightGBM 数据格式化辅助
# ============================================================================


def build_lgbm_arrays(
    panel: pd.DataFrame,
    dates: list[pd.Timestamp],
    feature_cols: list[str],
    label_col: str = "forward_return_21d",
    n_quantiles: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将 panel 子集转换为 LightGBM lambdarank 所需的数组格式.

    LightGBM 要求数据按 group 连续排列（同一截面日期的所有行排在一起）。

    Returns:
        (X, y, groups) 三元组:
            X:      (n_samples, n_features) float32
            y:      (n_samples,) int32  0..n_quantiles-1
            groups: (n_dates,) int   每个 group（截面）的样本数
    """
    labeler = CrossSectionalLabel(n_quantiles=n_quantiles)
    label_series = labeler.transform(panel, dates, label_col)

    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    group_sizes: list[int] = []

    for d in sorted(dates):
        if d not in panel.index.get_level_values("as_of"):
            continue
        xs = panel.xs(d, level="as_of")
        # 只保留有标签的行
        idx_with_label = label_series.xs(d, level="as_of").index if d in label_series.index.get_level_values("as_of") else pd.Index([])
        if len(idx_with_label) == 0:
            continue
        xs_valid = xs.loc[xs.index.intersection(idx_with_label)]
        if xs_valid.empty:
            continue

        x = xs_valid[feature_cols].fillna(0.0).values.astype(np.float32)
        y_vals = label_series.xs(d, level="as_of").reindex(xs_valid.index).values.astype(np.int32)

        X_rows.append(x)
        y_rows.extend(y_vals.tolist())
        group_sizes.append(len(xs_valid))

    if not X_rows:
        return np.empty((0, len(feature_cols)), dtype=np.float32), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    X_out = np.vstack(X_rows)
    y_out = np.array(y_rows, dtype=np.int32)
    groups_out = np.array(group_sizes, dtype=np.int32)

    return X_out, y_out, groups_out
