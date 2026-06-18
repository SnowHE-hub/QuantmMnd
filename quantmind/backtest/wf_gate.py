"""quantmind.backtest.wf_gate — Walk-forward 闸门编排器（P1 决定性闸门）.

依据 ``docs/plans/walkforward_design_plan.md`` §4–§8、§12。**新建**，复用既有
``LGBMRankerModel`` / ``build_lgbm_arrays`` / ``ExecutionSimulator`` / ``PerformanceMetrics``，
**不覆盖既有两套 walk-forward**。

硬条件落实
----------
- **H-A 根除 OOS 方向泄漏**：方向（符号翻转）**只用训练/验证段**决定，``decide_direction``
  绝不读 OOS 标签；``auto_flip=False``。
- **H-C purge 消融反证**：``purge_ablation`` 用同一 leak-probe 在 purge 开/关两切分上跑，
  关掉必见 IC 虚高（见 ``tests/test_wf_split.py``）。
- 幸存者上界标注：``gate_meta`` 强制含。
- 标签 ≠ NAV 来源：本模块只产 IC（用标签）与"目标持仓"（喂 BacktestEngine），
  NAV 由引擎实际成交价链算，不引用 ``forward_return_*``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np
import pandas as pd

from quantmind.backtest.wf_metrics import (
    aggregate_ic, cross_sectional_ic, quantile_monotonicity,
)
from quantmind.backtest.wf_split import PurgedWalkForwardSplit, WFFold


# ============================================================================
# Predictor 接口（model-agnostic，便于注入 leak-probe 做消融反证）
# ============================================================================
class Predictor(Protocol):
    def fit(self, panel: pd.DataFrame, train_dates: list[pd.Timestamp],
            val_dates: list[pd.Timestamp], feature_cols: list[str],
            label_col: str) -> "Predictor": ...

    def score(self, panel: pd.DataFrame, test_date: pd.Timestamp,
              feature_cols: list[str]) -> pd.Series: ...


class LGBMPredictor:
    """生产用：复用 build_lgbm_arrays + LGBMRankerModel（``auto_flip=False``）.

    方向**不**在模型内部用 IC 决定（auto_flip 关）；方向由 gate 的
    ``decide_direction`` 仅凭训练/验证段决定（H-A）。

    **Categorical 支持**：``feature_cols`` 含 string/object/category 列时，
    自动检测并以 LightGBM 原生 categorical_feature 形式传入（无 one-hot；
    NaN→-1 单独成类）。编码字典由 ``panel`` 列的全局 unique 决定，train/test
    一致，无数据泄漏。
    """

    def __init__(self, n_quantiles: int = 5, **lgbm_kwargs):
        self.n_quantiles = n_quantiles
        self.lgbm_kwargs = lgbm_kwargs
        self._model = None
        self._feature_cols: list[str] = []
        self._cat_cols: list[str] = []

    def fit(self, panel, train_dates, val_dates, feature_cols, label_col):
        from quantmind.models.factor_model import (
            build_lgbm_arrays, detect_categorical_cols,
        )
        from quantmind.models.lgbm_ranker import LGBMRankerModel

        self._feature_cols = list(feature_cols)
        self._cat_cols = detect_categorical_cols(panel, feature_cols)
        train_core = [d for d in train_dates if d not in set(val_dates)] or train_dates
        X, y, g = build_lgbm_arrays(panel, train_core, feature_cols, label_col, self.n_quantiles)
        Xv, yv, gv = build_lgbm_arrays(panel, val_dates, feature_cols, label_col, self.n_quantiles)
        self._model = LGBMRankerModel(auto_flip=False, **self.lgbm_kwargs)
        cat_arg = self._cat_cols if self._cat_cols else None
        if len(Xv) and len(yv):
            self._model.fit(X, y, groups=g, X_val=Xv, y_val=yv, groups_val=gv,
                            feature_names=feature_cols, categorical_feature=cat_arg)
        else:
            self._model.fit(X, y, groups=g, feature_names=feature_cols,
                            categorical_feature=cat_arg)
        return self

    def score(self, panel, test_date, feature_cols):
        from quantmind.models.factor_model import encode_categorical_codes
        # score 时用与 fit 一致的 categorical 编码（同 panel 同 unique → 同 codes）
        if self._cat_cols:
            panel = encode_categorical_codes(panel, self._cat_cols)
        xs = panel.xs(pd.Timestamp(test_date), level="as_of")
        X = xs[feature_cols].fillna(0.0).values.astype(np.float32)
        s = self._model.predict(X)
        return pd.Series(s, index=xs.index, name="score")


class CarryForwardLabelProbe:
    """**仅用于 purge 消融反证** 的泄漏探针（非生产模型）.

    模拟"靠标签窗口重叠作弊"的模型：fit 时记下每只票在【最后一个训练 as_of】的标签；
    score 时直接用该记忆标签当预测分。

    - purge **关**：最后训练 as_of 紧贴 cutoff，其标签窗口与测试标签重叠 → 记忆标签
      与测试实现收益强相关 → IC 虚高。
    - purge **开**：最后训练 as_of 距 cutoff ≥ H+E，标签窗口不重叠 → 记忆标签与测试无关
      → IC 塌回 ~0。
    两者相等即 purge 未生效。等同面板侧 PIT 篡改探针。
    """

    def __init__(self) -> None:
        self._memo: pd.Series | None = None

    def fit(self, panel, train_dates, val_dates, feature_cols, label_col):
        if not train_dates:
            self._memo = pd.Series(dtype="float64")
            return self
        last = max(train_dates)
        xs = panel.xs(pd.Timestamp(last), level="as_of")
        self._memo = xs[label_col].astype("float64")
        return self

    def score(self, panel, test_date, feature_cols):
        xs = panel.xs(pd.Timestamp(test_date), level="as_of")
        memo = self._memo if self._memo is not None else pd.Series(dtype="float64")
        return memo.reindex(xs.index).rename("score")


# ============================================================================
# 方向（H-A：只用训练/验证段，绝不碰 OOS）
# ============================================================================
def decide_direction(
    predictor: Predictor,
    panel: pd.DataFrame,
    val_dates: list[pd.Timestamp],
    feature_cols: list[str],
    label_col: str,
) -> int:
    """符号方向 = sign(验证段 in-sample IC 均值)。**只读 val（属训练段），绝不读 OOS。**

    Returns +1（正向）或 -1（翻转）。val 为空时回退 +1（先验正向）。
    """
    ics = []
    for d in val_dates:
        try:
            s = predictor.score(panel, d, feature_cols)
            realized = panel.xs(pd.Timestamp(d), level="as_of")[label_col]
            ic = cross_sectional_ic(s, realized)
            if ic == ic:
                ics.append(ic)
        except KeyError:
            continue
    if not ics:
        return 1
    return 1 if float(np.mean(ics)) >= 0 else -1


# ============================================================================
# OOS 截面 IC 评估
# ============================================================================
@dataclass
class OOSEvalResult:
    ic_series: pd.Series           # index=test as_of, value=rank IC（已乘方向）
    direction: int
    monotonicity: dict             # test_date -> Spearman(组序,组均)

    @property
    def ic_summary(self) -> dict:
        return aggregate_ic(self.ic_series)


def evaluate_oos_ic(
    panel: pd.DataFrame,
    fold: WFFold,
    predictor: Predictor,
    feature_cols: list[str],
    label_col: str,
    *,
    direction: int | None = None,
    n_quantiles: int = 5,
) -> OOSEvalResult:
    """训练 predictor，方向由 val 定（H-A），再在每个 OOS 截面算 rank IC。"""
    predictor.fit(panel, fold.train_dates, fold.val_dates, feature_cols, label_col)
    if direction is None:
        direction = decide_direction(predictor, panel, fold.val_dates, feature_cols, label_col)

    ics: dict[pd.Timestamp, float] = {}
    mono: dict[str, float] = {}
    for d in fold.test_dates:
        try:
            xs = panel.xs(pd.Timestamp(d), level="as_of")
        except KeyError:
            continue
        s = predictor.score(panel, d, feature_cols) * direction
        realized = xs[label_col]
        ics[pd.Timestamp(d)] = cross_sectional_ic(s, realized)
        mono[str(pd.Timestamp(d).date())] = quantile_monotonicity(s, realized, n_quantiles).order_corr
    ic_series = pd.Series(ics, name="rank_ic").sort_index()
    return OOSEvalResult(ic_series=ic_series, direction=direction, monotonicity=mono)


# ============================================================================
# 🔒 H-C：purge 消融反证
# ============================================================================
@dataclass
class PurgeAblationResult:
    ic_purged: float
    ic_nopurge: float
    purged_n: int
    nopurge_n: int

    @property
    def inflation(self) -> float:
        """无 purge 相对 purge 的 IC 抬升量（应 > 0）。"""
        return self.ic_nopurge - self.ic_purged

    @property
    def purge_effective(self) -> bool:
        """purge 是否在生效：关掉后 IC 明显虚高。"""
        return self.inflation > 1e-6


def purge_ablation(
    panel: pd.DataFrame,
    trading_days: Sequence,
    as_of_grid: Sequence,
    cutoff,
    feature_cols: list[str],
    label_col: str,
    *,
    predictor_factory: Callable[[], Predictor] = CarryForwardLabelProbe,
    horizon: int = 12,
    embargo: int = 12,
    oos_start=None,
    oos_end=None,
) -> PurgeAblationResult:
    """同一 cutoff、同一 predictor，跑 purge 开/关两次，对比 OOS IC 均值。

    purge **关** = ``H=0, E=0``（train 紧贴 test，复刻既有无 purge 行为）。
    期望 ``ic_nopurge`` 明显 > ``ic_purged``；相等 = purge 未生效 = FAIL（调用方断言）。
    """
    sp_purged = PurgedWalkForwardSplit(trading_days, horizon=horizon, embargo=embargo)
    sp_nopurge = PurgedWalkForwardSplit(trading_days, horizon=0, embargo=0)
    f_purged = sp_purged.split(as_of_grid, cutoff, oos_start=oos_start, oos_end=oos_end)
    f_nopurge = sp_nopurge.split(as_of_grid, cutoff, oos_start=oos_start, oos_end=oos_end)

    r_purged = evaluate_oos_ic(panel, f_purged, predictor_factory(), feature_cols, label_col, direction=1)
    r_nopurge = evaluate_oos_ic(panel, f_nopurge, predictor_factory(), feature_cols, label_col, direction=1)
    return PurgeAblationResult(
        ic_purged=float(r_purged.ic_series.mean()),
        ic_nopurge=float(r_nopurge.ic_series.mean()),
        purged_n=int(r_purged.ic_series.notna().sum()),
        nopurge_n=int(r_nopurge.ic_series.notna().sum()),
    )


# ============================================================================
# 通过线（§6，事前定死）+ 幸存者标注（§8）
# ============================================================================
@dataclass(frozen=True)
class PassLines:
    ic_mean: float = 0.03
    ic_ir: float = 0.4
    net_excess_annual: float = 0.05
    win_rate: float = 0.52
    max_drawdown: float = 0.25       # 上限（< 才过）
    monotonicity_min: float = 0.0    # 组序 vs 收益 Spearman > 0


def evaluate_gate(metrics: dict, lines: PassLines | None = None) -> dict:
    """对照通过线给 过/不过。基础线全满足=过闸（幸存者上界意义下）。"""
    lines = lines or PassLines()
    checks = {
        "ic_mean": metrics.get("ic_mean", float("nan")) > lines.ic_mean,
        "ic_ir": metrics.get("ic_ir", float("nan")) > lines.ic_ir,
        "net_excess_annual": metrics.get("net_excess_annual", float("nan")) > lines.net_excess_annual,
        "win_rate": metrics.get("win_rate", float("nan")) > lines.win_rate,
        "max_drawdown": metrics.get("max_drawdown", float("nan")) < lines.max_drawdown,
        "monotonicity": metrics.get("monotonicity", float("nan")) > lines.monotonicity_min,
    }
    passed = all(bool(v) for v in checks.values())
    return {"passed": passed, "checks": checks,
            "verdict": "PASS（乐观上界）" if passed else "FAIL"}


def gate_meta(extra: dict | None = None) -> dict:
    """闸门输出 meta，**强制含**幸存者上界标注（§8）。"""
    meta = {
        "survivorship_caveat": (
            "面板 v5 源头即幸存者池（1374 票全交易到 2026、stock_basic 100% L、delist_date 0）。"
            "本闸门一切结论 = 乐观上界（optimistic upper bound）。"
            "过闸≠真有 alpha（必要非充分，真实含退市表现更低）；不过闸=强负面信号。"
        ),
        "deferred": "退市票行情回补为后续项（承接面板 §0.5 第3点），回补后须复核。",
        "label_not_nav": "IC 用 forward_return_12d 标签；NAV 由 BacktestEngine 实际成交价链算，不引用标签列。",
        "direction_policy": "auto_flip=False；方向仅由训练/验证段决定，绝不碰 OOS（H-A）。",
    }
    if extra:
        meta.update(extra)
    return meta


__all__ = [
    "Predictor", "LGBMPredictor", "CarryForwardLabelProbe",
    "decide_direction", "OOSEvalResult", "evaluate_oos_ic",
    "PurgeAblationResult", "purge_ablation",
    "PassLines", "evaluate_gate", "gate_meta",
]
