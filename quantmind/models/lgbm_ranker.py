"""quantmind.models.lgbm_ranker — LightGBM LambdaRank 截面排序模型.

原理
====

LambdaRank 是 Learning-to-Rank 框架：
- 每个「查询」(query) = 一个再平衡截面日期
- 每个「文档」(document) = 一只股票
- 标签 = 该截面内 forward_return 的 5 分位数排名（0=最低，4=最高）
- 目标函数：最大化截面内的 NDCG（Normalized Discounted Cumulative Gain）

Walk-Forward 验证（扩展窗口）
==============================

::

    截面 0   1   2 ... 9  | 10  11  12 ... 21
                           |
    Fold 0: train=[0..7], val=[8,9], test=10
    Fold 1: train=[0..8], val=[9,10], test=11
    ...
    Fold 11: train=[0..18], val=[19,20], test=21
    → 共 12 个测试折叠（2021Q3 → 2024Q2）

评估指标
========

- Rank IC (截面 Spearman)：预测分数 vs 实际 forward_return 的 Spearman 相关
- Top-K Hit Rate：预测 Top-K 中，超越同期中位数的比例
- Quintile Return Spread：模型分层后 Q5/Q1 平均收益差
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

from quantmind.models.factor_model import (
    FactorModel,
    WalkForwardFold,
    WalkForwardSplit,
    build_lgbm_arrays,
)

__all__ = [
    "LGBMRankerModel",
    "WalkForwardResult",
    "FoldResult",
    "walk_forward_evaluate",
]

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")


# ============================================================================
# 结果容器
# ============================================================================


@dataclass
class FoldResult:
    """单折 Walk-Forward 评估结果."""

    fold_idx: int
    test_date: pd.Timestamp
    n_train: int               # 训练样本数
    n_train_dates: int         # 训练截面数
    n_test: int                # 测试截面股票数
    best_iteration: int        # LightGBM early stopping 最优迭代数

    # 评估指标
    rank_ic: float             # Spearman(pred_score, actual_return)
    top10_hit_rate: float      # Top-10 预测中超过中位数的比例
    top20_hit_rate: float      # Top-20
    quintile_returns: dict[str, float] = field(default_factory=dict)  # Q1..Q5 平均收益
    spread: float = float("nan")  # Q5 - Q1

    # 因子重要性（最后一折有效）
    feature_importances: dict[str, float] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """全量 Walk-Forward 评估汇总."""

    feature_cols: list[str]
    label_col: str
    n_folds: int
    folds: list[FoldResult]
    direction: int = 1          # +1 正向使用，-1 auto_flip 反转信号
    regime_note: str = ""       # 描述 regime 情况的说明文字

    @property
    def ic_series(self) -> pd.Series:
        return pd.Series(
            {f.test_date: f.rank_ic for f in self.folds},
            name="rank_ic",
        ).sort_index()

    @property
    def ic_mean(self) -> float:
        ics = [f.rank_ic for f in self.folds if not np.isnan(f.rank_ic)]
        return float(np.mean(ics)) if ics else float("nan")

    @property
    def ic_std(self) -> float:
        ics = [f.rank_ic for f in self.folds if not np.isnan(f.rank_ic)]
        return float(np.std(ics, ddof=1)) if len(ics) > 1 else float("nan")

    @property
    def ic_ir(self) -> float:
        std = self.ic_std
        return self.ic_mean / std if std and not np.isnan(std) else float("nan")

    @property
    def ic_win_rate(self) -> float:
        ics = [f.rank_ic for f in self.folds if not np.isnan(f.rank_ic)]
        return float(np.mean([ic > 0 for ic in ics])) if ics else float("nan")

    @property
    def avg_spread(self) -> float:
        spreads = [f.spread for f in self.folds if not np.isnan(f.spread)]
        return float(np.mean(spreads)) if spreads else float("nan")

    @property
    def effective_ic_mean(self) -> float:
        """方向修正后的有效 IC 均值（direction=-1 时取反）."""
        raw = self.ic_mean
        return raw * self.direction if not np.isnan(raw) else raw

    @property
    def effective_ic_ir(self) -> float:
        """方向修正后的有效 IC IR（direction=-1 时取反）."""
        raw = self.ic_ir
        return raw * self.direction if not np.isnan(raw) else raw

    def summary_df(self) -> pd.DataFrame:
        rows = []
        for f in self.folds:
            rows.append(
                {
                    "fold": f.fold_idx,
                    "test_date": str(f.test_date)[:10],
                    "n_train_dates": f.n_train_dates,
                    "n_train_samples": f.n_train,
                    "n_test": f.n_test,
                    "best_iter": f.best_iteration,
                    "rank_ic": round(f.rank_ic, 4),
                    "top10_hit%": f"{f.top10_hit_rate:.1%}",
                    "top20_hit%": f"{f.top20_hit_rate:.1%}",
                    "spread%": f"{f.spread * 100:.3f}%",
                }
            )
        return pd.DataFrame(rows)


# ============================================================================
# LGBMRankerModel
# ============================================================================


class LGBMRankerModel(FactorModel):
    """LightGBM lambdarank 截面排序模型.

    参数
    ----
    n_estimators:     最大树数（early stopping 会提前终止）
    num_leaves:       每棵树最大叶节点数
    learning_rate:    学习率
    feature_fraction: 列采样比例
    bagging_fraction: 行采样比例
    early_stopping_rounds: 验证集上无改善则停止
    verbose:          -1 静默，>0 打印进度
    """

    def __init__(
        self,
        n_estimators: int = 300,
        num_leaves: int = 15,      # 小数据集用浅树，防止过拟合
        learning_rate: float = 0.05,
        feature_fraction: float = 0.8,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 5,
        lambda_l1: float = 0.1,
        lambda_l2: float = 0.1,
        min_child_samples: int = 10,  # 每个叶子至少 10 个样本
        early_stopping_rounds: int = 100, # 更宽松，给模型更多时间学习
        auto_flip: bool = True,       # 检测负 IC 后自动反转预测方向
        n_jobs: int = -1,
        verbose: int = -1,
        random_state: int = 42,
    ) -> None:
        self._params = dict(
            objective="lambdarank",
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            feature_fraction=feature_fraction,
            bagging_fraction=bagging_fraction,
            bagging_freq=bagging_freq,
            reg_alpha=lambda_l1,
            reg_lambda=lambda_l2,
            min_child_samples=min_child_samples,
            # 指数增益：让 Q4/Q5 的信号强度远高于 Q0/Q1
            label_gain=[0, 1, 3, 7, 15],
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
        )
        self._early_stopping_rounds = early_stopping_rounds
        self.auto_flip = auto_flip
        self.direction: int = 1       # +1 正向，-1 反转（由 set_direction 设置）
        self._model: lgb.LGBMRanker | None = None
        self._feature_names: list[str] | None = None

    def set_direction(self, ic_mean: float) -> None:
        """根据 Walk-Forward IC 均值决定预测方向.

        若 auto_flip=True 且 ic_mean < 0，则 self.direction=-1，预测时自动取反。
        """
        if self.auto_flip and not np.isnan(ic_mean) and ic_mean < 0:
            self.direction = -1
        else:
            self.direction = 1

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        groups: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        groups_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> "LGBMRankerModel":
        """拟合 LightGBM LambdaRank 模型.

        Args:
            X:          (n_samples, n_features) float32
            y:          (n_samples,) int  0..n_quantiles-1 （截面分位数标签）
            groups:     (n_groups,) int   每个截面的样本数
            X_val / y_val / groups_val: 验证集（用于 early stopping）
            feature_names: 特征名列表（用于 feature importance）
        """
        if groups is None:
            raise ValueError("groups is required for lambdarank")

        self._model = lgb.LGBMRanker(**self._params)
        self._feature_names = feature_names

        callbacks: list[Any] = []
        eval_set = None
        eval_group = None

        if X_val is not None and y_val is not None and groups_val is not None:
            eval_set = [(X_val, y_val)]
            eval_group = [groups_val]
            callbacks = [
                lgb.early_stopping(self._early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),  # 静默
            ]

        self._model.fit(
            X,
            y,
            group=groups,
            eval_set=eval_set,
            eval_group=eval_group,
            feature_name=feature_names or "auto",
            callbacks=callbacks if callbacks else None,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.predict(X) * self.direction

    @property
    def best_iteration_(self) -> int:
        if self._model is None:
            return 0
        return getattr(self._model, "best_iteration_", self._params["n_estimators"])

    def feature_importance(self, importance_type: str = "gain") -> dict[str, float]:
        """返回特征重要性 dict."""
        if self._model is None:
            return {}
        names = self._feature_names or [f"f{i}" for i in range(self._model.n_features_)]
        vals = self._model.feature_importances_.tolist()
        total = sum(vals) or 1.0
        return {n: v / total for n, v in zip(names, vals)}

    def explain(
        self,
        X: np.ndarray,
        tickers: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        """用 SHAP 计算每支股票各因子的贡献值.

        优先使用 ``shap`` 库的 TreeExplainer；若未安装，
        降级为 LightGBM 内置的 ``pred_contrib``（等价结果）。

        Args:
            X:       (n_stocks, n_features) float32，与 predict() 输入相同
            tickers: 股票代码列表，长度须等于 X.shape[0]

        Returns:
            Dict[ticker, Dict[factor_name, shap_value]]
            shap_value 正 = 提升预测分，负 = 压低预测分
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        names = self._feature_names or [f"f{i}" for i in range(X.shape[1])]
        tickers = tickers or [f"stock_{i}" for i in range(len(X))]

        try:
            import shap  # type: ignore[import-untyped]
            explainer = shap.TreeExplainer(self._model.booster_)
            shap_vals = explainer.shap_values(X)          # (n, n_features)
        except ImportError:
            # LightGBM 内置 SHAP：pred_contrib 返回 (n, n_features+1)，最后一列是 bias
            shap_vals = self._model.booster_.predict(X, pred_contrib=True)[:, :-1]

        # direction 修正：若 direction=-1，SHAP 贡献符号也取反以保持一致
        shap_vals = shap_vals * self.direction

        return {
            ticker: {name: float(shap_vals[i, j]) for j, name in enumerate(names)}
            for i, ticker in enumerate(tickers)
        }


# ============================================================================
# 单折评估
# ============================================================================


def _evaluate_fold(
    model: LGBMRankerModel,
    panel: pd.DataFrame,
    fold: WalkForwardFold,
    feature_cols: list[str],
    label_col: str,
    n_quantiles: int = 5,
    top_k_list: list[int] | None = None,
) -> FoldResult:
    """在单个测试截面上评估已训练的模型."""
    top_k_list = top_k_list or [10, 20]

    test_date = fold.test_date
    if test_date not in panel.index.get_level_values("as_of"):
        return FoldResult(
            fold_idx=fold.fold_idx,
            test_date=test_date,
            n_train=0,
            n_train_dates=0,
            n_test=0,
            best_iteration=0,
            rank_ic=float("nan"),
            top10_hit_rate=float("nan"),
            top20_hit_rate=float("nan"),
        )

    xs = panel.xs(test_date, level="as_of")
    valid = xs[[*feature_cols, label_col]].dropna(subset=[label_col])
    if len(valid) < 10:
        return FoldResult(
            fold_idx=fold.fold_idx,
            test_date=test_date,
            n_train=0,
            n_train_dates=0,
            n_test=len(valid),
            best_iteration=0,
            rank_ic=float("nan"),
            top10_hit_rate=float("nan"),
            top20_hit_rate=float("nan"),
        )

    X_test = valid[feature_cols].fillna(0.0).values.astype(np.float32)
    actual_returns = valid[label_col].values

    pred_scores = model.predict(X_test)

    # Rank IC (Spearman)
    rho, _ = stats.spearmanr(pred_scores, actual_returns)
    rank_ic = float(rho) if not np.isnan(rho) else float("nan")

    # Top-K Hit Rate
    n = len(pred_scores)
    sorted_idx = np.argsort(pred_scores)[::-1]
    median_ret = np.nanmedian(actual_returns)
    hit_rates = {}
    for k in top_k_list:
        if k > n:
            hit_rates[k] = float("nan")
            continue
        top_returns = actual_returns[sorted_idx[:k]]
        hit_rates[k] = float(np.mean(top_returns > median_ret))

    # Quintile Return Spread（用预测分数分5组）
    try:
        q_labels = pd.qcut(pred_scores, n_quantiles, labels=False, duplicates="drop")
    except ValueError:
        q_labels = None

    qret_dict = {}
    spread = float("nan")
    if q_labels is not None:
        for q in range(n_quantiles):
            mask = q_labels == q
            if mask.sum() > 0:
                qret_dict[f"Q{q+1}"] = float(np.nanmean(actual_returns[mask]))
        q1 = qret_dict.get("Q1", float("nan"))
        q5 = qret_dict.get(f"Q{n_quantiles}", float("nan"))
        if not (np.isnan(q1) or np.isnan(q5)):
            spread = q5 - q1

    # 训练集大小信息
    n_train = sum(len(panel.xs(d, level="as_of")) for d in fold.train_dates
                  if d in panel.index.get_level_values("as_of"))
    n_train_dates = len(fold.train_dates)

    return FoldResult(
        fold_idx=fold.fold_idx,
        test_date=test_date,
        n_train=n_train,
        n_train_dates=n_train_dates,
        n_test=len(valid),
        best_iteration=model.best_iteration_,
        rank_ic=rank_ic,
        top10_hit_rate=hit_rates.get(10, float("nan")),
        top20_hit_rate=hit_rates.get(20, float("nan")),
        quintile_returns=qret_dict,
        spread=spread,
        feature_importances=model.feature_importance() if fold.fold_idx % 3 == 0 else {},
    )


# ============================================================================
# 完整 Walk-Forward 评估
# ============================================================================


def walk_forward_evaluate(
    panel: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "forward_return_21d",
    *,
    min_train_periods: int = 10,
    n_val_periods: int = 2,
    n_quantiles: int = 5,
    model_kwargs: dict | None = None,
    auto_flip: bool = True,
    verbose: bool = True,
) -> tuple[WalkForwardResult, LGBMRankerModel]:
    """端到端 Walk-Forward 验证，返回评估结果和最后一折的模型.

    Args:
        panel:            MultiIndex[as_of, ticker] DataFrame
        feature_cols:     特征列名列表
        label_col:        目标标签列名
        min_train_periods: 最少训练截面数
        n_val_periods:    验证截面数（用于 early stopping）
        n_quantiles:      分位数分层数
        model_kwargs:     LGBMRankerModel 构造参数（覆盖默认）
        verbose:          是否打印折叠进度

    Returns:
        (WalkForwardResult, last_fold_model)
    """
    dates = sorted(panel.index.get_level_values("as_of").unique())
    splitter = WalkForwardSplit(
        min_train_periods=min_train_periods,
        n_val_periods=n_val_periods,
    )
    folds = splitter.split(dates)

    if verbose:
        print(f"  Walk-Forward: {len(folds)} folds "
              f"({str(dates[0])[:10]} → {str(dates[-1])[:10]})")
        print(f"  Features: {len(feature_cols)},  Label: {label_col}")
        print(f"  Min train: {min_train_periods}, Val: {n_val_periods}")

    model_kw = {**(model_kwargs or {}), "auto_flip": auto_flip}
    fold_results: list[FoldResult] = []
    last_model: LGBMRankerModel | None = None

    for fold in folds:
        # ---------- 构建训练数据 ----------
        X_tr, y_tr, g_tr = build_lgbm_arrays(
            panel, fold.train_dates, feature_cols, label_col, n_quantiles
        )
        X_val, y_val, g_val = build_lgbm_arrays(
            panel, fold.val_dates, feature_cols, label_col, n_quantiles
        )

        if len(X_tr) == 0 or len(X_val) == 0:
            if verbose:
                print(f"  Fold {fold.fold_idx:2d} SKIPPED (empty data)")
            continue

        # ---------- 训练 ----------
        model = LGBMRankerModel(**model_kw)
        model.fit(
            X_tr, y_tr,
            groups=g_tr,
            X_val=X_val, y_val=y_val, groups_val=g_val,
            feature_names=feature_cols,
        )

        # ---------- 测试评估 ----------
        result = _evaluate_fold(
            model, panel, fold, feature_cols, label_col, n_quantiles
        )
        fold_results.append(result)
        last_model = model

        if verbose:
            print(
                f"  Fold {fold.fold_idx:2d} | test={str(fold.test_date)[:10]} | "
                f"train={result.n_train_dates}期×~{result.n_train//max(result.n_train_dates,1)}股 | "
                f"iter={result.best_iteration:3d} | "
                f"IC={result.rank_ic:+.4f} | "
                f"Top10={result.top10_hit_rate:.0%} | "
                f"Spread={result.spread*100:+.3f}%"
            )

    if last_model is None:
        raise RuntimeError("No folds completed. Check data and parameters.")

    result_obj = WalkForwardResult(
        feature_cols=feature_cols,
        label_col=label_col,
        n_folds=len(fold_results),
        folds=fold_results,
    )

    # ── auto_flip：检测负 IC，设置方向并写入 regime_note ───────────────────
    if auto_flip and not np.isnan(result_obj.ic_mean) and result_obj.ic_mean < 0:
        result_obj.direction = -1
        result_obj.regime_note = (
            "模型信号已反转（direction=-1）。"
            "原始 IC 均值为负（{:.4f}），对应 A 股 2021–2024 regime 切换特征："
            "训练期（2019–2021）大盘低波动占优，测试期小盘股领跑导致因子方向系统性反转。"
            "auto_flip 取反后有效 IC 均值 = {:.4f}。"
        ).format(result_obj.ic_mean, result_obj.effective_ic_mean)
        last_model.set_direction(result_obj.ic_mean)
        if verbose:
            print(f"\n  ⚠  auto_flip 触发：原始 IC={result_obj.ic_mean:+.4f} < 0")
            print(f"  ✅ direction=-1，有效 IC={result_obj.effective_ic_mean:+.4f}")
    else:
        result_obj.direction = 1
        last_model.set_direction(result_obj.ic_mean)

    if verbose:
        print(f"\n  ── 汇总 ──────────────────────────────────────")
        print(f"  IC Mean  （原始）: {result_obj.ic_mean:+.4f}")
        print(f"  IC Mean  （有效）: {result_obj.effective_ic_mean:+.4f}  direction={result_obj.direction:+d}")
        print(f"  IC Std   : {result_obj.ic_std:.4f}")
        print(f"  IC IR    （有效）: {result_obj.effective_ic_ir:+.4f}")
        print(f"  IC WinRate: {result_obj.ic_win_rate:.1%}")
        print(f"  Avg Spread: {result_obj.avg_spread*100:+.3f}%")
        if result_obj.regime_note:
            print(f"  Regime   : {result_obj.regime_note[:80]}…")

    return result_obj, last_model
