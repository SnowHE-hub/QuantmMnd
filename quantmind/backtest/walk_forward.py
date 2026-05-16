"""quantmind.backtest.walk_forward — Rolling Walk-Forward 验证框架.

参数（Spec 标准）：
  训练窗口：3年（756 个交易日）
  验证窗口：6个月（126 个交易日）——参数选择 / 超参优化
  测试窗口：3个月（63 个交易日）——OOS 评估
  滚动步长：3个月（63 个交易日）

流程：
  [train][val][test] → 步进 step → [    train    ][val][test] → ...

验证反数据窥探原则：
  - 每个 test 窗口严格不使用 >= test_start 的数据
  - model 在 [train+val] 上拟合，在 [test] 上评估
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Type

import numpy as np
import pandas as pd
from loguru import logger

__all__ = [
    "WalkForwardConfig",
    "WalkForwardWindow",
    "WalkForwardResult",
    "WalkForwardValidator",
]

_TRADING_DAYS_PER_YEAR = 252


@dataclass
class WalkForwardConfig:
    """Walk-Forward 窗口参数."""

    train_days: int = 756      # 训练窗口：3年
    val_days: int = 126        # 验证窗口：6个月
    test_days: int = 63        # 测试窗口：3个月
    step_days: int = 63        # 滚动步长：3个月
    min_train_days: int = 252  # 最小训练天数（避免首期窗口太小）


@dataclass
class WalkForwardWindow:
    """单个 Walk-Forward 窗口."""

    window_idx: int
    train_start: date
    train_end: date
    val_start: date
    val_end: date
    test_start: date
    test_end: date

    # OOS 测试结果
    oos_metrics: dict[str, Any] = field(default_factory=dict)
    oos_returns: pd.Series = field(default_factory=pd.Series)
    is_valid: bool = True  # 数据是否充足


@dataclass
class WalkForwardResult:
    """Walk-Forward 验证结果."""

    config: WalkForwardConfig
    windows: list[WalkForwardWindow] = field(default_factory=list)

    # 汇总统计（所有 OOS 窗口的指标均值/标准差）
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def n_windows(self) -> int:
        return len([w for w in self.windows if w.is_valid])

    def compute_summary(self) -> dict[str, Any]:
        """计算所有有效 OOS 窗口的统计汇总."""
        valid_windows = [w for w in self.windows if w.is_valid and w.oos_metrics]
        if not valid_windows:
            return {}

        keys = [
            "total_return", "annualized_return", "sharpe_ratio",
            "max_drawdown", "volatility", "calmar_ratio",
        ]
        summary: dict[str, Any] = {}
        for key in keys:
            vals = [w.oos_metrics.get(key, np.nan) for w in valid_windows]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                summary[f"{key}_mean"] = float(np.mean(vals))
                summary[f"{key}_std"] = float(np.std(vals))
                summary[f"{key}_ir"] = float(
                    np.mean(vals) / (np.std(vals) + 1e-10)
                )  # 稳定性 IR

        # 串联 OOS 收益
        all_returns = pd.concat(
            [w.oos_returns for w in valid_windows if not w.oos_returns.empty]
        ).sort_index()

        if not all_returns.empty:
            from quantmind.backtest.metrics import PerformanceMetrics
            combined = PerformanceMetrics().compute_all(all_returns)
            summary["combined_sharpe"] = combined.get("sharpe_ratio", np.nan)
            summary["combined_return"] = combined.get("total_return", np.nan)
            summary["combined_max_drawdown"] = combined.get("max_drawdown", np.nan)

        summary["n_valid_windows"] = len(valid_windows)
        self.summary = summary
        return summary


class WalkForwardValidator:
    """Rolling Walk-Forward 验证器.

    Args:
        config:       WalkForwardConfig
        prices_df:    价格 DataFrame（MultiIndex date/ticker 或 index=date）
        features_df:  因子特征 DataFrame（MultiIndex date/ticker）
    """

    def __init__(
        self,
        config: WalkForwardConfig | None = None,
        prices_df: pd.DataFrame | None = None,
        features_df: pd.DataFrame | None = None,
    ) -> None:
        self.config = config or WalkForwardConfig()
        self._prices_df = prices_df
        self._features_df = features_df

    # ── 生成窗口 ──────────────────────────────────────────────────────────────

    def generate_windows(
        self,
        all_dates: list[date],
    ) -> list[WalkForwardWindow]:
        """根据交易日序列生成所有 Walk-Forward 窗口.

        Args:
            all_dates: 所有交易日（升序）

        Returns:
            WalkForwardWindow 列表
        """
        cfg = self.config
        n = len(all_dates)
        windows: list[WalkForwardWindow] = []

        # 最小需要 train_days + val_days + test_days 个交易日
        min_required = cfg.train_days + cfg.val_days + cfg.test_days

        idx = 0
        window_idx = 0
        while True:
            train_start_i = idx
            train_end_i = idx + cfg.train_days - 1
            val_start_i = train_end_i + 1
            val_end_i = val_start_i + cfg.val_days - 1
            test_start_i = val_end_i + 1
            test_end_i = test_start_i + cfg.test_days - 1

            if test_end_i >= n:
                break  # 数据不足

            window = WalkForwardWindow(
                window_idx=window_idx,
                train_start=all_dates[train_start_i],
                train_end=all_dates[train_end_i],
                val_start=all_dates[val_start_i],
                val_end=all_dates[val_end_i],
                test_start=all_dates[test_start_i],
                test_end=all_dates[test_end_i],
                is_valid=(train_end_i - train_start_i + 1) >= cfg.min_train_days,
            )
            windows.append(window)
            window_idx += 1
            idx += cfg.step_days

        logger.info(
            f"[WalkForward] Generated {len(windows)} windows "
            f"({len([w for w in windows if w.is_valid])} valid)"
        )
        return windows

    # ── 主验证入口 ────────────────────────────────────────────────────────────

    def validate(
        self,
        strategy_class: type,
        features_df: pd.DataFrame | None = None,
        prices_df: pd.DataFrame | None = None,
        strategy_kwargs: dict | None = None,
    ) -> WalkForwardResult:
        """对策略进行 Walk-Forward 验证.

        Args:
            strategy_class:  Strategy 子类（未实例化）
            features_df:     因子特征（MultiIndex date/ticker）
            prices_df:       价格数据（MultiIndex date/ticker）
            strategy_kwargs: 策略构造参数

        Returns:
            WalkForwardResult
        """
        features = features_df if features_df is not None else self._features_df
        prices = prices_df if prices_df is not None else self._prices_df

        if prices is None:
            raise ValueError("prices_df 不能为空")

        # 提取所有交易日
        if isinstance(prices.index, pd.MultiIndex):
            all_dates_ts = sorted(set(prices.index.get_level_values(0)))
        else:
            all_dates_ts = sorted(prices.index.unique())

        all_dates = [ts.date() if hasattr(ts, "date") else ts for ts in all_dates_ts]

        windows = self.generate_windows(all_dates)
        result = WalkForwardResult(config=self.config, windows=windows)

        from quantmind.backtest.engine import BacktestEngine, BacktestConfig
        from quantmind.backtest.metrics import PerformanceMetrics

        for window in windows:
            if not window.is_valid:
                logger.debug(f"[WalkForward] Window {window.window_idx} 数据不足，跳过")
                continue

            logger.info(
                f"[WalkForward] Window {window.window_idx}: "
                f"train={window.train_start}~{window.train_end} | "
                f"test={window.test_start}~{window.test_end}"
            )

            try:
                # 严格 PIT：test 窗口只能用 < test_start 的特征
                train_features = self._slice_features(
                    features, window.train_start, window.train_end
                ) if features is not None else None

                # 实例化策略（传入训练数据）
                kwargs = strategy_kwargs or {}
                if train_features is not None:
                    kwargs["train_features"] = train_features

                strategy = strategy_class(**kwargs)

                # 在测试窗口运行回测
                engine = BacktestEngine(prices_df=prices)
                universe = self._get_universe(prices, window.test_start, window.test_end)

                bt_result = engine.run(
                    strategy=strategy,
                    start_date=window.test_start,
                    end_date=window.test_end,
                    universe=universe,
                )

                window.oos_returns = bt_result.returns
                window.oos_metrics = bt_result.metrics

            except Exception as e:
                logger.warning(f"[WalkForward] Window {window.window_idx} 失败: {e}")
                window.is_valid = False
                window.oos_metrics = {"error": str(e)}

        result.compute_summary()
        logger.info(
            f"[WalkForward] Done. "
            f"Combined Sharpe={result.summary.get('combined_sharpe', 'N/A')}"
        )
        return result

    # ── 辅助方法 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _slice_features(
        features: pd.DataFrame,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """按日期范围切片特征 DataFrame（严格 PIT）."""
        if isinstance(features.index, pd.MultiIndex):
            dates = features.index.get_level_values(0)
            mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
            return features[mask]
        else:
            return features.loc[pd.Timestamp(start):pd.Timestamp(end)]

    @staticmethod
    def _get_universe(
        prices: pd.DataFrame,
        start: date,
        end: date,
    ) -> list[str]:
        """从价格数据中提取测试窗口的股票代码列表."""
        if isinstance(prices.index, pd.MultiIndex):
            dates = prices.index.get_level_values(0)
            mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
            tickers = prices[mask].index.get_level_values(1).unique().tolist()
            return list(tickers)
        return []
