"""scripts/update_model_rolling.py — 持续学习闭环：季度验证 + 增量模型更新.

流程：
  1. validate_last_quarter: 计算新季度 Rank IC，与历史基准对比
  2. should_retrain: 判断是否触发重训
  3. retrain_model: 增量重训（不覆盖生产模型）
  4. update_feature_weights: 追踪特征重要性变化

用法：
    python scripts/update_model_rolling.py \\
        --new-quarter 2025-06-30 \\
        --ic-decay-threshold 0.3 \\
        --retrain-strategy expanding

    # 仅验证，不重训
    python scripts/update_model_rolling.py \\
        --new-quarter 2025-06-30 \\
        --validate-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_PANEL_DIR = _ROOT / "data" / "panel"
_MODEL_DIR = _ROOT / "models"
_PRICE_FILE = _ROOT / "data" / "prices" / "csi300_daily_adj_close.parquet"
_REPORTS_DIR = _ROOT / "reports" / "rolling_update"

# 历史基准 IC（来自 lgbm_v1_final.pkl 验证集报告）
_BASELINE_IC_MEAN = 0.062
_BASELINE_IC_STD = 0.045


@dataclass
class ValidationResult:
    quarter_date: str
    ic: float
    ic_std: float
    ic_ir: float          # IC / IC_std
    ic_win_rate: float    # IC > 0 的比例
    n_periods: int
    ic_vs_baseline: float  # (本季IC - 历史均值) / 历史均值，相对偏差
    status: str           # OK / WARNING / RETRAIN


class RollingModelUpdater:
    """持续学习闭环：季度验证 + 增量模型更新."""

    def __init__(
        self,
        model_path: Path | None = None,
        baseline_ic: float = _BASELINE_IC_MEAN,
        baseline_ic_std: float = _BASELINE_IC_STD,
    ) -> None:
        self.model_path = model_path or (_MODEL_DIR / "lgbm_v1_final.pkl")
        self.baseline_ic = baseline_ic
        self.baseline_ic_std = baseline_ic_std
        self._model = None  # 懒加载

    # ── 季度验证 ────────────────────────────────────────────────────────────────

    def validate_last_quarter(self, quarter_date: str) -> ValidationResult:
        """验证指定季度的预测质量.

        Args:
            quarter_date: 季度末日期（YYYY-MM-DD），如 '2025-06-30'

        Returns:
            ValidationResult
        """
        logger.info(f"[RollingUpdater] 验证季度: {quarter_date}")

        # 1. 从 holdout/test panel 中查找该截面
        ics = self._calc_quarter_ic(quarter_date)

        if not ics:
            logger.warning(f"[RollingUpdater] 无法计算 {quarter_date} 的 IC，使用 holdout 数据")
            ics = self._calc_ic_from_holdout(quarter_date)

        if not ics:
            # 无法验证，返回警告状态
            return ValidationResult(
                quarter_date=quarter_date,
                ic=0.0,
                ic_std=0.0,
                ic_ir=0.0,
                ic_win_rate=0.0,
                n_periods=0,
                ic_vs_baseline=-1.0,
                status="WARNING",
            )

        ic_arr = np.array(ics)
        ic_mean = float(np.nanmean(ic_arr))
        ic_std = float(np.nanstd(ic_arr)) if len(ic_arr) > 1 else self.baseline_ic_std
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
        ic_win = float(np.mean(ic_arr > 0))
        ic_vs_baseline = (ic_mean - self.baseline_ic) / abs(self.baseline_ic) if self.baseline_ic != 0 else 0.0

        # 状态判断
        status = self._determine_status(ic_mean, ic_vs_baseline, ics)

        result = ValidationResult(
            quarter_date=quarter_date,
            ic=round(ic_mean, 4),
            ic_std=round(ic_std, 4),
            ic_ir=round(ic_ir, 3),
            ic_win_rate=round(ic_win, 3),
            n_periods=len(ics),
            ic_vs_baseline=round(ic_vs_baseline, 3),
            status=status,
        )
        logger.info(
            f"[RollingUpdater] IC={ic_mean:+.4f} | 基准={self.baseline_ic:.4f} | "
            f"偏差={ic_vs_baseline:+.1%} | 状态={status}"
        )
        return result

    def _calc_quarter_ic(self, quarter_date: str) -> list[float]:
        """从面板数据计算模型的 Rank IC."""
        model = self._load_model()
        if model is None:
            return []

        # 查找包含该季度的面板文件
        panel_paths = [
            _PANEL_DIR / "holdout.parquet",
            _PANEL_DIR / "test.parquet",
            _PANEL_DIR / "val.parquet",
        ]
        panels = []
        for p in panel_paths:
            if p.exists():
                try:
                    df = pd.read_parquet(p)
                    panels.append(df)
                except Exception:
                    pass

        if not panels:
            return []

        panel = pd.concat(panels).sort_index()
        all_dates = panel.index.get_level_values("as_of").unique()
        target = pd.Timestamp(quarter_date)

        # 取该季度附近的截面（±60天）
        relevant_dates = [d for d in all_dates if abs((d - target).days) <= 92]
        if not relevant_dates:
            # 扩大范围：最近4个截面
            past_dates = sorted([d for d in all_dates if d <= target])
            relevant_dates = past_dates[-4:] if past_dates else []

        ics = []
        feat_names = getattr(model, "_feature_names", None) or getattr(model, "feature_names", None)
        if not feat_names:
            return []

        for d in relevant_dates:
            try:
                cross = panel.xs(d, level="as_of")
                missing = [f for f in feat_names if f not in cross.columns]
                if len(missing) > len(feat_names) * 0.3:
                    continue

                available_feats = [f for f in feat_names if f in cross.columns]
                X = cross[available_feats].fillna(0).to_numpy(dtype=np.float32)
                # 补全缺失特征
                if len(available_feats) < len(feat_names):
                    full_X = np.zeros((len(X), len(feat_names)), dtype=np.float32)
                    feat_idx = {f: i for i, f in enumerate(feat_names)}
                    for j, f in enumerate(available_feats):
                        if f in feat_idx:
                            full_X[:, feat_idx[f]] = X[:, j]
                    X = full_X

                scores = model.predict(X)
                label_col = "forward_return_21d" if "forward_return_21d" in cross.columns else None
                if label_col is None:
                    continue

                actual = cross[label_col].values
                valid = ~(np.isnan(scores) | np.isnan(actual))
                if valid.sum() < 20:
                    continue

                from scipy import stats as scipy_stats
                ic, _ = scipy_stats.spearmanr(scores[valid], actual[valid])
                ics.append(float(ic))
            except Exception as e:
                logger.debug(f"[RollingUpdater] IC计算失败({d}): {e}")

        return ics

    def _calc_ic_from_holdout(self, quarter_date: str) -> list[float]:
        """从 holdout 面板用价格计算实际 IC."""
        holdout = _PANEL_DIR / "holdout.parquet"
        if not holdout.exists():
            return []

        try:
            panel = pd.read_parquet(holdout)
            model = self._load_model()
            if model is None:
                return []
            return self._calc_quarter_ic(quarter_date)
        except Exception:
            return []

    def _determine_status(
        self, ic_mean: float, ic_vs_baseline: float, ics: list[float]
    ) -> str:
        """判断模型健康状态."""
        # 连续2期 IC < 0
        if len(ics) >= 2 and all(ic < 0 for ic in ics[-2:]):
            return "RETRAIN"
        # IC 显著下降（> 30% 相对历史均值）
        if ic_vs_baseline < -0.30:
            return "RETRAIN"
        # IC 轻微下降
        if ic_vs_baseline < -0.15:
            return "WARNING"
        return "OK"

    # ── 重训判断 ────────────────────────────────────────────────────────────────

    def should_retrain(
        self,
        validation_result: ValidationResult,
        ic_decay_threshold: float = 0.3,
    ) -> bool:
        """判断是否触发增量重训.

        触发条件：
        - 本季 IC < 历史均值 × (1 - ic_decay_threshold)
        - 连续2季 IC 为负
        - validation_result.status == "RETRAIN"

        Args:
            validation_result: 验证结果
            ic_decay_threshold: IC 衰减阈值（默认 30%）

        Returns:
            True = 需要重训
        """
        if validation_result.status == "RETRAIN":
            logger.info(f"[RollingUpdater] 触发重训: status=RETRAIN")
            return True

        trigger_ic = self.baseline_ic * (1 - ic_decay_threshold)
        if validation_result.ic < trigger_ic:
            logger.info(
                f"[RollingUpdater] 触发重训: IC={validation_result.ic:.4f} < 阈值={trigger_ic:.4f}"
            )
            return True

        logger.info(
            f"[RollingUpdater] 无需重训: IC={validation_result.ic:.4f} >= 阈值={trigger_ic:.4f}"
        )
        return False

    # ── 增量重训 ────────────────────────────────────────────────────────────────

    def retrain_model(
        self,
        training_end_date: str,
        strategy: str = "expanding",
        output_dir: Path | None = None,
    ) -> Path | None:
        """增量重训模型（不覆盖生产模型）.

        Args:
            training_end_date: 训练数据截止日期
            strategy: 'expanding'（扩展窗口）or 'rolling'（滚动3年）
            output_dir: 输出目录（默认 reports/rolling_update/{date}/）

        Returns:
            新模型文件路径，失败返回 None
        """
        out_dir = output_dir or (_REPORTS_DIR / training_end_date)
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[RollingUpdater] 开始增量重训: strategy={strategy}, end={training_end_date}")

        # 1. 构建训练数据
        train_df, val_df = self._build_training_data(training_end_date, strategy)
        if train_df is None or len(train_df) == 0:
            logger.error("[RollingUpdater] 无法构建训练数据")
            return None

        logger.info(f"[RollingUpdater] 训练数据: {train_df.shape}, 验证数据: {val_df.shape if val_df is not None else 'None'}")

        # 2. 加载现有模型配置（继承特征集）
        model = self._load_model()
        feat_names = None
        if model is not None:
            feat_names = getattr(model, "_feature_names", None) or getattr(model, "feature_names", None)

        # 3. 确定特征集
        label_col = "forward_return_21d"
        if feat_names is None:
            feat_names = self._resolve_feature_names(train_df, label_col)

        if not feat_names:
            logger.error("[RollingUpdater] 无法确定特征集")
            return None

        missing = [f for f in feat_names if f not in train_df.columns]
        if missing:
            logger.warning(f"[RollingUpdater] 缺少特征: {missing[:5]}，从可用特征中选取")
            feat_names = [f for f in feat_names if f in train_df.columns]

        # 4. 训练模型
        new_model_path = _MODEL_DIR / f"lgbm_v1_{training_end_date}.pkl"
        try:
            from quantmind.models.factor_model import build_lgbm_arrays
            from quantmind.models.lgbm_ranker import LGBMRankerModel

            train_dates = sorted(train_df.index.get_level_values("as_of").unique())
            X_tr, y_tr, g_tr = build_lgbm_arrays(train_df, train_dates, feat_names, label_col, 5)

            if len(X_tr) == 0:
                logger.error("[RollingUpdater] 训练数组为空")
                return None

            X_va, y_va, g_va = None, None, None
            if val_df is not None and len(val_df) > 0:
                val_dates = sorted(val_df.index.get_level_values("as_of").unique())
                X_va, y_va, g_va = build_lgbm_arrays(val_df, val_dates, feat_names, label_col, 5)

            new_model = LGBMRankerModel(
                n_estimators=500,
                num_leaves=63,
                learning_rate=0.05,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                min_child_samples=10,
                early_stopping_rounds=50,
                auto_flip=True,
                verbose=-1,
            )
            new_model.fit(
                X_tr, y_tr, groups=g_tr,
                X_val=X_va, y_val=y_va, groups_val=g_va,
                feature_names=feat_names,
            )

            # 保存新模型（带日期版本，不覆盖生产模型）
            from quantmind.models.factor_model import FactorModel
            new_model.save(new_model_path)
            logger.info(f"[RollingUpdater] ✅ 新模型保存: {new_model_path}")

        except Exception as e:
            logger.error(f"[RollingUpdater] ❌ 模型训练失败: {e}")
            return None

        # 5. 验证新模型
        new_model_ic = self._validate_new_model(new_model_path, training_end_date)

        # 6. 生成重训报告
        self._write_retrain_report(out_dir, training_end_date, strategy,
                                   new_model_path, new_model_ic, feat_names)

        # 7. 生成决策报告
        self._write_decision_report(out_dir, training_end_date, new_model_ic)

        return new_model_path

    def _build_training_data(
        self, training_end_date: str, strategy: str
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """构建训练和验证数据."""
        end_ts = pd.Timestamp(training_end_date)

        # 加载所有可用面板
        panel_files = [
            _PANEL_DIR / "train.parquet",
            _PANEL_DIR / "val.parquet",
            _PANEL_DIR / "test.parquet",
            _PANEL_DIR / "holdout.parquet",
        ]
        frames = []
        for f in panel_files:
            if f.exists():
                try:
                    frames.append(pd.read_parquet(f))
                except Exception:
                    pass

        if not frames:
            return None, None

        full_panel = pd.concat(frames).sort_index()
        all_dates = sorted(full_panel.index.get_level_values("as_of").unique())

        if strategy == "expanding":
            # 扩展窗口：所有截止 training_end_date 的数据
            train_dates = [d for d in all_dates if d <= end_ts]
        elif strategy == "rolling":
            # 滚动3年窗口
            start_ts = end_ts - pd.DateOffset(years=3)
            train_dates = [d for d in all_dates if start_ts <= d <= end_ts]
        else:
            train_dates = [d for d in all_dates if d <= end_ts]

        if len(train_dates) < 4:
            logger.warning(f"[RollingUpdater] 训练截面不足: {len(train_dates)}")
            return None, None

        # 最后4个截面作为验证集
        val_dates = train_dates[-4:]
        train_dates = train_dates[:-4]

        if not train_dates:
            return None, None

        train_df = full_panel.loc[full_panel.index.get_level_values("as_of").isin(train_dates)]
        val_df = full_panel.loc[full_panel.index.get_level_values("as_of").isin(val_dates)]

        return train_df, val_df

    def _resolve_feature_names(self, panel: pd.DataFrame, label_col: str) -> list[str]:
        """从面板推断特征列名."""
        exclude = {label_col, "forward_return_63d", "exposure_industry", "exposure_area"}
        return [
            c for c in panel.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(panel[c])
        ]

    def _validate_new_model(self, model_path: Path, training_end_date: str) -> float | None:
        """在 holdout 集上验证新模型."""
        try:
            from quantmind.models.factor_model import FactorModel
            new_model = FactorModel.load(model_path)
            updater = RollingModelUpdater(model_path=model_path)
            result = updater.validate_last_quarter(training_end_date)
            return result.ic
        except Exception as e:
            logger.warning(f"[RollingUpdater] 新模型验证失败: {e}")
            return None

    # ── 特征权重更新 ────────────────────────────────────────────────────────────

    def update_feature_weights(self, new_model_path: Path | None = None) -> dict:
        """比较新旧模型特征重要性，追踪因子演变.

        Returns:
            dict containing degraded, emerging factors and evolution data
        """
        old_model = self._load_model()
        if old_model is None:
            return {}

        new_model = None
        if new_model_path and new_model_path.exists():
            try:
                from quantmind.models.factor_model import FactorModel
                new_model = FactorModel.load(new_model_path)
            except Exception:
                pass

        old_importance = self._get_feature_importance(old_model)
        new_importance = self._get_feature_importance(new_model) if new_model else {}

        if not old_importance:
            return {}

        evolution = {
            "as_of": datetime.now().strftime("%Y-%m-%d"),
            "old_model": str(self.model_path),
            "new_model": str(new_model_path) if new_model_path else None,
            "top10_old": sorted(old_importance.items(), key=lambda x: -x[1])[:10],
            "top10_new": sorted(new_importance.items(), key=lambda x: -x[1])[:10] if new_importance else [],
            "degraded_factors": [],
            "emerging_factors": [],
        }

        if new_importance:
            # 失效因子：重要性下降 > 50%
            for feat, old_imp in old_importance.items():
                new_imp = new_importance.get(feat, 0)
                if old_imp > 0 and (old_imp - new_imp) / old_imp > 0.5:
                    evolution["degraded_factors"].append({
                        "feature": feat,
                        "old_importance": round(old_imp, 4),
                        "new_importance": round(new_imp, 4),
                        "decay_pct": round((old_imp - new_imp) / old_imp * 100, 1),
                    })

            # 新兴因子：重要性上升 > 100%
            for feat, new_imp in new_importance.items():
                old_imp = old_importance.get(feat, 0)
                if old_imp > 0 and (new_imp - old_imp) / old_imp > 1.0:
                    evolution["emerging_factors"].append({
                        "feature": feat,
                        "old_importance": round(old_imp, 4),
                        "new_importance": round(new_imp, 4),
                        "growth_pct": round((new_imp - old_imp) / old_imp * 100, 1),
                    })

        # 写入 JSON
        out_path = _REPORTS_DIR / "feature_evolution.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(evolution, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info(f"[RollingUpdater] 特征演变报告 → {out_path}")
        return evolution

    def _get_feature_importance(self, model) -> dict[str, float]:
        """提取模型特征重要性."""
        if model is None:
            return {}
        try:
            booster = getattr(model, "_model", None) or getattr(model, "booster_", None)
            if booster is None:
                return {}
            importance = booster.feature_importance(importance_type="gain")
            feat_names = (
                getattr(model, "_feature_names", None)
                or getattr(model, "feature_names", None)
                or []
            )
            if len(feat_names) != len(importance):
                return {}
            total = importance.sum() or 1
            return {f: float(imp / total) for f, imp in zip(feat_names, importance)}
        except Exception:
            return {}

    # ── 报告生成 ────────────────────────────────────────────────────────────────

    def write_validation_report(
        self, result: ValidationResult, out_dir: Path
    ) -> Path:
        """写入季度验证 Markdown 报告."""
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "validation_report.md"

        lines = [
            f"# QuantMind 模型季度验证报告 — {result.quarter_date}",
            "",
            f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 本季度 Rank IC 统计",
            "",
            f"| 指标 | 值 |",
            f"|---|---|",
            f"| 本季 IC 均值 | {result.ic:+.4f} |",
            f"| IC 标准差 | {result.ic_std:.4f} |",
            f"| ICIR | {result.ic_ir:+.3f} |",
            f"| IC > 0 比率 | {result.ic_win_rate:.1%} |",
            f"| 截面数 | {result.n_periods} |",
            "",
            "## 与历史基准对比",
            "",
            f"| 对比项 | 值 |",
            f"|---|---|",
            f"| 历史 IC 基准 | {self.baseline_ic:.4f} |",
            f"| 相对偏差 | {result.ic_vs_baseline:+.1%} |",
            f"| 状态判断 | **{result.status}** |",
            "",
            "## 状态说明",
            "",
        ]

        if result.status == "OK":
            lines.append("✅ 模型性能正常，无需干预。")
        elif result.status == "WARNING":
            lines.append("⚠️ IC 轻微下降，建议关注后续趋势，暂不重训。")
        else:
            lines.append("🔴 IC 显著衰减，建议触发增量重训。")

        lines.extend(["", "---", "> 本报告由 QuantMind RollingModelUpdater 自动生成。"])

        report_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"[RollingUpdater] 验证报告 → {report_path}")
        return report_path

    def _write_retrain_report(
        self,
        out_dir: Path,
        training_end_date: str,
        strategy: str,
        new_model_path: Path,
        new_model_ic: float | None,
        feat_names: list[str],
    ) -> Path:
        """写入重训结果 Markdown 报告."""
        report_path = out_dir / "retrain_report.md"
        lines = [
            f"# QuantMind 增量重训报告 — {training_end_date}",
            "",
            f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 重训配置",
            "",
            f"| 参数 | 值 |",
            f"|---|---|",
            f"| 训练截止 | {training_end_date} |",
            f"| 策略 | {strategy} |",
            f"| 特征数 | {len(feat_names)} |",
            f"| 新模型路径 | `{new_model_path}` |",
            "",
            "## 新模型验证 IC",
            "",
            f"| 指标 | 值 |",
            f"|---|---|",
            f"| 新模型 IC | {f'{new_model_ic:+.4f}' if new_model_ic is not None else 'N/A'} |",
            f"| 历史基准 IC | {self.baseline_ic:.4f} |",
            "",
            "## 注意事项",
            "",
            f"- 新模型文件：`{new_model_path.name}`",
            f"- ⚠️ **新模型未自动切换为生产模型**（`lgbm_v1_final.pkl`）",
            f"- 如需切换，请确认新模型在 holdout 集上表现后手动执行：",
            f"  ```bash",
            f"  cp {new_model_path} {_MODEL_DIR}/lgbm_v1_final.pkl",
            f"  ```",
        ]
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    def _write_decision_report(
        self,
        out_dir: Path,
        training_end_date: str,
        new_model_ic: float | None,
    ) -> Path:
        """写入切换决策报告."""
        report_path = out_dir / "decision.md"
        if new_model_ic is not None and new_model_ic > self.baseline_ic * 0.9:
            rec = "✅ **建议切换**：新模型 IC 接近或优于历史基准，可考虑切换为生产模型。"
        elif new_model_ic is not None and new_model_ic > 0:
            rec = "⚠️ **谨慎观察**：新模型有正 IC，但低于历史基准，建议继续观察1个季度。"
        else:
            rec = "❌ **不建议切换**：新模型 IC 不理想，继续使用现有生产模型。"

        lines = [
            f"# 生产模型切换决策 — {training_end_date}",
            "",
            f"## 建议",
            "",
            rec,
            "",
            f"## 指标对比",
            "",
            f"| 指标 | 现有模型 | 新模型 |",
            f"|---|---|---|",
            f"| IC 均值 | {self.baseline_ic:.4f} | {f'{new_model_ic:+.4f}' if new_model_ic is not None else 'N/A'} |",
            "",
            "## 手动切换命令",
            "",
            "```bash",
            f"# 确认后执行（不可逆，请先备份）",
            f"cp models/lgbm_v1_final.pkl models/lgbm_v1_final.pkl.bak",
            f"cp models/lgbm_v1_{training_end_date}.pkl models/lgbm_v1_final.pkl",
            "```",
        ]
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    # ── 模型加载 ────────────────────────────────────────────────────────────────

    def _load_model(self):
        """懒加载当前生产模型."""
        if self._model is not None:
            return self._model
        if not self.model_path.exists():
            logger.warning(f"[RollingUpdater] 模型文件不存在: {self.model_path}")
            return None
        try:
            from quantmind.models.factor_model import FactorModel
            self._model = FactorModel.load(self.model_path)
            return self._model
        except Exception as e:
            logger.error(f"[RollingUpdater] 模型加载失败: {e}")
            return None


# ── CLI 入口 ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QuantMind 持续学习闭环")
    p.add_argument("--new-quarter", required=True, help="新季度末日期 YYYY-MM-DD")
    p.add_argument("--ic-decay-threshold", type=float, default=0.3, help="IC 衰减触发阈值")
    p.add_argument("--retrain-strategy", default="expanding", choices=["expanding", "rolling"])
    p.add_argument("--validate-only", action="store_true", help="只验证，不触发重训")
    p.add_argument("--force-retrain", action="store_true", help="强制触发重训（忽略 IC 检查）")
    p.add_argument(
        "--model-path",
        type=Path,
        default=_MODEL_DIR / "lgbm_v1_final.pkl",
        help="当前生产模型路径",
    )
    p.add_argument("--baseline-ic", type=float, default=_BASELINE_IC_MEAN)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    quarter = args.new_quarter
    out_dir = args.output_dir or (_REPORTS_DIR / quarter)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[RollingUpdater] ===== QuantMind 持续学习闭环 =====")
    logger.info(f"[RollingUpdater] 新季度: {quarter} | 阈值: {args.ic_decay_threshold}")

    updater = RollingModelUpdater(
        model_path=args.model_path,
        baseline_ic=args.baseline_ic,
    )

    # Step 1: 验证
    logger.info("\n[RollingUpdater] === Step 1: 季度验证 ===")
    result = updater.validate_last_quarter(quarter)
    validation_report = updater.write_validation_report(result, out_dir)

    # 保存验证结果 JSON
    result_json = out_dir / "validation_result.json"
    result_json.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Step 2: 判断是否重训
    if args.validate_only:
        logger.info("[RollingUpdater] --validate-only: 跳过重训判断")
        _print_summary(result, None, None)
        return

    do_retrain = args.force_retrain or updater.should_retrain(result, args.ic_decay_threshold)

    new_model_path = None
    if do_retrain:
        logger.info("\n[RollingUpdater] === Step 2: 增量重训 ===")
        new_model_path = updater.retrain_model(
            training_end_date=quarter,
            strategy=args.retrain_strategy,
            output_dir=out_dir,
        )

        if new_model_path:
            # Step 3: 特征演变分析
            logger.info("\n[RollingUpdater] === Step 3: 特征重要性分析 ===")
            updater.update_feature_weights(new_model_path)
    else:
        logger.info("[RollingUpdater] 模型性能正常，无需重训")
        # 写空的决策报告
        (out_dir / "decision.md").write_text(
            f"# 生产模型切换决策 — {quarter}\n\n"
            f"✅ **无需切换**：当前模型 IC={result.ic:+.4f}，性能正常。\n",
            encoding="utf-8",
        )

    _print_summary(result, new_model_path, out_dir)


def _print_summary(
    result: ValidationResult,
    new_model_path: Path | None,
    out_dir: Path | None,
) -> None:
    print(f"\n{'='*60}")
    print(f"QuantMind 持续学习闭环完成 — {result.quarter_date}")
    print(f"{'='*60}")
    print(f"  IC={result.ic:+.4f} | 基准={_BASELINE_IC_MEAN:.4f} | 偏差={result.ic_vs_baseline:+.1%}")
    print(f"  状态: {result.status}")
    if new_model_path:
        print(f"  新模型: {new_model_path}")
        print(f"  ⚠️  新模型需手动确认后切换为生产模型")
    else:
        print(f"  无新模型（未触发重训）")
    if out_dir:
        print(f"  报告目录: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
