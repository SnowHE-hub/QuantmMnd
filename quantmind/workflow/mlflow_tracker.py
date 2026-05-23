"""quantmind/workflow/mlflow_tracker.py

QuantMind 标准化 MLflow 实验追踪器。

背景
----
每次 NAV 回测结果存储在 reports/nav_v4/{method}_e3/ 目录，手工管理，
版本对比靠文件名区分，难以跨实验比较超参和指标。
本模块用 MLflow 自动版本化每次实验，提供：

自动记录
--------
- 参数（Params）：weight_method, top_n, holding_period,
                  transaction_cost, regime, model_version
- 指标（Metrics）：annual_return, sharpe, max_dd,
                   ic_mean, icir, turnover, win_rate
- 产出物（Artifacts）：nav_daily.csv, factor_weights.json,
                       reports/figures/*.png, nav_metrics.json
- 标签（Tags）：phase(E1/E2/E3), regime(bull/neutral/bear),
                git_commit（自动从 git log 读取）

用法
----
>>> from quantmind.workflow.mlflow_tracker import ExperimentTracker
>>> tracker = ExperimentTracker()
>>> run_id = tracker.log_backtest_run(
...     params={'weight_method': 'hrp', 'top_n': 30},
...     metrics={'ann_return': 0.245, 'sharpe': 0.996},
...     artifacts_dir='reports/nav_v4/hrp_e3/'
... )
>>> df = tracker.compare_runs(metric='sharpe', n_top=5)
>>> best = tracker.get_best_run(metric='sharpe')
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

log = logging.getLogger(__name__)

# 默认实验名
_EXPERIMENT_NAME = "QuantMind-NAV-Backtest"
# 默认 MLflow 追踪目录（相对项目根）
_DEFAULT_TRACKING_URI = "mlruns"
# 允许记录为 artifact 的扩展名
_ARTIFACT_EXTS = {".csv", ".json", ".html", ".png", ".jpg", ".pdf", ".parquet"}


def _get_git_commit(length: int = 8) -> str:
    """从 git log 读取当前 commit hash（失败时返回 'unknown'）。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", f"HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:length]
    except Exception:
        pass
    return "unknown"


def _resolve_tracking_uri(uri: str) -> str:
    """将相对路径转换为 file:// 绝对 URI（MLflow 要求）。"""
    p = Path(uri)
    if not p.is_absolute():
        # 尝试相对于项目根（当前工作目录）
        p = Path.cwd() / p
    p.mkdir(parents=True, exist_ok=True)
    return p.as_uri()   # → file:///absolute/path/to/mlruns


class ExperimentTracker:
    """封装 MLflow，为 QuantMind 提供标准化实验追踪。

    Parameters
    ----------
    tracking_uri : str
        MLflow tracking server URI 或本地目录路径。
        默认 ``"mlruns"``（相对于当前工作目录）。
    experiment_name : str
        MLflow 实验名称，默认 ``"QuantMind-NAV-Backtest"``。

    Examples
    --------
    >>> tracker = ExperimentTracker()
    >>> run_id = tracker.log_backtest_run(
    ...     params={'weight_method': 'hrp', 'phase': 'E3'},
    ...     metrics={'sharpe': 0.996, 'ann_return': 0.245},
    ...     artifacts_dir='reports/nav_v4/hrp_e3/'
    ... )
    >>> tracker.compare_runs('sharpe', n_top=5)
    """

    EXPERIMENT_NAME: str = _EXPERIMENT_NAME

    def __init__(
        self,
        tracking_uri: str = _DEFAULT_TRACKING_URI,
        experiment_name: str = _EXPERIMENT_NAME,
    ) -> None:
        import mlflow

        self._mlflow = mlflow
        self.experiment_name = experiment_name
        self.tracking_uri = _resolve_tracking_uri(tracking_uri)

        mlflow.set_tracking_uri(self.tracking_uri)
        # 若实验不存在则自动创建
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

        log.info(
            "ExperimentTracker 初始化：tracking_uri=%s  experiment=%s",
            self.tracking_uri, experiment_name,
        )

    # ── 核心：记录一次回测 ────────────────────────────────────────────────────

    def log_backtest_run(
        self,
        params:        Dict[str, Any],
        metrics:       Dict[str, float],
        artifacts_dir: Optional[str | Path] = None,
        tags:          Optional[Dict[str, str]] = None,
        run_name:      Optional[str] = None,
    ) -> str:
        """记录一次完整回测，返回 run_id。

        Parameters
        ----------
        params : dict
            超参字典，例如 ``{'weight_method': 'hrp', 'top_n': 30,
            'transaction_cost': 0.0013, 'model_version': 'v6'}``。
        metrics : dict
            绩效指标字典，例如 ``{'ann_return': 0.245, 'sharpe': 0.996,
            'max_dd': -0.226, 'win_rate': 0.605}``。
            值必须为 float（非 float 自动转换）。
        artifacts_dir : str or Path, optional
            结果目录路径；目录下所有允许扩展名的文件均被上传。
            目录不存在时打印 WARNING 而不抛出异常。
        tags : dict, optional
            额外标签；``git_commit`` 如未提供则自动填充。
        run_name : str, optional
            Run 显示名称；None 时使用 ``{weight_method}_{phase}`` 格式。

        Returns
        -------
        str  MLflow run_id（UUID）
        """
        mlflow = self._mlflow

        # ── 自动 tag ──────────────────────────────────────────────────────────
        auto_tags: Dict[str, str] = {
            "git_commit":   _get_git_commit(),
            "weight_method": str(params.get("weight_method", "unknown")),
            "phase":         str(params.get("phase", "unknown")),
        }
        if tags:
            auto_tags.update(tags)

        # ── 自动 run_name ─────────────────────────────────────────────────────
        if run_name is None:
            method = params.get("weight_method", "unknown")
            phase  = params.get("phase", "")
            run_name = f"{method}_{phase}" if phase else str(method)

        # ── 确保 metrics 全为 float ───────────────────────────────────────────
        clean_metrics: Dict[str, float] = {}
        for k, v in metrics.items():
            try:
                clean_metrics[k] = float(v)
            except (TypeError, ValueError):
                log.warning("log_backtest_run: metric '%s'=%r 无法转换为 float，跳过", k, v)

        # ── 开始 MLflow run ───────────────────────────────────────────────────
        with mlflow.start_run(run_name=run_name) as run:
            # 参数
            for k, v in params.items():
                mlflow.log_param(k, v)

            # 指标
            mlflow.log_metrics(clean_metrics)

            # 标签
            mlflow.set_tags(auto_tags)

            # 产出物
            if artifacts_dir is not None:
                art_path = Path(artifacts_dir)
                if not art_path.is_absolute():
                    art_path = Path.cwd() / art_path

                if art_path.exists() and art_path.is_dir():
                    logged = 0
                    for f in sorted(art_path.iterdir()):
                        if f.is_file() and f.suffix.lower() in _ARTIFACT_EXTS:
                            mlflow.log_artifact(str(f))
                            logged += 1
                    log.info(
                        "log_backtest_run: 上传 %d 个 artifact（%s）",
                        logged, art_path,
                    )
                else:
                    log.warning(
                        "log_backtest_run: artifacts_dir='%s' 不存在，跳过上传",
                        artifacts_dir,
                    )

            run_id = run.info.run_id

        log.info(
            "log_backtest_run: ✅ run_id=%s  run_name=%s  sharpe=%.3f",
            run_id[:8], run_name,
            clean_metrics.get("sharpe", clean_metrics.get("sharpe_ratio", float("nan"))),
        )
        return run_id

    # ── 实验对比 ──────────────────────────────────────────────────────────────

    def compare_runs(
        self,
        metric:  str = "sharpe",
        n_top:   int = 5,
        filters: Optional[str] = None,
    ) -> pd.DataFrame:
        """返回指定指标 Top-N 的历史实验记录。

        Parameters
        ----------
        metric : str
            排序指标名（与 ``log_backtest_run`` 中的 metrics key 一致）。
        n_top : int
            返回前 N 条记录，默认 5。
        filters : str, optional
            MLflow 搜索过滤表达式，例如 ``"params.phase = 'E3'"``。

        Returns
        -------
        pd.DataFrame
            columns: run_id, run_name, status, 各 param/metric 列
        """
        mlflow = self._mlflow

        exp = mlflow.get_experiment_by_name(self.experiment_name)
        if exp is None:
            log.warning("compare_runs: 实验 '%s' 不存在", self.experiment_name)
            return pd.DataFrame()

        # 检索所有 runs
        try:
            df = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string=filters or "",
                order_by=[f"metrics.{metric} DESC"],
                max_results=n_top,
            )
        except Exception as exc:
            log.warning("compare_runs: search_runs 失败 — %s", exc)
            return pd.DataFrame()

        if df.empty:
            return df

        # 整理列名：去掉 'metrics.' / 'params.' 前缀方便阅读
        rename = {}
        for col in df.columns:
            if col.startswith("metrics."):
                rename[col] = col[len("metrics."):]
            elif col.startswith("params."):
                rename[col] = f"p_{col[len('params.'):]}"
            elif col.startswith("tags."):
                rename[col] = f"t_{col[len('tags.'):]}"
        df = df.rename(columns=rename)

        # 只保留关键列
        keep_cols = ["run_id", "status"]
        p_cols = [c for c in df.columns if c.startswith("p_")]
        m_cols = [c for c in df.columns if c in {
            "sharpe", "sharpe_ratio", "ann_return", "annual_return",
            "max_dd", "max_drawdown", "win_rate", "monthly_win_rate",
            "ic_mean", "icir", "turnover", "avg_turnover",
        }]
        tag_cols = [c for c in df.columns if c in {"t_git_commit", "t_phase", "t_weight_method"}]

        out_cols = keep_cols + p_cols + m_cols + tag_cols
        out_cols = [c for c in out_cols if c in df.columns]

        return df[out_cols].reset_index(drop=True)

    def get_best_run(self, metric: str = "sharpe") -> Dict[str, Any]:
        """返回历史最优实验的参数和指标字典。

        Parameters
        ----------
        metric : str
            最优化指标，默认 'sharpe'。

        Returns
        -------
        dict 含键：``run_id``, ``params``, ``metrics``, ``tags``。
        空实验时返回空 dict。
        """
        mlflow = self._mlflow

        exp = mlflow.get_experiment_by_name(self.experiment_name)
        if exp is None:
            log.warning("get_best_run: 实验 '%s' 不存在", self.experiment_name)
            return {}

        try:
            df = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=[f"metrics.{metric} DESC"],
                max_results=1,
            )
        except Exception as exc:
            log.warning("get_best_run: search_runs 失败 — %s", exc)
            return {}

        if df.empty:
            return {}

        row = df.iloc[0]
        params  = {k[len("params."):]: v for k, v in row.items() if k.startswith("params.")}
        metrics = {k[len("metrics."):]: v for k, v in row.items() if k.startswith("metrics.")}
        tags    = {k[len("tags."):]: v for k, v in row.items() if k.startswith("tags.")}

        return {
            "run_id":  row["run_id"],
            "params":  params,
            "metrics": metrics,
            "tags":    tags,
        }

    # ── 批量导入历史结果 ──────────────────────────────────────────────────────

    def import_from_reports_dir(
        self,
        reports_root: str | Path,
        phase: str = "E3",
        model_version: str = "v6",
    ) -> List[str]:
        """扫描 reports 目录，批量导入 nav_metrics.json 结果到 MLflow。

        Parameters
        ----------
        reports_root : str or Path
            包含各 method 子目录的根目录，例如 ``reports/nav_v4/``。
        phase : str
            实验阶段标签，默认 'E3'。
        model_version : str
            模型版本标签，默认 'v6'。

        Returns
        -------
        list[str]  成功记录的 run_id 列表
        """
        import json

        reports_root = Path(reports_root)
        if not reports_root.is_absolute():
            reports_root = Path.cwd() / reports_root

        run_ids: List[str] = []

        for metrics_file in sorted(reports_root.rglob("nav_metrics.json")):
            sub_dir = metrics_file.parent
            method = sub_dir.name   # e.g. 'hrp_e3', 'kelly_e3'

            try:
                raw = json.loads(metrics_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("import_from_reports_dir: 跳过 %s — %s", metrics_file, exc)
                continue

            # 标准化指标键名
            metrics_map = {
                "ann_return":        raw.get("ann_return"),
                "sharpe":            raw.get("sharpe_ratio"),
                "max_dd":            raw.get("max_drawdown"),
                "win_rate":          raw.get("monthly_win_rate"),
                "total_return":      raw.get("total_return"),
                "ann_volatility":    raw.get("ann_volatility"),
                "calmar":            raw.get("calmar_ratio"),
                "avg_turnover":      raw.get("avg_turnover"),
                "total_cost_drag":   raw.get("total_cost_drag"),
                "excess_ann_return": raw.get("excess_ann_return"),
            }
            metrics_map = {k: v for k, v in metrics_map.items() if v is not None}

            params = {
                "weight_method":    method,
                "phase":            phase,
                "model_version":    model_version,
                "transaction_cost": raw.get("cost_bps", 0) / 10000,
                "start_date":       raw.get("start_date", ""),
                "end_date":         raw.get("end_date", ""),
                "n_rebalances":     raw.get("n_rebalances", ""),
            }

            try:
                run_id = self.log_backtest_run(
                    params=params,
                    metrics=metrics_map,
                    artifacts_dir=str(sub_dir),
                    run_name=f"{method}_{phase}",
                )
                run_ids.append(run_id)
                log.info("import_from_reports_dir: ✅ %s → run_id=%s", method, run_id[:8])
            except Exception as exc:
                log.error("import_from_reports_dir: ❌ %s — %s", method, exc)

        log.info("import_from_reports_dir: 共导入 %d 条记录", len(run_ids))
        return run_ids

    # ── __repr__ ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"ExperimentTracker("
            f"experiment='{self.experiment_name}', "
            f"uri='{self.tracking_uri}')"
        )


__all__ = ["ExperimentTracker"]
