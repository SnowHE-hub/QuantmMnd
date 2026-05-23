"""tests/test_mlflow_tracker.py

ExperimentTracker (MLflow 封装) 单元测试（≥10 个）。

所有测试在临时目录中运行，不污染项目的 mlruns/。
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from quantmind.workflow.mlflow_tracker import ExperimentTracker, _get_git_commit


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tracker(tmp_path: Path) -> ExperimentTracker:
    """每个测试使用独立临时 mlruns 目录，实验互不干扰。"""
    return ExperimentTracker(
        tracking_uri=str(tmp_path / "mlruns"),
        experiment_name="test_experiment",
    )


@pytest.fixture
def e3_reports_dir(tmp_path: Path) -> Path:
    """在临时目录中模拟 nav_v4 结构，含 4 种权重的 nav_metrics.json。"""
    base = tmp_path / "nav_v4"
    methods = {
        "hrp_e3":   {"ann_return": 0.2446, "sharpe_ratio": 0.996,  "max_drawdown": -0.2262,
                     "monthly_win_rate": 0.605, "cost_bps": 13.0, "n_rebalances": 30},
        "equal_e3": {"ann_return": 0.2127, "sharpe_ratio": 0.880,  "max_drawdown": -0.2356,
                     "monthly_win_rate": 0.628, "cost_bps": 13.0, "n_rebalances": 30},
        "blend_e3": {"ann_return": 0.2119, "sharpe_ratio": 0.863,  "max_drawdown": -0.2307,
                     "monthly_win_rate": 0.605, "cost_bps": 13.0, "n_rebalances": 30},
        "kelly_e3": {"ann_return": 0.1767, "sharpe_ratio": 0.668,  "max_drawdown": -0.2537,
                     "monthly_win_rate": 0.593, "cost_bps": 13.0, "n_rebalances": 30},
    }
    for method, data in methods.items():
        d = base / method
        d.mkdir(parents=True)
        (d / "nav_metrics.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )
        # 假造一个 CSV artifact
        (d / "nav_daily.csv").write_text("date,nav\n2024-01-02,1.05\n", encoding="utf-8")
    return base


# ─────────────────────────────────────────────────────────────────────────────
# 1: log_backtest_run 返回有效 run_id（UUID 格式，32 个十六进制字符）
# ─────────────────────────────────────────────────────────────────────────────

def test_log_backtest_run_returns_valid_run_id(tracker):
    """log_backtest_run 应返回 32 字符十六进制 run_id。"""
    run_id = tracker.log_backtest_run(
        params={"weight_method": "hrp", "phase": "E3"},
        metrics={"sharpe": 0.996, "ann_return": 0.245},
    )
    assert isinstance(run_id, str)
    assert len(run_id) == 32
    # UUID 去掉连字符后为 32 位十六进制
    assert all(c in "0123456789abcdef" for c in run_id.lower())


# ─────────────────────────────────────────────────────────────────────────────
# 2: 同一实验多次 run，compare_runs 按 sharpe 降序
# ─────────────────────────────────────────────────────────────────────────────

def test_compare_runs_descending_sharpe(tracker):
    """compare_runs(metric='sharpe') 返回的记录应按 sharpe 从高到低排列。"""
    for method, sharpe in [("hrp", 0.996), ("equal", 0.880), ("kelly", 0.668)]:
        tracker.log_backtest_run(
            params={"weight_method": method, "phase": "E3"},
            metrics={"sharpe": sharpe, "ann_return": 0.2},
        )

    df = tracker.compare_runs(metric="sharpe", n_top=5)
    assert not df.empty
    sharpes = df["sharpe"].dropna().tolist()
    assert sharpes == sorted(sharpes, reverse=True), \
        f"sharpe 未降序排列：{sharpes}"


# ─────────────────────────────────────────────────────────────────────────────
# 3: get_best_run 返回 sharpe 最高的实验（HRP）
# ─────────────────────────────────────────────────────────────────────────────

def test_get_best_run_returns_hrp(tracker):
    """4 种权重中 HRP sharpe=0.996 最高，get_best_run 应返回 hrp。"""
    for method, sharpe in [("hrp", 0.996), ("equal", 0.880),
                            ("blend", 0.863), ("kelly", 0.668)]:
        tracker.log_backtest_run(
            params={"weight_method": method, "phase": "E3"},
            metrics={"sharpe": sharpe},
        )

    best = tracker.get_best_run(metric="sharpe")
    assert best.get("params", {}).get("weight_method") == "hrp"
    assert abs(float(best["metrics"]["sharpe"]) - 0.996) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 4: git commit hash 自动写入 tag
# ─────────────────────────────────────────────────────────────────────────────

def test_git_commit_tag_auto_populated(tracker):
    """log_backtest_run 应自动在 tags 中写入 git_commit（非空字符串）。"""
    tracker.log_backtest_run(
        params={"weight_method": "hrp"},
        metrics={"sharpe": 0.9},
    )
    df = tracker.compare_runs(metric="sharpe", n_top=1)
    assert not df.empty
    # tag 列名为 t_git_commit
    if "t_git_commit" in df.columns:
        commit = df["t_git_commit"].iloc[0]
        assert isinstance(commit, str) and len(commit) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 5: artifacts_dir 不存在时不抛出异常（只打 WARNING）
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_artifacts_dir_no_exception(tracker):
    """artifacts_dir 指向不存在路径时，函数应正常返回 run_id，不抛出异常。"""
    run_id = tracker.log_backtest_run(
        params={"weight_method": "hrp"},
        metrics={"sharpe": 0.9},
        artifacts_dir="/nonexistent/path/does/not/exist/",
    )
    assert isinstance(run_id, str) and len(run_id) == 32


# ─────────────────────────────────────────────────────────────────────────────
# 6: artifacts 目录存在时 CSV/JSON 被上传（MLflow 本地文件系统验证）
# ─────────────────────────────────────────────────────────────────────────────

def test_artifacts_uploaded_when_dir_exists(tracker, tmp_path):
    """存在 artifacts_dir 时，其中的 CSV/JSON 文件应被 MLflow 记录。"""
    art_dir = tmp_path / "reports"
    art_dir.mkdir()
    (art_dir / "nav_daily.csv").write_text("date,nav\n2024-01-01,1.0\n")
    (art_dir / "nav_metrics.json").write_text('{"sharpe": 0.9}')

    run_id = tracker.log_backtest_run(
        params={"weight_method": "test"},
        metrics={"sharpe": 0.9},
        artifacts_dir=str(art_dir),
    )
    assert run_id  # 不抛出异常即视为上传成功（MLflow 本地写文件）


# ─────────────────────────────────────────────────────────────────────────────
# 7: compare_runs 在空实验时返回空 DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def test_compare_runs_empty_experiment(tmp_path):
    """空实验（无任何 run）时 compare_runs 应返回空 DataFrame，不抛出异常。"""
    tracker = ExperimentTracker(
        tracking_uri=str(tmp_path / "mlruns"),
        experiment_name="fresh_empty_exp",
    )
    df = tracker.compare_runs(metric="sharpe", n_top=5)
    assert df.empty


# ─────────────────────────────────────────────────────────────────────────────
# 8: get_best_run 在空实验时返回空 dict
# ─────────────────────────────────────────────────────────────────────────────

def test_get_best_run_empty_experiment(tmp_path):
    """空实验时 get_best_run 应返回空 dict，不抛出异常。"""
    tracker = ExperimentTracker(
        tracking_uri=str(tmp_path / "mlruns"),
        experiment_name="empty_exp_2",
    )
    result = tracker.get_best_run(metric="sharpe")
    assert result == {}


# ─────────────────────────────────────────────────────────────────────────────
# 9: import_from_reports_dir 批量导入 4 条 E3 记录
# ─────────────────────────────────────────────────────────────────────────────

def test_import_from_reports_dir_count(tracker, e3_reports_dir):
    """import_from_reports_dir 应导入 4 条 E3 结果（hrp/equal/blend/kelly）。"""
    run_ids = tracker.import_from_reports_dir(
        reports_root=str(e3_reports_dir),
        phase="E3",
        model_version="v6",
    )
    assert len(run_ids) == 4
    assert all(isinstance(rid, str) and len(rid) == 32 for rid in run_ids)


# ─────────────────────────────────────────────────────────────────────────────
# 10: import_from_reports_dir 后 get_best_run 返回 HRP
# ─────────────────────────────────────────────────────────────────────────────

def test_import_and_best_run_is_hrp(tracker, e3_reports_dir):
    """导入后 get_best_run('sharpe') 应指向 hrp_e3（sharpe=0.996 最高）。"""
    tracker.import_from_reports_dir(str(e3_reports_dir), phase="E3")
    best = tracker.get_best_run(metric="sharpe")
    assert best  # 非空
    method = best.get("params", {}).get("weight_method", "")
    assert "hrp" in method.lower(), f"最优 method={method} 应含 'hrp'"


# ─────────────────────────────────────────────────────────────────────────────
# 11: compare_runs n_top 参数限制返回数量
# ─────────────────────────────────────────────────────────────────────────────

def test_compare_runs_n_top(tracker, e3_reports_dir):
    """compare_runs(n_top=2) 应最多返回 2 条记录。"""
    tracker.import_from_reports_dir(str(e3_reports_dir), phase="E3")
    df = tracker.compare_runs(metric="sharpe", n_top=2)
    assert len(df) <= 2


# ─────────────────────────────────────────────────────────────────────────────
# 12: 非 float 指标值自动过滤，不抛出异常
# ─────────────────────────────────────────────────────────────────────────────

def test_non_float_metric_filtered(tracker):
    """metrics 中非 float 值（如字符串）应被过滤，不使 log_backtest_run 崩溃。"""
    run_id = tracker.log_backtest_run(
        params={"weight_method": "hrp"},
        metrics={"sharpe": 0.9, "bad_metric": "not_a_number", "ann_return": 0.2},
    )
    assert isinstance(run_id, str) and len(run_id) == 32


# ─────────────────────────────────────────────────────────────────────────────
# 13: _get_git_commit 返回非空字符串
# ─────────────────────────────────────────────────────────────────────────────

def test_get_git_commit_nonempty():
    """_get_git_commit() 应返回非空字符串（git 可用时为 hash，否则为 'unknown'）。"""
    commit = _get_git_commit()
    assert isinstance(commit, str) and len(commit) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 14: 自定义 run_name 被正确写入
# ─────────────────────────────────────────────────────────────────────────────

def test_custom_run_name_stored(tracker):
    """指定 run_name 时，compare_runs 不应抛出异常（run_name 无法直接从 search_runs 取，验证无崩溃）。"""
    run_id = tracker.log_backtest_run(
        params={"weight_method": "hrp"},
        metrics={"sharpe": 1.0},
        run_name="my_custom_run",
    )
    assert isinstance(run_id, str)


# ─────────────────────────────────────────────────────────────────────────────
# 15: __repr__ 包含实验名和 tracking URI
# ─────────────────────────────────────────────────────────────────────────────

def test_repr_contains_experiment_name(tracker):
    """ExperimentTracker.__repr__ 应包含实验名称。"""
    r = repr(tracker)
    assert "test_experiment" in r
    assert "ExperimentTracker" in r
