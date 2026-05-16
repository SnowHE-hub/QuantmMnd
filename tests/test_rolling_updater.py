"""tests/test_rolling_updater.py — RollingModelUpdater 单元测试.

使用 mock 数据，不调用真实 Tushare，不依赖完整模型文件。
运行：
    conda run -n quantmind python -m pytest tests/test_rolling_updater.py -v
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import tempfile

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.update_model_rolling import (
    RollingModelUpdater,
    ValidationResult,
    _BASELINE_IC_MEAN,
    _BASELINE_IC_STD,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

QUARTER = "2025-06-30"


def _make_updater(model_path: Path | None = None) -> RollingModelUpdater:
    """创建 RollingModelUpdater，使用不存在的模型路径（跳过模型加载）。"""
    return RollingModelUpdater(
        model_path=model_path or Path("/nonexistent/model.pkl"),
        baseline_ic=_BASELINE_IC_MEAN,
    )


def _make_validation_result(ic: float, status: str = "OK") -> ValidationResult:
    ic_vs_baseline = (ic - _BASELINE_IC_MEAN) / abs(_BASELINE_IC_MEAN)
    return ValidationResult(
        quarter_date=QUARTER,
        ic=ic,
        ic_std=0.04,
        ic_ir=ic / 0.04 if ic else 0.0,
        ic_win_rate=0.6,
        n_periods=3,
        ic_vs_baseline=ic_vs_baseline,
        status=status,
    )


# ── Test 1: validate_last_quarter 对 mock 数据返回 ic 和 status ────────────

def test_validate_returns_ic_and_status():
    """validate_last_quarter 应返回有效的 ValidationResult（含 ic 和 status）。"""
    updater = _make_updater()

    # mock _calc_quarter_ic 返回 3 个 IC 值
    mock_ics = [0.05, 0.06, 0.07]
    with patch.object(updater, "_calc_quarter_ic", return_value=mock_ics):
        result = updater.validate_last_quarter(QUARTER)

    assert isinstance(result, ValidationResult)
    assert result.ic == pytest.approx(np.mean(mock_ics), abs=1e-4)
    assert result.status in ("OK", "WARNING", "RETRAIN")
    assert result.n_periods == 3
    assert result.quarter_date == QUARTER


# ── Test 2: should_retrain — ic=0.02 (<阈值 0.043) → True ────────────────

def test_should_retrain_low_ic():
    """IC=0.02 < 历史均值×0.7=0.043，应触发重训。"""
    updater = _make_updater()
    result = _make_validation_result(ic=0.02, status="RETRAIN")

    do_retrain = updater.should_retrain(result, ic_decay_threshold=0.3)
    assert do_retrain is True, f"IC=0.02应触发重训，触发线={_BASELINE_IC_MEAN * 0.7:.4f}"


# ── Test 3: should_retrain — ic=0.06 (正常) → False ──────────────────────

def test_should_retrain_normal_ic():
    """IC=0.06 接近历史均值 0.062，不应触发重训。"""
    updater = _make_updater()
    result = _make_validation_result(ic=0.06, status="OK")

    do_retrain = updater.should_retrain(result, ic_decay_threshold=0.3)
    assert do_retrain is False, "IC=0.06应正常，不触发重训"


# ── Test 4: should_retrain — status=RETRAIN 总是触发 ─────────────────────

def test_should_retrain_status_override():
    """status=RETRAIN 时，无论 IC 值如何都应触发重训。"""
    updater = _make_updater()
    # IC=0.08 高于基准，但 status=RETRAIN（连续负IC场景）
    result = _make_validation_result(ic=0.08, status="RETRAIN")

    do_retrain = updater.should_retrain(result, ic_decay_threshold=0.3)
    assert do_retrain is True


# ── Test 5: should_retrain — ic=-0.01 (连续负) → True ───────────────────

def test_should_retrain_negative_ic():
    """连续负 IC 应触发重训。"""
    updater = _make_updater()
    # IC 为负，ic_vs_baseline = (-0.01 - 0.062) / 0.062 ≈ -1.16
    result = _make_validation_result(ic=-0.01, status="RETRAIN")

    do_retrain = updater.should_retrain(result, ic_decay_threshold=0.3)
    assert do_retrain is True


# ── Test 6: retrain_model 在 mock 模式不调真实 Tushare ─────────────────────

def test_retrain_model_mock_mode():
    """retrain_model 应在 mock 模式下正常运行，不调真实 Tushare。"""
    updater = _make_updater()
    n_tickers = 20
    feats = ["pe_ttm", "pb", "roe_ttm", "momentum_6m", "accruals",
             "volatility_3m", "rsi_14", "debt_to_assets"]

    n_periods = 8
    dates = pd.date_range("2022-01-01", periods=n_periods, freq="QE")
    index = pd.MultiIndex.from_product(
        [dates, [f"60{i:04d}.SH" for i in range(n_tickers)]],
        names=["as_of", "ticker"],
    )
    mock_panel = pd.DataFrame(
        np.random.randn(len(index), len(feats) + 1),
        index=index,
        columns=feats + ["forward_return_21d"],
    )
    mock_panel["forward_return_21d"] = np.random.choice([0, 1, 2, 3, 4], size=len(index))

    # mock X/y/g arrays
    mock_X = np.random.randn(100, len(feats)).astype(np.float32)
    mock_y = np.random.choice([0, 1, 2, 3, 4], 100)
    mock_g = np.array([n_tickers] * 5)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Mock 模型实例
        mock_model_inst = MagicMock()
        mock_model_inst.best_iteration_ = 100
        mock_model_inst.save = MagicMock()

        with (
            patch.object(updater, "_build_training_data",
                         return_value=(mock_panel.iloc[:n_tickers * 6],
                                       mock_panel.iloc[-n_tickers * 2:])),
            patch.object(updater, "_resolve_feature_names", return_value=feats),
            patch.object(updater, "_validate_new_model", return_value=0.05),
            # 在方法的本地 import 命名空间 patch
            patch("quantmind.models.factor_model.build_lgbm_arrays",
                  return_value=(mock_X, mock_y, mock_g)),
            patch("quantmind.models.lgbm_ranker.LGBMRankerModel",
                  return_value=mock_model_inst),
        ):
            result_path = updater.retrain_model(
                training_end_date=QUARTER,
                strategy="expanding",
                output_dir=tmp_path,
            )

        # 验证：报告文件应存在（即使 model.save 被 mock 了）
        assert (tmp_path / "retrain_report.md").exists(), "应生成 retrain_report.md"
        assert (tmp_path / "decision.md").exists(), "应生成 decision.md"


# ── Test 7: write_validation_report 格式正确 ─────────────────────────────

def test_write_validation_report():
    """验证报告 Markdown 应包含必要字段。"""
    updater = _make_updater()
    result = _make_validation_result(ic=0.055, status="OK")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        report_path = updater.write_validation_report(result, out_dir)

        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "Rank IC" in content, "报告应包含 Rank IC"
        assert "OK" in content or "WARNING" in content or "RETRAIN" in content
        assert QUARTER in content


# ── Test 8: update_feature_weights 在模型不存在时不崩溃 ──────────────────

def test_update_feature_weights_no_model():
    """当模型文件不存在时，update_feature_weights 应返回空字典而不崩溃。"""
    updater = _make_updater()
    result = updater.update_feature_weights(new_model_path=None)
    # 应返回空字典（无法加载模型）
    assert isinstance(result, dict)


# ── Test 9: ValidationResult 字段完整性 ──────────────────────────────────

def test_validation_result_fields():
    """ValidationResult 应包含所有必要字段。"""
    result = _make_validation_result(ic=0.04, status="WARNING")

    assert hasattr(result, "ic")
    assert hasattr(result, "ic_std")
    assert hasattr(result, "ic_ir")
    assert hasattr(result, "ic_win_rate")
    assert hasattr(result, "n_periods")
    assert hasattr(result, "ic_vs_baseline")
    assert hasattr(result, "status")
    assert result.status in ("OK", "WARNING", "RETRAIN")


# ── Test 10: should_retrain 边界条件 ─────────────────────────────────────

def test_should_retrain_boundary():
    """IC 刚好在阈值边界的情况。"""
    updater = _make_updater()

    # ic_decay_threshold=0.3，触发线 = 0.062 × 0.7 = 0.0434
    threshold_ic = _BASELINE_IC_MEAN * (1 - 0.3)

    # 刚好在阈值上（不触发）
    result_ok = _make_validation_result(ic=threshold_ic + 0.001, status="OK")
    assert updater.should_retrain(result_ok, ic_decay_threshold=0.3) is False

    # 刚好低于阈值（触发）
    result_trigger = _make_validation_result(ic=threshold_ic - 0.001, status="WARNING")
    assert updater.should_retrain(result_trigger, ic_decay_threshold=0.3) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
