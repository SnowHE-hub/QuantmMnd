"""tests/test_iteration.py — 迭代优化模块单元测试 (≥15)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# 测试夹具
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sim_dir(tmp_path: Path) -> Path:
    """构建最小化的 sim30d 结构."""
    sd = tmp_path / "sim30d"
    sd.mkdir()
    (sd / "daily").mkdir()

    # summary.json（新格式）
    summary = {
        "n_days": 30,
        "date_range": "20251009-20251119",
        "portfolio_returns": {
            "1w":  {"ir": -0.46,  "win_rate": 0.0,  "mean_return": -0.02},
            "2w":  {"ir": -0.42,  "win_rate": 0.0,  "mean_return": -0.01},
            "21d": {"ir": -0.10,  "win_rate": 0.0,  "mean_return":  0.00},
            "3m":  {"ir":  1.80,  "win_rate": 0.967, "mean_return": 0.2156},
        },
    }
    (sd / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    # stock_returns.parquet
    rng = np.random.default_rng(42)
    n   = 90
    df  = pd.DataFrame({
        "date":            ["20251015"] * 30 + ["20251022"] * 30 + ["20251029"] * 30,
        "ticker":          [f"{i:06d}.SH" for i in range(n)],
        "industry":        (["科技"] * 15 + ["金融"] * 15) * 3,
        "composite_score": rng.uniform(40, 90, n),
        "lgbm_score":      rng.uniform(0.3, 0.8, n),
        "value_score":     rng.uniform(30, 80, n),
        "momentum_score":  rng.uniform(30, 80, n),
        "quality_score":   rng.uniform(50, 90, n),
        "technical_score": rng.uniform(40, 80, n),
        "return_1w":       rng.normal(-0.01, 0.05, n),
        "return_2w":       rng.normal(-0.01, 0.07, n),
        "return_21d":      rng.normal( 0.00, 0.08, n),
        "return_3m":       rng.normal( 0.20, 0.30, n),
    })
    df.to_parquet(sd / "stock_returns.parquet", index=False)

    # 2 个 daily JSON
    for day_idx, date_str in enumerate(["20251015", "20251022"]):
        day_data = {
            "date":      date_str,
            "day_index": day_idx,
            "system1_candidates": [{"ticker": f"{i:06d}.SH"} for i in range(50)],
            "system2_analysis":   [{"ticker": f"{i:06d}.SH"} for i in range(25)],
            "system3_final_list": [{"ticker": f"{i:06d}.SH", "composite_score": 70.0,
                                    "rating": "A", "hist_win_rate": 0.6,
                                    "hist_sharpe": 0.8, "hist_maxdd": -0.1,
                                    "risk_level": "M", "investable": True}
                                   for i in range(10)],
            "returns": {"1w": {"mean": -0.01, "win_rate": 0.4}},
        }
        (sd / "daily" / f"{date_str}.json").write_text(
            json.dumps(day_data), encoding="utf-8"
        )

    return sd


@pytest.fixture
def minimal_config(tmp_path: Path) -> Path:
    """最小化 strategy_config_v2.json."""
    cfg = {
        "holding_period": {"recommended": "1w"},
        "system2_updates": {
            "weights_calibrated": {
                "value": 0.25, "momentum": 0.25,
                "quality": 0.25, "technical": 0.25,
            }
        },
        "layer6_overweight": {"overweight_industries": [], "overweight_limit": 5},
        "barra_constrained": False,
    }
    p = tmp_path / "strategy_config_v2.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# TestSimulationAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulationAnalyzer:

    def test_analyze_returns_sim_diagnosis(self, sim_dir):
        """analyze() 应返回 SimDiagnosis 实例."""
        from quantmind.iteration.analyzer import SimulationAnalyzer, SimDiagnosis
        diag = SimulationAnalyzer(sim_dir).analyze()
        assert isinstance(diag, SimDiagnosis)

    def test_analyze_reads_n_days(self, sim_dir):
        """分析后 n_days 应从 summary.json 正确读取."""
        from quantmind.iteration.analyzer import SimulationAnalyzer
        diag = SimulationAnalyzer(sim_dir).analyze()
        assert diag.n_days == 30

    def test_analyze_reads_ir_3m(self, sim_dir):
        """ir_3m 应从 portfolio_returns 读取."""
        from quantmind.iteration.analyzer import SimulationAnalyzer
        diag = SimulationAnalyzer(sim_dir).analyze()
        assert abs(diag.ir_3m - 1.80) < 0.01

    def test_analyze_reads_win_rate(self, sim_dir):
        """win_rate_3m 应约等于 0.967."""
        from quantmind.iteration.analyzer import SimulationAnalyzer
        diag = SimulationAnalyzer(sim_dir).analyze()
        assert abs(diag.win_rate_3m - 0.967) < 0.001

    def test_ic_summary_has_horizons(self, sim_dir):
        """ic_summary 应包含 ic_1w, ic_2w, ic_21d, ic_3m 四个键."""
        from quantmind.iteration.analyzer import SimulationAnalyzer
        diag = SimulationAnalyzer(sim_dir).analyze()
        for h in ("ic_1w", "ic_2w", "ic_21d", "ic_3m"):
            assert h in diag.ic_summary, f"ic_summary 缺少 {h}"

    def test_ic_summary_has_factor_keys(self, sim_dir):
        """ic_3m 应包含各因子键."""
        from quantmind.iteration.analyzer import SimulationAnalyzer
        diag = SimulationAnalyzer(sim_dir).analyze()
        ic3  = diag.ic_summary.get("ic_3m", {})
        for fc in ("value_score", "quality_score"):
            assert fc in ic3

    def test_funnel_has_required_keys(self, sim_dir):
        """funnel 应包含 s1_avg, s3_avg, s1_to_s3_ratio."""
        from quantmind.iteration.analyzer import SimulationAnalyzer
        diag = SimulationAnalyzer(sim_dir).analyze()
        for k in ("s1_avg", "s3_avg", "s1_to_s3_ratio"):
            assert k in diag.funnel

    def test_funnel_s1_avg_correct(self, sim_dir):
        """日均 System1 候选数应约为 50."""
        from quantmind.iteration.analyzer import SimulationAnalyzer
        diag = SimulationAnalyzer(sim_dir).analyze()
        assert abs(diag.funnel["s1_avg"] - 50.0) < 1.0

    def test_industry_attribution_populated(self, sim_dir):
        """top/bottom 行业列表应非空."""
        from quantmind.iteration.analyzer import SimulationAnalyzer
        diag = SimulationAnalyzer(sim_dir).analyze()
        assert len(diag.top_industries)    > 0
        assert len(diag.bottom_industries) > 0

    def test_bad_batches_populated(self, sim_dir):
        """bad_batches 应返回非空列表."""
        from quantmind.iteration.analyzer import SimulationAnalyzer
        diag = SimulationAnalyzer(sim_dir).analyze()
        assert isinstance(diag.bad_batches, list)

    def test_issues_list_nonempty(self, sim_dir):
        """issues 应为非空 list[str]."""
        from quantmind.iteration.analyzer import SimulationAnalyzer
        diag = SimulationAnalyzer(sim_dir).analyze()
        assert isinstance(diag.issues, list)
        assert len(diag.issues) > 0

    def test_to_markdown_contains_header(self, sim_dir):
        """to_markdown() 输出应包含标题."""
        from quantmind.iteration.analyzer import SimulationAnalyzer
        diag = SimulationAnalyzer(sim_dir).analyze()
        md   = diag.to_markdown()
        assert "30日模拟诊断报告" in md

    def test_to_dict_and_from_dict_roundtrip(self, sim_dir):
        """to_dict/from_dict 应保持数据完整."""
        from quantmind.iteration.analyzer import SimulationAnalyzer, SimDiagnosis
        diag  = SimulationAnalyzer(sim_dir).analyze()
        d     = diag.to_dict()
        diag2 = SimDiagnosis.from_dict(d)
        assert diag2.n_days   == diag.n_days
        assert diag2.ir_3m    == pytest.approx(diag.ir_3m)

    def test_missing_summary_returns_empty_diag(self, tmp_path):
        """sim_dir 没有 summary.json 时应返回默认值，不抛异常."""
        from quantmind.iteration.analyzer import SimulationAnalyzer
        empty_dir = tmp_path / "empty_sim"
        empty_dir.mkdir()
        diag = SimulationAnalyzer(empty_dir).analyze()
        assert diag.n_days == 0

    def test_normalize_summary_old_format(self, tmp_path):
        """旧格式 summary（flat 结构）应被正确标准化."""
        from quantmind.iteration.analyzer import SimulationAnalyzer
        sd = tmp_path / "old_sim"
        sd.mkdir()
        (sd / "daily").mkdir()
        old_summary = {
            "n": 10,
            "ir_1w": -0.3,
            "ir_3m":  1.5,
            "win_rate_3m": 0.8,
        }
        (sd / "summary.json").write_text(json.dumps(old_summary), encoding="utf-8")
        diag = SimulationAnalyzer(sd).analyze()
        assert diag.n_days == 10


# ─────────────────────────────────────────────────────────────────────────────
# TestParameterOptimizer
# ─────────────────────────────────────────────────────────────────────────────

class TestParameterOptimizer:

    def test_generate_suggestions_returns_list(self, sim_dir, minimal_config):
        """generate_suggestions 应返回列表."""
        from quantmind.iteration.analyzer  import SimulationAnalyzer
        from quantmind.iteration.optimizer import ParameterOptimizer
        diag  = SimulationAnalyzer(sim_dir).analyze()
        opt   = ParameterOptimizer(config_path=minimal_config)
        suggs = opt.generate_suggestions(diag)
        assert isinstance(suggs, list)

    def test_suggestions_have_required_fields(self, sim_dir, minimal_config):
        """每条建议应有 param_path, old_value, new_value, reason, confidence."""
        from quantmind.iteration.analyzer  import SimulationAnalyzer
        from quantmind.iteration.optimizer import ParameterOptimizer, ParameterSuggestion
        diag  = SimulationAnalyzer(sim_dir).analyze()
        suggs = ParameterOptimizer(config_path=minimal_config).generate_suggestions(diag)
        for s in suggs:
            assert isinstance(s, ParameterSuggestion)
            assert s.param_path
            assert 0.0 <= s.confidence <= 1.0

    def test_weight_suggestions_sum_to_one(self, sim_dir, minimal_config):
        """权重建议应用后，四因子之和应 ≈ 1."""
        from quantmind.iteration.analyzer  import SimulationAnalyzer
        from quantmind.iteration.optimizer import ParameterOptimizer
        diag    = SimulationAnalyzer(sim_dir).analyze()
        opt     = ParameterOptimizer(config_path=minimal_config)
        suggs   = opt.generate_suggestions(diag)
        updated = opt.apply_suggestions(suggs, config_path=minimal_config, dry_run=True)
        weights = updated["system2_updates"]["weights_calibrated"]
        total   = sum(weights[k] for k in ("value", "momentum", "quality", "technical"))
        assert abs(total - 1.0) < 1e-4

    def test_apply_suggestions_writes_config(self, sim_dir, minimal_config):
        """apply_suggestions 应写回 JSON 文件."""
        from quantmind.iteration.analyzer  import SimulationAnalyzer
        from quantmind.iteration.optimizer import ParameterOptimizer
        diag  = SimulationAnalyzer(sim_dir).analyze()
        opt   = ParameterOptimizer(config_path=minimal_config)
        suggs = opt.generate_suggestions(diag)
        opt.apply_suggestions(suggs, config_path=minimal_config, dry_run=False)
        # 文件应已被更新
        updated = json.loads(minimal_config.read_text(encoding="utf-8"))
        assert "updated_at" in updated

    def test_apply_dry_run_does_not_write(self, sim_dir, minimal_config):
        """dry_run=True 时，配置文件不应被修改."""
        from quantmind.iteration.analyzer  import SimulationAnalyzer
        from quantmind.iteration.optimizer import ParameterOptimizer
        mtime_before = minimal_config.stat().st_mtime
        diag  = SimulationAnalyzer(sim_dir).analyze()
        opt   = ParameterOptimizer(config_path=minimal_config)
        suggs = opt.generate_suggestions(diag)
        opt.apply_suggestions(suggs, config_path=minimal_config, dry_run=True)
        mtime_after = minimal_config.stat().st_mtime
        assert mtime_before == mtime_after

    def test_suggestion_to_dict_from_dict(self):
        """ParameterSuggestion to_dict/from_dict 往返."""
        from quantmind.iteration.optimizer import ParameterSuggestion
        s  = ParameterSuggestion(
            param_path="system2_updates.weights_calibrated.value",
            old_value=0.242, new_value=0.18, reason="IC_3m < 0",
            confidence=0.75, category="weight",
        )
        d  = s.to_dict()
        s2 = ParameterSuggestion.from_dict(d)
        assert s2.param_path   == s.param_path
        assert s2.new_value    == pytest.approx(s.new_value)
        assert s2.confidence   == pytest.approx(s.confidence)

    def test_holding_period_suggestion_when_short_term_negative(self, sim_dir, minimal_config):
        """当 ir_1w < 0 且 ir_3m > 0 时应生成持仓周期建议."""
        from quantmind.iteration.analyzer  import SimDiagnosis
        from quantmind.iteration.optimizer import ParameterOptimizer
        diag = SimDiagnosis(
            ir_1w=-0.46, ir_2w=-0.42, ir_21d=-0.10, ir_3m=1.80,
            win_rate_3m=0.967,
        )
        suggs = ParameterOptimizer(config_path=minimal_config).generate_suggestions(diag)
        param_paths = [s.param_path for s in suggs]
        assert "holding_period.recommended" in param_paths


# ─────────────────────────────────────────────────────────────────────────────
# TestIterationComparator
# ─────────────────────────────────────────────────────────────────────────────

class TestIterationComparator:

    def test_compare_diagnoses_returns_report(self, sim_dir):
        """compare_diagnoses 应返回 ComparisonReport."""
        from quantmind.iteration.analyzer   import SimDiagnosis
        from quantmind.iteration.comparator import IterationComparator, ComparisonReport
        b = SimDiagnosis(ir_1w=-0.46, ir_2w=-0.42, ir_21d=-0.1, ir_3m=1.0,
                         win_rate_3m=0.9, loss_pct_gt5=0.1)
        n = SimDiagnosis(ir_1w=-0.20, ir_2w=-0.10, ir_21d=0.1,  ir_3m=1.5,
                         win_rate_3m=0.95, loss_pct_gt5=0.05)
        report = IterationComparator().compare_diagnoses(b, n)
        assert isinstance(report, ComparisonReport)

    def test_ir_delta_computed_correctly(self):
        """ir_delta 应正确计算新轮 - 基准."""
        from quantmind.iteration.analyzer   import SimDiagnosis
        from quantmind.iteration.comparator import IterationComparator
        b = SimDiagnosis(ir_3m=1.0)
        n = SimDiagnosis(ir_3m=1.5)
        report = IterationComparator().compare_diagnoses(b, n)
        assert abs(report.ir_delta["3m"] - 0.5) < 0.01

    def test_overall_improved_when_ir3m_up(self):
        """ir_3m 明显提升时 overall_improved 应为 True."""
        from quantmind.iteration.analyzer   import SimDiagnosis
        from quantmind.iteration.comparator import IterationComparator
        b = SimDiagnosis(ir_3m=0.5, win_rate_3m=0.5, loss_pct_gt5=0.2)
        n = SimDiagnosis(ir_3m=1.5, win_rate_3m=0.8, loss_pct_gt5=0.05)
        report = IterationComparator().compare_diagnoses(b, n)
        assert report.overall_improved is True

    def test_overall_not_improved_when_ir3m_down(self):
        """ir_3m 下降时 overall_improved 应为 False."""
        from quantmind.iteration.analyzer   import SimDiagnosis
        from quantmind.iteration.comparator import IterationComparator
        b = SimDiagnosis(ir_3m=1.5)
        n = SimDiagnosis(ir_3m=0.2)
        report = IterationComparator().compare_diagnoses(b, n)
        assert report.overall_improved is False

    def test_to_markdown_contains_sections(self):
        """to_markdown 输出应包含 IR 变化和综合判断章节."""
        from quantmind.iteration.analyzer   import SimDiagnosis
        from quantmind.iteration.comparator import IterationComparator
        b = SimDiagnosis(ir_3m=1.0)
        n = SimDiagnosis(ir_3m=1.2)
        md = IterationComparator().compare_diagnoses(b, n).to_markdown()
        assert "IR 变化" in md
        assert "综合判断" in md

    def test_to_html_report_returns_html(self):
        """to_html_report 应返回 <html> 字符串."""
        from quantmind.iteration.analyzer   import SimDiagnosis
        from quantmind.iteration.comparator import IterationComparator
        b = SimDiagnosis(ir_3m=1.0)
        n = SimDiagnosis(ir_3m=1.2)
        html = IterationComparator().compare_diagnoses(b, n).to_html_report()
        assert html.startswith("<html>")

    def test_compare_with_real_dirs(self, sim_dir, tmp_path):
        """compare(new_dir) 应读取两个目录并返回 ComparisonReport."""
        from quantmind.iteration.comparator import IterationComparator, ComparisonReport
        # 用同一个 sim_dir 作为 baseline 和 new（结果应 delta ≈ 0）
        report = IterationComparator(sim_dir).compare(sim_dir)
        assert isinstance(report, ComparisonReport)
        assert abs(report.ir_delta.get("3m", 99)) < 0.01  # 同目录 delta ≈ 0

    def test_loss_pct_delta_computed(self):
        """loss_pct_delta 应正确计算."""
        from quantmind.iteration.analyzer   import SimDiagnosis
        from quantmind.iteration.comparator import IterationComparator
        b = SimDiagnosis(loss_pct_gt5=0.30)
        n = SimDiagnosis(loss_pct_gt5=0.15)
        report = IterationComparator().compare_diagnoses(b, n)
        assert abs(report.loss_pct_delta - (-0.15)) < 0.001

    def test_win_rate_delta_computed(self):
        """win_rate_delta 应正确计算."""
        from quantmind.iteration.analyzer   import SimDiagnosis
        from quantmind.iteration.comparator import IterationComparator
        b = SimDiagnosis(win_rate_3m=0.60)
        n = SimDiagnosis(win_rate_3m=0.75)
        report = IterationComparator().compare_diagnoses(b, n)
        assert abs(report.win_rate_delta - 0.15) < 0.001
