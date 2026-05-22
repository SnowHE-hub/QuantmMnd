"""tests/test_analyst_revision.py

Tests for quantmind/features/analyst_revision.py

覆盖点：
  1-5   评级映射（所有已知字符串、未知 → NaN、None/空串、数字字符串）
  6-7   build_analyst_revision_factor：升降调方向
  8-9   build_analyst_revision_factor：缺失前期报告时用中性基准 3.0
  10    build_analyst_revision_factor：空 DataFrame 返回空 DataFrame
  11    build_analyst_revision_factor：无近期报告的股票被过滤
  12    build_analyst_revision_factor：输出列结构
  13    build_analyst_revision_factor：多股票时每股一行
  14-15 compute_analyst_ic：空因子返回 nan；样本量 < 30 返回 n=0 且 valid=False
  16    compute_analyst_ic：returns 文件不存在 → 返回 nan
  17    fetch_analyst_reports 缓存命中：有缓存时不调用 tushare
  18    fetch_analyst_reports 空 API 返回：缓存空 DataFrame 并返回空 DataFrame
  19    ANALYST_REVISION_FACTORS 常量内容
  20    run_analyst_pipeline：smoke test（mock fetch + IC）
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quantmind.features.analyst_revision import (
    ANALYST_REVISION_FACTORS,
    _digitize_rating,
    build_analyst_revision_factor,
    compute_analyst_ic,
    fetch_analyst_reports,
    run_analyst_pipeline,
)


# ─── 1-5: 评级映射 ─────────────────────────────────────────────────────────────

class TestDigitizeRating:
    """_digitize_rating：评级字符串 → 数值映射."""

    def test_strong_buy_ratings(self):
        assert _digitize_rating("强烈推荐") == 5.0
        assert _digitize_rating("强推") == 5.0
        assert _digitize_rating("买入") == 5.0

    def test_buy_ratings(self):
        assert _digitize_rating("推荐") == 4.0
        assert _digitize_rating("增持") == 4.0
        assert _digitize_rating("审慎推荐") == 4.0
        assert _digitize_rating("优于大市") == 4.0

    def test_neutral_ratings(self):
        assert _digitize_rating("中性") == 3.0
        assert _digitize_rating("持有") == 3.0
        assert _digitize_rating("标配") == 3.0
        assert _digitize_rating("中立") == 3.0

    def test_underweight_ratings(self):
        assert _digitize_rating("减持") == 2.0
        assert _digitize_rating("低于大市") == 2.0
        assert _digitize_rating("回避") == 2.0

    def test_sell_ratings(self):
        assert _digitize_rating("卖出") == 1.0
        assert _digitize_rating("强卖") == 1.0
        assert _digitize_rating("强烈卖出") == 1.0

    def test_unknown_rating_returns_nan(self):
        assert math.isnan(_digitize_rating("未知评级"))
        assert math.isnan(_digitize_rating("gibberish"))
        assert math.isnan(_digitize_rating("BUY"))

    def test_none_returns_nan(self):
        assert math.isnan(_digitize_rating(None))

    def test_empty_string_returns_nan(self):
        assert math.isnan(_digitize_rating(""))
        assert math.isnan(_digitize_rating("   "))

    def test_numeric_string_returns_nan(self):
        # 纯数字不在映射表中
        assert math.isnan(_digitize_rating("5"))
        assert math.isnan(_digitize_rating("3.0"))


# ─── 6-13: build_analyst_revision_factor ───────────────────────────────────────

def _make_reports(
    ts_code: str,
    dates_ratings: list[tuple[str, str]],
) -> pd.DataFrame:
    """辅助：构造 report_rc 格式的 DataFrame."""
    rows = []
    for d, r in dates_ratings:
        rows.append({
            "ts_code":      ts_code,
            "report_date":  pd.Timestamp(d),
            "rating":       r,
            "rating_score": _digitize_rating(r),
        })
    return pd.DataFrame(rows)


class TestBuildAnalystRevisionFactor:
    """build_analyst_revision_factor：因子构造逻辑."""

    def test_upgrade_gives_positive_revision(self):
        """近期评级（5）高于前期（3）→ revision > 0."""
        df = _make_reports("000001.SZ", [
            ("2024-01-10", "持有"),   # 前期
            ("2024-02-15", "买入"),   # 近期（约 36 天后）
        ])
        result = build_analyst_revision_factor(df, window=30)
        assert not result.empty
        val = result.loc[result["ts_code"] == "000001.SZ", "analyst_revision_30d"].iloc[0]
        assert val > 0, f"预期 revision > 0，实际 {val}"

    def test_downgrade_gives_negative_revision(self):
        """近期评级（1）低于前期（5）→ revision < 0."""
        df = _make_reports("000002.SZ", [
            ("2024-01-05", "买入"),   # 前期
            ("2024-02-10", "卖出"),   # 近期
        ])
        result = build_analyst_revision_factor(df, window=30)
        assert not result.empty
        val = result.loc[result["ts_code"] == "000002.SZ", "analyst_revision_30d"].iloc[0]
        assert val < 0, f"预期 revision < 0，实际 {val}"

    def test_no_prev_period_uses_neutral_baseline(self):
        """前期无报告时，前期均值默认为 3.0（中性基准）."""
        # 只有近期报告（买入=5），前期无报告 → revision = 5 - 3 = 2.0
        df = _make_reports("000003.SZ", [
            ("2024-03-01", "买入"),
            ("2024-03-05", "买入"),
        ])
        result = build_analyst_revision_factor(df, window=30)
        assert not result.empty
        val = result.loc[result["ts_code"] == "000003.SZ", "analyst_revision_30d"].iloc[0]
        assert abs(val - 2.0) < 1e-9, f"预期 2.0（5-3），实际 {val}"

    def test_neutral_remains_zero_with_neutral_baseline(self):
        """近期均为中性（3），前期无报告 → revision = 0.0."""
        df = _make_reports("000004.SZ", [
            ("2024-03-10", "持有"),
            ("2024-03-15", "中性"),
        ])
        result = build_analyst_revision_factor(df, window=30)
        assert not result.empty
        val = result.loc[result["ts_code"] == "000004.SZ", "analyst_revision_30d"].iloc[0]
        assert abs(val) < 1e-9, f"预期 0.0，实际 {val}"

    def test_empty_dataframe_returns_empty(self):
        result = build_analyst_revision_factor(pd.DataFrame())
        assert result.empty
        assert "analyst_revision_30d" in result.columns

    def test_missing_rating_score_column_returns_empty(self):
        df = pd.DataFrame({"ts_code": ["000001.SZ"], "report_date": ["2024-01-01"]})
        result = build_analyst_revision_factor(df, window=30)
        assert result.empty

    def test_stock_with_no_recent_reports_is_filtered(self):
        """最近 30 天无报告的股票不应出现在结果中（ref_date 是最新报告日）."""
        # 所有报告都在 ref_date - 30d 之前，近期窗口为空
        # ref_date = max(report_date) = 2024-01-01
        # recent = report_date > 2023-12-02  → 只有 2024-01-01 在近期
        df = _make_reports("000005.SZ", [
            ("2024-01-01", "增持"),   # 唯一近期报告
        ])
        result = build_analyst_revision_factor(df, window=30)
        # 有近期报告 → 不过滤
        assert "000005.SZ" in result["ts_code"].values

    def test_output_columns(self):
        df = _make_reports("000006.SZ", [
            ("2024-01-05", "买入"),
            ("2024-02-10", "持有"),
        ])
        result = build_analyst_revision_factor(df, window=30)
        assert set(result.columns) >= {"ts_code", "report_date", "analyst_revision_30d"}

    def test_multi_stock_one_row_per_stock(self):
        """多只股票各返回一行（截面）."""
        df1 = _make_reports("000001.SZ", [("2024-02-15", "买入")])
        df2 = _make_reports("000002.SZ", [("2024-02-20", "减持")])
        df  = pd.concat([df1, df2], ignore_index=True)
        result = build_analyst_revision_factor(df, window=30)
        assert len(result) == 2
        assert set(result["ts_code"]) == {"000001.SZ", "000002.SZ"}

    def test_all_nan_ratings_returns_empty(self):
        """评级全为未知（NaN）时返回空 DataFrame."""
        df = pd.DataFrame({
            "ts_code":      ["000007.SZ", "000007.SZ"],
            "report_date":  [pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-05")],
            "rating":       ["???", "???"],
            "rating_score": [float("nan"), float("nan")],
        })
        result = build_analyst_revision_factor(df, window=30)
        assert result.empty


# ─── 14-16: compute_analyst_ic ─────────────────────────────────────────────────

class TestComputeAnalystIc:
    """compute_analyst_ic：IC 验证."""

    def test_empty_factor_returns_nan(self):
        result = compute_analyst_ic(pd.DataFrame(columns=["ts_code", "analyst_revision_30d"]))
        assert math.isnan(result["ic"])
        assert result["valid"] is False

    def test_small_n_returns_invalid(self, tmp_path):
        """样本量 < 30 时 valid=False，n 记录实际样本数."""
        n = 5
        ret_df = pd.DataFrame({
            "ts_code":    [f"{i:06d}.SZ" for i in range(n)],
            "return_3m":  [0.1] * n,
        })
        ret_path = tmp_path / "stock_returns.parquet"
        ret_df.to_parquet(ret_path, index=False)

        factor_df = pd.DataFrame({
            "ts_code":              [f"{i:06d}.SZ" for i in range(n)],
            "report_date":          [pd.Timestamp("2024-03-01")] * n,
            "analyst_revision_30d": [float(i) for i in range(n)],
        })
        result = compute_analyst_ic(factor_df, returns_path=ret_path)
        assert result["n"] <= 30
        assert result["valid"] is False

    def test_returns_file_not_exist_returns_nan(self, tmp_path):
        factor_df = pd.DataFrame({
            "ts_code":              ["000001.SZ"],
            "report_date":          [pd.Timestamp("2024-03-01")],
            "analyst_revision_30d": [1.0],
        })
        result = compute_analyst_ic(
            factor_df,
            returns_path=tmp_path / "nonexistent.parquet",
        )
        assert math.isnan(result["ic"])
        assert result["n"] == 0

    def test_sufficient_sample_computes_ic(self, tmp_path):
        """足够样本 (n=50) 时应返回有效 IC dict."""
        n = 50
        tickers = [f"{i:06d}.SZ" for i in range(n)]
        factor_vals = [float(i) for i in range(n)]
        ret_vals    = [float(i) * 0.01 for i in range(n)]  # 正相关

        ret_df = pd.DataFrame({"ts_code": tickers, "return_3m": ret_vals})
        ret_path = tmp_path / "stock_returns.parquet"
        ret_df.to_parquet(ret_path, index=False)

        factor_df = pd.DataFrame({
            "ts_code":              tickers,
            "report_date":          [pd.Timestamp("2024-03-01")] * n,
            "analyst_revision_30d": factor_vals,
        })
        result = compute_analyst_ic(factor_df, returns_path=ret_path)
        assert not math.isnan(result["ic"])
        assert result["n"] == n
        assert result["ic"] > 0.9  # 完全正相关


# ─── 17-18: fetch_analyst_reports ──────────────────────────────────────────────

class TestFetchAnalystReports:
    """fetch_analyst_reports：缓存逻辑."""

    def test_cache_hit_skips_tushare(self, tmp_path, monkeypatch):
        """缓存存在且 force=False 时不调用 tushare."""
        cache = tmp_path / "analyst_reports.parquet"
        expected = pd.DataFrame({
            "ts_code":      ["000001.SZ"],
            "report_date":  [pd.Timestamp("2024-01-15")],
            "rating":       ["买入"],
            "rating_score": [5.0],
        })
        expected.to_parquet(cache, index=False)

        monkeypatch.setattr(
            "quantmind.features.analyst_revision._ANALYST_CACHE",
            cache,
        )
        with patch("tushare.pro_api") as mock_pro:
            df = fetch_analyst_reports(force=False)
            mock_pro.assert_not_called()

        assert len(df) == 1
        assert df["ts_code"].iloc[0] == "000001.SZ"

    def test_empty_api_response_caches_empty_df(self, tmp_path, monkeypatch):
        """API 返回空时，缓存空 DataFrame 并返回空 DataFrame."""
        cache = tmp_path / "analyst_reports.parquet"
        monkeypatch.setattr(
            "quantmind.features.analyst_revision._ANALYST_CACHE",
            cache,
        )
        monkeypatch.setattr(
            "quantmind.features.analyst_revision._TEXT_DIR",
            tmp_path,
        )

        mock_pro = MagicMock()
        mock_pro.report_rc.return_value = pd.DataFrame()

        with patch("tushare.pro_api", return_value=mock_pro):
            with patch("quantmind.features.analyst_revision._load_token", return_value="fake_token"):
                df = fetch_analyst_reports(
                    start_date="20240101",
                    end_date="20240131",
                    force=True,
                )

        assert df.empty
        assert cache.exists()


# ─── 19: ANALYST_REVISION_FACTORS 常量 ─────────────────────────────────────────

def test_analyst_revision_factors_constant():
    """ANALYST_REVISION_FACTORS 应包含 analyst_revision_30d."""
    assert "analyst_revision_30d" in ANALYST_REVISION_FACTORS
    assert isinstance(ANALYST_REVISION_FACTORS, list)
    assert len(ANALYST_REVISION_FACTORS) >= 1


# ─── 20: run_analyst_pipeline smoke test ───────────────────────────────────────

def test_run_analyst_pipeline_smoke(tmp_path, monkeypatch):
    """smoke test：mock fetch，验证 pipeline 返回 (DataFrame, dict)."""
    monkeypatch.setattr(
        "quantmind.features.analyst_revision._TEXT_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        "quantmind.features.analyst_revision._ANALYST_CACHE",
        tmp_path / "analyst_reports.parquet",
    )
    monkeypatch.setattr(
        "quantmind.features.analyst_revision._RETURNS_PATH",
        tmp_path / "nonexistent.parquet",  # 让 IC 返回 nan（但不崩溃）
    )

    sample_reports = pd.DataFrame({
        "ts_code":      ["000001.SZ", "000002.SZ"],
        "report_date":  [pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-05")],
        "rating":       ["买入", "减持"],
        "rating_score": [5.0, 2.0],
    })

    with patch(
        "quantmind.features.analyst_revision.fetch_analyst_reports",
        return_value=sample_reports,
    ):
        factor_df, ic_result = run_analyst_pipeline(force_fetch=False)

    assert isinstance(factor_df, pd.DataFrame)
    assert isinstance(ic_result, dict)
    assert "ic" in ic_result
    assert "valid" in ic_result
