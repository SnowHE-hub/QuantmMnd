"""pytest tests for quantmind/features/text_sentiment.py

Test coverage
-------------
1. fetch_announcements  — mock Tushare, 验证缓存写入 + 重复使用缓存
2. score_announcements  — 词典法 + mock transformer pipeline
3. _score_dict          — 正面/负面/中性文本各自的得分方向
4. build_ann_sentiment_factor — MultiIndex 格式、值域、缺失填 0
5. compute_ic           — 有效样本 IC 路径 + 样本不足路径
6. 端到端 run_full_pipeline — 3 条公告，无 Tushare 联网，验证因子输出
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quantmind.features.text_sentiment import (
    TEXT_SENTIMENT_FACTORS,
    _FACTOR_CACHE,
    _ANN_CACHE,
    _score_dict,
    _results_to_score,
    build_ann_sentiment_factor,
    compute_ic,
    fetch_announcements,
    run_full_pipeline,
    score_announcements,
)


# ─── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def three_announcements() -> pd.DataFrame:
    """3 条公告（2 只股票），供多个测试复用."""
    return pd.DataFrame({
        "ts_code":  ["000001.SZ", "000001.SZ", "000002.SZ"],
        "ann_date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-02"]),
        "title":    ["公司盈利增长超预期", "公司亏损下滑风险", "中性公告"],
        "content":  ["净利润同比增长30%，分红回购", "净利润亏损，减持风险", "无特别内容"],
    })


@pytest.fixture()
def scored_announcements(three_announcements) -> pd.DataFrame:
    """已打分的公告 DataFrame（使用词典法）."""
    return score_announcements(three_announcements)


# ─── 1. fetch_announcements (mock Tushare) ────────────────────────────────────

class TestFetchAnnouncements:
    def test_cache_is_written(self, tmp_path, three_announcements, monkeypatch):
        """mock tushare 返回 3 条公告后，缓存文件应存在且行数正确."""
        # 重定向缓存路径到临时目录
        cache_path = tmp_path / "announcements.parquet"
        monkeypatch.setattr(
            "quantmind.features.text_sentiment._ANN_CACHE", cache_path
        )
        monkeypatch.setattr(
            "quantmind.features.text_sentiment._TEXT_DIR", tmp_path
        )

        # Mock tushare
        mock_pro = MagicMock()
        mock_pro.anns_d.return_value = three_announcements.rename(
            columns={"ann_date": "ann_date"}  # keep same
        ).assign(ann_date=three_announcements["ann_date"].dt.strftime("%Y%m%d"))

        with patch("tushare.pro_api", return_value=mock_pro), \
             patch("quantmind.features.text_sentiment._load_token",
                   return_value="fake_token"):
            df = fetch_announcements(start_date="20250101", end_date="20250131")

        assert cache_path.exists(), "缓存文件应被写入"
        assert len(df) == 3, f"预期 3 行，得到 {len(df)}"

    def test_cache_is_reused(self, tmp_path, three_announcements, monkeypatch):
        """缓存已存在时，不再调用 Tushare API."""
        cache_path = tmp_path / "announcements.parquet"
        three_announcements.to_parquet(cache_path, index=False)

        monkeypatch.setattr(
            "quantmind.features.text_sentiment._ANN_CACHE", cache_path
        )
        monkeypatch.setattr(
            "quantmind.features.text_sentiment._TEXT_DIR", tmp_path
        )

        with patch("tushare.pro_api") as mock_pro_api:
            df = fetch_announcements(start_date="20250101")
            mock_pro_api.assert_not_called()   # 不应联网

        assert len(df) == 3

    def test_empty_result_returns_empty_df(self, tmp_path, monkeypatch):
        """Tushare 返回空时，应返回空 DataFrame 并写入缓存（不崩溃）."""
        cache_path = tmp_path / "announcements.parquet"
        monkeypatch.setattr(
            "quantmind.features.text_sentiment._ANN_CACHE", cache_path
        )
        monkeypatch.setattr(
            "quantmind.features.text_sentiment._TEXT_DIR", tmp_path
        )

        mock_pro = MagicMock()
        mock_pro.anns_d.return_value = pd.DataFrame()

        with patch("tushare.pro_api", return_value=mock_pro), \
             patch("quantmind.features.text_sentiment._load_token",
                   return_value="tok"):
            df = fetch_announcements(start_date="20250101", end_date="20250115")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert cache_path.exists()


# ─── 2. _score_dict ──────────────────────────────────────────────────────────

class TestScoreDict:
    def test_positive_text_returns_positive_score(self):
        score = _score_dict("公司盈利增长超预期，分红回购")
        assert score > 0, f"正面文本应得 >0 分，got {score}"

    def test_negative_text_returns_negative_score(self):
        score = _score_dict("公司亏损下滑，减持风险，违规处罚")
        assert score < 0, f"负面文本应得 <0 分，got {score}"

    def test_neutral_text_returns_zero(self):
        score = _score_dict("公告无特别内容")
        assert score == pytest.approx(0.0), f"中性文本应得 ~0，got {score}"

    def test_empty_string_returns_zero(self):
        assert _score_dict("") == 0.0
        assert _score_dict(None) == 0.0   # type: ignore[arg-type]

    def test_score_in_range(self):
        score = _score_dict("增长盈利回购 亏损减持违规诉讼")
        assert -1.0 <= score <= 1.0


# ─── 3. _results_to_score ────────────────────────────────────────────────────

class TestResultsToScore:
    def test_positive_dominant(self):
        results = [{"label": "positive", "score": 0.9},
                   {"label": "negative", "score": 0.1}]
        assert _results_to_score(results) == pytest.approx(0.8)

    def test_negative_dominant(self):
        results = [{"label": "positive", "score": 0.2},
                   {"label": "negative", "score": 0.8}]
        assert _results_to_score(results) == pytest.approx(-0.6)

    def test_label_pos_neg_variants(self):
        """LABEL_1 / LABEL_0 风格的标签."""
        results = [{"label": "1", "score": 0.7},
                   {"label": "0", "score": 0.3}]
        score = _results_to_score(results)
        assert score == pytest.approx(0.4)


# ─── 4. score_announcements ──────────────────────────────────────────────────

class TestScoreAnnouncements:
    def test_dict_fallback_scores_all_rows(self, three_announcements):
        """不注入 transformer 时，词典法应对全部行打分."""
        result = score_announcements(three_announcements)
        assert "sentiment_score" in result.columns
        assert len(result) == 3

    def test_scores_in_range(self, three_announcements):
        result = score_announcements(three_announcements)
        assert result["sentiment_score"].between(-1.0, 1.0).all(), \
            f"情绪分应在 [-1,1]：\n{result['sentiment_score']}"

    def test_positive_title_gets_positive_score(self, three_announcements):
        result = score_announcements(three_announcements)
        # 第 0 行是正面公告
        assert result.iloc[0]["sentiment_score"] > 0

    def test_negative_title_gets_negative_score(self, three_announcements):
        result = score_announcements(three_announcements)
        # 第 1 行是负面公告
        assert result.iloc[1]["sentiment_score"] < 0

    def test_mock_transformer_pipeline(self, three_announcements):
        """注入 mock transformer pipeline，验证得分路径正确调用."""
        def mock_pipe(batch):
            return [
                [{"label": "positive", "score": 0.8},
                 {"label": "negative", "score": 0.2}]
            ] * len(batch)

        result = score_announcements(three_announcements, _pipe_override=mock_pipe)
        # pytest.approx 不直接配合 Series.all() 使用，改用 np.testing
        np.testing.assert_allclose(
            result["sentiment_score"].values,
            0.6,
            atol=1e-6,
            err_msg="mock pipeline 应对所有行返回 0.6",
        )
        assert (result["model_used"] == "mock").all()

    def test_empty_df_handled(self):
        result = score_announcements(pd.DataFrame())
        assert result.empty or "sentiment_score" in result.columns


# ─── 5. build_ann_sentiment_factor ───────────────────────────────────────────

class TestBuildAnnSentimentFactor:
    def test_multiindex_format(self, scored_announcements):
        factor = build_ann_sentiment_factor(scored_announcements)
        assert isinstance(factor, pd.Series)
        assert factor.index.names == ["ts_code", "trade_date"], \
            f"MultiIndex names 应为 [ts_code, trade_date]，got {factor.index.names}"

    def test_values_in_range(self, scored_announcements):
        factor = build_ann_sentiment_factor(scored_announcements)
        assert factor.between(-1.0, 1.0).all(), \
            f"因子值应在 [-1,1]：{factor.describe()}"

    def test_missing_values_filled_with_zero(self):
        """无公告的日期 → 滚动窗口衰减，6 天后（5日窗口外）应回到 0."""
        df = pd.DataFrame({
            "ts_code":         ["A.SZ"],
            "ann_date":        pd.to_datetime(["2025-01-02"]),
            "title":           ["盈利增长"],
            "content":         ["好"],
            "sentiment_score": [0.8],
        })
        # 日历须覆盖 5 日窗口之外：公告在 Jan-02，窗口 5 天，Jan-09+ 的值应为 0
        cal = pd.DatetimeIndex(pd.date_range("2025-01-02", "2025-02-10", freq="B"))
        factor = build_ann_sentiment_factor(df, trading_cal=cal)
        assert factor.notna().all(), "因子不应含 NaN"
        # 5 日滚动窗口：5 天后公告不再在窗口内，值应衰减到 0
        zero_mask = np.abs(factor.values) < 1e-9
        assert zero_mask.any(), (
            "应有 0 值（公告日期超出 5 日滚动窗口后的日期）"
        )

    def test_two_tickers_independent(self, scored_announcements):
        """两只股票的因子应独立计算."""
        factor = build_ann_sentiment_factor(scored_announcements)
        tickers = factor.index.get_level_values("ts_code").unique()
        assert len(tickers) == 2, f"应有 2 只股票，得到 {tickers.tolist()}"

    def test_empty_df_returns_empty_series(self):
        factor = build_ann_sentiment_factor(pd.DataFrame())
        assert isinstance(factor, pd.Series)
        assert factor.empty

    def test_no_sentiment_score_column_returns_empty(self, three_announcements):
        factor = build_ann_sentiment_factor(three_announcements)   # 无 sentiment_score
        assert factor.empty

    def test_factor_name(self, scored_announcements):
        factor = build_ann_sentiment_factor(scored_announcements)
        assert factor.name == "ann_sentiment_5d"


# ─── 6. compute_ic ───────────────────────────────────────────────────────────

class TestComputeIC:
    def _make_factor(self) -> pd.Series:
        """构造与 stock_returns 格式兼容的因子 Series."""
        n = 50
        rng = np.random.default_rng(42)
        tickers = [f"{i:06d}.SZ" for i in range(1, n + 1)]
        dates   = pd.to_datetime(["2025-10-09"] * n)
        idx     = pd.MultiIndex.from_arrays([tickers, dates],
                                            names=["ts_code", "trade_date"])
        return pd.Series(rng.uniform(-1, 1, n), index=idx, name="ann_sentiment_5d")

    def _make_returns_parquet(self, tmp_path: Path, factor: pd.Series) -> Path:
        """创建与 factor 对齐的 mock stock_returns.parquet."""
        tickers   = factor.index.get_level_values("ts_code").tolist()
        dates     = factor.index.get_level_values("trade_date").tolist()
        rng       = np.random.default_rng(0)
        return_3m = rng.normal(0.1, 0.2, len(tickers))
        df = pd.DataFrame({"ticker": tickers, "date": dates,
                           "return_3m": return_3m})
        p = tmp_path / "stock_returns.parquet"
        df.to_parquet(p, index=False)
        return p

    def test_valid_sample_returns_ic_dict(self, tmp_path):
        factor = self._make_factor()
        ret_path = self._make_returns_parquet(tmp_path, factor)
        result = compute_ic(factor, returns_path=ret_path)
        assert "ic" in result and "p_value" in result and "n" in result
        assert np.isfinite(result["ic"]), "有效样本应返回有限 IC 值"
        assert result["n"] == 50

    def test_insufficient_sample_returns_nan(self, tmp_path):
        """样本 < 30 应返回 ic=NaN."""
        n = 10
        rng = np.random.default_rng(1)
        tickers = [f"{i:06d}.SZ" for i in range(1, n + 1)]
        dates   = pd.to_datetime(["2025-10-09"] * n)
        idx     = pd.MultiIndex.from_arrays([tickers, dates],
                                            names=["ts_code", "trade_date"])
        factor  = pd.Series(rng.uniform(-1, 1, n), index=idx)
        df = pd.DataFrame({
            "ticker": tickers, "date": dates,
            "return_3m": rng.normal(0.1, 0.2, n),
        })
        p = tmp_path / "sr.parquet"
        df.to_parquet(p, index=False)
        result = compute_ic(factor, returns_path=p)
        assert np.isnan(result["ic"])

    def test_missing_returns_file(self, tmp_path):
        factor = self._make_factor()
        result = compute_ic(factor, returns_path=tmp_path / "nonexistent.parquet")
        assert np.isnan(result["ic"])
        assert result["valid"] is False


# ─── 7. TEXT_SENTIMENT_FACTORS 常量 ──────────────────────────────────────────

def test_text_sentiment_factors_constant():
    assert "ann_sentiment_5d" in TEXT_SENTIMENT_FACTORS


# ─── 8. 端到端 run_full_pipeline (mock, 无联网) ──────────────────────────────

class TestRunFullPipeline:
    def test_pipeline_returns_factor_and_ic(self, tmp_path, three_announcements,
                                             monkeypatch):
        """mock Tushare + 无 transformer → 词典打分 → 验证输出格式."""
        monkeypatch.setattr(
            "quantmind.features.text_sentiment._ANN_CACHE",
            tmp_path / "anns.parquet",
        )
        monkeypatch.setattr(
            "quantmind.features.text_sentiment._FACTOR_CACHE",
            tmp_path / "factor.parquet",
        )
        monkeypatch.setattr(
            "quantmind.features.text_sentiment._TEXT_DIR", tmp_path
        )
        # stock_returns: 无匹配 ticker，IC 会是 n=0 路径
        monkeypatch.setattr(
            "quantmind.features.text_sentiment._RETURNS_PATH",
            tmp_path / "nonexistent.parquet",
        )

        mock_pro = MagicMock()
        mock_pro.anns_d.return_value = three_announcements.assign(
            ann_date=three_announcements["ann_date"].dt.strftime("%Y%m%d")
        )

        with patch("tushare.pro_api", return_value=mock_pro), \
             patch("quantmind.features.text_sentiment._load_token",
                   return_value="tok"):
            factor, ic = run_full_pipeline(start_date="20250101", force_fetch=True)

        # 因子格式
        assert isinstance(factor, pd.Series)
        assert factor.index.names == ["ts_code", "trade_date"]
        assert factor.between(-1.0, 1.0).all(), f"因子超出 [-1,1]: {factor}"

        # IC 结构完整
        assert {"ic", "p_value", "n", "valid"} == set(ic.keys())
