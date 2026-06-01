"""tests/test_watchlist.py — 自选股模块单元测试 (≥15)."""
from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ── 夹具 ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_wl_path(tmp_path):
    """临时 watchlist.json 路径."""
    return tmp_path / "watchlist.json"


@pytest.fixture
def mgr(tmp_wl_path):
    """使用临时 JSON 路径的 WatchlistManager."""
    from quantmind.watchlist.manager import WatchlistManager
    m = WatchlistManager(path=tmp_wl_path)
    return m


@pytest.fixture
def mgr_with_stock(mgr):
    """已添加贵州茅台的 WatchlistManager（不走网络）."""
    with patch.object(mgr, "_fetch_stock_name", return_value="贵州茅台"), \
         patch.object(mgr, "_fetch_current_price", return_value=1450.0):
        mgr.add_stock("600519.SH", note="白酒龙头")
    return mgr


@pytest.fixture
def scorer(tmp_path):
    """使用临时目录的 WatchlistDailyScorer."""
    from quantmind.watchlist.daily_scorer import WatchlistDailyScorer
    return WatchlistDailyScorer(
        alpha_panel_path=None,  # 不加载真实 panel
        prices_path=None,
        cache_dir=tmp_path / "scores",
    )

# ── WatchlistManager 测试 ─────────────────────────────────────────────────────

class TestWatchlistManager:

    def test_add_stock_writes_json(self, mgr, tmp_wl_path):
        """add_stock 应写入 JSON 文件."""
        with patch.object(mgr, "_fetch_stock_name", return_value="贵州茅台"), \
             patch.object(mgr, "_fetch_current_price", return_value=1450.0):
            result = mgr.add_stock("600519.SH")

        assert result["success"] is True
        assert tmp_wl_path.exists()
        data = json.loads(tmp_wl_path.read_text(encoding="utf-8"))
        assert len(data["stocks"]) == 1
        assert data["stocks"][0]["ticker"] == "600519.SH"

    def test_add_stock_sets_name_and_price(self, mgr):
        """add_stock 应从工具函数取名称和价格."""
        with patch.object(mgr, "_fetch_stock_name", return_value="平安银行"), \
             patch.object(mgr, "_fetch_current_price", return_value=10.5):
            result = mgr.add_stock("000001.SZ")

        assert result["success"] is True
        stock = result["stock"]
        assert stock["name"] == "平安银行"
        assert stock["add_price"] == pytest.approx(10.5)

    def test_add_stock_duplicate_returns_exists(self, mgr_with_stock):
        """重复添加同一股票应返回 success=False 并提示已存在."""
        with patch.object(mgr_with_stock, "_fetch_stock_name", return_value="X"), \
             patch.object(mgr_with_stock, "_fetch_current_price", return_value=0.0):
            result = mgr_with_stock.add_stock("600519.SH")

        assert result["success"] is False
        assert "已在" in result["message"]

    def test_add_stock_records_add_date(self, mgr):
        """add_stock 应记录今日日期."""
        with patch.object(mgr, "_fetch_stock_name", return_value="T"), \
             patch.object(mgr, "_fetch_current_price", return_value=None):
            result = mgr.add_stock("600036.SH")

        today = date.today().isoformat()
        assert result["stock"]["add_date"] == today

    def test_remove_stock_deletes_entry(self, mgr_with_stock):
        """remove_stock 应从 JSON 中删除目标股票."""
        removed = mgr_with_stock.remove_stock("600519.SH")
        assert removed is True
        assert mgr_with_stock.get_all() == []

    def test_remove_stock_nonexistent_returns_false(self, mgr):
        """remove_stock 不存在的 ticker 应返回 False."""
        removed = mgr.remove_stock("999999.SH")
        assert removed is False

    def test_get_all_returns_list(self, mgr_with_stock):
        """get_all 应返回 list，每项有必需字段."""
        all_stocks = mgr_with_stock.get_all()
        assert isinstance(all_stocks, list)
        assert len(all_stocks) == 1
        s = all_stocks[0]
        for key in ("ticker", "name", "add_date", "add_price", "note"):
            assert key in s

    def test_update_note_success(self, mgr_with_stock):
        """update_note 应更新已有股票的备注."""
        ok = mgr_with_stock.update_note("600519.SH", "新备注")
        assert ok is True
        s = mgr_with_stock.get_stock("600519.SH")
        assert s["note"] == "新备注"

    def test_update_note_nonexistent_returns_false(self, mgr):
        """update_note 不存在的 ticker 应返回 False."""
        ok = mgr.update_note("000000.SH", "note")
        assert ok is False

    def test_fetch_stock_name_fallback_to_ticker(self, mgr):
        """_fetch_stock_name 找不到时应返回 ticker 本身."""
        # 没有本地 stock_basic，且 TUSHARE_TOKEN 未设置
        with patch("os.environ.get", return_value=""), \
             patch("pathlib.Path.exists", return_value=False):
            name = mgr._fetch_stock_name("900999.SH")
        assert name == "900999.SH"

    def test_add_empty_ticker_returns_error(self, mgr):
        """空字符串 ticker 应返回 success=False."""
        result = mgr.add_stock("")
        assert result["success"] is False

    def test_get_stock_returns_none_for_unknown(self, mgr):
        """get_stock 对不存在的 ticker 应返回 None."""
        assert mgr.get_stock("999.SH") is None

# ── StockScore 测试 ──────────────────────────────────────────────────────────

class TestStockScore:

    def test_stock_score_defaults(self):
        """StockScore 应有所有默认字段."""
        from quantmind.watchlist.daily_scorer import StockScore
        sc = StockScore(ticker="600519.SH", name="贵州茅台", as_of="2026-05-31")
        assert sc.value_score     == 50.0
        assert sc.momentum_score  == 50.0
        assert sc.quality_score   == 50.0
        assert sc.technical_score == 50.0
        assert sc.in_alpha_universe is False

    def test_stock_score_to_dict_and_from_dict(self):
        """to_dict / from_dict 应保持数据完整."""
        from quantmind.watchlist.daily_scorer import StockScore
        sc = StockScore(
            ticker="000001.SZ", name="平安银行", as_of="2026-05-31",
            current_price=10.5, value_score=65.0, agent_recommendation="买入",
            in_alpha_universe=True,
        )
        d    = sc.to_dict()
        sc2  = StockScore.from_dict(d)
        assert sc2.ticker             == "000001.SZ"
        assert sc2.current_price      == pytest.approx(10.5)
        assert sc2.value_score        == pytest.approx(65.0)
        assert sc2.agent_recommendation == "买入"
        assert sc2.in_alpha_universe  is True

# ── WatchlistDailyScorer 测试 ────────────────────────────────────────────────

class TestWatchlistDailyScorer:

    def test_score_all_returns_dict(self, scorer):
        """score_all 应返回 dict[ticker, StockScore]."""
        with patch.object(scorer, "_score_one", return_value=MagicMock(
            to_dict=lambda: {"ticker": "000001.SZ", "name": "X",
                             "as_of": "2026-05-31", "composite_score": 55.0}
        )):
            result = scorer.score_all(["000001.SZ"], "2026-05-31", use_cache=False)
        assert "000001.SZ" in result

    def test_score_all_uses_cache(self, scorer, tmp_path):
        """第二次调用应命中缓存，不重新计算."""
        from quantmind.watchlist.daily_scorer import StockScore
        cache_dir = tmp_path / "scores"
        cache_dir.mkdir(exist_ok=True)
        sc = StockScore(ticker="600519.SH", name="贵州茅台", as_of="2026-05-31",
                        composite_score=72.0)
        cache_file = cache_dir / "2026-05-31.json"
        cache_file.write_text(
            json.dumps({"600519.SH": sc.to_dict()}), encoding="utf-8"
        )
        from quantmind.watchlist.daily_scorer import WatchlistDailyScorer
        scorer2 = WatchlistDailyScorer(cache_dir=cache_dir)

        called = []
        original = scorer2._score_one
        def mock_score_one(*args, **kwargs):
            called.append(args[0])
            return original(*args, **kwargs)

        with patch.object(scorer2, "_score_one", side_effect=mock_score_one):
            result = scorer2.score_all(["600519.SH"], "2026-05-31", use_cache=True)

        # 缓存命中，_score_one 不应被调用
        assert "600519.SH" not in called
        assert result["600519.SH"].composite_score == pytest.approx(72.0)

    def test_get_price_info_returns_required_fields(self, scorer):
        """_get_price_info 对无数据的 ticker 应返回有必需字段的 dict."""
        info = scorer._get_price_info("999999.SH")
        for key in ("current_price", "price_change_1d", "price_change_5d",
                    "ma20", "ma60", "drawdown_from_high_52w", "volume_ratio"):
            assert key in info

    def test_not_in_alpha_universe(self, scorer):
        """不在 alpha panel 里的股票 in_alpha_universe 应为 False."""
        # scorer._alpha_panel 为 None（路径不存在），所以不在宇宙里
        result = scorer._get_factor_scores("888888.SH", "2026-05-31")
        assert result["in_alpha_universe"] is False

    def test_score_all_cache_invalidation(self, scorer, tmp_path):
        """invalidate_cache 应清除指定日期缓存文件."""
        cache_dir = tmp_path / "scores"
        cache_dir.mkdir(exist_ok=True)
        from quantmind.watchlist.daily_scorer import WatchlistDailyScorer, StockScore
        sc2 = WatchlistDailyScorer(cache_dir=cache_dir)
        cache_file = cache_dir / "2026-05-31.json"
        cache_file.write_text("{}", encoding="utf-8")
        assert cache_file.exists()
        sc2.invalidate_cache("2026-05-31")
        assert not cache_file.exists()
