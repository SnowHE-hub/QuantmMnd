"""quantmind.data.backtest_data — 回测系统专用数据提供者.

为回测验证系统提供：精确历史价格、指数收益、交易成本配置。
专注 PIT（Point-In-Time）正确性，不依赖外部 API。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

_ROOT = Path(__file__).resolve().parent.parent.parent
_PRICES_DIR = _ROOT / "data" / "prices"
_CSI300_PRICE_FILE = _PRICES_DIR / "csi300_daily_adj_close.parquet"
_FULL_MARKET_PRICE_FILE = _PRICES_DIR / "full_market_daily_adj_close.parquet"
_INDEX_FILE = _PRICES_DIR / "index_daily.parquet"

# 备选文件路径
_RAW_PANEL = _ROOT / "data" / "raw" / "daily_prices_panel.parquet"


class BacktestDataProvider:
    """回测系统专用数据提供者.

    特点：
    - 严格 PIT：只返回 [start_date, end_date] 内的数据
    - 不依赖外部 API，只读本地文件
    - 缓存策略：read-only，不写入
    """

    def __init__(self) -> None:
        self._price_cache: dict[str, pd.DataFrame] = {}

    # ── 股票价格序列 ───────────────────────────────────────────────────────────

    def get_price_series(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        adjust: str = "hfq",
    ) -> pd.Series:
        """获取单只股票复权价格序列（PIT 严格）.

        Args:
            ticker: 股票代码（如 600519.SH）
            start_date: 开始日期（含，YYYY-MM-DD）
            end_date: 结束日期（含，YYYY-MM-DD），严格 PIT
            adjust: 复权方式（hfq=后复权，默认）

        Returns:
            pd.Series，index=DatetimeIndex(trade_date)，name=ticker
        """
        panel = self._load_price_panel(ticker)
        if panel is None or ticker not in panel.columns:
            logger.warning(f"[BacktestData] {ticker} 不在价格面板中")
            return pd.Series(dtype=float, name=ticker)

        series = panel[ticker].dropna()

        # PIT 严格：只返回 [start_date, end_date] 范围内的数据
        if start_date:
            series = series[series.index >= pd.Timestamp(start_date)]
        if end_date:
            series = series[series.index <= pd.Timestamp(end_date)]

        return series

    def get_price_panel(
        self,
        tickers: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """获取多只股票价格面板.

        Returns:
            DataFrame，index=DatetimeIndex，columns=tickers
        """
        # 确定要读取的文件
        panel_file = self._resolve_price_file()
        if panel_file is None:
            return pd.DataFrame()

        cache_key = str(panel_file)
        if cache_key not in self._price_cache:
            logger.info(f"[BacktestData] 加载价格面板: {panel_file}")
            self._price_cache[cache_key] = pd.read_parquet(panel_file)

        df = self._price_cache[cache_key]

        # 过滤 ticker
        if tickers:
            available = [t for t in tickers if t in df.columns]
            df = df[available]

        # PIT 时间过滤
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]

        return df

    def get_forward_returns(
        self,
        tickers: list[str],
        signal_date: str,
        horizon_days: int = 21,
    ) -> pd.Series:
        """计算信号日后 horizon_days 个交易日的前向收益.

        Args:
            tickers: 股票列表
            signal_date: 信号日期（用作 entry price 参考）
            horizon_days: 持有天数（交易日）

        Returns:
            pd.Series，index=ticker，value=前向收益率
        """
        panel = self._load_price_panel_full()
        if panel is None:
            return pd.Series(dtype=float)

        signal_ts = pd.Timestamp(signal_date)
        result = {}

        for ticker in tickers:
            if ticker not in panel.columns:
                continue
            series = panel[ticker].dropna()
            future = series[series.index >= signal_ts]
            if len(future) < horizon_days + 1:
                continue
            entry = float(future.iloc[0])
            exit_price = float(future.iloc[horizon_days])
            if entry > 0:
                result[ticker] = exit_price / entry - 1

        return pd.Series(result)

    # ── 指数数据 ───────────────────────────────────────────────────────────────

    def get_index_returns(
        self,
        index_code: str = "000300.SH",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.Series:
        """获取指数日收益率序列.

        Args:
            index_code: 指数代码（000300.SH=沪深300）
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            pd.Series，index=DatetimeIndex，日收益率
        """
        if not _INDEX_FILE.exists():
            logger.warning(f"[BacktestData] 指数文件不存在: {_INDEX_FILE}")
            return pd.Series(dtype=float)

        try:
            df = pd.read_parquet(_INDEX_FILE)

            # 尝试多种列名格式
            idx_col = None
            for col in [index_code, index_code.split(".")[0], "close", "000300"]:
                if col in df.columns:
                    idx_col = col
                    break

            if idx_col is None:
                logger.warning(f"[BacktestData] 指数 {index_code} 不在文件中，可用列: {df.columns[:5].tolist()}")
                # 取第一列
                if len(df.columns) > 0:
                    idx_col = df.columns[0]
                else:
                    return pd.Series(dtype=float)

            series = df[idx_col].dropna()
            if start_date:
                series = series[series.index >= pd.Timestamp(start_date)]
            if end_date:
                series = series[series.index <= pd.Timestamp(end_date)]

            return series.pct_change().dropna()
        except Exception as e:
            logger.warning(f"[BacktestData] 指数数据读取失败: {e}")
            return pd.Series(dtype=float)

    # ── 交易成本配置 ───────────────────────────────────────────────────────────

    def get_cost_config(self) -> dict:
        """返回交易成本参数（A股标准）.

        Returns:
            dict:
              commission_bps: 佣金（买卖双向，bps）
              stamp_tax_bps: 印花税（仅卖出，bps）
              slippage_bps: 滑点估计（bps）
              total_buy_bps: 买入总成本
              total_sell_bps: 卖出总成本
        """
        commission = 3    # 3 bps 佣金（双向）
        stamp_tax = 10    # 10 bps 印花税（仅卖出）
        slippage = 10     # 10 bps 滑点

        return {
            "commission_bps": commission,
            "stamp_tax_bps": stamp_tax,
            "slippage_bps": slippage,
            "total_buy_bps": commission + slippage,
            "total_sell_bps": commission + stamp_tax + slippage,
            "round_trip_bps": 2 * commission + stamp_tax + 2 * slippage,
            "round_trip_pct": (2 * commission + stamp_tax + 2 * slippage) / 10000,
        }

    # ── 因子面板 ───────────────────────────────────────────────────────────────

    def get_factor_panel(
        self,
        panel_paths: list[str | Path] | None = None,
    ) -> pd.DataFrame | None:
        """获取因子面板（用于历史截面信号）.

        Args:
            panel_paths: 面板文件路径列表（默认使用 data/panel/test.parquet）

        Returns:
            MultiIndex DataFrame (as_of × ticker)
        """
        if panel_paths is None:
            panel_paths = [
                _ROOT / "data/panel/test.parquet",
                _ROOT / "data/panel/val.parquet",
            ]

        frames = []
        for p in panel_paths:
            p = Path(p)
            if p.exists():
                frames.append(pd.read_parquet(p))
        if not frames:
            return None

        return pd.concat(frames).sort_index()

    # ── 辅助方法 ───────────────────────────────────────────────────────────────

    def _load_price_panel(self, ticker: str) -> pd.DataFrame | None:
        """加载含指定 ticker 的价格面板（自动选择文件）."""
        panel_file = self._resolve_price_file(ticker)
        if panel_file is None:
            return None
        cache_key = str(panel_file)
        if cache_key not in self._price_cache:
            self._price_cache[cache_key] = pd.read_parquet(panel_file)
        return self._price_cache[cache_key]

    def _load_price_panel_full(self) -> pd.DataFrame | None:
        """加载完整价格面板."""
        file = self._resolve_price_file()
        if file is None:
            return None
        key = str(file)
        if key not in self._price_cache:
            self._price_cache[key] = pd.read_parquet(file)
        return self._price_cache[key]

    def _resolve_price_file(self, ticker: str | None = None) -> Path | None:
        """选择合适的价格文件."""
        # 优先检查全市场文件
        if _FULL_MARKET_PRICE_FILE.exists():
            return _FULL_MARKET_PRICE_FILE
        # 其次 CSI300
        if _CSI300_PRICE_FILE.exists():
            return _CSI300_PRICE_FILE
        # 最后检查 raw 目录
        if _RAW_PANEL.exists():
            return _RAW_PANEL
        logger.warning("[BacktestData] 找不到价格面板文件")
        return None
