"""quantmind.data.market_data — 选股系统专用轻量数据提供者.

为漏斗选股系统提供：全A股列表、日线价格、轻量基本面。
不依赖 ChromaDB 或 Tushare 高权限接口。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

_ROOT = Path(__file__).resolve().parent.parent.parent
_MARKET_DIR = _ROOT / "data" / "market"
_PRICES_DIR = _ROOT / "data" / "prices"
_SNAPSHOTS_DIR = _ROOT / "data" / "snapshots"
_CSI300_PRICE_FILE = _PRICES_DIR / "csi300_daily_adj_close.parquet"
_FULL_MARKET_PRICE_FILE = _PRICES_DIR / "full_market_daily_adj_close.parquet"


class MarketDataProvider:
    """选股系统轻量数据提供者，用于漏斗过滤.

    数据层优先级：
      1. 本地缓存 parquet
      2. AkShare 实时拉取（需网络）
      3. 降级：从已有快照提取
    """

    def __init__(self, as_of: str | None = None, use_cache: bool = True) -> None:
        self.as_of = as_of or ""
        self.use_cache = use_cache
        _MARKET_DIR.mkdir(parents=True, exist_ok=True)

    # ── 全A股股票池 ────────────────────────────────────────────────────────────

    def get_full_universe(self) -> pd.DataFrame:
        """获取全A股列表.

        Returns:
            DataFrame with columns: ticker, name, industry, total_mv,
                circ_mv, pe_ttm, pb, list_date
        """
        cache_path = _MARKET_DIR / f"universe_full_{self.as_of or 'latest'}.parquet"
        if self.use_cache and cache_path.exists():
            logger.info(f"[MarketData] 从缓存读取全A股列表: {cache_path}")
            return pd.read_parquet(cache_path)

        # 尝试从 AkShare 获取
        df = self._fetch_universe_akshare()
        if df is not None and len(df) > 0:
            df.to_parquet(cache_path, index=False)
            logger.info(f"[MarketData] 全A股列表缓存: {len(df)}只 → {cache_path}")
            return df

        # 降级：从 CSI300 价格面板提取
        logger.warning("[MarketData] AkShare 不可用，从价格面板构建 universe")
        return self._fallback_universe_from_prices()

    def _fetch_universe_akshare(self) -> pd.DataFrame | None:
        """从 AkShare 获取全A股实时行情."""
        try:
            import akshare as ak

            logger.info("[MarketData] 从 AkShare 获取全A股实时行情...")
            spot = ak.stock_zh_a_spot_em()
            # 标准化列名
            col_map = {
                "代码": "code",
                "名称": "name",
                "总市值": "total_mv",
                "流通市值": "circ_mv",
                "市盈率-动态": "pe_ttm",
                "市净率": "pb",
                "行业": "industry",
            }
            spot = spot.rename(columns={k: v for k, v in col_map.items() if k in spot.columns})

            # 格式化 ticker
            if "code" in spot.columns:
                spot["ticker"] = spot["code"].apply(_format_ticker)
            elif "ticker" not in spot.columns:
                return None

            # 转换数值列
            for col in ["total_mv", "circ_mv", "pe_ttm", "pb"]:
                if col in spot.columns:
                    spot[col] = pd.to_numeric(spot[col], errors="coerce")

            # 添加上市日期（若无则用占位）
            if "list_date" not in spot.columns:
                spot["list_date"] = "2010-01-01"

            keep_cols = [c for c in ["ticker", "name", "industry", "total_mv",
                                      "circ_mv", "pe_ttm", "pb", "list_date"]
                         if c in spot.columns]
            return spot[keep_cols].dropna(subset=["ticker"])
        except Exception as e:
            logger.warning(f"[MarketData] AkShare 全A股行情失败: {e}")
            return None

    def _fallback_universe_from_prices(self) -> pd.DataFrame:
        """从价格文件提取 ticker 列表（降级方案）."""
        price_file = _FULL_MARKET_PRICE_FILE if _FULL_MARKET_PRICE_FILE.exists() else _CSI300_PRICE_FILE
        try:
            df = pd.read_parquet(price_file, columns=[])
            tickers = list(df.columns)
        except Exception:
            tickers = []

        return pd.DataFrame({
            "ticker": tickers,
            "name": tickers,
            "industry": "未知",
            "total_mv": 500000.0,
            "circ_mv": 300000.0,
            "pe_ttm": None,
            "pb": None,
            "list_date": "2015-01-01",
        })

    # ── 日线价格 ───────────────────────────────────────────────────────────────

    def get_daily_prices(
        self,
        tickers: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """获取日线复权收盘价面板.

        优先读本地 full_market_daily_adj_close.parquet，
        若 ticker 不在文件中则从 AkShare 补充下载。

        Returns:
            DataFrame, index=trade_date (DatetimeIndex), columns=tickers
        """
        price_file = _FULL_MARKET_PRICE_FILE if _FULL_MARKET_PRICE_FILE.exists() else _CSI300_PRICE_FILE

        if not price_file.exists():
            logger.warning(f"[MarketData] 价格文件不存在: {price_file}")
            return pd.DataFrame()

        logger.info(f"[MarketData] 读取价格面板: {price_file}")
        df = pd.read_parquet(price_file)

        # 筛选时间范围
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]

        # 若指定 tickers，过滤列
        if tickers:
            missing = [t for t in tickers if t not in df.columns]
            available = [t for t in tickers if t in df.columns]
            df = df[available] if available else df

            # 补充下载缺失 ticker
            if missing:
                logger.info(f"[MarketData] {len(missing)} 只 ticker 不在面板中，尝试补充下载")
                supplemented = self._fetch_missing_prices(
                    missing, start_date or "2019-01-01", end_date or self.as_of
                )
                if not supplemented.empty:
                    df = pd.concat([df, supplemented], axis=1)

        return df

    def _fetch_missing_prices(
        self, tickers: list[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """从 AkShare 补充下载缺失 ticker 的价格（分批，带 sleep）."""
        frames = {}
        for ticker in tickers[:20]:  # 最多补充20只，避免太慢
            try:
                import akshare as ak
                code = ticker.split(".")[0]
                hist = ak.stock_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=start_date.replace("-", ""),
                    end_date=(end_date or "").replace("-", "") or "20991231",
                    adjust="hfq",
                )
                if not hist.empty and "收盘" in hist.columns:
                    hist.index = pd.to_datetime(hist["日期"])
                    frames[ticker] = hist["收盘"].rename(ticker)
                time.sleep(0.5)
            except Exception as e:
                logger.debug(f"[MarketData] 补充下载 {ticker} 失败: {e}")

        return pd.DataFrame(frames) if frames else pd.DataFrame()

    # ── 轻量基本面 ─────────────────────────────────────────────────────────────

    def get_fundamentals_light(self, as_of: str | None = None) -> pd.DataFrame:
        """获取轻量基本面数据（PE/PB/ROE/营收增速）.

        数据来源（优先级顺序）：
          1. data/market/fundamentals_light_{as_of}.parquet
          2. 最新快照目录的 daily_basic.parquet
          3. 特征文件提取
        """
        target_date = as_of or self.as_of
        cache_path = _MARKET_DIR / f"fundamentals_light_{target_date}.parquet"
        if self.use_cache and cache_path.exists():
            return pd.read_parquet(cache_path)

        # 从快照目录提取
        df = self._extract_from_snapshots(target_date)
        if df is not None:
            df.to_parquet(cache_path, index=False)
            return df

        # 从特征文件提取
        df = self._extract_from_features(target_date)
        if df is not None:
            df.to_parquet(cache_path, index=False)
            return df

        logger.warning("[MarketData] 轻量基本面数据不可用")
        return pd.DataFrame(columns=["ticker", "pe_ttm", "pb", "roe_ttm", "revenue_yoy"])

    def _extract_from_snapshots(self, as_of: str) -> pd.DataFrame | None:
        """从快照目录提取基本面数据."""
        # 找最近的快照目录
        snapshot_dates = sorted(_SNAPSHOTS_DIR.glob("????-??-??"))
        if not snapshot_dates:
            return None

        target = pd.Timestamp(as_of) if as_of else pd.Timestamp.now()
        past_dates = [d for d in snapshot_dates if pd.Timestamp(d.name) <= target]
        if not past_dates:
            past_dates = snapshot_dates
        snap_dir = past_dates[-1]

        daily_basic = snap_dir / "daily_basic.parquet"
        if not daily_basic.exists():
            return None

        try:
            df = pd.read_parquet(daily_basic)
            col_map = {"ts_code": "ticker", "pe_ttm": "pe_ttm", "pb": "pb"}
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            if "ticker" not in df.columns:
                return None

            # 标准化 ticker
            if df["ticker"].iloc[0].endswith((".SH", ".SZ", ".BJ")):
                pass  # already formatted
            else:
                df["ticker"] = df["ticker"].apply(_format_tushare_ticker)

            keep = [c for c in ["ticker", "pe_ttm", "pb", "roe_ttm", "revenue_yoy",
                                  "circ_mv", "total_mv"] if c in df.columns]
            return df[keep]
        except Exception as e:
            logger.debug(f"[MarketData] 快照基本面提取失败: {e}")
            return None

    def _extract_from_features(self, as_of: str) -> pd.DataFrame | None:
        """从特征文件提取基本面字段."""
        feat_dir = _ROOT / "data" / "features"
        feat_files = sorted(feat_dir.glob("*.parquet"))
        if not feat_files:
            return None

        try:
            df = pd.read_parquet(feat_files[-1])
            if df.index.nlevels > 1:
                dates = df.index.get_level_values(0).unique()
                closest = min(dates, key=lambda d: abs((pd.Timestamp(d) - pd.Timestamp(as_of)).days))
                df = df.xs(closest, level=0)

            df = df.reset_index()
            ticker_col = "ticker" if "ticker" in df.columns else df.columns[0]
            df = df.rename(columns={ticker_col: "ticker"})

            cols = [c for c in ["ticker", "pe_ttm", "pb", "roe_ttm", "revenue_yoy"] if c in df.columns]
            return df[cols] if len(cols) > 1 else None
        except Exception as e:
            logger.debug(f"[MarketData] 特征文件基本面提取失败: {e}")
            return None


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def _format_ticker(code: str) -> str:
    """将6位代码格式化为 ticker（600519 → 600519.SH）."""
    code = str(code).strip().zfill(6)
    if code.startswith(("6", "5", "900")):
        return f"{code}.SH"
    elif code.startswith(("00", "30", "002")):
        return f"{code}.SZ"
    elif code.startswith(("43", "83", "87", "88")):
        return f"{code}.BJ"
    return f"{code}.SH"


def _format_tushare_ticker(ts_code: str) -> str:
    """将 Tushare 格式 ticker 转为标准格式（600519.SH → 600519.SH）."""
    if "." in str(ts_code):
        parts = ts_code.split(".")
        suffix = parts[1].upper()
        if suffix in ("SH", "SZ", "BJ"):
            return ts_code
    return ts_code
