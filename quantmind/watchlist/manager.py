"""quantmind/watchlist/manager.py — 用户自选股持久化管理.

存储格式（data/watchlist/watchlist.json）：
{
  "stocks": [
    {
      "ticker":    "600519.SH",
      "name":      "贵州茅台",
      "add_date":  "2026-05-31",
      "add_price": 1450.0,
      "note":      "白酒龙头，关注估值"
    },
    ...
  ],
  "last_updated": "2026-05-31"
}
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]


class WatchlistManager:
    """管理用户自选股列表，持久化到 JSON 文件."""

    WATCHLIST_PATH = _ROOT / "data" / "watchlist" / "watchlist.json"
    PRICES_PATH    = _ROOT / "data" / "raw" / "daily_prices_panel.parquet"
    ALPHA_PATH     = _ROOT / "data" / "panel" / "alpha_panel_v4.parquet"

    def __init__(self, path: Path | None = None) -> None:
        self._path = Path(path) if path else self.WATCHLIST_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def add_stock(self, ticker: str, note: str = "") -> dict[str, Any]:
        """添加股票到自选股.

        Returns
        -------
        dict with keys:
            success : bool
            message : str
            stock   : dict | None
        """
        ticker = ticker.strip().upper()
        if not ticker:
            return {"success": False, "message": "股票代码不能为空", "stock": None}

        data = self._load()
        existing = [s for s in data["stocks"] if s["ticker"] == ticker]
        if existing:
            return {
                "success": False,
                "message": f"{ticker} 已在自选股列表中",
                "stock": existing[0],
            }

        name       = self._fetch_stock_name(ticker)
        add_price  = self._fetch_current_price(ticker)
        stock_entry: dict[str, Any] = {
            "ticker":    ticker,
            "name":      name,
            "add_date":  date.today().isoformat(),
            "add_price": add_price,
            "note":      note,
        }
        data["stocks"].append(stock_entry)
        data["last_updated"] = date.today().isoformat()
        self._save(data)
        log.info("[Watchlist] 添加 %s（%s）入场价=%.2f", ticker, name, add_price or 0)
        return {"success": True, "message": f"已添加 {ticker}（{name}）", "stock": stock_entry}

    def remove_stock(self, ticker: str) -> bool:
        """从自选股中删除指定股票，返回是否删除成功."""
        ticker = ticker.strip().upper()
        data   = self._load()
        before = len(data["stocks"])
        data["stocks"] = [s for s in data["stocks"] if s["ticker"] != ticker]
        if len(data["stocks"]) < before:
            data["last_updated"] = date.today().isoformat()
            self._save(data)
            log.info("[Watchlist] 删除 %s", ticker)
            return True
        return False

    def get_all(self) -> list[dict]:
        """返回所有自选股条目列表."""
        return self._load()["stocks"]

    def update_note(self, ticker: str, note: str) -> bool:
        """更新指定股票的备注，返回是否更新成功."""
        ticker = ticker.strip().upper()
        data   = self._load()
        for stock in data["stocks"]:
            if stock["ticker"] == ticker:
                stock["note"] = note
                data["last_updated"] = date.today().isoformat()
                self._save(data)
                return True
        return False

    def get_stock(self, ticker: str) -> dict | None:
        """返回单只股票的存储条目，不存在返回 None."""
        ticker = ticker.strip().upper()
        for s in self._load()["stocks"]:
            if s["ticker"] == ticker:
                return s
        return None

    # ── 私有：数据获取 ────────────────────────────────────────────────────────

    def _fetch_stock_name(self, ticker: str) -> str:
        """查询股票名称：alpha_panel → Tushare → ticker 本身."""
        # 1. 从 alpha_panel 的 MultiIndex 中找名称（快速）
        #    alpha_panel 没有 name 列，跳到 2
        # 2. 从本地 stock_basic 快照找
        try:
            import pandas as pd
            for snap_path in [
                _ROOT / "data" / "raw" / "stock_basic.parquet",
                _ROOT / "data" / "raw" / "stock_basic.csv",
                _ROOT / "data" / "sim30d" / "raw" / "stock_basic.parquet",
            ]:
                if snap_path.exists():
                    ext = snap_path.suffix
                    sb  = pd.read_parquet(snap_path) if ext == ".parquet" else pd.read_csv(snap_path)
                    col = "ts_code" if "ts_code" in sb.columns else None
                    if col:
                        row = sb[sb[col] == ticker]
                        if not row.empty and "name" in sb.columns:
                            return str(row.iloc[0]["name"])
        except Exception as e:
            log.debug("[Watchlist] 本地 stock_basic 读取失败: %s", e)

        # 3. Tushare stock_basic
        try:
            token = os.environ.get("TUSHARE_TOKEN", "")
            if token:
                import tushare as ts
                ts.set_token(token)
                pro = ts.pro_api()
                sb  = pro.stock_basic(ts_code=ticker, fields="ts_code,name")
                if not sb.empty:
                    return str(sb.iloc[0]["name"])
        except Exception as e:
            log.debug("[Watchlist] Tushare stock_basic 查询失败: %s", e)

        # 4. 降级：返回 ticker 本身
        return ticker

    def _fetch_current_price(self, ticker: str) -> float | None:
        """从 daily_prices_panel.parquet 读取最新收盘价."""
        if not self.PRICES_PATH.exists():
            return None
        try:
            import pandas as pd
            import pyarrow.parquet as pq

            schema_names = pq.ParquetFile(str(self.PRICES_PATH)).schema_arrow.names
            # 显示价格用 close（与市场行情一致），adj_close 留给收益率计算
            price_col    = "close" if "close" in schema_names else "adj_close"
            date_col     = "trade_date" if "trade_date" in schema_names else "date"

            df = pd.read_parquet(
                self.PRICES_PATH,
                filters=[("ts_code", "==", ticker)],
                columns=[date_col, price_col],
            )
            if df.empty:
                return None
            df[date_col] = pd.to_datetime(df[date_col])
            latest = df.sort_values(date_col).iloc[-1][price_col]
            return float(latest)
        except Exception as e:
            log.debug("[Watchlist] 价格读取失败(%s): %s", ticker, e)
            return None

    # ── 私有：JSON IO ────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if not self._path.exists():
            return {"stocks": [], "last_updated": date.today().isoformat()}
        try:
            with self._path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            if "stocks" not in data:
                data["stocks"] = []
            return data
        except Exception as e:
            log.warning("[Watchlist] JSON 读取失败，重置: %s", e)
            return {"stocks": [], "last_updated": date.today().isoformat()}

    def _save(self, data: dict) -> None:
        with self._path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
