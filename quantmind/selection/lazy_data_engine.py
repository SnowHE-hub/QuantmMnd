"""quantmind.selection.lazy_data_engine — 漏斗懒加载数据引擎.

核心思想：
  每层 filter 只为"上层存活"的股票拉取"本层需要"的数据。
  同一次漏斗运行内，已拉取的数据通过内存缓存自动复用。
  跨漏斗运行，通过 SharedDataCache 磁盘缓存按 TTL 复用。
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

from quantmind.data.shared_cache import SharedDataCache

if TYPE_CHECKING:
    pass

_ROOT = Path(__file__).resolve().parent.parent.parent
_PRICE_FILE = _ROOT / "data" / "prices" / "full_market_daily_adj_close.parquet"
_CSI300_PRICE_FILE = _ROOT / "data" / "prices" / "csi300_daily_adj_close.parquet"
_LONG_PRICE_PANEL = _ROOT / "data" / "raw" / "daily_prices_panel.parquet"
_SNAPSHOTS_DIR = _ROOT / "data" / "snapshots"


def _ticker_to_short(ticker: str) -> str:
    """600519.SH → 600519"""
    return ticker.split(".")[0] if "." in ticker else ticker


def _tickers_hash(tickers: list[str]) -> str:
    """生成 ticker 列表的短哈希（用于缓存 key）."""
    s = ",".join(sorted(tickers))
    return hashlib.md5(s.encode()).hexdigest()[:8]


def _merge_tushare_daily_adj(daily: pd.DataFrame, adj: pd.DataFrame | None) -> pd.DataFrame:
    """Tushare daily + adj_factor → 含 trade_date、adj_close 的表（与 build_daily_price_panel 一致）."""
    if daily is None or daily.empty:
        return pd.DataFrame()
    df = daily.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    if adj is None or adj.empty:
        df["adj_factor"] = pd.NA
    else:
        a = adj.copy()
        a["trade_date"] = pd.to_datetime(a["trade_date"], format="%Y%m%d", errors="coerce")
        df = df.merge(a[["trade_date", "adj_factor"]], on="trade_date", how="left")
    df["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce")
    df = df.sort_values("trade_date").reset_index(drop=True)
    latest = df["adj_factor"].iloc[-1]
    if pd.notna(latest) and latest != 0:
        df["adj_close"] = pd.to_numeric(df["close"], errors="coerce") * df["adj_factor"] / float(latest)
    else:
        df["adj_close"] = pd.to_numeric(df["close"], errors="coerce")
    return df


_SPOT_COLUMNS = [
    "ticker",
    "name",
    "industry",
    "total_mv",
    "circ_mv",
    "pe_ttm",
    "pb",
    "list_date",
    "is_st",
    "close",
]


class LazyDataEngine:
    """懒加载数据引擎.

    维护两级缓存：
    - 内存缓存（_mem）：本次进程运行内有效，即时命中
    - 磁盘缓存（SharedDataCache）：跨运行复用，按 TTL 过期
    """

    def __init__(
        self,
        as_of: str,
        cache_dir: str | Path | None = None,
        ttl_hours: int = 24,
        shared_cache: SharedDataCache | None = None,
        universe: str = "csi300",
        custom_universe_file: str | Path | None = None,
    ) -> None:
        self.as_of = as_of
        self.as_of_date = as_of
        self.universe = universe
        self.custom_universe_file = Path(custom_universe_file) if custom_universe_file else None
        self.ttl_hours = ttl_hours
        self._shared = shared_cache or SharedDataCache.get_instance(
            cache_dir or (_ROOT / "data" / "cache")
        )
        self._mem: dict[str, pd.DataFrame | dict] = {}
        self._mem_hits = 0
        self._disk_hits = 0
        self._misses = 0

    # ── 内部缓存读写 ──────────────────────────────────────────────────────────

    def _get(self, key: str) -> pd.DataFrame | dict | None:
        if key in self._mem:
            self._mem_hits += 1
            return self._mem[key]
        val = self._shared.read(key)
        if val is not None:
            self._disk_hits += 1
            self._mem[key] = val
            return val
        self._misses += 1
        return None

    def _put(self, key: str, data: pd.DataFrame | dict, ttl_hours: int | None = None) -> None:
        self._mem[key] = data
        self._shared.write(key, data, ttl_hours=ttl_hours or self.ttl_hours)

    # ── Layer1：全市场实时行情 ────────────────────────────────────────────────

    def get_spot_data(self, universe: str | None = None) -> pd.DataFrame:
        """Layer1 用：现货截面.

        - ``csi300``：返回空表，由 ``FunnelSelector`` 使用本地 CSI300 面板。
        - ``full_a`` / ``csi1000``：AkShare ``stock_zh_a_spot_em``（单次超时 20s，调用后 sleep 0.3s），
          剔除北交所；失败则降级 ``_fetch_csi300_spot_from_snapshot`` → snapshot universe。
        - ``custom``：从 ``custom_universe_file`` 读取代码列表。
        """
        uni = universe or self.universe
        if uni == "csi300":
            return pd.DataFrame(columns=_SPOT_COLUMNS)

        if uni == "custom" and self.custom_universe_file:
            key = f"spot_custom_{self.as_of}_{hashlib.md5(str(self.custom_universe_file).resolve()).hexdigest()[:10]}"
        elif uni in ("full_a", "csi1000"):
            key = f"spot_full_a_{self.as_of}"
        else:
            key = f"spot_{uni}_{self.as_of}"

        cached = self._get(key)
        if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
            logger.debug(f"[LazyDataEngine] 命中缓存: {key}")
            return cached

        logger.info(f"[LazyDataEngine] Layer1: 拉取现货截面 universe={uni} ...")
        if uni == "custom":
            df = self._load_custom_universe()
        elif uni in ("full_a", "csi1000"):
            df = self._fetch_full_a_spot()
            if df.empty:
                logger.warning("[LazyDataEngine] full_a 为空，降级 CSI300 快照现货")
                df = self._fetch_csi300_spot_from_snapshot()
            if df.empty:
                df = self._fallback_spot_from_snapshot_universe()
        else:
            df = pd.DataFrame(columns=_SPOT_COLUMNS)

        if not df.empty:
            self._put(key, df, ttl_hours=self.ttl_hours)
        logger.info(f"[LazyDataEngine] Layer1 现货截面: {len(df)} 只")
        return df

    def _load_custom_universe(self) -> pd.DataFrame:
        """从文本/csv 读取 ticker 列表（custom universe）."""
        p = self.custom_universe_file
        if p is None or not p.is_file():
            logger.warning("[LazyDataEngine] custom universe 文件缺失")
            return pd.DataFrame(columns=_SPOT_COLUMNS)
        tickers: list[str] = []
        try:
            text = p.read_text(encoding="utf-8")
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                tickers.append(line.split(",")[0].strip())
        except OSError as e:
            logger.warning(f"[LazyDataEngine] 读取 custom universe 失败: {e}")
            return pd.DataFrame(columns=_SPOT_COLUMNS)
        seen: set[str] = set()
        out_t = []
        for t in tickers:
            if t not in seen:
                seen.add(t)
                out_t.append(t)
        df = pd.DataFrame({"ticker": out_t})
        df["name"] = df["ticker"]
        df["list_date"] = pd.Timestamp("2010-01-01")
        df["is_st"] = False
        df["close"] = np.nan
        df["industry"] = None
        df["total_mv"] = np.nan
        df["circ_mv"] = np.nan
        df["pe_ttm"] = np.nan
        df["pb"] = np.nan
        snap_dir = _SNAPSHOTS_DIR / self.as_of
        db = snap_dir / "daily_basic.parquet"
        if db.is_file():
            try:
                db_df = pd.read_parquet(db)
                if "ts_code" in db_df.columns:
                    db_df = db_df.rename(columns={"ts_code": "ticker"})
                cols = [c for c in ["ticker", "total_mv", "circ_mv", "pe_ttm", "pb"] if c in db_df.columns]
                if "trade_date" in db_df.columns and cols:
                    latest = db_df.sort_values("trade_date", ascending=False).drop_duplicates("ticker")
                    df = df.merge(latest[cols], on="ticker", how="left")
            except Exception as e:
                logger.debug(f"[LazyDataEngine] custom+daily_basic 合并失败: {e}")
        for c in _SPOT_COLUMNS:
            if c not in df.columns:
                df[c] = np.nan
        df["total_mv"] = pd.to_numeric(df["total_mv"], errors="coerce").fillna(5e5)
        df["circ_mv"] = pd.to_numeric(df["circ_mv"], errors="coerce").fillna(df["total_mv"])
        return df[_SPOT_COLUMNS].reset_index(drop=True)

    def _fetch_full_a_spot(self) -> pd.DataFrame:
        """全 A 现货：AkShare → Tushare stock_basic → CSI300 快照。"""
        old_to = socket.getdefaulttimeout()
        try:
            socket.setdefaulttimeout(20.0)

            # ── A：AkShare 东方财富现货 ──────────────────────────────────────
            try:
                import akshare as ak

                raw = ak.stock_zh_a_spot_em()
                if raw is not None and len(raw) > 1000:
                    rename = {
                        "代码": "ticker_short",
                        "名称": "name",
                        "最新价": "close",
                        "总市值": "total_mv",
                        "流通市值": "circ_mv",
                        "市盈率-动态": "pe_ttm",
                        "市净率": "pb",
                        "所属行业": "industry",
                        "换手率": "turnover_rate",
                        "成交量": "vol",
                        "成交额": "amount",
                    }
                    df = raw.rename(columns={k: v for k, v in rename.items() if k in raw.columns})
                    code_cn = "代码" if "代码" in raw.columns else None
                    if code_cn is None:
                        raise ValueError("AkShare 缺少代码列")
                    df["_code6"] = raw[code_cn].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
                    df = df[~df["_code6"].str.startswith("8")].copy()

                    def _full_ticker(code: str) -> str:
                        c = str(code).strip().zfill(6)
                        return f"{c}.SH" if c.startswith(("6", "9")) else f"{c}.SZ"

                    df["ticker"] = df["ticker_short"].map(_full_ticker)
                    df = df[~df["ticker"].str.endswith(".BJ")]
                    list_cn = next((c for c in ("上市日期", "上市时间", "首发日期") if c in raw.columns), None)
                    if list_cn:
                        df["list_date"] = pd.to_datetime(raw.loc[df.index, list_cn], errors="coerce")
                    else:
                        df["list_date"] = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
                    df["is_st"] = df["name"].astype(str).str.contains(r"ST|\*ST", regex=True, na=False)
                    for c in _SPOT_COLUMNS:
                        if c not in df.columns:
                            df[c] = np.nan
                    time.sleep(0.3)
                    logger.info(f"[LazyData] AkShare full_a: {len(df)} 只")
                    return df[_SPOT_COLUMNS].reset_index(drop=True)
                logger.warning("[LazyData] AkShare 返回行数过少，尝试 Tushare")
            except Exception as e:
                logger.warning(f"[LazyData] AkShare 失败 ({e})")

            # ── B：Tushare stock_basic（无行情，市值填 0）────────────────────
            try:
                import tushare as ts

                token = os.environ.get("TUSHARE_TOKEN", "").strip()
                if not token:
                    logger.warning("[LazyData] TUSHARE_TOKEN 未设置，跳过 Tushare stock_basic fallback")
                else:
                    ts.set_token(token)
                    pro = ts.pro_api()
                    sb = pro.stock_basic(
                        exchange="",
                        list_status="L",
                        fields="ts_code,name,industry,market,list_date",
                    )
                    if sb is not None and len(sb) > 1000:
                        sb = sb[~sb["ts_code"].astype(str).str.startswith("8")].copy()
                        out = sb.rename(columns={"ts_code": "ticker"})
                        out["list_date"] = pd.to_datetime(out["list_date"], errors="coerce")
                        out["total_mv"] = 0.0
                        out["circ_mv"] = 0.0
                        out["pe_ttm"] = np.nan
                        out["pb"] = np.nan
                        out["close"] = np.nan
                        out["is_st"] = out["name"].astype(str).str.contains(r"ST|\*ST", regex=True, na=False)
                        if "industry" not in out.columns:
                            out["industry"] = None
                        for c in _SPOT_COLUMNS:
                            if c not in out.columns:
                                out[c] = np.nan
                        logger.info(f"[LazyData] Tushare stock_basic fallback: {len(out)} 只（无行情）")
                        return out[_SPOT_COLUMNS].reset_index(drop=True)
            except Exception as e:
                logger.warning(f"[LazyData] Tushare fallback 失败 ({e})")

            logger.warning("[LazyData] 全部接口失败，降级为本地 CSI300 快照")
            return self._fetch_csi300_spot_from_snapshot()
        finally:
            socket.setdefaulttimeout(old_to)

    def _fetch_csi300_spot_from_snapshot(self) -> pd.DataFrame:
        """从快照目录构建 CSI300 截面（stock_basic + daily_basic），用于 full_a 降级。"""
        snap_dir = _SNAPSHOTS_DIR / self.as_of
        uni_path = snap_dir / "universe.parquet"
        if not uni_path.is_file():
            candidates = sorted(
                [d for d in _SNAPSHOTS_DIR.iterdir() if d.is_dir() and (d / "universe.parquet").is_file()],
                key=lambda p: p.name,
                reverse=True,
            )
            picked = None
            for d in candidates:
                if d.name <= self.as_of:
                    picked = d
                    break
            if picked is None and candidates:
                picked = candidates[-1]
            if picked is None:
                return pd.DataFrame(columns=_SPOT_COLUMNS)
            snap_dir = picked
            uni_path = snap_dir / "universe.parquet"

        try:
            u = pd.read_parquet(uni_path)
            if "ticker" not in u.columns:
                return pd.DataFrame(columns=_SPOT_COLUMNS)
            out = pd.DataFrame({"ticker": u["ticker"].astype(str)})
            out["name"] = out["ticker"]
            out["industry"] = None
            out["list_date"] = pd.Timestamp("2010-01-01")
            out["is_st"] = False
            out["close"] = np.nan
            sb_csv = snap_dir / "stock_basic.csv"
            sb_pq = snap_dir / "stock_basic.parquet"
            if sb_pq.is_file():
                try:
                    sb_df = pd.read_parquet(sb_pq)
                    code_col = "ts_code" if "ts_code" in sb_df.columns else None
                    if code_col:
                        sb_df = sb_df.rename(columns={code_col: "ticker"})
                        sub = ["ticker"]
                        if "name" in sb_df.columns:
                            sub.append("name")
                        if "industry" in sb_df.columns:
                            sub.append("industry")
                        m = out.merge(sb_df[sub].drop_duplicates("ticker"), on="ticker", how="left")
                        if "name" in m.columns:
                            out["name"] = m["name"].fillna(out["name"])
                        if "industry" in m.columns:
                            out["industry"] = m["industry"]
                except Exception as e:
                    logger.debug(f"[LazyDataEngine] stock_basic 读取失败: {e}")
            elif sb_csv.is_file():
                try:
                    sb_df = pd.read_csv(sb_csv)
                    if "ts_code" in sb_df.columns:
                        sb_df = sb_df.rename(columns={"ts_code": "ticker"})
                        sub = ["ticker"] + [c for c in ("name", "industry") if c in sb_df.columns]
                        m = out.merge(sb_df[sub].drop_duplicates("ticker"), on="ticker", how="left")
                        for c in ("name", "industry"):
                            if c in m.columns:
                                out[c] = m[c].fillna(out[c])
                except Exception as e:
                    logger.debug(f"[LazyDataEngine] stock_basic.csv 读取失败: {e}")

            db = snap_dir / "daily_basic.parquet"
            if db.is_file():
                try:
                    db_df = pd.read_parquet(db)
                    if "ts_code" in db_df.columns:
                        db_df = db_df.rename(columns={"ts_code": "ticker"})
                    cols = [c for c in ["ticker", "total_mv", "circ_mv", "pe_ttm", "pb"] if c in db_df.columns]
                    if "trade_date" in db_df.columns and cols:
                        latest = db_df.sort_values("trade_date", ascending=False).drop_duplicates("ticker")
                        m = out.merge(latest[cols], on="ticker", how="left")
                        for c in ["total_mv", "circ_mv", "pe_ttm", "pb"]:
                            if c in m.columns:
                                out[c] = m[c]
                except Exception as e:
                    logger.debug(f"[LazyDataEngine] daily_basic 合并失败: {e}")

            for c in _SPOT_COLUMNS:
                if c not in out.columns:
                    out[c] = np.nan
            out["total_mv"] = pd.to_numeric(out["total_mv"], errors="coerce").fillna(5e5)
            out["circ_mv"] = pd.to_numeric(out["circ_mv"], errors="coerce").fillna(out["total_mv"])
            return out[_SPOT_COLUMNS].reset_index(drop=True)
        except Exception as e:
            logger.warning(f"[LazyDataEngine] _fetch_csi300_spot_from_snapshot 失败: {e}")
            return pd.DataFrame(columns=_SPOT_COLUMNS)

    def _fallback_spot_from_snapshot_universe(self) -> pd.DataFrame:
        """AkShare 不可用时的降级：使用某日快照 ``universe.parquet``（CSI300）近似全 A。"""
        snap_dir = _SNAPSHOTS_DIR / self.as_of
        uni_path = snap_dir / "universe.parquet"
        if not uni_path.is_file():
            candidates = sorted(
                [
                    d
                    for d in _SNAPSHOTS_DIR.iterdir()
                    if d.is_dir() and (d / "universe.parquet").is_file()
                ],
                key=lambda p: p.name,
                reverse=True,
            )
            picked = None
            for d in candidates:
                if d.name <= self.as_of:
                    picked = d
                    break
            if picked is None and candidates:
                picked = candidates[-1]
            if picked is None:
                logger.warning("[LazyDataEngine] 无 snapshots universe.parquet，返回空表")
                return pd.DataFrame(columns=_SPOT_COLUMNS)
            uni_path = picked / "universe.parquet"
            snap_dir = picked

        logger.warning("AkShare 不可用，降级为 CSI300 数据")
        logger.info("[LazyDataEngine] 降级快照目录: {}", snap_dir)
        u = pd.read_parquet(uni_path)
        if "ticker" not in u.columns:
            return pd.DataFrame(columns=_SPOT_COLUMNS)

        out = pd.DataFrame({"ticker": u["ticker"].astype(str)})
        out["name"] = out["ticker"]
        out["industry"] = None
        out["list_date"] = pd.Timestamp("2010-01-01")
        out["is_st"] = False
        out["close"] = np.nan

        sb = snap_dir / "stock_basic.parquet"
        if sb.is_file():
            try:
                sb_df = pd.read_parquet(sb)
                code_col = "ts_code" if "ts_code" in sb_df.columns else None
                if code_col:
                    sb_df = sb_df.rename(columns={code_col: "ticker"})
                    sub_cols = ["ticker"]
                    if "name" in sb_df.columns:
                        sub_cols.append("name")
                    if "industry" in sb_df.columns:
                        sub_cols.append("industry")
                    m = out.merge(sb_df[sub_cols].drop_duplicates("ticker"), on="ticker", how="left")
                    if "name" in m.columns:
                        out["name"] = m["name"].fillna(out["name"])
                    if "industry" in m.columns:
                        out["industry"] = m["industry"]
            except Exception as e:
                logger.debug(f"[LazyDataEngine] stock_basic 合并失败: {e}")

        db = snap_dir / "daily_basic.parquet"
        if db.is_file():
            try:
                db_df = pd.read_parquet(db)
                if "ts_code" in db_df.columns:
                    db_df = db_df.rename(columns={"ts_code": "ticker"})
                cols = [c for c in ["ticker", "total_mv", "circ_mv", "pe_ttm", "pb"] if c in db_df.columns]
                if "trade_date" in db_df.columns and cols:
                    latest = db_df.sort_values("trade_date", ascending=False).drop_duplicates("ticker")
                    m = out.merge(latest[cols], on="ticker", how="left")
                    for c in ["total_mv", "circ_mv", "pe_ttm", "pb"]:
                        if c in m.columns:
                            out[c] = m[c]
            except Exception as e:
                logger.debug(f"[LazyDataEngine] daily_basic 合并失败: {e}")

        for c in _SPOT_COLUMNS:
            if c not in out.columns:
                out[c] = np.nan
        out["total_mv"] = pd.to_numeric(out["total_mv"], errors="coerce").fillna(5e5)
        out["circ_mv"] = pd.to_numeric(out["circ_mv"], errors="coerce").fillna(out["total_mv"])
        return out[_SPOT_COLUMNS].reset_index(drop=True)

    # ── Layer2/3/5：价格窗口 ──────────────────────────────────────────────────

    def get_price_window(
        self,
        tickers: list[str],
        window_days: int = 63,
    ) -> pd.DataFrame:
        """Layer2/3/5 用：只拉指定 tickers 的近 N 日价格.

        优先级：本地宽表 → 本地 ``daily_prices_panel`` 长表 → **Tushare** daily+adj_factor
        → （可选）AkShare，仅当环境变量 ``QUANTMIND_USE_AKSHARE_PRICES=1``。

        返回：DataFrame，行为日期，列为 ticker，值为**后复权**收盘价（与宽表一致）。
        缓存 key：prices_{as_of}_{window_days}_{tickers_hash}
        """
        if not tickers:
            return pd.DataFrame()

        key = f"prices_{self.as_of}_{window_days}_{_tickers_hash(tickers)}"
        cached = self._get(key)
        if cached is not None and isinstance(cached, pd.DataFrame):
            logger.debug(f"[LazyDataEngine] 命中缓存: {key}（{len(cached.columns)} 只）")
            return cached

        df = self._load_prices_from_local(tickers, window_days)
        missing = [t for t in tickers if t not in df.columns]
        if missing:
            token = os.environ.get("TUSHARE_TOKEN", "").strip()
            if token:
                logger.info(
                    f"[LazyDataEngine] 本地无数据的 {len(missing)} 只，优先用 Tushare 补充（daily+adj_factor）"
                )
                fetched_ts = self._fetch_prices_tushare(missing, window_days)
                if not fetched_ts.empty:
                    df = fetched_ts if df.empty else df.combine_first(fetched_ts)

            missing = [t for t in tickers if t not in df.columns]
            use_ak = os.environ.get("QUANTMIND_USE_AKSHARE_PRICES", "0").lower() in (
                "1",
                "true",
                "yes",
            )
            if missing and use_ak:
                logger.info(
                    f"[LazyDataEngine] 仍缺 {len(missing)} 只，使用 AkShare（QUANTMIND_USE_AKSHARE_PRICES=1）"
                )
                fetched = self._fetch_prices_akshare(missing, window_days)
                if not fetched.empty:
                    df = fetched if df.empty else df.combine_first(fetched)
            elif missing and not token:
                logger.warning(
                    f"[LazyDataEngine] 仍缺 {len(missing)} 只无本地/Tushare 行情；"
                    "请配置 TUSHARE_TOKEN，或设 QUANTMIND_USE_AKSHARE_PRICES=1 启用 AkShare"
                )
            elif missing:
                logger.warning(
                    f"[LazyDataEngine] Tushare 补价后仍缺 {len(missing)} 只；"
                    "可检查积分/限频，或设 QUANTMIND_USE_AKSHARE_PRICES=1"
                )

        self._put(key, df, ttl_hours=self.ttl_hours)
        return df

    def _load_prices_from_local(
        self, tickers: list[str], window_days: int
    ) -> pd.DataFrame:
        """从本地 parquet 宽表 + 长表读取价格，只读需要的列和最近 N 日."""
        frames: list[pd.DataFrame] = []
        for price_file in [_PRICE_FILE, _CSI300_PRICE_FILE]:
            if not price_file.exists():
                continue
            try:
                import pyarrow.parquet as pq

                available = pq.read_schema(price_file).names
                want = [t for t in tickers if t in available]
                if not want:
                    continue
                df = pd.read_parquet(price_file, columns=want)
                cutoff = pd.Timestamp(self.as_of)
                df = df[df.index <= cutoff].tail(window_days + 5)
                frames.append(df[want])
            except Exception as e:
                logger.debug(f"[LazyDataEngine] 读取 {price_file.name} 失败: {e}")

        if frames:
            result = frames[0]
            for f in frames[1:]:
                result = result.combine_first(f)
        else:
            result = pd.DataFrame()

        from_long = self._load_prices_from_long_panel(tickers, window_days)
        if from_long.empty:
            return result
        if result.empty:
            return from_long
        return result.combine_first(from_long)

    def _load_prices_from_long_panel(self, tickers: list[str], window_days: int) -> pd.DataFrame:
        """从 ``data/raw/daily_prices_panel.parquet`` 透视为宽表（adj_close 优先）."""
        if not _LONG_PRICE_PANEL.is_file():
            return pd.DataFrame()
        try:
            lp = pd.read_parquet(_LONG_PRICE_PANEL)
            if "ts_code" not in lp.columns or "trade_date" not in lp.columns:
                return pd.DataFrame()
            want_set = set(tickers)
            lp = lp[lp["ts_code"].astype(str).isin(want_set)].copy()
            if lp.empty:
                return pd.DataFrame()
            val_col = "adj_close" if "adj_close" in lp.columns else "close"
            if val_col not in lp.columns:
                return pd.DataFrame()
            lp["trade_date"] = pd.to_datetime(lp["trade_date"], errors="coerce")
            cutoff = pd.Timestamp(self.as_of)
            lp = lp[lp["trade_date"] <= cutoff]
            wide = lp.pivot_table(
                index="trade_date",
                columns="ts_code",
                values=val_col,
                aggfunc="last",
            )
            wide = wide.sort_index().tail(window_days + 5)
            cols = [c for c in wide.columns if str(c) in want_set]
            return wide[cols] if cols else pd.DataFrame()
        except Exception as e:
            logger.debug(f"[LazyDataEngine] 长表面板读取失败: {e}")
            return pd.DataFrame()

    def _fetch_prices_tushare(self, tickers: list[str], window_days: int) -> pd.DataFrame:
        """用 Tushare daily + adj_factor 逐只拉取后复权收盘价（适合境外网络）。"""
        import tushare as ts

        token = os.environ.get("TUSHARE_TOKEN", "").strip()
        if not token:
            return pd.DataFrame()
        try:
            ts.set_token(token)
            pro = ts.pro_api(timeout=120)
        except Exception as e:
            logger.warning(f"[LazyDataEngine] Tushare 初始化失败: {e}")
            return pd.DataFrame()

        cutoff = pd.Timestamp(self.as_of)
        start = (cutoff - pd.Timedelta(days=max(window_days * 2, 120))).strftime("%Y%m%d")
        end = cutoff.strftime("%Y%m%d")

        results: dict[str, pd.Series] = {}
        sleep_s = float(os.environ.get("QUANTMIND_TUSHARE_PRICE_SLEEP", "0.35"))
        n = len(tickers)
        for i, ticker in enumerate(tickers):
            try:
                daily = pro.daily(ts_code=ticker, start_date=start, end_date=end)
                time.sleep(sleep_s)
                adj = pro.adj_factor(ts_code=ticker, start_date=start, end_date=end)
                time.sleep(sleep_s)
                merged = _merge_tushare_daily_adj(
                    daily if daily is not None else pd.DataFrame(),
                    adj if adj is not None else pd.DataFrame(),
                )
                if merged.empty or "adj_close" not in merged.columns:
                    continue
                s = merged.set_index("trade_date")["adj_close"].astype(float).rename(ticker)
                results[ticker] = s.loc[:cutoff].tail(window_days + 5)
            except Exception as e:
                logger.debug(f"[LazyDataEngine] Tushare 拉价格失败 {ticker}: {e}")
            if (i + 1) % 50 == 0 or i + 1 == n:
                logger.info(f"[LazyDataEngine] Tushare 补价进度 {i + 1}/{n}（已成功 {len(results)} 只）")

        if not results:
            return pd.DataFrame()
        return pd.DataFrame(results)

    def _fetch_prices_akshare(
        self, tickers: list[str], window_days: int
    ) -> pd.DataFrame:
        """用 AkShare 批量拉取价格（需 ``QUANTMIND_USE_AKSHARE_PRICES=1`` 才会被调用）。"""
        import akshare as ak

        cutoff = pd.Timestamp(self.as_of)
        start = (cutoff - pd.Timedelta(days=window_days * 2)).strftime("%Y%m%d")
        end = cutoff.strftime("%Y%m%d")

        results: dict[str, pd.Series] = {}
        batch_size = 50
        old_to = socket.getdefaulttimeout()
        try:
            socket.setdefaulttimeout(20.0)
            for i in range(0, len(tickers), batch_size):
                batch = tickers[i : i + batch_size]
                for ticker in batch:
                    try:
                        short = _ticker_to_short(ticker)
                        raw = ak.stock_zh_a_hist(
                            symbol=short,
                            period="daily",
                            start_date=start,
                            end_date=end,
                            adjust="hfq",
                        )
                        if raw is None or raw.empty:
                            continue
                        date_col = "日期" if "日期" in raw.columns else raw.columns[0]
                        close_col = "收盘" if "收盘" in raw.columns else raw.columns[4]
                        s = raw.set_index(date_col)[close_col].rename(ticker)
                        s.index = pd.to_datetime(s.index)
                        results[ticker] = s.tail(window_days + 5)
                    except Exception as e:
                        logger.debug(f"[LazyDataEngine] AkShare 拉价格失败 {ticker}: {e}")
                if i + batch_size < len(tickers):
                    time.sleep(0.3)
        finally:
            socket.setdefaulttimeout(old_to)

        if not results:
            return pd.DataFrame()
        return pd.DataFrame(results)

    # ── Layer4：轻量基本面 ────────────────────────────────────────────────────

    def get_fundamentals_light(self, tickers: list[str]) -> pd.DataFrame:
        """Layer4 用：只拉指定 tickers 的轻量基本面.

        来源：优先 data/snapshots/{latest}/daily_basic.parquet；缺失则 AkShare 补充。
        返回：ticker, roe_ttm, pe_ttm, pb, revenue_yoy
        缓存 key：fundamentals_{as_of}_{tickers_hash}，TTL：24h
        """
        if not tickers:
            return pd.DataFrame(columns=["ticker", "roe_ttm", "pe_ttm", "pb", "revenue_yoy"])

        key = f"fundamentals_{self.as_of}_{_tickers_hash(tickers)}"
        cached = self._get(key)
        if cached is not None and isinstance(cached, pd.DataFrame):
            logger.debug(f"[LazyDataEngine] 命中缓存: {key}")
            return cached

        df = self._load_fundamentals_from_snapshot(tickers)
        missing = [t for t in tickers if t not in df["ticker"].values]
        if missing:
            logger.info(f"[LazyDataEngine] 快照缺 {len(missing)} 只基本面，用 AkShare 补充")
            fetched = self._fetch_fundamentals_akshare(missing)
            if not fetched.empty:
                df = pd.concat([df, fetched], ignore_index=True)

        self._put(key, df, ttl_hours=self.ttl_hours)
        return df

    def _load_fundamentals_from_snapshot(self, tickers: list[str]) -> pd.DataFrame:
        """从最新快照目录读取 daily_basic.parquet."""
        if not _SNAPSHOTS_DIR.exists():
            return pd.DataFrame(columns=["ticker", "roe_ttm", "pe_ttm", "pb", "revenue_yoy"])

        snap_dates = sorted(
            [d for d in _SNAPSHOTS_DIR.iterdir() if d.is_dir()], reverse=True
        )
        for snap_dir in snap_dates[:3]:
            basic_file = snap_dir / "daily_basic.parquet"
            if not basic_file.exists():
                continue
            try:
                df = pd.read_parquet(basic_file)
                # 标准化列名
                col_map = {
                    "ts_code": "ticker",
                    "roe": "roe_ttm",
                    "pe_ttm": "pe_ttm",
                    "pb": "pb",
                    "revenue_yoy": "revenue_yoy",
                }
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                if "ticker" not in df.columns:
                    continue
                want = [t for t in tickers if t in df["ticker"].values]
                if not want:
                    continue
                keep = ["ticker"] + [c for c in ["roe_ttm", "pe_ttm", "pb", "revenue_yoy"] if c in df.columns]
                return df[df["ticker"].isin(tickers)][keep].drop_duplicates("ticker")
            except Exception as e:
                logger.debug(f"[LazyDataEngine] 快照读取失败 {snap_dir}: {e}")

        return pd.DataFrame(columns=["ticker", "roe_ttm", "pe_ttm", "pb", "revenue_yoy"])

    def _fetch_fundamentals_akshare(self, tickers: list[str]) -> pd.DataFrame:
        """用 AkShare stock_a_indicator_lg 批量拉取轻量基本面."""
        import akshare as ak

        rows = []
        for i, ticker in enumerate(tickers):
            try:
                short = _ticker_to_short(ticker)
                raw = ak.stock_a_indicator_lg(symbol=short)
                if raw is None or raw.empty:
                    continue
                latest = raw.sort_values("trade_date", ascending=False).iloc[0]
                rows.append({
                    "ticker": ticker,
                    "pe_ttm": latest.get("pe_ttm"),
                    "pb": latest.get("pb"),
                    "roe_ttm": latest.get("roe_ttm"),
                    "revenue_yoy": None,
                })
            except Exception as e:
                logger.debug(f"[LazyDataEngine] AkShare 基本面失败 {ticker}: {e}")
            if (i + 1) % 50 == 0:
                time.sleep(0.3)

        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["ticker", "roe_ttm", "pe_ttm", "pb", "revenue_yoy"]
        )

    # ── Layer6/研究：KB 上下文 ────────────────────────────────────────────────

    def get_kb_context(self, tickers: list[str]) -> dict[str, dict]:
        """Layer6/投资研究 用：只拉指定 tickers 的 KB 上下文.

        复用现有 retrieve_stock_context 逻辑，TTL 7天。
        返回：{ticker: context_dict}
        """
        result: dict[str, dict] = {}
        for ticker in tickers:
            key = f"kb_context_{ticker}_{self.as_of}"
            cached = self._get(key)
            if cached is not None and isinstance(cached, dict):
                result[ticker] = cached
                continue

            ctx = self._fetch_kb_context(ticker)
            self._put(key, ctx, ttl_hours=7 * 24)
            result[ticker] = ctx

        return result

    def _fetch_kb_context(self, ticker: str) -> dict:
        """调用现有 retrieve_stock_context."""
        try:
            from quantmind.data.retriever import retrieve_stock_context
            return retrieve_stock_context(ticker, self.as_of)
        except Exception as e:
            logger.debug(f"[LazyDataEngine] KB 上下文获取失败 {ticker}: {e}")
            return {}

    # ── 研究系统：详细财务 ────────────────────────────────────────────────────

    def get_detailed_financials(self, ticker: str) -> pd.DataFrame:
        """投资研究用：单只股票详细财务（多期），TTL 7天."""
        key = f"financials_{ticker}"
        cached = self._get(key)
        if cached is not None and isinstance(cached, pd.DataFrame):
            return cached

        df = self._fetch_detailed_financials(ticker)
        self._put(key, df, ttl_hours=7 * 24)
        return df

    def _fetch_detailed_financials(self, ticker: str) -> pd.DataFrame:
        """从研究数据 Provider 拉取详细财务."""
        try:
            from quantmind.data.research_data import ResearchDataProvider
            rdp = ResearchDataProvider()
            return rdp.get_detailed_financials(ticker)
        except Exception as e:
            logger.debug(f"[LazyDataEngine] 详细财务获取失败 {ticker}: {e}")
            return pd.DataFrame()

    # ── 缓存统计 ─────────────────────────────────────────────────────────────

    def cache_stats(self) -> dict:
        """返回本次运行的缓存命中率统计."""
        total = self._mem_hits + self._disk_hits + self._misses
        return {
            "mem_hits": self._mem_hits,
            "disk_hits": self._disk_hits,
            "misses": self._misses,
            "hit_rate": (self._mem_hits + self._disk_hits) / total if total > 0 else 0.0,
            "mem_hit_rate": self._mem_hits / total if total > 0 else 0.0,
        }
