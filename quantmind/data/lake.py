"""quantmind.data.lake — 连续日频数据湖共享读写契约.

数据湖把外部日频序列物化为 ``data/lake/<series>.parquet`` 连续日频表，
供周频面板重采样按 ``trade_date <= as_of`` 严格 PIT 取窗。

序列（series）
-------------
- ``daily_basic``   估值/市值/换手/流通股（纯回补，无快照）
- ``index_daily``   指数日线
- ``north_bound``   市场北向（仅 trade_date 键）
- ``margin``        融资融券明细
- ``hk_hold``       沪深股通个股持股
- ``stock_basic``   股票基本信息（仅 ts_code 键，无 trade_date）

契约
----
- ``read_lake_window(series, as_of, lookback_trading_days)``：严格 ``trade_date <= as_of``
  的 PIT 窗口，按交易日历（以 ``alpha_prices_panel`` 为准）回看 ``lookback_trading_days``
  个交易日；**保留原生列**。lookback 可 >= 252（支持 beta_252d）。
- ``write_lake(series, df)``：幂等。按 key 去重 merge 到 parquet，``keep="last"``
  使新写入（回补）覆盖旧行（快照）。
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd

# ============================================================================
# 路径与常量
# ============================================================================

# 仓库根：本文件位于 <root>/quantmind/data/lake.py
_REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_LAKE_DIR = _REPO_ROOT / "data" / "lake"
CHECKPOINT_DIR = DATA_LAKE_DIR / "_checkpoints"
PRICE_PANEL_PATH = _REPO_ROOT / "data" / "raw" / "alpha_prices_panel.parquet"

# 去重主键：默认 (ts_code, trade_date)；以下序列特殊
LAKE_KEYS: dict[str, list[str]] = {
    "north_bound": ["trade_date"],   # 市场聚合，无个股维度
    "stock_basic": ["ts_code"],      # 静态信息，无 trade_date
}

#: 所有已知序列（用于校验/遍历）
LAKE_SERIES = (
    "daily_basic",
    "index_daily",
    "north_bound",
    "margin",
    "hk_hold",
    "stock_basic",
)


def lake_path(series: str) -> Path:
    """``data/lake/<series>.parquet``."""
    return DATA_LAKE_DIR / f"{series}.parquet"


def lake_key(series: str) -> list[str]:
    """该序列的去重主键。"""
    return LAKE_KEYS.get(series, ["ts_code", "trade_date"])


# ============================================================================
# 交易日历（以 alpha_prices_panel 为准）
# ============================================================================

_CALENDAR_CACHE: pd.DatetimeIndex | None = None


def load_trading_calendar(refresh: bool = False) -> pd.DatetimeIndex:
    """返回升序去重的交易日历（``DatetimeIndex``），来源 ``alpha_prices_panel``.

    缓存于模块级；``refresh=True`` 强制重读。
    """
    global _CALENDAR_CACHE
    if _CALENDAR_CACHE is not None and not refresh:
        return _CALENDAR_CACHE
    if not PRICE_PANEL_PATH.exists():
        raise FileNotFoundError(
            f"交易日历源缺失：{PRICE_PANEL_PATH}（read_lake_window 依赖它）"
        )
    td = pd.read_parquet(PRICE_PANEL_PATH, columns=["trade_date"])["trade_date"]
    cal = pd.DatetimeIndex(pd.to_datetime(td).dropna().unique()).sort_values()
    _CALENDAR_CACHE = cal
    return cal


def _to_ts(d: str | date | datetime | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(d)


# ============================================================================
# 读：PIT 窗口
# ============================================================================


def read_lake_window(
    series: str,
    as_of: str | date | datetime | pd.Timestamp,
    lookback_trading_days: int,
) -> pd.DataFrame:
    """读 ``series`` 在 ``as_of`` 时点、回看 ``lookback_trading_days`` 个交易日的 PIT 窗口.

    Args:
        series: 序列名（见 LAKE_SERIES）。
        as_of: 时点；严格保证返回行 ``trade_date <= as_of``（不含未来）。
        lookback_trading_days: 回看交易日数（按 alpha_prices_panel 日历计），>=1。
            须能支持 >=252（beta_252d）。

    Returns:
        原生列 DataFrame（``trade_date`` 已转 datetime64），按 ``trade_date`` 升序。
        无数据 / 无文件 → 空 DataFrame。

    Note:
        无 ``trade_date`` 的序列（stock_basic）不支持窗口读，请直接 ``pd.read_parquet(lake_path(...))``。
    """
    if lookback_trading_days < 1:
        raise ValueError("lookback_trading_days must be >= 1")

    p = lake_path(series)
    if not p.exists():
        return pd.DataFrame()

    df = pd.read_parquet(p)
    if "trade_date" not in df.columns:
        raise ValueError(
            f"series '{series}' 无 trade_date 列，不支持 read_lake_window"
        )
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    as_of_ts = _to_ts(as_of).normalize()

    # 用交易日历定位回看窗口起点（严格 <= as_of）
    cal = load_trading_calendar()
    cal_le = cal[cal <= as_of_ts]
    if len(cal_le) == 0:
        return df.iloc[0:0].reset_index(drop=True)
    window = cal_le[-lookback_trading_days:]
    start_ts = window[0]

    # 严格 PIT：start <= trade_date <= as_of（上界用 as_of 本身，杜绝未来泄漏）
    mask = (df["trade_date"] >= start_ts) & (df["trade_date"] <= as_of_ts)
    out = df.loc[mask].sort_values("trade_date").reset_index(drop=True)
    return out


# ============================================================================
# 写：幂等 merge
# ============================================================================


def write_lake(series: str, df: pd.DataFrame) -> int:
    """幂等写入：按 key 去重 merge 到 ``data/lake/<series>.parquet``.

    - key：``lake_key(series)``（默认 (ts_code, trade_date)）。
    - ``drop_duplicates(keep="last")``：``df`` 中的行（新数据/回补）覆盖磁盘旧行。
    - 重复调用同一批数据安全（同 key 覆盖，不增行）。

    Returns:
        合并后磁盘表总行数。
    """
    if df is None or len(df) == 0:
        p = lake_path(series)
        return len(pd.read_parquet(p)) if p.exists() else 0

    key = lake_key(series)
    new = df.copy()
    if "trade_date" in new.columns:
        new["trade_date"] = pd.to_datetime(new["trade_date"])

    missing = [k for k in key if k not in new.columns]
    if missing:
        raise ValueError(f"series '{series}' 缺少主键列 {missing}；现有列 {list(new.columns)}")

    p = lake_path(series)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        old = pd.read_parquet(p)
        if "trade_date" in old.columns:
            old["trade_date"] = pd.to_datetime(old["trade_date"])
        merged = pd.concat([old, new], ignore_index=True)
    else:
        merged = new

    merged = (
        merged.drop_duplicates(subset=key, keep="last")
        .sort_values(key)
        .reset_index(drop=True)
    )
    # 原子写：先写同目录临时文件，再 os.replace 换名（POSIX/NTFS rename 原子）。
    # 单写者语义——本函数非并发安全，不可多进程并行写同一 series。
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    try:
        merged.to_parquet(tmp, index=False)
        os.replace(tmp, p)  # 崩溃只会留下半写的 tmp，原文件 p 保持完好
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return len(merged)


# ============================================================================
# 覆盖自检
# ============================================================================


def coverage(series: str, max_gap_calendar_days: int = 11) -> dict:
    """返回该序列的覆盖统计（行数、交易日数、起止、缺口）.

    缺口按**交易日历**定义：相邻已覆盖交易日在日历上跨过 >1 个交易日即为缺口。
    """
    p = lake_path(series)
    if not p.exists():
        return {"series": series, "exists": False}
    df = pd.read_parquet(p)
    out: dict = {"series": series, "exists": True, "rows": int(len(df))}
    if "trade_date" not in df.columns:
        if "ts_code" in df.columns:
            out["unique_ts_code"] = int(df["ts_code"].nunique())
        return out

    td = pd.to_datetime(df["trade_date"])
    days = pd.DatetimeIndex(td.dropna().unique()).sort_values()
    out["unique_days"] = int(len(days))
    if "ts_code" in df.columns:
        out["unique_ts_code"] = int(df["ts_code"].nunique())
    if len(days) == 0:
        return out
    out["start"] = str(days.min().date())
    out["end"] = str(days.max().date())

    # 用日历找缺口：列出区间内日历交易日中本表缺失的日子
    cal = load_trading_calendar()
    cal_span = cal[(cal >= days.min()) & (cal <= days.max())]
    covered = set(days)
    missing = [d for d in cal_span if d not in covered]
    out["calendar_days_in_span"] = int(len(cal_span))
    out["missing_trading_days"] = int(len(missing))
    out["coverage_pct_vs_calendar"] = (
        round(100.0 * len(covered & set(cal_span)) / len(cal_span), 3)
        if len(cal_span)
        else None
    )
    # 最大连续缺口（日历交易日数）
    if missing:
        out["missing_first"] = str(min(missing).date())
        out["missing_last"] = str(max(missing).date())
    # 日历相邻覆盖日的最大间隔（自然日），辅助参考
    if len(days) > 1:
        gaps = days.to_series().diff().dt.days.dropna()
        out["max_gap_calendar_days"] = int(gaps.max())
    return out


# ============================================================================
# Checkpoint（断点续传）— 供 backfill 复用
# ============================================================================


def checkpoint_path(series: str) -> Path:
    return CHECKPOINT_DIR / f"{series}.json"


def load_checkpoint(series: str) -> set[str]:
    """返回已完成 trade_date 字符串集合（YYYYMMDD）。"""
    p = checkpoint_path(series)
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return set(data.get("done", []))
    except Exception:
        return set()


def save_checkpoint(series: str, done: set[str], extra: dict | None = None) -> None:
    """落盘已完成集合（先于/伴随数据落盘调用）。"""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "series": series,
        "done": sorted(done),
        "count": len(done),
        "updated": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        payload.update(extra)
    checkpoint_path(series).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


__all__ = [
    "DATA_LAKE_DIR",
    "CHECKPOINT_DIR",
    "LAKE_SERIES",
    "LAKE_KEYS",
    "lake_path",
    "lake_key",
    "load_trading_calendar",
    "read_lake_window",
    "write_lake",
    "coverage",
    "checkpoint_path",
    "load_checkpoint",
    "save_checkpoint",
]
