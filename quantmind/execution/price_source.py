"""quantmind/execution/price_source.py — E3 执行层的 parquet 数据源.

背景（见 docs/STORAGE_STRATEGY.md）：
E3 原本从 PostgreSQL 的 ``daily_prices_panel`` / ``realized_pnl`` / ``alpha_universe``
读价格与推荐，但这些 PG 表是中断迁移的残留、**全空**，导致回放/回填静默退化成
``time_expired @ 入场价``（"假装在工作"）。本模块统一改读 parquet 真源。

价格源选 ``alpha_prices_panel.parquet`` 而非 ``daily_prices_panel.parquet``：
  - daily_prices_panel.parquet 只到 2024-12-31、508 票 → 仅覆盖 realized_pnl 的 10/80 笔
  - alpha_prices_panel.parquet 到 2026-05、1374 票 → 覆盖全部 80 笔（0 缺失）

唯一仍在 PG 的 E3 表是 ``simulated_orders``（E3 真正拥有的表），不在本模块范围。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]

# 全 universe 日线（覆盖 2019-2026），E3 价格真源
PRICE_PARQUET = _ROOT / "data" / "raw" / "alpha_prices_panel.parquet"
# 已实现 PnL（历史推荐池）真源
REALIZED_PNL_PARQUET = _ROOT / "data" / "feedback" / "realized_pnl.parquet"
# 股票池 name/industry 真源
UNIVERSE_PARQUET = _ROOT / "data" / "alpha_universe" / "alpha_universe.parquet"

_PRICE_COLS = ["ts_code", "trade_date", "open", "high", "low", "close"]


@lru_cache(maxsize=1)
def load_price_panel() -> pd.DataFrame:
    """全 universe 日线（``trade_date`` 归一为 python ``date``）。

    整文件加载一次，进程内 LRU 缓存（避免回填/回放 N 次重复读 77MB parquet）。
    """
    if not PRICE_PARQUET.exists():
        raise FileNotFoundError(
            f"E3 价格源 parquet 不存在: {PRICE_PARQUET}（无法回放/回填）"
        )
    df = pd.read_parquet(PRICE_PARQUET, columns=_PRICE_COLS)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df


def load_price_bars(
    ticker: str,
    start,
    end,
    *,
    start_exclusive: bool = False,
) -> pd.DataFrame:
    """某 ticker 在 [start, end] 内的日线，按 trade_date 升序。

    Args:
        start_exclusive: True → ``trade_date > start``（回填用，避免含入场日当根）；
                         False → ``trade_date >= start``（K线/区间展示用）。
    Returns:
        含列 [trade_date, open, high, low, close] 的 DataFrame；无数据则空 DataFrame。
    """
    df = load_price_panel()
    start = pd.to_datetime(start).date()
    end = pd.to_datetime(end).date()
    mask = (df["ts_code"] == str(ticker)) & (df["trade_date"] <= end)
    mask &= (df["trade_date"] > start) if start_exclusive else (df["trade_date"] >= start)
    return df.loc[mask].sort_values("trade_date").reset_index(drop=True)


def load_realized_pnl() -> pd.DataFrame:
    """已实现 PnL（parquet 真源），按 ``as_of_date`` 升序。"""
    if not REALIZED_PNL_PARQUET.exists():
        raise FileNotFoundError(f"realized_pnl parquet 不存在: {REALIZED_PNL_PARQUET}")
    df = pd.read_parquet(REALIZED_PNL_PARQUET)
    if "as_of_date" in df.columns:
        df = df.sort_values("as_of_date").reset_index(drop=True)
    return df


def load_name_industry_map() -> dict[str, dict]:
    """``ts_code -> {name, industry}``（parquet 真源）。"""
    if not UNIVERSE_PARQUET.exists():
        return {}
    df = pd.read_parquet(UNIVERSE_PARQUET, columns=["ts_code", "name", "industry"])
    return {
        str(r.ts_code): {"name": r.name, "industry": r.industry}
        for r in df.itertuples()
    }


def clear_cache() -> None:
    """清空价格面板缓存（测试/数据更新后调用）。"""
    load_price_panel.cache_clear()
