"""quantmind/data/handler.py

QuantMind 统一数据接口层（Qlib-compatible DataHandler）。

设计目标
--------
1. **适配器模式**：上层代码不感知底层是 parquet / Qlib / Arrow mmap。
   当前后端 = parquet（保持现有行为）；未来可热插拔为 Qlib Data Server。

2. **LRU 缓存**：
   - 大文件（alpha_panel、prices）在首次加载后驻留内存。
   - 相同过滤参数（as_of、ts_codes、date_range）额外缓存 → 避免重复切片。
   - ``invalidate_cache()`` 供 daily_update.py 跑完后调用，强制重新加载。

3. **懒加载 + 内存映射**：
   - 文件只在第一次 ``fetch_*()`` 时读取，构造函数不做 IO。
   - 大文件（> 10 MB）使用 ``memory_map=True`` 减少 RSS。

4. **单例**：
   - 通过 ``QuantMindDataHandler.get_instance()`` 获取全局唯一实例，
     避免不同模块重复加载同一文件。

Qlib DataHandler 对齐接口
--------------------------
    .setup_data()       — 初始化 / 预热（可选调用）
    .fetch_panel()      — alpha_panel_v4（因子面板，MultiIndex as_of×ticker）
    .fetch_prices()     — 日线价格（long-format）
    .fetch_positions()  — 持仓记录（paper_trading/positions.parquet）
    .fetch_returns()    — 已实现收益（feedback/realized_pnl.parquet）
    .invalidate_cache() — 清除所有缓存

迁移策略（调用点替换示例）
---------------------------
旧代码::

    panel = pd.read_parquet("data/panel/alpha_panel_v4.parquet")

新代码::

    from quantmind.data.handler import QuantMindDataHandler
    panel = QuantMindDataHandler.get_instance().fetch_panel()

两者返回同一个 DataFrame（引用缓存副本），性能差异在第二次调用时体现为 >100x 加速。

用法示例
--------
>>> from quantmind.data.handler import QuantMindDataHandler
>>> h = QuantMindDataHandler.get_instance()
>>> panel = h.fetch_panel()                              # 全量
>>> panel_q = h.fetch_panel(as_of="2024-03-31")        # 单季截面
>>> panel_f = h.fetch_panel(ts_codes=["600519.SH"])    # 单股全历史
>>> prices = h.fetch_prices(ts_code="600519.SH",
...                          start_date="2024-01-01",
...                          end_date="2024-06-30")
>>> positions = h.fetch_positions(latest_only=True)
>>> pnl = h.fetch_returns()
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

log = logging.getLogger(__name__)

# ── 默认文件路径 ───────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent

_DEFAULT_PATHS: Dict[str, Path] = {
    # Top-1: alpha factor panel (29K rows × 75 cols, ~18 MB)
    "panel":     _ROOT / "data" / "panel" / "alpha_panel_v4.parquet",
    # Top-2: daily prices long-format (728K rows, ~28 MB)
    "prices":    _ROOT / "data" / "raw" / "daily_prices_panel.parquet",
    # Top-3: paper-trading positions (~90 rows, tiny)
    "positions": _ROOT / "data" / "paper_trading" / "positions.parquet",
    # Bonus: realized PnL (feedback loop)
    "returns":   _ROOT / "data" / "feedback" / "realized_pnl.parquet",
}

# 大于此字节数的文件使用 memory_map=True
_MMAP_THRESHOLD = 5 * 1024 * 1024   # 5 MB


class QuantMindDataHandler:
    """统一数据接口，屏蔽底层存储格式。

    Parameters
    ----------
    paths : dict, optional
        覆盖默认文件路径，格式 ``{"panel": Path(...), ...}``。
        未指定的键使用 ``_DEFAULT_PATHS`` 中的默认值。

    Examples
    --------
    >>> h = QuantMindDataHandler.get_instance()
    >>> panel = h.fetch_panel()
    >>> panel_q = h.fetch_panel(as_of="2024-03-31")
    >>> prices = h.fetch_prices(ts_code="600519.SH", start_date="2024-01-01")
    """

    _instance: "QuantMindDataHandler | None" = None
    _lock = threading.Lock()

    # ── 单例 ──────────────────────────────────────────────────────────────────

    @classmethod
    def get_instance(
        cls,
        paths: Optional[Dict[str, Union[str, Path]]] = None,
    ) -> "QuantMindDataHandler":
        """获取全局唯一 DataHandler 实例（线程安全）。

        首次调用会构造实例；后续调用直接返回缓存实例。
        若需要不同路径配置，用 ``QuantMindDataHandler(paths=...)`` 直接构造。
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(paths=paths)
        return cls._instance

    def __init__(
        self,
        paths: Optional[Dict[str, Union[str, Path]]] = None,
    ) -> None:
        # 合并路径配置
        self._paths: Dict[str, Path] = dict(_DEFAULT_PATHS)
        if paths:
            for k, v in paths.items():
                self._paths[k] = Path(v)

        # 底层数据缓存：key → DataFrame（整表）
        self._raw_cache:    Dict[str, pd.DataFrame] = {}
        # 过滤结果缓存：(key, filter_hash) → DataFrame（切片）
        self._slice_cache:  Dict[Tuple[str, str], pd.DataFrame] = {}
        # 缓存统计
        self._stats: Dict[str, int] = {
            "panel_loads": 0, "prices_loads": 0,
            "positions_loads": 0, "returns_loads": 0,
            "cache_hits": 0, "cache_misses": 0,
        }
        self._lock = threading.Lock()

    # ── 初始化 / 预热（可选）──────────────────────────────────────────────────

    def setup_data(self, preload: bool = False) -> None:
        """初始化数据源（Qlib DataHandler 兼容接口）。

        Parameters
        ----------
        preload : bool
            若 True，立即读取所有数据文件到内存（适合批量回测场景）。
            默认 False（懒加载）。
        """
        if preload:
            log.info("DataHandler.setup_data: 预热所有数据源...")
            self._load_raw("panel")
            self._load_raw("prices")
            self._load_raw("positions")
            self._load_raw("returns")
            log.info("DataHandler.setup_data: 预热完成")
        else:
            log.debug("DataHandler.setup_data: 懒加载模式，文件将按需读取")

    # ── fetch_panel ────────────────────────────────────────────────────────────

    def fetch_panel(
        self,
        as_of:    Optional[Union[str, "pd.Timestamp"]] = None,
        ts_codes: Optional[List[str]] = None,
        columns:  Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """加载 alpha_panel_v4（MultiIndex: as_of × ticker）。

        Parameters
        ----------
        as_of : str or Timestamp, optional
            过滤特定季度截面，例如 ``"2024-03-31"``。
            None = 返回全量历史数据。
        ts_codes : list[str], optional
            过滤指定股票代码，例如 ``["600519.SH", "000858.SZ"]``。
            None = 全部股票。
        columns : list[str], optional
            只返回指定列（按需裁剪，节省内存）。None = 全部列。

        Returns
        -------
        pd.DataFrame
            MultiIndex (as_of, ticker) 的因子面板。
        """
        filter_key = _make_hash("panel", as_of=as_of, ts_codes=ts_codes, columns=columns)

        hit = self._get_slice("panel", filter_key)
        if hit is not None:
            return hit

        raw = self._load_raw("panel")
        df  = self._filter_panel(raw, as_of=as_of, ts_codes=ts_codes, columns=columns)
        self._put_slice("panel", filter_key, df)
        self._stats["panel_loads"] += 1
        return df

    @staticmethod
    def _filter_panel(
        df:       pd.DataFrame,
        as_of:    Optional[Union[str, "pd.Timestamp"]],
        ts_codes: Optional[List[str]],
        columns:  Optional[List[str]],
    ) -> pd.DataFrame:
        if as_of is not None:
            as_of_ts = pd.Timestamp(as_of)
            # MultiIndex: level 0 = as_of
            idx = df.index.get_level_values(0)
            df  = df[idx == as_of_ts]

        if ts_codes is not None:
            ticker_set = set(ts_codes)
            idx = df.index.get_level_values(1)
            df  = df[idx.isin(ticker_set)]

        if columns is not None:
            existing = [c for c in columns if c in df.columns]
            df = df[existing]

        return df

    # ── fetch_prices ──────────────────────────────────────────────────────────

    def fetch_prices(
        self,
        ts_code:    Optional[str] = None,
        start_date: Optional[Union[str, "pd.Timestamp"]] = None,
        end_date:   Optional[Union[str, "pd.Timestamp"]] = None,
        columns:    Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """加载日线价格（long-format: trade_date × ts_code）。

        Parameters
        ----------
        ts_code : str, optional
            过滤单只股票，例如 ``"600519.SH"``。None = 全市场。
        start_date : str or Timestamp, optional
            开始日期（含），例如 ``"2024-01-01"``。
        end_date : str or Timestamp, optional
            结束日期（含），例如 ``"2024-06-30"``。
        columns : list[str], optional
            只返回指定列；若不指定返回全部列。

        Returns
        -------
        pd.DataFrame
            Long-format 日线价格，columns ⊇ [trade_date, ts_code, close]。
        """
        filter_key = _make_hash(
            "prices",
            ts_code=ts_code, start=start_date, end=end_date, columns=columns,
        )

        hit = self._get_slice("prices", filter_key)
        if hit is not None:
            return hit

        raw = self._load_raw("prices")
        df  = self._filter_prices(raw, ts_code=ts_code,
                                  start_date=start_date, end_date=end_date,
                                  columns=columns)
        self._put_slice("prices", filter_key, df)
        self._stats["prices_loads"] += 1
        return df

    @staticmethod
    def _filter_prices(
        df:         pd.DataFrame,
        ts_code:    Optional[str],
        start_date: Optional[Union[str, "pd.Timestamp"]],
        end_date:   Optional[Union[str, "pd.Timestamp"]],
        columns:    Optional[List[str]],
    ) -> pd.DataFrame:
        mask = pd.Series(True, index=df.index)

        date_col = "trade_date" if "trade_date" in df.columns else df.index.name
        if date_col and date_col in df.columns:
            dates = pd.to_datetime(df[date_col])
            if start_date is not None:
                mask &= dates >= pd.Timestamp(start_date)
            if end_date is not None:
                mask &= dates <= pd.Timestamp(end_date)

        code_col = "ts_code" if "ts_code" in df.columns else (
            "ticker" if "ticker" in df.columns else None
        )
        if ts_code is not None and code_col:
            mask &= df[code_col] == ts_code

        df = df[mask]

        if columns is not None:
            existing = [c for c in columns if c in df.columns]
            df = df[existing]

        return df.reset_index(drop=True)

    # ── fetch_positions ───────────────────────────────────────────────────────

    def fetch_positions(self, latest_only: bool = False) -> pd.DataFrame:
        """加载持仓记录（data/paper_trading/positions.parquet）。

        Parameters
        ----------
        latest_only : bool
            True = 只返回最新 as_of 日期的持仓；False = 全量历史。

        Returns
        -------
        pd.DataFrame
            持仓记录，columns ⊇ [as_of, ticker, predicted_rank, ...]。
        """
        filter_key = _make_hash("positions", latest_only=latest_only)
        hit = self._get_slice("positions", filter_key)
        if hit is not None:
            return hit

        raw = self._load_raw("positions")
        df  = raw.copy()

        if latest_only and not df.empty:
            date_col = "as_of" if "as_of" in df.columns else df.columns[0]
            latest   = df[date_col].max()
            df       = df[df[date_col] == latest]

        self._put_slice("positions", filter_key, df)
        self._stats["positions_loads"] += 1
        return df

    # ── fetch_returns ─────────────────────────────────────────────────────────

    def fetch_returns(
        self,
        period:   str = "63d",
        ts_codes: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """加载已实现收益（data/feedback/realized_pnl.parquet）。

        Parameters
        ----------
        period : str
            收益期（"21d" / "63d" 等），用于选择 forward_return_* 列。
            当前实现：返回全量 realized_pnl，调用方自行选列。
        ts_codes : list[str], optional
            过滤指定股票。None = 全部。

        Returns
        -------
        pd.DataFrame
            已实现收益记录，columns ⊇ [as_of_date, ticker, actual_return_63d, ...]。
        """
        filter_key = _make_hash("returns", period=period, ts_codes=ts_codes)
        hit = self._get_slice("returns", filter_key)
        if hit is not None:
            return hit

        raw = self._load_raw("returns")
        df  = raw.copy()

        if ts_codes is not None:
            code_col = "ticker" if "ticker" in df.columns else (
                "ts_code" if "ts_code" in df.columns else None
            )
            if code_col:
                df = df[df[code_col].isin(set(ts_codes))]

        self._put_slice("returns", filter_key, df)
        self._stats["returns_loads"] += 1
        return df

    # ── 缓存管理 ──────────────────────────────────────────────────────────────

    def invalidate_cache(self) -> None:
        """清除所有缓存，下次调用时重新从磁盘读取。

        应在 daily_update.py 数据更新完成后调用，确保后续查询使用最新数据。
        """
        with self._lock:
            self._raw_cache.clear()
            self._slice_cache.clear()
        log.info("DataHandler: 缓存已清除（raw + slice）")

    def invalidate_key(self, key: str) -> None:
        """清除单个数据源的缓存（'panel' / 'prices' / 'positions' / 'returns'）。"""
        with self._lock:
            self._raw_cache.pop(key, None)
            to_remove = [k for k in self._slice_cache if k[0] == key]
            for k in to_remove:
                del self._slice_cache[k]
        log.info("DataHandler: 已清除 '%s' 的缓存", key)

    def cache_info(self) -> Dict[str, int]:
        """返回缓存统计信息（命中/未命中次数、各 key 加载次数）。"""
        with self._lock:
            return {
                **self._stats,
                "raw_keys": len(self._raw_cache),
                "slice_keys": len(self._slice_cache),
            }

    # ── 内部：原始数据加载 ────────────────────────────────────────────────────

    def _load_raw(self, key: str) -> pd.DataFrame:
        """懒加载原始 parquet 文件（线程安全，只加载一次）。"""
        with self._lock:
            if key in self._raw_cache:
                return self._raw_cache[key]

        path = self._paths.get(key)
        if path is None:
            raise KeyError(f"QuantMindDataHandler: 未知数据键 '{key}'")
        if not path.exists():
            raise FileNotFoundError(
                f"QuantMindDataHandler: 数据文件不存在: {path}"
            )

        use_mmap = path.stat().st_size > _MMAP_THRESHOLD
        t0 = time.perf_counter()
        try:
            df = pd.read_parquet(path, memory_map=use_mmap)
        except TypeError:
            # 旧版 pyarrow 不支持 memory_map 参数
            df = pd.read_parquet(path)

        elapsed = time.perf_counter() - t0
        log.info(
            "DataHandler._load_raw('%s'): shape=%s  %.3fs  mmap=%s",
            key, df.shape, elapsed, use_mmap,
        )

        with self._lock:
            self._raw_cache[key] = df
        return df

    # ── 内部：切片缓存 ────────────────────────────────────────────────────────

    def _get_slice(self, key: str, filter_hash: str) -> Optional[pd.DataFrame]:
        cache_key = (key, filter_hash)
        with self._lock:
            if cache_key in self._slice_cache:
                self._stats["cache_hits"] += 1
                return self._slice_cache[cache_key]
        self._stats["cache_misses"] += 1
        return None

    def _put_slice(self, key: str, filter_hash: str, df: pd.DataFrame) -> None:
        cache_key = (key, filter_hash)
        with self._lock:
            self._slice_cache[cache_key] = df

    # ── 便捷属性 ──────────────────────────────────────────────────────────────

    @property
    def panel_path(self) -> Path:
        return self._paths["panel"]

    @property
    def prices_path(self) -> Path:
        return self._paths["prices"]

    @property
    def positions_path(self) -> Path:
        return self._paths["positions"]

    @property
    def returns_path(self) -> Path:
        return self._paths["returns"]

    def __repr__(self) -> str:
        with self._lock:
            loaded = list(self._raw_cache.keys())
        return (
            f"QuantMindDataHandler("
            f"loaded={loaded}, "
            f"cache_hits={self._stats['cache_hits']}, "
            f"misses={self._stats['cache_misses']})"
        )


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _make_hash(key: str, **kwargs) -> str:
    """将数据键 + 过滤参数转换为短 hash 字符串，用作切片缓存键。"""
    parts = [f"__key__={key}"] + sorted(
        f"{k}={v}" for k, v in kwargs.items() if v is not None
    )
    raw = "|".join(parts).encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:12]


# ── 模块级便捷函数（直接替换 pd.read_parquet 调用）──────────────────────────

def get_handler() -> QuantMindDataHandler:
    """返回全局单例 DataHandler（最简调用形式）。"""
    return QuantMindDataHandler.get_instance()


def load_panel(
    as_of:    Optional[Union[str, "pd.Timestamp"]] = None,
    ts_codes: Optional[List[str]] = None,
    columns:  Optional[List[str]] = None,
) -> pd.DataFrame:
    """模块级快捷函数：加载 alpha_panel_v4，直接替换 pd.read_parquet(PANEL_PATH)。"""
    return get_handler().fetch_panel(as_of=as_of, ts_codes=ts_codes, columns=columns)


def load_prices(
    ts_code:    Optional[str] = None,
    start_date: Optional[Union[str, "pd.Timestamp"]] = None,
    end_date:   Optional[Union[str, "pd.Timestamp"]] = None,
    columns:    Optional[List[str]] = None,
) -> pd.DataFrame:
    """模块级快捷函数：加载日线价格，直接替换 pd.read_parquet(PRICES_PATH)。"""
    return get_handler().fetch_prices(
        ts_code=ts_code, start_date=start_date, end_date=end_date, columns=columns
    )


def load_positions(latest_only: bool = False) -> pd.DataFrame:
    """模块级快捷函数：加载持仓数据，直接替换 pd.read_parquet(POSITIONS_PATH)。"""
    return get_handler().fetch_positions(latest_only=latest_only)


def load_returns(
    period:   str = "63d",
    ts_codes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """模块级快捷函数：加载已实现收益，直接替换 pd.read_parquet(RETURNS_PATH)。"""
    return get_handler().fetch_returns(period=period, ts_codes=ts_codes)


def open_parquet_cached(
    path: Union[str, Path],
    memory_map: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """通用缓存 parquet 读取器——共享 DataHandler 的全局缓存。

    等价于 ``pd.read_parquet(path)``，但首次加载后结果驻留全局缓存。
    相同路径的第二次调用直接返回内存中的 DataFrame（0.00x ms 级）。

    Parameters
    ----------
    path : str or Path
        parquet 文件路径。
    memory_map : bool
        对大文件（> 5 MB）启用内存映射，默认 True。
    **kwargs
        透传给 ``pd.read_parquet``（columns / filters 等）。

    Returns
    -------
    pd.DataFrame

    Examples
    --------
    替换前::

        df = pd.read_parquet(panel_file)

    替换后::

        from quantmind.data.handler import open_parquet_cached
        df = open_parquet_cached(panel_file)

    Notes
    -----
    - 若传入了 ``columns`` 或 ``filters``，则以 ``str(path)+str(kwargs)`` 为缓存键，
      不复用无过滤的完整缓存——调用方应自行决定是否传参。
    - 调用 ``QuantMindDataHandler.get_instance().invalidate_cache()`` 可清除全部缓存。
    """
    path = Path(path)
    h    = QuantMindDataHandler.get_instance()

    # 生成缓存键：若有额外 kwargs，加入到 key 中避免缓存污染
    key  = str(path) if not kwargs else f"{path}|{sorted(kwargs.items())}"

    with h._lock:
        if key in h._raw_cache:
            h._stats["cache_hits"] += 1
            return h._raw_cache[key]

    h._stats["cache_misses"] += 1

    use_mmap = memory_map and path.exists() and path.stat().st_size > _MMAP_THRESHOLD
    try:
        df = pd.read_parquet(path, memory_map=use_mmap, **kwargs)
    except TypeError:
        df = pd.read_parquet(path, **kwargs)

    with h._lock:
        h._raw_cache[key] = df

    log.debug("open_parquet_cached: loaded '%s' shape=%s", path, df.shape)
    return df


__all__ = [
    "QuantMindDataHandler",
    "get_handler",
    "load_panel",
    "load_prices",
    "load_positions",
    "load_returns",
    "open_parquet_cached",
]
