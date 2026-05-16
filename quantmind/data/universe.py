"""quantmind.data.universe — 历史成分股管理（PIT-correct）.

核心问题
========

如果直接用「当前」沪深 300 成分股做回测，会引入 **survivorship bias**：
今天的强势股不代表 2018 年的成分股。

解决方案
========

1. **主源（Tushare）**：``index_weight`` 接口按 ``trade_date`` 返回历史成分股权重
2. **备源（Akshare）**：仅作当前成分股交叉校验
3. **退市过滤**：用 ``stock_basic`` 的 ``list_date / delist_date`` 排除非交易状态

输出格式
========

- ``get_universe(name, as_of)`` -> ``List[str]``  当时点成分股代码列表
- ``get_universe_with_weights(name, as_of)`` -> ``DataFrame``  含权重
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Literal

import pandas as pd

from quantmind.core.cache import cached
from quantmind.core.logger import get_logger, operation_logger
from quantmind.data.tushare_provider import (
    TushareProvider,
    _raw_index_weight,
    _raw_stock_basic_all,
)

if TYPE_CHECKING:
    from quantmind.data.akshare_provider import AkshareProvider

log = get_logger(__name__)


# 常用指数别名 -> tushare 代码
INDEX_ALIASES: dict[str, str] = {
    "csi300": "000300.SH",
    "hs300": "000300.SH",
    "沪深300": "000300.SH",
    "csi500": "000905.SH",
    "zz500": "000905.SH",
    "中证500": "000905.SH",
    "csi1000": "000852.SH",
    "中证1000": "000852.SH",
    "sse50": "000016.SH",
    "上证50": "000016.SH",
    "csi800": "000906.SH",
    "中证800": "000906.SH",
}


_ALPHA_UNIVERSE_FILE = (
    __import__("pathlib").Path(__file__).resolve().parents[2]
    / "data"
    / "alpha_universe"
    / "alpha_universe.txt"
)

_ALPHA_UNIVERSE_NAMES = {"alpha", "alpha1374", "full_alpha"}


def _load_alpha_universe() -> list[str]:
    """从 alpha_universe.txt 读取 1374 只股票代码列表。"""
    if not _ALPHA_UNIVERSE_FILE.is_file():
        log.warning(f"alpha universe 文件不存在：{_ALPHA_UNIVERSE_FILE}")
        return []
    tickers = [
        line.strip()
        for line in _ALPHA_UNIVERSE_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    log.info(f"加载 alpha universe：{len(tickers)} 只股票")
    return tickers


def _resolve_index(name: str) -> str:
    """把别名 / 短名解析为 tushare 标准代码。alpha 宇宙返回特殊标识。"""
    key = name.strip().lower()
    if key in INDEX_ALIASES:
        return INDEX_ALIASES[key]
    if key in _ALPHA_UNIVERSE_NAMES:
        return "_ALPHA_"
    return TushareProvider._normalize_index_code(name)


@cached(ttl_hours=24 * 7)
def _stock_basic_lookup() -> pd.DataFrame:
    """全市场上市/退市表（含历史），用于 list_date / delist_date 校验."""
    raw = _raw_stock_basic_all()
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    for col in ("list_date", "delist_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _filter_active(tickers: list[str], as_of: date) -> list[str]:
    """剔除 ``as_of`` 时未上市或已退市的票."""
    basic = _stock_basic_lookup()
    if basic.empty:
        return tickers
    cutoff = pd.Timestamp(as_of)
    active = set(
        basic.loc[
            (basic["list_date"].notna() & (basic["list_date"] <= cutoff))
            & (basic["delist_date"].isna() | (basic["delist_date"] > cutoff)),
            "ts_code",
        ].tolist()
    )
    return [t for t in tickers if t in active]


# ============================================================================
# 公开 API
# ============================================================================


def get_universe(
    name: str,
    as_of: date | None = None,
    *,
    drop_inactive: bool = True,
    fallback_to_current: bool = True,
) -> list[str]:
    """返回 ``as_of`` 时点的历史成分股代码列表（PIT-correct）.

    Args:
        name: 指数名称或代码（如 ``csi300`` / ``000300.SH``）
        as_of: 时点；None = 今天
        drop_inactive: 剔除当时未上市或已退市的票
        fallback_to_current: tushare 拿不到时是否退回到当前成分股（带 warning）

    Returns:
        sorted unique tickers, e.g. ``['000001.SZ', '000063.SZ', ...]``
    """
    if as_of is None:
        as_of = date.today()
    code = _resolve_index(name)

    # alpha 宇宙：直接读文件，不走 Tushare
    if code == "_ALPHA_":
        tickers = _load_alpha_universe()
        if drop_inactive and tickers:
            before = len(tickers)
            tickers = _filter_active(tickers, as_of)
            log.info(f"alpha universe filter: {before} -> {len(tickers)} active")
        return sorted(set(tickers))

    with operation_logger("universe.get_universe", index=code, as_of=str(as_of)):
        # 主源：tushare index_weight
        try:
            tushare = TushareProvider()
            tickers = tushare.get_index_constituents(code, as_of=as_of)
        except Exception as e:  # noqa: BLE001
            log.warning(f"tushare index_weight failed: {e}")
            tickers = []

        # 失败时备源
        if not tickers and fallback_to_current:
            log.warning(
                f"tushare returned empty for {code} @ {as_of}; "
                "falling back to akshare CURRENT constituents (survivorship-biased!)"
            )
            try:
                from quantmind.data.akshare_provider import AkshareProvider

                ak_provider: AkshareProvider = AkshareProvider()
                tickers = ak_provider.get_index_constituents(code, as_of=as_of)
            except Exception as e:  # noqa: BLE001
                log.error(f"akshare fallback also failed: {e}")
                tickers = []

        if drop_inactive and tickers:
            before = len(tickers)
            tickers = _filter_active(tickers, as_of)
            log.info(f"universe filter: {before} -> {len(tickers)} active tickers")

        return sorted(set(tickers))


def get_universe_with_weights(
    name: str,
    as_of: date | None = None,
) -> pd.DataFrame:
    """同 ``get_universe`` 但返回 DataFrame，含权重列 ``weight`` (%).

    返回列：``ticker, weight, trade_date``
    """
    if as_of is None:
        as_of = date.today()
    code = _resolve_index(name)
    end_str = TushareProvider._to_yyyymmdd(as_of)
    # index_weight 仅在指数调样后才有新 trade_date；60 日窗口在季初常为空（如 2020-03）
    lookback_start = date(as_of.year - 1, 1, 1)
    start_str = TushareProvider._to_yyyymmdd(lookback_start)

    with operation_logger("universe.with_weights", index=code, as_of=str(as_of)):
        raw = _raw_index_weight(code, start_str, end_str)
        if raw is None or raw.empty:
            return pd.DataFrame()
        df = raw.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df[df["trade_date"] <= pd.Timestamp(as_of)]
        if df.empty:
            return pd.DataFrame()
        latest = df["trade_date"].max()
        df = df[df["trade_date"] == latest].copy()
        from quantmind.data.base import normalize_ticker

        df["ticker"] = df["con_code"].apply(normalize_ticker)
        out = df[["ticker", "weight", "trade_date"]].sort_values("weight", ascending=False)
        return out.reset_index(drop=True)


def cross_validate_universe(
    name: str,
    as_of: date | None = None,
    *,
    tolerance: float = 0.05,
) -> dict:
    """双源交叉校验：tushare vs akshare.

    返回 dict 含：
        - ``tushare_count`` / ``akshare_count``
        - ``intersection`` / ``only_tushare`` / ``only_akshare``
        - ``agreement_ratio``  交集 / max(并集, 1)
        - ``warning_level`` ``ok | warn | fail`` （依据 tolerance）
    """
    if as_of is None:
        as_of = date.today()
    code = _resolve_index(name)

    tushare_set = set(TushareProvider().get_index_constituents(code, as_of=as_of))
    try:
        from quantmind.data.akshare_provider import AkshareProvider

        ak_set = set(AkshareProvider().get_index_constituents(code))
    except Exception as e:  # noqa: BLE001
        log.warning(f"akshare cross-check unavailable: {e}")
        ak_set = set()

    union = tushare_set | ak_set
    inter = tushare_set & ak_set
    ratio = len(inter) / len(union) if union else 0.0
    level = "ok" if ratio >= 1 - tolerance else "warn" if ratio >= 1 - 3 * tolerance else "fail"

    return {
        "as_of": as_of,
        "index": code,
        "tushare_count": len(tushare_set),
        "akshare_count": len(ak_set),
        "intersection": len(inter),
        "only_tushare": sorted(tushare_set - ak_set)[:10],
        "only_akshare": sorted(ak_set - tushare_set)[:10],
        "agreement_ratio": ratio,
        "warning_level": level,
    }


__all__ = [
    "INDEX_ALIASES",
    "cross_validate_universe",
    "get_universe",
    "get_universe_with_weights",
]


if __name__ == "__main__":
    from datetime import date as _date

    for as_of in [_date(2020, 6, 30), _date(2023, 6, 30), _date(2025, 6, 30)]:
        u = get_universe("csi300", as_of=as_of)
        print(f"  csi300 @ {as_of}: {len(u)} stocks  e.g. {u[:5]}")

    print("\n--- weights ---")
    df = get_universe_with_weights("csi300", as_of=_date(2024, 6, 30))
    print(df.head(10))
    print(f"  total weight sum: {df['weight'].sum():.2f}%")

    print("\n--- cross validate ---")
    print(cross_validate_universe("csi300", as_of=_date.today()))
