"""日频面板构建的提速替身（只在 T-030 的 build_daily_base 进程内替换，生产代码不动）.

`expansion.margin_buy_amount_20d` 对每只票做一次 `m[m["ticker"] == t]` 全表扫描（O(票数 × 窗口行数)，
2026 年的截面单日约 110 秒）。这里先用 groupby 取每只票的行号，再对同样的行、同样的顺序做同样的
`sort_values("trade_date").tail(20)["rzmre"].fillna(0).sum()`，结果逐位相同（见测试）。
`margin_buy_intensity` 不在 WHITELIST_35 里，且面板构建的 daily_basic 只有 ticker 列（无 total_mv），
原函数恒返回 NaN；替身直接返回同样的 NaN 序列，省掉它内部重复的一次 margin_buy_amount_20d。
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from quantmind.features import expansion as ex


def margin_buy_amount_20d(snapshot: dict, as_of: date) -> pd.Series:
    m = ex._margin_prep(ex._snapshot_df(snapshot, "margin"), as_of)
    u = snapshot.get("universe")
    if u is None or u.empty or m.empty or "rzmre" not in m.columns:
        return pd.Series(np.nan, index=u["ticker"], dtype="float64")
    pos = {str(k): v for k, v in m.groupby("ticker", sort=False).indices.items()}
    sums: dict[str, float] = {}
    for t in u["ticker"]:
        ix = pos.get(str(t))
        g = m.iloc[ix].sort_values("trade_date").tail(20) if ix is not None else m.iloc[:0]
        sums[str(t)] = float(g["rzmre"].fillna(0).sum()) if len(g) else np.nan
    return pd.Series({t: sums[str(t)] for t in u["ticker"]}, name="margin_buy_amount_20d")


def margin_buy_intensity_no_mv(snapshot: dict, as_of: date) -> pd.Series:
    db = snapshot.get("daily_basic")
    u = snapshot.get("universe")
    if db is not None and not db.empty and "total_mv" in db.columns and u is not None:
        return ex.margin_buy_intensity(snapshot, as_of)
    idx = list(u["ticker"]) if u is not None else []
    return pd.Series(np.nan, index=pd.Index(idx), dtype="float64")


def install() -> None:
    """替换 EXPANSION_FACTORS 里的两项（compute_all_expansion_factors 遍历的就是这张表）。"""
    rep = {"margin_buy_amount_20d": margin_buy_amount_20d, "margin_buy_intensity": margin_buy_intensity_no_mv}
    for i, (name, _) in enumerate(ex.EXPANSION_FACTORS):
        if name in rep:
            ex.EXPANSION_FACTORS[i] = (name, rep[name])
