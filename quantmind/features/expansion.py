"""quantmind.features.expansion — Feature Expansion v1（Data Expansion 表 → 因子）.

PIT：所有时序表仅使用 ``trade_date <= as_of`` 的行；当前点取 as_of 前最近交易日。

说明（行业暴露）
================

``exposure_industry`` / ``exposure_area`` 来自 ``stock_basic`` 的 **静态** 标签，
**未** 处理历史行业变更；如需历史一致的行业中性化，需另行维护行业迁移表。

市场类因子
==========

``market_momentum_60d``、``market_volatility_60d``、``market_drawdown_60d`` 为
**全市场同值**（沪深300基准），在 ``standardize`` 中不参与横截面 z-score，
以免被抹成 0。
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

from quantmind.features.utils import pivot_prices, safe_divide

# 默认市场基准：沪深300（Tushare index ts_code）
BENCHMARK_INDEX_TS = "000300.SH"

# 新股阈值：上市不满该年限视为「近端 IPO」
RECENT_IPO_MAX_YEARS = 1.0

# 标准化阶段排除横截面 z-score / 中性化（与 pipeline / standardize 共享）
MARKET_LEVEL_FEATURES: frozenset[str] = frozenset(
    {
        "market_momentum_60d",
        "market_volatility_60d",
        "market_drawdown_60d",
        # 情绪组市场级同值，与 expansion 一致处理，避免横截面 z 后端常数列归 0
        "north_bound_30d_net_inflow",
    }
)

# 宽表中的字符串暴露列（不参与 winsorize / zscore / 中性化）
CATEGORICAL_EXPOSURE_FIELDS: frozenset[str] = frozenset(
    {
        "exposure_industry",
        "exposure_area",
    }
)


def _as_timestamp(as_of: date) -> pd.Timestamp:
    return pd.Timestamp(as_of)


def _snapshot_df(snapshot: dict, key: str) -> pd.DataFrame:
    """避免 ``get(k) or DataFrame()`` 对空表/真值二义性报错。"""
    v = snapshot.get(key)
    if v is None or not isinstance(v, pd.DataFrame):
        return pd.DataFrame()
    return v


def _filter_pit(df: pd.DataFrame, as_of: date, date_col: str = "trade_date") -> pd.DataFrame:
    if df is None or df.empty or date_col not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    # 按日历日比较，避免「同日收盘时间戳 > as_of 当日 00:00」被误剔除
    day_ns = d[date_col].dt.normalize()
    as_ns = pd.Timestamp(as_of).normalize()
    return d[day_ns <= as_ns]


def _benchmark_index_frame(index_daily: pd.DataFrame | None, as_of: date) -> pd.DataFrame:
    """提取沪深300日线子表（PIT），index 为 sorted trade_date."""
    if index_daily is None or index_daily.empty:
        return pd.DataFrame()
    df = _filter_pit(index_daily, as_of)
    if "ts_code" not in df.columns:
        return pd.DataFrame()
    ts = df["ts_code"].astype(str).str.upper()
    mask = ts == BENCHMARK_INDEX_TS.upper()
    if not mask.any():
        mask = ts == "000300.SH"
    out = df.loc[mask].dropna(subset=["trade_date"]).sort_values("trade_date")
    return out


def _index_close_series(index_daily: pd.DataFrame | None, as_of: date) -> pd.Series:
    """沪深300收盘序列，index=日期."""
    bi = _benchmark_index_frame(index_daily, as_of)
    if bi.empty or "close" not in bi.columns:
        return pd.Series(dtype="float64")
    s = bi.set_index("trade_date")["close"].astype("float64")
    return s.sort_index()


def _returns_from_close(close: pd.Series) -> pd.Series:
    return close.astype("float64").pct_change(fill_method=None).dropna()


def _rolling_ols_beta(y: np.ndarray, x: np.ndarray) -> float:
    """Cov(x,y)/Var(x)，x 为市场收益."""
    if len(y) < 5 or len(x) < 5 or len(y) != len(x):
        return float("nan")
    vx = np.var(x, ddof=1)
    if vx < 1e-18:
        return float("nan")
    cov = np.cov(x, y, ddof=1)[0, 1]
    return float(cov / vx)


def exposure_industry(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """静态行业标签（字符串），用于行业暴露与分组.

    stock_basic.industry 为时点可见的截面标签；历史行业变更未追溯。
    """
    del as_of
    u = snapshot.get("universe")
    if u is None or u.empty:
        return pd.Series(dtype="string")
    idx = u["ticker"].values
    sb = snapshot.get("stock_basic")
    if sb is None or sb.empty or "industry" not in sb.columns:
        return pd.Series(pd.NA, index=idx, name="exposure_industry", dtype="string")
    s = sb.drop_duplicates(subset=["ticker"]).set_index("ticker")["industry"].astype("string")
    return s.reindex(idx).rename("exposure_industry")


def exposure_area(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    del as_of
    u = snapshot.get("universe")
    if u is None or u.empty:
        return pd.Series(dtype="string")
    idx = u["ticker"].values
    sb = snapshot.get("stock_basic")
    if sb is None or sb.empty or "area" not in sb.columns:
        return pd.Series(pd.NA, index=idx, name="exposure_area", dtype="string")
    s = sb.drop_duplicates(subset=["ticker"]).set_index("ticker")["area"].astype("string")
    return s.reindex(idx).rename("exposure_area")


def list_age_years(snapshot: dict, as_of: date) -> pd.Series:
    """上市年限（年）."""
    u = snapshot.get("universe")
    if u is None or u.empty:
        return pd.Series(dtype="float64")
    idx = u["ticker"].values
    sb = snapshot.get("stock_basic")
    if sb is None or sb.empty or "list_date" not in sb.columns:
        return pd.Series(np.nan, index=idx, name="list_age_years", dtype="float64")
    sb = sb.drop_duplicates(subset=["ticker"])
    ld = pd.to_datetime(sb["list_date"], errors="coerce")
    ticker = sb["ticker"].values
    ages: dict[str, float] = {}
    cutoff = _as_timestamp(as_of)
    for t, dt in zip(ticker, ld, strict=False):
        if pd.isna(dt) or dt > cutoff:
            ages[str(t)] = float("nan")
        else:
            ages[str(t)] = (cutoff - dt).days / 365.25
    return pd.Series({t: ages.get(t, np.nan) for t in idx}, name="list_age_years")


def is_recent_ipo(snapshot: dict, as_of: date) -> pd.Series:
    """上市不足 RECENT_IPO_MAX_YEARS 记 1，否则 0."""
    la = list_age_years(snapshot, as_of)
    v = np.where(la.isna(), np.nan, np.where(la < RECENT_IPO_MAX_YEARS, 1.0, 0.0))
    return pd.Series(v, index=la.index, name="is_recent_ipo", dtype="float64")


def _hk_prep(hk: pd.DataFrame, as_of: date) -> pd.DataFrame:
    h = _filter_pit(hk, as_of)
    if h.empty or "ticker" not in h.columns:
        return pd.DataFrame()
    h = h.sort_values(["ticker", "trade_date"])
    return h


def _hk_amount_col(h: pd.DataFrame) -> str | None:
    if "hold_amount" in h.columns:
        return "hold_amount"
    if "hold_vol" in h.columns:
        return "hold_vol"
    return None


def north_hold_ratio(snapshot: dict, as_of: date) -> pd.Series:
    h = _hk_prep(_snapshot_df(snapshot, "hk_hold"), as_of)
    u = snapshot.get("universe")
    if u is None or u.empty or h.empty or "hold_ratio" not in h.columns:
        return pd.Series(np.nan, index=u["ticker"] if u is not None else [], dtype="float64")
    last = h.groupby("ticker", sort=False).tail(1).set_index("ticker")["hold_ratio"]
    return last.astype("float64").reindex(u["ticker"]).rename("north_hold_ratio")


def north_hold_amount(snapshot: dict, as_of: date) -> pd.Series:
    """持股金额或持股量列名因源表而异；无 hold_amount 时用 hold_vol 作为持股数量代理."""
    h = _hk_prep(_snapshot_df(snapshot, "hk_hold"), as_of)
    u = snapshot.get("universe")
    if u is None or u.empty or h.empty:
        return pd.Series(np.nan, index=u["ticker"], dtype="float64")
    col = _hk_amount_col(h)
    if col is None:
        return pd.Series(np.nan, index=u["ticker"], dtype="float64")
    last = h.groupby("ticker", sort=False).tail(1).set_index("ticker")[col]
    return last.astype("float64").reindex(u["ticker"]).rename("north_hold_amount")


def _hk_ratio_series_at_lag(
    h: pd.DataFrame, tickers: pd.Index, as_of: date, lag_trading: int
) -> pd.Series:
    """每只票 lag_trading 个交易日前的 hold_ratio（不足则为 NaN）."""
    out: dict[str, float] = {str(t): np.nan for t in tickers}
    if h.empty or "hold_ratio" not in h.columns:
        return pd.Series(out, dtype="float64")
    h = _filter_pit(h, as_of)
    for t, g in h.groupby("ticker", sort=False):
        g = g.sort_values("trade_date")
        if len(g) <= lag_trading:
            continue
        out[str(t)] = float(g.iloc[-(lag_trading + 1)]["hold_ratio"])
    return pd.Series({t: out.get(t, np.nan) for t in tickers}, dtype="float64")


def north_hold_ratio_change_20d(snapshot: dict, as_of: date) -> pd.Series:
    h = _hk_prep(_snapshot_df(snapshot, "hk_hold"), as_of)
    u = snapshot.get("universe")
    if u is None or u.empty:
        return pd.Series(dtype="float64")
    cur = north_hold_ratio(snapshot, as_of)
    lag = _hk_ratio_series_at_lag(h, u["ticker"], as_of, 20)
    return (cur - lag.reindex(cur.index)).rename("north_hold_ratio_change_20d")


def north_hold_ratio_change_60d(snapshot: dict, as_of: date) -> pd.Series:
    h = _hk_prep(_snapshot_df(snapshot, "hk_hold"), as_of)
    u = snapshot.get("universe")
    if u is None or u.empty:
        return pd.Series(dtype="float64")
    cur = north_hold_ratio(snapshot, as_of)
    lag = _hk_ratio_series_at_lag(h, u["ticker"], as_of, 60)
    return (cur - lag.reindex(cur.index)).rename("north_hold_ratio_change_60d")


def north_hold_amount_change_20d(snapshot: dict, as_of: date) -> pd.Series:
    h = _hk_prep(_snapshot_df(snapshot, "hk_hold"), as_of)
    u = snapshot.get("universe")
    if u is None or u.empty or h.empty:
        return pd.Series(np.nan, index=u["ticker"], dtype="float64")
    col = _hk_amount_col(h)
    if col is None:
        return pd.Series(np.nan, index=u["ticker"], dtype="float64")
    cur = h.groupby("ticker", sort=False).tail(1).set_index("ticker")[col].astype("float64")
    out = []
    for t in u["ticker"]:
        g = h[h["ticker"] == t].sort_values("trade_date")
        if len(g) <= 20:
            out.append(np.nan)
        else:
            out.append(float(cur.get(t, np.nan)) - float(g.iloc[-21][col]))
    s = pd.Series(out, index=u["ticker"], dtype="float64")
    return s.rename("north_hold_amount_change_20d")


def north_hold_trend_60d(snapshot: dict, as_of: date) -> pd.Series:
    """最近至多 60 条 hk_hold 记录上 hold_ratio 对时间的线性回归斜率."""
    h = _hk_prep(_snapshot_df(snapshot, "hk_hold"), as_of)
    u = snapshot.get("universe")
    if u is None or u.empty or h.empty or "hold_ratio" not in h.columns:
        return pd.Series(np.nan, index=u["ticker"], dtype="float64")
    slopes: dict[str, float] = {}
    for t in u["ticker"]:
        g = h[h["ticker"] == t].sort_values("trade_date").tail(60)
        if len(g) < 5:
            slopes[str(t)] = float("nan")
            continue
        y = g["hold_ratio"].astype("float64").values
        x = np.arange(len(y), dtype="float64")
        slope, _, _, _, _ = stats.linregress(x, y)
        slopes[str(t)] = float(slope)
    return pd.Series({t: slopes[str(t)] for t in u["ticker"]}, name="north_hold_trend_60d")


def _margin_prep(m: pd.DataFrame, as_of: date) -> pd.DataFrame:
    if m is None or m.empty:
        return pd.DataFrame()
    return _filter_pit(m, as_of).sort_values(["ticker", "trade_date"])


def margin_balance(snapshot: dict, as_of: date) -> pd.Series:
    m = _margin_prep(_snapshot_df(snapshot, "margin"), as_of)
    u = snapshot.get("universe")
    if u is None or u.empty or m.empty or "rzye" not in m.columns:
        return pd.Series(np.nan, index=u["ticker"] if u is not None else [], dtype="float64")
    last = m.groupby("ticker", sort=False).tail(1).set_index("ticker")["rzye"]
    return last.astype("float64").reindex(u["ticker"]).rename("margin_balance")


def _margin_field_at_lag(m: pd.DataFrame, tickers: pd.Index, field: str, lag: int) -> pd.Series:
    out = {t: np.nan for t in tickers}
    if m.empty or field not in m.columns:
        return pd.Series(out, dtype="float64")
    for t, g in m.groupby("ticker", sort=False):
        g = g.sort_values("trade_date")
        if len(g) <= lag:
            continue
        out[str(t)] = float(g.iloc[-(lag + 1)][field])
    return pd.Series({t: out.get(t, np.nan) for t in tickers}, dtype="float64")


def margin_balance_change_20d(snapshot: dict, as_of: date) -> pd.Series:
    m = _margin_prep(_snapshot_df(snapshot, "margin"), as_of)
    u = snapshot.get("universe")
    if u is None or u.empty:
        return pd.Series(dtype="float64")
    cur = margin_balance(snapshot, as_of)
    lag = _margin_field_at_lag(m, u["ticker"], "rzye", 20)
    return (cur - lag.reindex(cur.index)).rename("margin_balance_change_20d")


def margin_buy_amount_20d(snapshot: dict, as_of: date) -> pd.Series:
    """最近 20 个交易日融资买入额 rzmre 之和."""
    m = _margin_prep(_snapshot_df(snapshot, "margin"), as_of)
    u = snapshot.get("universe")
    if u is None or u.empty or m.empty or "rzmre" not in m.columns:
        return pd.Series(np.nan, index=u["ticker"], dtype="float64")
    sums: dict[str, float] = {}
    for t in u["ticker"]:
        g = m[m["ticker"] == t].sort_values("trade_date").tail(20)
        sums[str(t)] = float(g["rzmre"].fillna(0).sum()) if len(g) else np.nan
    return pd.Series({t: sums[str(t)] for t in u["ticker"]}, name="margin_buy_amount_20d")


def margin_buy_intensity(snapshot: dict, as_of: date) -> pd.Series:
    """融资买入额 20d 累计 / 总市值（万元口径与 Tushare 一致）."""
    amt = margin_buy_amount_20d(snapshot, as_of)
    db = snapshot.get("daily_basic")
    u = snapshot.get("universe")
    if db is None or db.empty or "total_mv" not in db.columns or u is None:
        return pd.Series(np.nan, index=amt.index, dtype="float64")
    mv = db.set_index("ticker")["total_mv"].astype("float64")
    return safe_divide(amt, mv.reindex(amt.index)).rename("margin_buy_intensity")


def short_balance_change_20d(snapshot: dict, as_of: date) -> pd.Series:
    m = _margin_prep(_snapshot_df(snapshot, "margin"), as_of)
    u = snapshot.get("universe")
    if u is None or u.empty or m.empty or "rqye" not in m.columns:
        return pd.Series(np.nan, index=u["ticker"], dtype="float64")
    cur = m.groupby("ticker", sort=False).tail(1).set_index("ticker")["rqye"].astype("float64")
    cur = cur.reindex(u["ticker"])
    lag = _margin_field_at_lag(m, u["ticker"], "rqye", 20)
    return (cur - lag.reindex(u["ticker"])).rename("short_balance_change_20d")


def short_sell_pressure(snapshot: dict, as_of: date) -> pd.Series:
    """融券余额 20 日变化 / 流通市值，反映相对规模的卖空边际增加."""
    m = _margin_prep(_snapshot_df(snapshot, "margin"), as_of)
    u = snapshot.get("universe")
    if u is None or u.empty:
        return pd.Series(dtype="float64")
    chg = short_balance_change_20d(snapshot, as_of)
    db = snapshot.get("daily_basic")
    if db is None or db.empty or "circ_mv" not in db.columns:
        return pd.Series(np.nan, index=u["ticker"], dtype="float64")
    cmv = db.set_index("ticker")["circ_mv"].astype("float64").reindex(u["ticker"])
    return safe_divide(chg, cmv).rename("short_sell_pressure")


def margin_short_ratio(snapshot: dict, as_of: date) -> pd.Series:
    """融资余额 / 融券余额（多空融资融券相对规模）."""
    m = _margin_prep(_snapshot_df(snapshot, "margin"), as_of)
    u = snapshot.get("universe")
    if u is None or u.empty or m.empty:
        return pd.Series(np.nan, index=u["ticker"], dtype="float64")
    need = {"rzye", "rqye"}
    if not need.issubset(m.columns):
        return pd.Series(np.nan, index=u["ticker"], dtype="float64")
    last = m.groupby("ticker", sort=False).tail(1).set_index("ticker")
    rz = last["rzye"].astype("float64")
    rq = last["rqye"].astype("float64").where(last["rqye"].astype("float64") != 0)
    return safe_divide(rz.reindex(u["ticker"]), rq.reindex(u["ticker"])).rename("margin_short_ratio")


def _stock_index_return_window(
    close_stock: pd.Series, close_idx: pd.Series, n: int
) -> tuple[float, float]:
    """个股与指数在共同交易日上的 n 日总收益 (1窗口需 n+1 个收盘点)."""
    common = close_stock.dropna().index.intersection(close_idx.dropna().index)
    common = common.sort_values()
    if len(common) < n + 1:
        return float("nan"), float("nan")
    common = common[-(n + 1) :]
    s0 = float(close_stock.loc[common[0]])
    s1 = float(close_stock.loc[common[-1]])
    i0 = float(close_idx.loc[common[0]])
    i1 = float(close_idx.loc[common[-1]])
    if s0 <= 0 or i0 <= 0:
        return float("nan"), float("nan")
    return s1 / s0 - 1.0, i1 / i0 - 1.0


def beta_n(snapshot: dict, as_of: date, window: int) -> pd.Series:
    """个股相对沪深300的 β（收益协方差/市场方差），窗口为交易日长度."""
    px = snapshot.get("prices")
    idf = snapshot.get("index_daily")
    u = snapshot.get("universe")
    if u is None or u.empty or px is None or px.empty:
        return pd.Series(np.nan, index=u["ticker"] if u is not None else [], dtype="float64")
    idx_ser = _index_close_series(idf, as_of) if idf is not None else pd.Series(dtype="float64")
    if idx_ser.empty:
        return pd.Series(np.nan, index=u["ticker"], dtype="float64")
    mret = _returns_from_close(idx_ser).tail(window + 50)
    close_w = pivot_prices(px, value_col="close")
    if close_w.empty:
        return pd.Series(np.nan, index=u["ticker"], dtype="float64")
    betas: dict[str, float] = {}
    for t in u["ticker"]:
        if str(t) not in close_w.columns:
            betas[str(t)] = float("nan")
            continue
        sret = _returns_from_close(close_w[str(t)]).tail(window + 50)
        common = mret.index.intersection(sret.index)
        common = common.sort_values()[-window:]
        if len(common) < max(10, window // 5):
            betas[str(t)] = float("nan")
            continue
        x = mret.loc[common].values
        y = sret.loc[common].values
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        if len(x) < max(10, window // 5):
            betas[str(t)] = float("nan")
            continue
        betas[str(t)] = _rolling_ols_beta(y, x)
    return pd.Series({t: betas[str(t)] for t in u["ticker"]}, dtype="float64")


def beta_252d(snapshot: dict, as_of: date) -> pd.Series:
    return beta_n(snapshot, as_of, 252).rename("beta_252d")


def beta_60d(snapshot: dict, as_of: date) -> pd.Series:
    return beta_n(snapshot, as_of, 60).rename("beta_60d")


def relative_strength_vs_csi300_60d(snapshot: dict, as_of: date) -> pd.Series:
    """(1+R_stock)/(1+R_index)-1，窗口 60 个交易日。"""
    return _relative_strength_n(snapshot, as_of, 60).rename("relative_strength_vs_csi300_60d")


def relative_strength_vs_csi300_120d(snapshot: dict, as_of: date) -> pd.Series:
    return _relative_strength_n(snapshot, as_of, 120).rename("relative_strength_vs_csi300_120d")


def _relative_strength_n(snapshot: dict, as_of: date, n: int) -> pd.Series:
    px = snapshot.get("prices")
    idf = snapshot.get("index_daily")
    u = snapshot.get("universe")
    if u is None or u.empty or px is None or px.empty:
        return pd.Series(dtype="float64")
    idx_ser = _index_close_series(idf, as_of) if idf is not None else pd.Series(dtype="float64")
    if idx_ser.empty:
        return pd.Series(np.nan, index=u["ticker"], dtype="float64")
    close_w = pivot_prices(px, value_col="close")
    out: dict[str, float] = {}
    for t in u["ticker"]:
        ts = str(t)
        if ts not in close_w.columns:
            out[ts] = float("nan")
            continue
        rs, ri = _stock_index_return_window(close_w[ts], idx_ser, n)
        if np.isnan(rs) or np.isnan(ri):
            out[ts] = float("nan")
        else:
            out[ts] = (1.0 + rs) / (1.0 + ri) - 1.0
    return pd.Series({t: out[str(t)] for t in u["ticker"]}, dtype="float64")


def market_momentum_60d(snapshot: dict, as_of: date) -> pd.Series:
    """沪深300 过去 60 交易日收益；全市场同值."""
    idx_ser = _index_close_series(snapshot.get("index_daily"), as_of)
    u = snapshot.get("universe")
    if u is None or u.empty or idx_ser.empty:
        return pd.Series(np.nan, index=u["ticker"], dtype="float64")
    if len(idx_ser) < 61:
        val = float("nan")
    else:
        series = idx_ser.sort_index()
        val = float(series.iloc[-1] / series.iloc[-61] - 1.0)
    return pd.Series(val, index=u["ticker"], name="market_momentum_60d", dtype="float64")


def market_volatility_60d(snapshot: dict, as_of: date) -> pd.Series:
    """沪深300 60 日收益率年化波动；全市场同值."""
    idx_ser = _index_close_series(snapshot.get("index_daily"), as_of)
    u = snapshot.get("universe")
    if u is None or u.empty or idx_ser.empty:
        return pd.Series(np.nan, index=u["ticker"], dtype="float64")
    r = _returns_from_close(idx_ser).tail(60)
    if len(r) < 20:
        val = float("nan")
    else:
        val = float(r.std(ddof=1) * np.sqrt(252.0))
    return pd.Series(val, index=u["ticker"], name="market_volatility_60d", dtype="float64")


def market_drawdown_60d(snapshot: dict, as_of: date) -> pd.Series:
    """沪深300 在 60 日窗口内自高点的回撤（正数越大跌得越深）；全市场同值."""
    idx_ser = _index_close_series(snapshot.get("index_daily"), as_of)
    u = snapshot.get("universe")
    if u is None or u.empty or idx_ser.empty:
        return pd.Series(np.nan, index=u["ticker"], dtype="float64")
    window = idx_ser.sort_index().tail(60)
    if len(window) < 5:
        val = float("nan")
    else:
        peak = float(window.max())
        last = float(window.iloc[-1])
        val = (peak - last) / peak if peak > 0 else float("nan")
    return pd.Series(val, index=u["ticker"], name="market_drawdown_60d", dtype="float64")


def _index_close_by_code(
    index_daily: pd.DataFrame | None, as_of: date, ts_code: str
) -> pd.Series:
    """提取指定指数的 PIT 收盘序列."""
    if index_daily is None or index_daily.empty:
        return pd.Series(dtype="float64")
    df = _filter_pit(index_daily, as_of)
    mask = df["ts_code"].astype(str).str.upper() == ts_code.upper()
    sub = df.loc[mask].dropna(subset=["trade_date"]).sort_values("trade_date")
    if sub.empty or "close" not in sub.columns:
        return pd.Series(dtype="float64")
    return sub.set_index("trade_date")["close"].astype("float64").sort_index()


def relative_strength_vs_csi500_60d(snapshot: dict, as_of: date) -> pd.Series:
    """个股相对中证500（小盘基准）的超额收益，窗口 60 交易日.

    = (1+R_stock)/(1+R_csi500) - 1

    与 relative_strength_vs_csi300_60d 互补——对中小盘股来说，
    超越中证500更能体现个股 alpha（超越小盘整体）。
    """
    px = snapshot.get("prices")
    idf = snapshot.get("index_daily")
    u = snapshot.get("universe")
    if u is None or u.empty or px is None or px.empty:
        return pd.Series(dtype="float64")
    idx_ser = _index_close_by_code(idf, as_of, "000905.SH")
    if idx_ser.empty:
        return pd.Series(np.nan, index=u["ticker"], dtype="float64")
    close_w = pivot_prices(px, value_col="close")
    out: dict[str, float] = {}
    for t in u["ticker"]:
        ts = str(t)
        if ts not in close_w.columns:
            out[ts] = float("nan")
            continue
        rs, ri = _stock_index_return_window(close_w[ts], idx_ser, 60)
        out[ts] = float("nan") if (np.isnan(rs) or np.isnan(ri)) else (1.0 + rs) / (1.0 + ri) - 1.0
    return pd.Series({t: out[str(t)] for t in u["ticker"]}, dtype="float64").rename(
        "relative_strength_vs_csi500_60d"
    )


def volume_price_corr_20d(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """量价相关性（20 日）= Pearson(日收益率, log成交量变化).

    正值 = 量升价涨（聪明钱买入）；负值 = 量升价跌（聪明钱出货）。
    在小盘 regime 中，量价共振是小市值股异动的早期信号。
    """
    px = snapshot.get("prices")
    u = snapshot.get("universe")
    if px is None or px.empty or u is None or u.empty:
        return pd.Series(dtype="float64")
    close = pivot_prices(px, value_col="close")
    vol = pivot_prices(px, value_col="volume")
    if len(close) < 21:
        return pd.Series(dtype="float64")
    ret = close.pct_change(fill_method=None).iloc[-20:]
    log_vol_chg = np.log(vol.replace(0, np.nan) + 1).diff().iloc[-20:]
    tickers = [str(t) for t in u["ticker"]]
    out: dict[str, float] = {}
    for t in tickers:
        if t not in ret.columns or t not in log_vol_chg.columns:
            out[t] = float("nan")
            continue
        r = ret[t].dropna()
        v = log_vol_chg[t].dropna()
        common = r.index.intersection(v.index)
        if len(common) < 10:
            out[t] = float("nan")
            continue
        try:
            c = float(np.corrcoef(r.loc[common].values, v.loc[common].values)[0, 1])
            out[t] = c if not np.isnan(c) else float("nan")
        except Exception:  # noqa: BLE001
            out[t] = float("nan")
    return pd.Series({t: out[t] for t in tickers}, dtype="float64").rename("volume_price_corr_20d")


# (name, fn) — 与 ALL_FACTOR_GROUPS 一致
EXPANSION_FACTORS: list[tuple[str, object]] = [
    ("exposure_industry", exposure_industry),
    ("exposure_area", exposure_area),
    ("list_age_years", list_age_years),
    ("is_recent_ipo", is_recent_ipo),
    ("north_hold_ratio", north_hold_ratio),
    ("north_hold_amount", north_hold_amount),
    ("north_hold_ratio_change_20d", north_hold_ratio_change_20d),
    ("north_hold_ratio_change_60d", north_hold_ratio_change_60d),
    ("north_hold_amount_change_20d", north_hold_amount_change_20d),
    ("north_hold_trend_60d", north_hold_trend_60d),
    ("margin_balance", margin_balance),
    ("margin_balance_change_20d", margin_balance_change_20d),
    ("margin_buy_amount_20d", margin_buy_amount_20d),
    ("margin_buy_intensity", margin_buy_intensity),
    ("short_balance_change_20d", short_balance_change_20d),
    ("short_sell_pressure", short_sell_pressure),
    ("margin_short_ratio", margin_short_ratio),
    ("beta_252d", beta_252d),
    ("beta_60d", beta_60d),
    ("relative_strength_vs_csi300_60d", relative_strength_vs_csi300_60d),
    ("relative_strength_vs_csi300_120d", relative_strength_vs_csi300_120d),
    ("market_momentum_60d", market_momentum_60d),
    ("market_volatility_60d", market_volatility_60d),
    ("market_drawdown_60d", market_drawdown_60d),
    # 小盘专属 (2)
    ("relative_strength_vs_csi500_60d", relative_strength_vs_csi500_60d),
    ("volume_price_corr_20d", volume_price_corr_20d),
]


def compute_all_expansion_factors(snapshot: dict, as_of: date) -> pd.DataFrame:
    """计算 Feature Expansion v1 全部列（含字符串暴露列）."""
    out: dict[str, pd.Series] = {}
    for name, fn in EXPANSION_FACTORS:
        try:
            s = fn(snapshot, as_of)
        except Exception:  # noqa: BLE001
            u = snapshot.get("universe")
            idx = u["ticker"] if (u is not None and not u.empty) else pd.Index([])
            if name in CATEGORICAL_EXPOSURE_FIELDS:
                s = pd.Series(pd.NA, index=idx, dtype="string")
            else:
                s = pd.Series(dtype="float64", name=name)
                if len(idx):
                    s = pd.Series(np.nan, index=idx, dtype="float64")
        s.name = name
        out[name] = s
    df = pd.DataFrame(out)
    df.index.name = "ticker"
    return df


__all__ = [
    "BENCHMARK_INDEX_TS",
    "CATEGORICAL_EXPOSURE_FIELDS",
    "EXPANSION_FACTORS",
    "MARKET_LEVEL_FEATURES",
    "RECENT_IPO_MAX_YEARS",
    "compute_all_expansion_factors",
    "is_recent_ipo",
    "list_age_years",
    "margin_short_ratio",
    "market_drawdown_60d",
    "market_momentum_60d",
    "market_volatility_60d",
    "beta_60d",
    "relative_strength_vs_csi500_60d",
    "volume_price_corr_20d",
]
