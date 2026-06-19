"""quantmind.features.short_horizon_factors — 短周期特异性因子（日线，PIT）.

依据 ``docs/plans/short_horizon_factor_plan.md``（已批准）+ bake-off R3 定稿。

设计
----
- 全部算子在 **wide DataFrame（index=trade_date 升序, columns=ticker）** 上向量化，
  所有时序算子**只回看**（rolling/shift 向过去），天然 PIT。
- 跨日用复权价（adj_close、adj_open=open×adj_factor…）；同日日内结构用原始价。
- 残差化因子（A2/A4）在【残差日收益 r̃】上构造：r̃ = (r − cap加权市场) 对
  行业哑变量 + z(log circ_mv) 截面 OLS 的残差。
- ``build_short_horizon_factors`` 返回 dict[name] -> 日频 wide 因子；
  ``sample_at_asof`` 把日频因子取到 (as_of,ticker) 网格（供 merge_increment）。

列名前缀 ``sh_`` 避免与 base 35 列碰撞。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ============================================================================
# 算子（wide frame；只回看）
# ============================================================================

def delay(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.shift(d)

def delta(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x - x.shift(d)

def ts_sum(x: pd.DataFrame, w: int) -> pd.DataFrame:
    return x.rolling(w, min_periods=max(2, w // 2)).sum()

def ts_mean(x: pd.DataFrame, w: int) -> pd.DataFrame:
    return x.rolling(w, min_periods=max(2, w // 2)).mean()

def ts_std(x: pd.DataFrame, w: int) -> pd.DataFrame:
    return x.rolling(w, min_periods=max(3, w // 2)).std()

def ts_median(x: pd.DataFrame, w: int) -> pd.DataFrame:
    return x.rolling(w, min_periods=max(2, w // 2)).median()

def ts_min(x: pd.DataFrame, w: int) -> pd.DataFrame:
    return x.rolling(w, min_periods=max(2, w // 2)).min()

def ts_max(x: pd.DataFrame, w: int) -> pd.DataFrame:
    return x.rolling(w, min_periods=max(2, w // 2)).max()

def ts_skew(x: pd.DataFrame, w: int) -> pd.DataFrame:
    return x.rolling(w, min_periods=max(5, w // 2)).skew()

def ts_kurt(x: pd.DataFrame, w: int) -> pd.DataFrame:
    return x.rolling(w, min_periods=max(5, w // 2)).kurt()

def ts_corr(x: pd.DataFrame, y: pd.DataFrame, w: int) -> pd.DataFrame:
    return x.rolling(w, min_periods=max(3, w // 2)).corr(y)

def sign(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(x)

def cs_rank(x: pd.DataFrame) -> pd.DataFrame:
    """截面百分位 rank（每个交易日跨 ticker，[0,1]）."""
    return x.rank(axis=1, pct=True)

def ts_rank(x: pd.DataFrame, w: int) -> pd.DataFrame:
    """末值在过去 w 窗口内的分位 = mean_{k=0..w-1}[x_t >= x_{t-k}]（向量化、只回看）."""
    acc = None
    for k in range(w):
        cmp = (x >= x.shift(k)).astype(float)
        acc = cmp if acc is None else acc + cmp
    return acc / w

def ts_cov(x: pd.DataFrame, y: pd.DataFrame, w: int) -> pd.DataFrame:
    return x.rolling(w, min_periods=max(3, w // 2)).cov(y)

def signedpower(x: pd.DataFrame, e: float) -> pd.DataFrame:
    return np.sign(x) * (x.abs() ** e)

def ts_argmax(x: pd.DataFrame, w: int) -> pd.DataFrame:
    """末窗 [t-w+1, t] 内最大值的位置（0=最旧 … w-1=最近），向量化只回看."""
    best_v = None; best_p = None
    for k in range(w):
        v = x.shift(k); pos = float(w - 1 - k)
        if best_v is None:
            best_v = v.copy()
            best_p = pd.DataFrame(pos, index=x.index, columns=x.columns)
        else:
            mask = v > best_v
            best_p = best_p.where(~mask, pos)
            best_v = best_v.where(~mask, v)
    return best_p


# ============================================================================
# 价量 wide 帧组装
# ============================================================================

def build_wide_frames(prices_long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """alpha_prices_panel 长表 → 一组 wide 帧（index=trade_date, columns=ticker）.

    需要列：ts_code, trade_date, open, high, low, close, vol, amount, adj_factor, adj_close.
    """
    df = prices_long.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["ticker"] = df["ts_code"].astype(str)
    df = df.drop_duplicates(["trade_date", "ticker"], keep="last")
    def piv(col):
        return df.pivot(index="trade_date", columns="ticker", values=col).sort_index()
    af = piv("adj_factor")
    out = {
        "open": piv("open"), "high": piv("high"), "low": piv("low"),
        "close": piv("close"), "vol": piv("vol"), "amount": piv("amount"),
        "adj_close": piv("adj_close"), "adj_factor": af,
    }
    out["adj_open"] = out["open"] * af
    out["adj_high"] = out["high"] * af
    out["adj_low"] = out["low"] * af
    # vwap（元）= 10 × amount(千元) / vol(手)；vol=0 → NaN
    out["vwap"] = (10.0 * out["amount"] / out["vol"].replace(0, np.nan))
    out["returns"] = out["adj_close"].pct_change(fill_method=None)
    out["adv20"] = ts_mean(out["vol"], 20)
    return out


def residual_returns(
    returns: pd.DataFrame, logcap: pd.DataFrame, industry_map: pd.Series
) -> pd.DataFrame:
    """逐日截面：(r − cap加权市场) 对 [1, 行业哑变量, z(log cap)] OLS 残差 r̃.

    - cap 权重 = circ_mv / Σcirc_mv（用 exp(logcap)）。
    - industry_map: ticker -> 行业字符串（静态）。
    - 只用当日截面（无跨日），PIT 安全。
    """
    tickers = list(returns.columns)
    ind = industry_map.reindex(tickers).astype("object").fillna("NA")
    # 行业哑变量矩阵（ticker × industry），一次构建
    ind_dummies = pd.get_dummies(ind, prefix="ind")
    ind_mat_full = ind_dummies.values.astype(float)      # (n_tickers, n_ind)
    ind_index = {t: i for i, t in enumerate(tickers)}

    resid = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    rv = returns.values
    lv = logcap.reindex(index=returns.index, columns=returns.columns).values
    cap = np.exp(lv)
    for ti in range(rv.shape[0]):
        r = rv[ti]; lc = lv[ti]; cp = cap[ti]
        mask = np.isfinite(r) & np.isfinite(lc) & np.isfinite(cp) & (cp > 0)
        if mask.sum() < 50:
            continue
        idx = np.where(mask)[0]
        rr = r[idx]
        w = cp[idx] / cp[idx].sum()
        mkt = float((w * rr).sum())            # cap 加权市场
        ex = rr - mkt                          # 超额
        z = (lc[idx] - lc[idx].mean()) / (lc[idx].std() + 1e-12)
        D = ind_mat_full[idx]
        # 去掉全 0 行业列（该日无该行业票）
        keep = D.sum(axis=0) > 0
        D = D[:, keep]
        X = np.column_stack([np.ones(len(idx)), z, D])
        beta, *_ = np.linalg.lstsq(X, ex, rcond=None)
        res = ex - X @ beta
        resid.values[ti, idx] = res
    return resid


# ============================================================================
# 因子构造（返回 dict[name]->日频 wide）
# ============================================================================

def build_short_horizon_factors(
    F: dict[str, pd.DataFrame], rtilde: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """F = build_wide_frames 输出；rtilde = residual_returns 输出。返回 sh_* 因子。"""
    o, c, h, l = F["open"], F["close"], F["high"], F["low"]
    ac, ao = F["adj_close"], F["adj_open"]
    vol, vwap, adv20, ret = F["vol"], F["vwap"], F["adv20"], F["returns"]
    out: dict[str, pd.DataFrame] = {}

    # ---- A1 日内反转（剔隔夜）+ 隔夜对照 ----
    intraday = c / o - 1.0                       # 同日，原始价
    overnight = ao / ac.shift(1) - 1.0           # 跨日，复权
    for w in (5, 10, 21):
        out[f"sh_rev_intraday_{w}"] = -ts_sum(intraday, w)
    out["sh_rev_overnight_10"] = -ts_sum(overnight, 10)
    out["sh_intraday_minus_overnight_21"] = ts_sum(intraday, 21) - ts_sum(overnight, 21)

    # ---- A2 残差化反转/波动 ----
    for w in (5, 10, 21):
        out[f"sh_resid_rev_{w}"] = -ts_sum(rtilde, w)
    out["sh_resid_vol_10"] = ts_std(rtilde, 10)
    out["sh_resid_vol_21"] = ts_std(rtilde, 21)

    # ---- A3 量条件反转 ----
    for w in (10, 21):
        low_mask = (vol < ts_median(vol, w)).astype(float)
        out[f"sh_lowvol_rev_{w}"] = -ts_sum(ret * low_mask, w)
    high_mask = (vol >= ts_median(vol, 21)).astype(float)
    out["sh_highvol_mom_21"] = ts_sum(ret * high_mask, 21)
    out["sh_volwt_rev_21"] = -ts_sum(ret * (1.0 / (1.0 + vol / adv20)), 21)

    # ---- A4 日线代理已实现矩（残差，弱代理）----
    out["sh_rskew_21"] = ts_skew(rtilde, 21)
    out["sh_rkurt_21"] = ts_kurt(rtilde, 21)
    up = rtilde.clip(lower=0.0); dn = rtilde.clip(upper=0.0)
    out["sh_downsemivol_10"] = ts_std(dn, 10)
    out["sh_downsemivol_21"] = ts_std(dn, 21)
    out["sh_updown_vol_ratio_21"] = ts_std(up, 21) / (ts_std(dn, 21) + 1e-12)

    # ---- B WQ101 / GTJA191 代表子集 ----
    eps = 1e-3
    out["sh_wq101"] = (c - o) / ((h - l) + eps)
    out["sh_wq6_corr_open_vol_10"] = -ts_corr(ao, vol, 10)
    out["sh_wq12_volsign_rev"] = sign(delta(vol, 1)) * (-delta(ac, 1))
    out["sh_wq3_rkopen_rkvol_10"] = -ts_corr(cs_rank(ao), cs_rank(vol), 10)
    out["sh_wq4_tsrank_low_9"] = -ts_rank(cs_rank(l), 9)
    out["sh_wq42_vwap_close"] = cs_rank(vwap - c) / (cs_rank(vwap + c) + 1e-12)
    out["sh_wq44_corr_high_rkvol_5"] = -ts_corr(h, cs_rank(vol), 5)
    out["sh_wq53"] = -delta(((c - l) - (h - c)) / ((c - l) + eps), 9)
    out["sh_gtja18_mom5"] = ac / ac.shift(5) - 1.0
    out["sh_gtja31_close_ma12"] = (ac - ts_mean(ac, 12)) / (ts_mean(ac, 12) + 1e-12)
    up_vol = ts_sum((ac > ac.shift(1)).astype(float) * vol, 26)
    dn_vol = ts_sum((ac <= ac.shift(1)).astype(float) * vol, 26)
    out["sh_gtja40_updown_volratio"] = up_vol / (dn_vol + 1e-12)
    out["sh_gtja58_winrate20"] = ts_mean((ac > ac.shift(1)).astype(float), 20)

    # ---- B 补全（恢复已批准的 ~28 个；仅用可得字段/算子）----
    logv = np.log(vol.replace(0, np.nan))
    base = ac.where(ret >= 0, ts_std(ret, 20))            # WQ#1: returns<0?std:close
    out["sh_wq1_argmax"] = cs_rank(ts_argmax(signedpower(base, 2.0), 5)) - 0.5
    out["sh_wq2_corr_dvol_intra"] = -ts_corr(cs_rank(delta(logv, 2)), cs_rank((c - o) / o), 6)
    out["sh_wq5_vwap_open_close"] = cs_rank(o - ts_sum(vwap, 10) / 10) * (-cs_rank(c - vwap).abs())
    cond7 = (vol > adv20).astype(float)
    out["sh_wq7_volcond"] = (-ts_rank(delta(ac, 7).abs(), 60) * sign(delta(ac, 7))) * cond7 + (-1.0) * (1 - cond7)
    d1 = delta(ac, 1)
    rev9 = d1.where((ts_min(d1, 5) > 0) | (ts_max(d1, 5) < 0), -d1)   # WQ#9
    out["sh_wq9_cond_rev"] = rev9
    out["sh_wq11_vwap_close_vol"] = (cs_rank(ts_max(vwap - c, 3)) + cs_rank(ts_min(vwap - c, 3))) * cs_rank(delta(vol, 3))
    out["sh_wq13_cov_close_vol"] = -cs_rank(ts_cov(cs_rank(c), cs_rank(vol), 5))
    out["sh_wq14_dret_corr"] = (-cs_rank(delta(ret, 3))) * ts_corr(o, vol, 10)
    out["sh_wq15_corr_high_vol"] = -ts_sum(cs_rank(ts_corr(cs_rank(h), cs_rank(vol), 3)), 3)
    out["sh_wq16_cov_high_vol"] = -cs_rank(ts_cov(cs_rank(h), cs_rank(vol), 5))
    out["sh_wq18"] = -cs_rank(ts_std((c - o).abs(), 5) + (c - o) + ts_corr(c, o, 10))
    out["sh_wq22_corr_std"] = -(delta(ts_corr(h, vol, 5), 5) * cs_rank(ts_std(c, 20)))
    out["sh_wq25"] = cs_rank((-ret) * adv20 * vwap * (h - c))
    out["sh_wq41_hl_vwap"] = ((h * l) ** 0.5) - vwap
    out["sh_wq43_vol_close"] = ts_rank(vol / adv20, 20) * ts_rank(-delta(ac, 7), 8)
    out["sh_gtja46_ma_revert"] = (ts_mean(ac, 3) + ts_mean(ac, 6) + ts_mean(ac, 12) + ts_mean(ac, 24)) / (4 * ac)

    return out


SHORT_HORIZON_NAMES = None  # 由 build_short_horizon_factors 的 keys 决定（运行时）


def sample_at_asof(
    factor_wide: dict[str, pd.DataFrame], as_of_list, tickers=None
) -> pd.DataFrame:
    """把日频 wide 因子取到 (as_of, ticker) 网格 → MultiIndex DataFrame.

    每个 as_of（交易日）取该日的因子值（PIT：只用 ≤ as_of 数据算出的值）。
    """
    as_of_list = [pd.Timestamp(a).normalize() for a in as_of_list]
    rows = []
    for name, wide in factor_wide.items():
        w = wide.copy()
        w.index = pd.to_datetime(w.index).normalize()
        sub = w.reindex(as_of_list)            # 行 = as_of
        s = sub.stack(future_stack=True)       # 保留 NaN（等价旧 dropna=False）
        s.index = s.index.set_names(["as_of", "ticker"])
        rows.append(s.rename(name))
    panel = pd.concat(rows, axis=1)
    if tickers is not None:
        panel = panel[panel.index.get_level_values("ticker").isin(set(tickers))]
    return panel.sort_index()


__all__ = [
    "delay", "delta", "ts_sum", "ts_mean", "ts_std", "ts_median", "ts_min", "ts_max",
    "ts_skew", "ts_kurt", "ts_corr", "sign", "cs_rank", "ts_rank",
    "build_wide_frames", "residual_returns", "build_short_horizon_factors",
    "sample_at_asof",
]
