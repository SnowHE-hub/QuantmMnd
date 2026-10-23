"""quantmind.line_a.m_market — M 层：市场状态与风格监测（03_final_plan_v2 第 5 节 M）.

只做监测，不改变参考臂（参考臂固定满仓）；按状态调仓位、拥挤时降小盘配额，都是单独登记的挑战臂。
所有数据只用 as_of 当天及以前；每个来源都报告实际用到的最后日期（两融通常晚一天）。
- 趋势：沪深 300、中证 500、中证 1000、中证 2000 的 5 / 20 / 60 日涨跌，相对 20 / 60 日均线的位置；
- 广度：全市场上涨家数占比、收盘高于 20 / 60 日均线的占比、20 日新高与新低家数（复权价）；
- 波动：沪深 300 日收益的 HAR-RV（日 / 周 / 月三项，滚动 750 天最小二乘）给出下一日波动预测，及 20 日已实现波动的三年分位；
- 两融：全市场两融余额，5 / 20 日变化与三年分位；
- 北向：2024-08-19 起口径改变（configs/line_a_v2/source_semantics.yaml），只报告、不用于判断；
- 风格：中证 2000 / 沪深 300 的 20 / 60 日相对强弱及三年分位；成交额比（20 日均值）的三年分位作为拥挤度代理。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INDICES = {"000300.SH": "沪深300", "000905.SH": "中证500", "000852.SH": "中证1000", "932000.CSI": "中证2000"}
YEAR3 = 750


def _pct_rank(s: pd.Series, v: float) -> float | None:
    s = s.dropna()
    return None if s.empty or v != v else float((s <= v).mean())


def _indices(lake: Path, as_of: pd.Timestamp) -> pd.DataFrame:
    parts = [pd.read_parquet(lake / "index_daily.parquet", columns=["ts_code", "trade_date", "close", "amount"])]
    extra = lake / "index_daily_extra.parquet"
    if extra.exists():
        parts.append(pd.read_parquet(extra, columns=["ts_code", "trade_date", "close", "amount"]))
    df = pd.concat(parts, ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str).str.replace("-", "").str.slice(0, 8), format="%Y%m%d")
    df = df[df["trade_date"] <= as_of].drop_duplicates(["ts_code", "trade_date"])
    return df.pivot(index="trade_date", columns="ts_code").sort_index()


def har_forecast(ret: pd.Series, window: int = YEAR3) -> dict:
    """HAR-RV（Corsi 2009）的日线代理：RV_t = r_t²；RV_{t+1} = b0 + b1·RV_t + b2·mean5 + b3·mean22。"""
    rv = ret.pow(2)
    X = pd.DataFrame({"d": rv, "w": rv.rolling(5).mean(), "m": rv.rolling(22).mean()})
    y = rv.shift(-1)
    D = pd.concat([X, y.rename("y")], axis=1).dropna().tail(window)
    if len(D) < 100:
        return {"status": "insufficient_history"}
    A = np.column_stack([np.ones(len(D)), D[["d", "w", "m"]].to_numpy()])
    beta, *_ = np.linalg.lstsq(A, D["y"].to_numpy(), rcond=None)
    last = X.iloc[-1]
    f = float(beta[0] + beta[1] * last["d"] + beta[2] * last["w"] + beta[3] * last["m"])
    rv20 = float(np.sqrt(ret.tail(20).pow(2).mean() * 252))
    hist = np.sqrt(ret.pow(2).rolling(20).mean() * 252).tail(YEAR3)
    return {"next_day_vol_ann": float(np.sqrt(max(f, 0.0) * 252)), "realized_vol_20d_ann": rv20,
            "realized_vol_20d_pct3y": _pct_rank(hist, rv20), "coef": [float(b) for b in beta], "n_fit": int(len(D))}


def _breadth(prices_path: Path, as_of: pd.Timestamp) -> dict:
    lo = as_of - pd.Timedelta(days=140)
    px = pd.read_parquet(prices_path, columns=["ts_code", "trade_date", "close", "adj_factor"],
                         filters=[("trade_date", ">=", lo), ("trade_date", "<=", as_of)])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    w = (px["close"] * px["adj_factor"]).groupby([px["trade_date"], px["ts_code"]]).last().unstack()
    w = w.sort_index()
    if w.empty or w.index[-1] != as_of:
        return {"status": "no_bar_on_as_of", "last_date": str(w.index[-1].date()) if len(w) else None}
    last, prev = w.iloc[-1], w.iloc[-2]
    ma20, ma60 = w.tail(20).mean(), w.tail(60).mean()
    hi20, lo20 = w.tail(20).max(), w.tail(20).min()
    ok = last.notna() & prev.notna()
    return {"n": int(ok.sum()), "up_share": float((last[ok] > prev[ok]).mean()),
            "above_ma20_share": float((last > ma20)[last.notna()].mean()),
            "above_ma60_share": float((last > ma60)[last.notna() & (w.tail(60).count() >= 40)].mean()),
            "new_high_20d": int((last >= hi20)[last.notna()].sum()), "new_low_20d": int((last <= lo20)[last.notna()].sum())}


def _margin(lake: Path, as_of: pd.Timestamp) -> dict:
    m = pd.read_parquet(lake / "margin.parquet", columns=["trade_date", "rzrqye"])
    m["trade_date"] = pd.to_datetime(m["trade_date"])
    s = m[m["trade_date"] <= as_of].groupby("trade_date")["rzrqye"].sum().sort_index()
    if s.empty:
        return {"status": "source_absent"}
    v = float(s.iloc[-1])
    return {"last_date": str(s.index[-1].date()), "balance_yuan": v,
            "chg_5d": float(v / s.iloc[-6] - 1) if len(s) > 5 else None,
            "chg_20d": float(v / s.iloc[-21] - 1) if len(s) > 20 else None,
            "pct3y": _pct_rank(s.tail(YEAR3), v)}


def market_state(data_root: Path, as_of, *, prices_path: Path | None = None) -> dict:
    D = Path(data_root) / "data"
    lake = D / "lake"
    d = pd.Timestamp(as_of)
    W = _indices(lake, d)
    out: dict = {"as_of": str(d.date()), "trend": {}, "style": {}}
    close, amount = W["close"], W["amount"]
    for code, name in INDICES.items():
        if code not in close.columns:
            out["trend"][name] = {"status": "source_absent"}
            continue
        c = close[code].dropna()
        out["trend"][name] = {"last_date": str(c.index[-1].date()), "close": float(c.iloc[-1]),
                              **{f"ret_{k}d": float(c.iloc[-1] / c.iloc[-1 - k] - 1) for k in (5, 20, 60)},
                              "vs_ma20": float(c.iloc[-1] / c.tail(20).mean() - 1),
                              "vs_ma60": float(c.iloc[-1] / c.tail(60).mean() - 1)}
    r300 = np.log(close["000300.SH"]).diff().dropna()
    out["volatility_hs300"] = har_forecast(r300)
    if "932000.CSI" in close.columns:
        ratio = (close["932000.CSI"] / close["000300.SH"]).dropna()
        rs60 = ratio / ratio.shift(60) - 1
        amt = (amount["932000.CSI"] / amount["000300.SH"]).rolling(20).mean().dropna()
        out["style"] = {"csi2000_vs_hs300_rs20": float(ratio.iloc[-1] / ratio.iloc[-21] - 1),
                        "csi2000_vs_hs300_rs60": float(rs60.iloc[-1]),
                        "rs60_pct3y": _pct_rank(rs60.tail(YEAR3), float(rs60.iloc[-1])),
                        "amount_ratio_20d": float(amt.iloc[-1]),
                        "amount_ratio_pct3y": _pct_rank(amt.tail(YEAR3), float(amt.iloc[-1]))}
    if prices_path is None:
        cands = sorted((D / "raw").glob("alpha_prices_panel_v6_2*.parquet"))
        prices_path = cands[-1] if cands else None
    out["breadth"] = _breadth(prices_path, d) if prices_path else {"status": "source_absent"}
    out["margin"] = _margin(lake, d)
    nb = pd.read_parquet(lake / "north_bound.parquet", columns=["trade_date", "north_money"])
    nb["trade_date"] = pd.to_datetime(nb["trade_date"])
    nb = nb[nb["trade_date"] <= d].sort_values("trade_date")
    out["northbound"] = {"status": "broken_series_audit_only", "note": "2024-08-19 起口径改变，只报告、不用于状态判断",
                         "last_date": str(nb["trade_date"].iloc[-1].date()) if len(nb) else None,
                         "north_money": float(v) if len(nb) and (v := nb["north_money"].iloc[-1]) is not None
                         and v == v else None}
    return out
