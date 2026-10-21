"""quantmind.line_a.risk_card — F5(c) 风险与执行卡（03_final_plan_v2 第 5 节 F5(c)；代码计算，不用语言模型）.

对宽候选与持仓逐只计算，结果是**事实**，不默认扣分：
- 风险暴露：对数总市值、252 日 beta（对沪深 300）、60 日年化波动、20 / 60 / 250 日动量、20 日日均成交额、行业
  （第一版用 Tushare 行业分类 industry_tushare；方案要求的申万一级按发布日口径待接入）；
- 预测波动：个股日收益的 HAR-RV 下一日年化波动（与 M 层同一函数）；
- 成交风险：
  - 250 日隔夜跳空 |开盘 / 前收 − 1| 的 50% 与 95% 分位；
  - 250 日收盘封涨停、封跌停的天数；
  - 复牌以来的交易日数；
  - as_of 之后 20 个自然日内、已公告的除权除息日；
  - 到异常波动认定阈值的距离：3 / 10 / 30 个交易日的累计偏离值与阈值之差。阈值取自 configs/line_a_v2/trading_rules.yaml，状态都是"待核实"，基准指数用近似（上证综指、创业板指，深市主板用沪深 300 代替）。
每个字段都有证据编号 `calc:risk_card:<股票>:<as_of>`，校验器可以据此回到这张卡。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quantmind.backtest.wf_costs import board_of
from quantmind.line_a.m_market import har_forecast

# (窗口交易日, 上行阈值, 下行阈值)；主板 3 日 ±20%，创业板 / 科创板 3 日 ±30%（待核实）
AV_RULES = {"3d": (3, 0.20, -0.20), "10d": (10, 1.00, -0.50), "30d": (30, 2.00, -0.70)}
AV_3D_GEM = (3, 0.30, -0.30)
BENCH = {"SH": "000001.SH", "CHINEXT": "399006.SZ", "SZ": "000300.SH"}


def _bench_code(t: str) -> str:
    b = board_of(t)
    if b == "CHINEXT":
        return BENCH["CHINEXT"]
    return BENCH["SH"] if t.endswith(".SH") else BENCH["SZ"]


def risk_cards(data_root: Path, as_of, tickers: list[str], *, prices_path: Path | None = None) -> pd.DataFrame:
    D = Path(data_root) / "data"
    lake = D / "lake"
    d = pd.Timestamp(as_of)
    if prices_path is None:
        prices_path = sorted((D / "raw").glob("alpha_prices_panel_v6_2*.parquet"))[-1]
    lo = d - pd.Timedelta(days=420)
    px = pd.read_parquet(prices_path, columns=["ts_code", "trade_date", "open", "close", "pre_close", "adj_factor",
                                               "amount"],
                         filters=[("trade_date", ">=", lo), ("trade_date", "<=", d), ("ts_code", "in", list(tickers))])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px.sort_values(["ts_code", "trade_date"])
    px["adj"] = px["close"] * px["adj_factor"]
    px["ret"] = px.groupby("ts_code")["adj"].pct_change(fill_method=None)
    px["gap"] = (px["open"] / px["pre_close"] - 1).abs()
    idx = pd.read_parquet(lake / "index_daily.parquet", columns=["ts_code", "trade_date", "close"])
    idx["trade_date"] = pd.to_datetime(idx["trade_date"])
    ir = idx[idx["trade_date"] <= d].pivot(index="trade_date", columns="ts_code", values="close").sort_index().pct_change()
    lim = pd.read_parquet(sorted((lake / "events").glob("stk_limit_2*.parquet"))[-1],
                          columns=["ts_code", "trade_date", "up_limit", "down_limit"],
                          filters=[("ts_code", "in", list(tickers))])
    lim["trade_date"] = pd.to_datetime(lim["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    px = px.merge(lim, on=["ts_code", "trade_date"], how="left")
    sus = pd.read_parquet(sorted((lake / "events").glob("suspend_d_2*.parquet"))[-1],
                          filters=[("ts_code", "in", list(tickers))])
    sus["trade_date"] = pd.to_datetime(sus["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    resumed = sus[(sus["suspend_type"].astype(str) == "R") & (sus["trade_date"] <= d)].groupby("ts_code")["trade_date"].max()
    divp = sorted((lake / "events").glob("dividend_2*.parquet"))
    div = pd.read_parquet(divp[-1], filters=[("ts_code", "in", list(tickers))]) if divp else pd.DataFrame()
    db_dates = pd.read_parquet(lake / "daily_basic.parquet", columns=["trade_date"],
                               filters=[("trade_date", "<=", d), ("trade_date", ">=", d - pd.Timedelta(days=10))])
    db_last = pd.to_datetime(db_dates["trade_date"]).max()
    db = pd.read_parquet(lake / "daily_basic.parquet", columns=["ts_code", "trade_date", "total_mv"],
                         filters=[("trade_date", "==", db_last), ("ts_code", "in", list(tickers))]).set_index("ts_code")
    sb = pd.read_parquet(sorted(lake.glob("stock_basic_full_merged_2*.parquet"))[-1], columns=["ts_code", "industry"])
    ind = sb.drop_duplicates("ts_code").set_index("ts_code")["industry"]
    rows = []
    for t, g in px.groupby("ts_code"):
        g = g.set_index("trade_date")
        r = g["ret"]
        last = g.tail(250)
        bench = ir[_bench_code(t)].reindex(g.index) if _bench_code(t) in ir.columns else pd.Series(np.nan, index=g.index)
        dev = (r - bench).dropna()
        b300 = ir["000300.SH"].reindex(g.index)
        j = pd.concat([r, b300], axis=1).dropna().tail(252)
        beta = float(np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1] / np.var(j.iloc[:, 1], ddof=1)) if len(j) > 60 else None
        mom = {f"mom_{k}": float(g["adj"].iloc[-1] / g["adj"].iloc[-1 - k] - 1) if len(g) > k else None for k in (20, 60, 250)}
        rec: dict = {"ticker": t, "as_of": str(d.date()), "evidence_id": f"calc:risk_card:{t}:{d.date()}",
                     "last_bar": str(g.index[-1].date()), "industry_tushare": ind.get(t),
                     "log_total_mv": float(np.log(db["total_mv"].get(t))) if t in db.index and db["total_mv"].get(t) else None,
                     "beta_252": beta, "vol_60_ann": float(r.tail(60).std() * np.sqrt(252)) if r.tail(60).count() > 30 else None,
                     **mom, "adv20_yuan": float(g["amount"].tail(20).mean() * 1000),
                     "gap_p50_250": float(last["gap"].quantile(0.5)), "gap_p95_250": float(last["gap"].quantile(0.95)),
                     "limit_up_close_250": int((last["close"] >= last["up_limit"] - 1e-9).sum()),
                     "limit_down_close_250": int((last["close"] <= last["down_limit"] + 1e-9).sum())}
        h = har_forecast(r.dropna().tail(800))
        rec["har_next_day_vol_ann"] = h.get("next_day_vol_ann")
        rd = resumed.get(t)
        rec["days_since_resumption"] = int((g.index > rd).sum()) if rd is not None and rd >= g.index[0] else None
        gem = board_of(t) in ("CHINEXT", "STAR")
        for k, (n, up, dn) in AV_RULES.items():
            if k == "3d" and gem:
                n, up, dn = AV_3D_GEM
            cum = float(dev.tail(n).sum()) if len(dev) >= n else None
            rec[f"av_{k}_cum_dev"] = cum
            rec[f"av_{k}_to_up"] = None if cum is None else up - cum
            rec[f"av_{k}_to_down"] = None if cum is None else cum - dn
        if len(div):
            dd = div[(div["ts_code"] == t) & (div["ann_date"].astype(str) <= d.strftime("%Y%m%d"))].copy()
            ex = pd.to_datetime(dd.get("ex_date", pd.Series(dtype=str)).astype(str), format="%Y%m%d", errors="coerce")
            soon = ex[(ex > d) & (ex <= d + pd.Timedelta(days=20))]
            rec["ex_dividend_next20d"] = sorted({str(x.date()) for x in soon})
        rec["rules_status"] = "abnormal-volatility thresholds and benchmark mapping: verify"
        rows.append(rec)
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()
