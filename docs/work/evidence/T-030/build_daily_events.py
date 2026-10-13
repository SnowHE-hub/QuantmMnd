"""T-030 P0：T-028 的 11 个事件因子在日频网格上的取值.

来源：t/028-events 分支 docs/work/evidence/T-028/build_event_factors.py（commit 94856145）。辅助函数逐字复制；
唯一改动是网格 = 日频面板索引（data/gpu_v1/panel/base），按月计算、按月写出（断点续跑）。
各因子在某个 (as_of, ticker) 上的取值只依赖该行自身与 < as_of 的事件 / 交易日数据；缩尾按 as_of 在面板行上做，
因此与周频网格逐值相同（由 p0_consistency.py 核对）。交易类的宽表滚动在全部交易日上一次算好。
输出 <data_root>/data/gpu_v1/panel/event/<YYYY-MM>.parquet；已存在的月份跳过，从不覆盖。
用法：python build_daily_events.py <data_root>
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

EV_END = "20261009"
POS_TYPES = {"预增", "扭亏", "续盈", "略增"}
NEG_TYPES = {"预减", "首亏", "续亏", "略减"}
CONTINUOUS = ["fc_pchg_mid", "express_np_growth", "holder_num_chg", "insider_net_ratio_90d", "unlock_ratio_next60",
              "inst_net_buy_20d", "block_discount_20d", "mf_lg_net_20d"]
FACTORS = ["fc_pchg_mid", "fc_type_score", "express_np_growth", "holder_num_chg", "insider_net_ratio_90d",
           "repurchase_flag_90d", "unlock_ratio_next60", "inst_net_buy_20d", "block_discount_20d", "mf_lg_net_20d",
           "limit_up_cnt_20d"]
WIN, MIN_OBS = 20, 15


def ymd(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str), format="%Y%m%d", errors="coerce")


def latest_before(grid: pd.DataFrame, ev: pd.DataFrame, cols: list[str], max_age_days: int,
                  order: list[str] | None = None) -> pd.DataFrame:
    """每个 (as_of, ticker) 取 date < as_of 的最近一条事件；超过 max_age_days 视为缺失."""
    ev = ev.dropna(subset=["date"]).sort_values(["date"] + (order or []), kind="mergesort")
    m = pd.merge_asof(grid.sort_values("as_of"), ev[["ticker", "date"] + cols], left_on="as_of", right_on="date",
                      by="ticker", allow_exact_matches=False, direction="backward")
    stale = ~((m["as_of"] - m["date"]).dt.days <= max_age_days)
    m.loc[stale, cols] = np.nan
    return m.set_index(["as_of", "ticker"])[cols]


def window_sum(grid: pd.DataFrame, ev: pd.DataFrame, col: str, days: int) -> pd.Series:
    """Σ col over events with date ∈ [as_of − days, as_of)；无事件 → 0."""
    e = ev.dropna(subset=["date"]).groupby(["ticker", "date"], as_index=False)[col].sum().sort_values("date")
    e["cum"] = e.groupby("ticker")[col].cumsum()
    g = grid.sort_values("as_of")
    hi = pd.merge_asof(g, e[["ticker", "date", "cum"]], left_on="as_of", right_on="date", by="ticker",
                       allow_exact_matches=False).set_index(["as_of", "ticker"])["cum"]
    g0 = g.assign(t0=g["as_of"] - pd.Timedelta(days=days)).sort_values("t0")
    lo = pd.merge_asof(g0, e[["ticker", "date", "cum"]], left_on="t0", right_on="date", by="ticker",
                       allow_exact_matches=False).set_index(["as_of", "ticker"])["cum"]
    return hi.fillna(0.0) - lo.reindex(hi.index).fillna(0.0)


def roll_before(wide: pd.DataFrame, as_of: pd.DatetimeIndex) -> pd.DataFrame:
    """前 WIN 个交易日（不含当天）的滚动和，取到 as_of 行."""
    r = wide.rolling(WIN, min_periods=MIN_OBS).sum().shift(1)
    return r.reindex(as_of)


def stack(w: pd.DataFrame, name: str) -> pd.Series:
    s = w.stack(future_stack=True)
    s.index = s.index.set_names(["as_of", "ticker"])
    return s.rename(name)


def winsorize(s: pd.Series) -> pd.Series:
    g = s.groupby(level="as_of")
    return s.clip(g.transform(lambda x: x.quantile(0.01)), g.transform(lambda x: x.quantile(0.99)))


def load(root: Path, tickers: list[str], as_of: pd.DatetimeIndex) -> dict:
    D = root / "data"
    E = D / "lake/events"
    S: dict = {}
    fc = pd.read_parquet(E / f"forecast_{EV_END}.parquet")
    S["fc"] = fc.assign(ticker=fc["ts_code"].astype(str), date=ymd(fc["ann_date"]),
                        pchg=fc[["p_change_min", "p_change_max"]].mean(axis=1),
                        tscore=fc["type"].astype(str).map(
                            lambda x: 1.0 if x in POS_TYPES else (-1.0 if x in NEG_TYPES else 0.0)),
                        uflag=pd.to_numeric(fc["update_flag"], errors="coerce").fillna(0))
    ex = pd.read_parquet(E / f"express_{EV_END}.parquet")
    base = ex["yoy_net_profit"].where(ex["yoy_net_profit"].abs() > 0)
    S["ex"] = ex.assign(ticker=ex["ts_code"].astype(str), date=ymd(ex["ann_date"]),
                        g=((ex["n_income"] - base) / base.abs()).clip(-5, 5),
                        uflag=pd.to_numeric(ex["update_flag"], errors="coerce").fillna(0))
    hn = pd.read_parquet(E / f"stk_holdernumber_{EV_END}.parquet")
    hn = hn.assign(ticker=hn["ts_code"].astype(str), date=ymd(hn["ann_date"]), end=ymd(hn["end_date"]))
    hn = hn.dropna(subset=["date", "end", "holder_num"])
    hn = hn[hn["holder_num"] > 0].sort_values(["ticker", "end", "date"]).drop_duplicates(["ticker", "end"], keep="first")
    hn = hn.sort_values(["ticker", "date", "end"])
    prev_h = hn.groupby("ticker")["holder_num"].shift(1)
    prev_end = hn.groupby("ticker")["end"].shift(1)
    hn["chg"] = np.where(prev_end < hn["end"], np.log(hn["holder_num"] / prev_h), np.nan)
    S["hn"] = hn
    ht = pd.read_parquet(E / f"stk_holdertrade_{EV_END}.parquet")
    sign = ht["in_de"].astype(str).map({"IN": 1.0, "DE": -1.0})
    S["ht"] = ht.assign(ticker=ht["ts_code"].astype(str), date=ymd(ht["ann_date"]),
                        v=sign * ht["change_ratio"]).dropna(subset=["v"])
    rp = pd.read_parquet(E / f"repurchase_{EV_END}.parquet")
    S["rp"] = rp.assign(ticker=rp["ts_code"].astype(str), date=ymd(rp["ann_date"]), one=1.0)
    sf = pd.read_parquet(E / f"share_float_{EV_END}.parquet",
                         columns=["ts_code", "ann_date", "float_date", "float_ratio"])
    sf = (sf.assign(ticker=sf["ts_code"].astype(str), ann=ymd(sf["ann_date"]), fl=ymd(sf["float_date"]))
          .dropna(subset=["ann", "fl", "float_ratio"]).groupby(["ticker", "ann", "fl"], as_index=False)["float_ratio"].sum())
    S["sf"] = (sf["ticker"].to_numpy(), sf["ann"].to_numpy(), sf["fl"].to_numpy(), sf["float_ratio"].to_numpy())

    daily = pd.read_parquet(D / "lake/daily.parquet", columns=["ts_code", "trade_date", "close", "amount"])
    daily = daily[(daily["trade_date"] >= "2018-11-01") & daily["ts_code"].isin(tickers)]
    close = daily.pivot(index="trade_date", columns="ts_code", values="close").sort_index()
    amount = daily.pivot(index="trade_date", columns="ts_code", values="amount").reindex_like(close) * 1000.0
    del daily
    days = close.index
    traded = close.notna().astype(float)
    W: dict = {}
    ti = pd.read_parquet(E / f"top_inst_{EV_END}.parquet")
    ti = ti[ti["exalter"].astype(str) == "机构专用"]
    ti = ti.drop_duplicates(["trade_date", "ts_code", "exalter", "buy", "sell"])
    ti = ti.assign(d=ymd(ti["trade_date"]), net=ti["net_buy"].fillna(ti["buy"].fillna(0) - ti["sell"].fillna(0)))
    net = ti.groupby(["d", "ts_code"])["net"].sum().unstack().reindex(index=days, columns=close.columns).fillna(0.0)
    net = net.where(traded > 0)
    W["inst_net_buy_20d"] = roll_before(net, as_of) / roll_before(amount, as_of)
    del net, ti

    bt = pd.read_parquet(E / f"block_trade_{EV_END}.parquet", columns=["ts_code", "trade_date", "price", "amount"])
    bt = bt.assign(d=ymd(bt["trade_date"]))
    cl = close.stack().rename("close").reset_index().rename(columns={"trade_date": "d"})
    bt = bt.merge(cl, on=["d", "ts_code"], how="inner")
    del cl
    bt = bt[(bt["close"] > 0) & (bt["amount"] > 0)]
    bt["w_disc"] = bt["amount"] * (bt["price"] / bt["close"] - 1.0)
    num = bt.groupby(["d", "ts_code"])["w_disc"].sum().unstack().reindex(index=days, columns=close.columns).fillna(0.0)
    den = bt.groupby(["d", "ts_code"])["amount"].sum().unstack().reindex(index=days, columns=close.columns).fillna(0.0)
    dn = den.shift(1).rolling(WIN, min_periods=1).sum().reindex(as_of)
    W["block_discount_20d"] = (num.shift(1).rolling(WIN, min_periods=1).sum().reindex(as_of) / dn).where(dn > 0)
    del bt, num, den, dn

    mf = pd.read_parquet(E / f"moneyflow_{EV_END}.parquet")
    mf = mf[mf["ts_code"].isin(tickers)].assign(d=lambda x: ymd(x["trade_date"]))
    amt_cols = [c for c in mf.columns if c.endswith("_amount") and c.startswith(("buy_", "sell_"))]
    mf["tot"] = mf[amt_cols].sum(axis=1)
    mf["lg"] = mf["buy_lg_amount"] + mf["buy_elg_amount"] - mf["sell_lg_amount"] - mf["sell_elg_amount"]
    lg = mf.pivot_table(index="d", columns="ts_code", values="lg", aggfunc="sum").reindex(index=days, columns=close.columns)
    tot = mf.pivot_table(index="d", columns="ts_code", values="tot", aggfunc="sum").reindex(index=days, columns=close.columns)
    del mf
    tr = roll_before(tot, as_of)
    W["mf_lg_net_20d"] = (roll_before(lg, as_of) / tr).where(tr > 0)
    del lg, tot, tr

    sl = pd.read_parquet(E / f"stk_limit_{EV_END}.parquet", columns=["ts_code", "trade_date", "up_limit"])
    sl = sl[sl["ts_code"].isin(tickers)].assign(d=lambda x: ymd(x["trade_date"]))
    up = sl.pivot_table(index="d", columns="ts_code", values="up_limit", aggfunc="last").reindex(index=days, columns=close.columns)
    hit = ((close >= up - 0.01) & up.notna() & close.notna()).astype(float).where(traded > 0)
    W["limit_up_cnt_20d"] = roll_before(hit, as_of)
    S["W"] = W
    return S


def one_month(S: dict, idx: pd.MultiIndex) -> pd.DataFrame:
    grid = idx.to_frame(index=False)
    grid["as_of"] = pd.to_datetime(grid["as_of"])
    as_of = pd.DatetimeIndex(sorted(grid["as_of"].unique()))
    res: dict[str, pd.Series] = {}
    x = latest_before(grid, S["fc"], ["pchg", "tscore"], 120, order=["uflag"])
    res["fc_pchg_mid"], res["fc_type_score"] = x["pchg"], x["tscore"]
    res["express_np_growth"] = latest_before(grid, S["ex"], ["g"], 120, order=["uflag"])["g"]
    res["holder_num_chg"] = latest_before(grid, S["hn"], ["chg"], 180)["chg"]
    res["insider_net_ratio_90d"] = window_sum(grid, S["ht"], "v", 90)
    res["repurchase_flag_90d"] = (window_sum(grid, S["rp"], "one", 90) > 0).astype(float)
    tk, ann, fl, fr = S["sf"]
    parts = []
    for t in as_of:
        tv = np.datetime64(t)
        m = (ann < tv) & (fl > tv) & (fl <= tv + np.timedelta64(60, "D"))
        if m.any():
            s = pd.Series(fr[m]).groupby(tk[m]).sum()
            parts.append(pd.DataFrame({"as_of": t, "ticker": s.index, "u": s.values}))
    un = pd.concat(parts).set_index(["as_of", "ticker"])["u"] if parts else pd.Series(dtype=float)
    res["unlock_ratio_next60"] = un.reindex(idx).fillna(0.0)
    W = S["W"]
    res["inst_net_buy_20d"] = stack(W["inst_net_buy_20d"].reindex(as_of), "inst").reindex(idx)
    res["block_discount_20d"] = stack(W["block_discount_20d"].reindex(as_of), "block").reindex(idx)
    res["mf_lg_net_20d"] = stack(W["mf_lg_net_20d"].reindex(as_of), "mf").reindex(idx)
    res["limit_up_cnt_20d"] = stack(W["limit_up_cnt_20d"].reindex(as_of), "lu").reindex(idx).fillna(0.0)
    df = pd.DataFrame({k: res[k].reindex(idx) for k in FACTORS}, index=idx)
    for c in CONTINUOUS:
        df[c] = winsorize(df[c])
    return df


def main() -> None:
    root = Path(sys.argv[1])
    t0 = time.time()
    bdir = root / "data/gpu_v1/panel/base"
    out_dir = root / "data/gpu_v1/panel/event"
    out_dir.mkdir(parents=True, exist_ok=True)
    months = sorted(p.stem for p in bdir.glob("*.parquet"))
    idxs = {m: pd.read_parquet(bdir / f"{m}.parquet", columns=["momentum_1m"]).index for m in months}
    tickers = sorted(set().union(*[set(i.get_level_values("ticker")) for i in idxs.values()]))
    all_asof = pd.DatetimeIndex(sorted(set().union(*[set(i.get_level_values("as_of")) for i in idxs.values()])))
    todo = [m for m in months if not (out_dir / f"{m}.parquet").exists()]
    print(f"[event] {len(months)} base months, {len(todo)} to do, {len(tickers)} tickers", flush=True)
    if not todo:
        return
    S = load(root, tickers, all_asof)
    print(f"[event] loaded ({time.time() - t0:.0f}s)", flush=True)
    for m in todo:
        df = one_month(S, idxs[m])
        out = out_dir / f"{m}.parquet"
        tmp = out.with_suffix(".tmp")
        df.to_parquet(tmp)
        os.replace(tmp, out)
        print(f"  {m} {df.shape} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[event] done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
