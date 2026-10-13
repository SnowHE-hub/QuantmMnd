"""T-030 P0：F6 逐日序列通道（稠密数组 seq[stock, day, channel]）与 F7 市场状态（逐日一行）.

PIT：第 t 天的每个值只用 ≤ t 收盘可得的数据；两融余额晚一天公布，F7 里一律取 t−1。
F6 通道（原始值，float32，缺失 = NaN；归一化在训练时只用该折训练日的统计量）：
  0 r_cc     log(复权收盘_t / 前一个有 bar 日的复权收盘)          8 turn_f   log1p(turnover_rate_f)
  1 r_co     log(复权开盘_t / 前一个有 bar 日的复权收盘)          9 log_cmv  log(circ_mv)
  2 r_oc     log(close_t / open_t)                               10 ex300    r_cc − log(沪深300_t / 沪深300_{t−1})
  3 hl       log(high / low)                                     11 ex500    r_cc − log(中证500_t / 中证500_{t−1})
  4 up_sh    log(high / max(open, close))                        12 vwap_gap log(vwap / close)，vwap = amount·10 / vol
  5 lo_sh    log(min(open, close) / low)                         13 traded   当天有 bar = 1，否则 0（非 NaN）
  6 log_amt  log1p(amount)                                       14 amt_rank 当日成交额截面分位（有 bar 的股票内）
  7 log_vol  log1p(vol)                                          15 cmv_rank 当日流通市值截面分位
F7（data/gpu_v1/panel/market/market_daily.parquet，index = trade_date）：沪深300/中证500/创业板指的 1/5/20/60 日对数收益，
  沪深300/中证500 的 20 日年化波动与 252 日高点回撤，上涨家数占比（当日、5 日均），截面收益离散度（当日、20 日均），
  全市场成交额相对 20 日均值的对数比，两融余额（t−1）20 日对数变化。
输出（拒绝覆盖）：data/gpu_v1/seq/{seq_raw_f32.npy, meta.json}、data/gpu_v1/panel/market/market_daily.parquet。
用法：python build_daily_seq.py <data_root>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = ["r_cc", "r_co", "r_oc", "hl", "up_sh", "lo_sh", "log_amt", "log_vol", "turn_f", "log_cmv",
            "ex300", "ex500", "vwap_gap", "traded", "amt_rank", "cmv_rank"]


def main() -> None:
    root = Path(sys.argv[1])
    D = root / "data"
    seq_dir, mkt_dir = D / "gpu_v1/seq", D / "gpu_v1/panel/market"
    seq_path, mkt_path = seq_dir / "seq_raw_f32.npy", mkt_dir / "market_daily.parquet"
    if seq_path.exists() or mkt_path.exists():
        raise SystemExit(f"refusing to overwrite {seq_path} / {mkt_path}")
    t0 = time.time()
    cols = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "adj_factor"]
    px = pd.read_parquet(D / "raw/alpha_prices_panel_v6_20261009.parquet", columns=cols)
    px = px.drop_duplicates(["trade_date", "ts_code"], keep="last")
    cal = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(D / "lake/trade_cal_sse.parquet")["cal_date"])).sort_values()
    days = cal[(cal >= px["trade_date"].min()) & (cal <= px["trade_date"].max())]
    assert set(days) == set(px["trade_date"].unique()), "price dates must equal the trade calendar"
    tickers = sorted(px["ts_code"].unique())

    def wide(df, col):
        return df.pivot(index="trade_date", columns="ts_code", values=col).reindex(index=days, columns=tickers)

    o, h, lo, c, vol, amt, af = (wide(px, k) for k in ("open", "high", "low", "close", "vol", "amount", "adj_factor"))
    del px
    adjc = c * af
    prev = adjc.ffill().shift(1)
    traded = c.notna()
    db = pd.read_parquet(D / "lake/daily_basic.parquet", columns=["ts_code", "trade_date", "turnover_rate_f", "circ_mv"])
    db["trade_date"] = pd.to_datetime(db["trade_date"])
    db = db.drop_duplicates(["trade_date", "ts_code"], keep="last")
    turn, cmv = wide(db, "turnover_rate_f"), wide(db, "circ_mv")
    del db
    ix = pd.read_parquet(D / "lake/index_daily.parquet", columns=["ts_code", "trade_date", "close"])
    ix["trade_date"] = pd.to_datetime(ix["trade_date"])
    ixc = ix.pivot(index="trade_date", columns="ts_code", values="close").reindex(days)
    ixr = np.log(ixc / ixc.shift(1))
    with np.errstate(divide="ignore", invalid="ignore"):
        r_cc = np.log(adjc / prev)
        ch = {"r_cc": r_cc, "r_co": np.log(o * af / prev), "r_oc": np.log(c / o), "hl": np.log(h / lo),
              "up_sh": np.log(h / np.maximum(o, c)), "lo_sh": np.log(np.minimum(o, c) / lo),
              "log_amt": np.log1p(amt), "log_vol": np.log1p(vol), "turn_f": np.log1p(turn).where(traded),
              "log_cmv": np.log(cmv).where(traded),
              "ex300": r_cc.sub(ixr["000300.SH"], axis=0), "ex500": r_cc.sub(ixr["000905.SH"], axis=0),
              "vwap_gap": np.log((amt * 10.0 / vol.where(vol > 0)) / c),
              "traded": traded.astype(float),
              "amt_rank": amt.where(traded).rank(axis=1, pct=True),
              "cmv_rank": cmv.where(traded).rank(axis=1, pct=True)}
    seq_dir.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(seq_path.with_suffix(".tmp.npy"), mode="w+", dtype=np.float32,
                                    shape=(len(tickers), len(days), len(CHANNELS)))
    for j, name in enumerate(CHANNELS):
        v = ch[name].to_numpy(dtype=np.float64)
        v[~np.isfinite(v)] = np.nan
        arr[:, :, j] = v.T.astype(np.float32)
    arr.flush()
    del arr
    seq_path.with_suffix(".tmp.npy").rename(seq_path)
    nan_share = {k: round(float(ch[k].isna().to_numpy().mean()), 4) for k in CHANNELS}

    # ── F7 ───────────────────────────────────────────────────────────────
    m = pd.DataFrame(index=days)
    for code, tag in (("000300.SH", "csi300"), ("000905.SH", "csi500"), ("399006.SZ", "chinext")):
        lc = np.log(ixc[code])
        for k in (1, 5, 20, 60):
            m[f"{tag}_ret_{k}"] = lc - lc.shift(k)
    for code, tag in (("000300.SH", "csi300"), ("000905.SH", "csi500")):
        m[f"{tag}_vol_20"] = ixr[code].rolling(20, min_periods=15).std() * np.sqrt(252)
        m[f"{tag}_dd_252"] = ixc[code] / ixc[code].rolling(252, min_periods=60).max() - 1.0
    rr = r_cc.where(traded)
    m["breadth_1"] = (rr > 0).sum(axis=1) / rr.notna().sum(axis=1)
    m["breadth_5"] = m["breadth_1"].rolling(5, min_periods=3).mean()
    m["disp_1"] = rr.std(axis=1)
    m["disp_20"] = m["disp_1"].rolling(20, min_periods=15).mean()
    tot = amt.sum(axis=1, min_count=1)
    m["amt_rel_20"] = np.log(tot / tot.rolling(20, min_periods=15).mean())
    mg = pd.read_parquet(D / "lake/margin.parquet", columns=["trade_date", "rzye"])
    mg["trade_date"] = pd.to_datetime(mg["trade_date"])
    rz = mg.groupby("trade_date")["rzye"].sum().reindex(days).shift(1)      # t−1：两融数据次日公布
    m["margin_chg_20"] = np.log(rz / rz.shift(20))
    m.index.name = "trade_date"
    mkt_dir.mkdir(parents=True, exist_ok=True)
    m.to_parquet(mkt_path)
    meta = {"shape": [len(tickers), len(days), len(CHANNELS)], "dtype": "float32", "channels": CHANNELS,
            "tickers": tickers, "days": [str(d.date()) for d in days], "nan_share": nan_share,
            "market_columns": list(m.columns), "market_nan_share": m.isna().mean().round(4).to_dict(),
            "seconds": round(time.time() - t0, 1)}
    (seq_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=0), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k not in ("tickers", "days")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
