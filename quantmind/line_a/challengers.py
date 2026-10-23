"""quantmind.line_a.challengers — 前瞻影子账本里的挑战臂（用户 2026-10-23 批准；规则与 T-042、T-043 的历史回测完全相同）.

挑战臂只改变"给引擎的分数表"，引擎、契约、成本、成交规则都照搬对应的参考臂：
- c5_filter（T-043 C5）：决策日先打分、后过滤，去掉 ST / *ST / 退、收盘价 < 2 元、总市值 < 10 亿元、复牌不满 5 个交易日的股票，
  引擎在 PIT 前 1500 里照常递补；
- quota_extended（T-042 Q）：原研究池 ENS 前 240 + 扩展域（top1500 以外、按时点 adv20 ≥ 1,000 万元）ENS 前 60，共 300 只等权，
  引擎的股票池上限放开。
所有判断只用决策日 as_of 及之前的数据。只读影子：不改变任何参考账本，资金批准状态 NOT_APPROVED。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_MIN, MV_MIN_WAN, RESUME_TD = 2.0, 100_000.0, 5


def load_arms(cfg_dir: Path) -> dict:
    return json.loads((Path(cfg_dir) / "challenger_arms.json").read_text(encoding="utf-8"))["arms"]


def _st_names(lake: Path) -> pd.DataFrame:
    parts = [*sorted((lake / "events").glob("namechange_full_*.parquet"))[-1:],
             *sorted((lake / "events").glob("namechange_2*.parquet"))[-1:]]
    n = pd.concat([pd.read_parquet(p, columns=["ts_code", "name", "start_date", "end_date"]) for p in parts],
                  ignore_index=True).drop_duplicates()
    n = n[n["name"].astype(str).str.contains("ST|退", regex=True)].copy()
    n["s"] = pd.to_datetime(n["start_date"].astype(str), errors="coerce")
    n["e"] = pd.to_datetime(n["end_date"].astype(str), errors="coerce").fillna(pd.Timestamp("2100-01-01"))
    return n


def _daily_basic_on(lake: Path, as_of: pd.Timestamp) -> pd.DataFrame:
    import pyarrow.dataset as pds
    dset = pds.dataset(str(lake / "daily_basic.parquet"), format="parquet")
    typ = str(dset.schema.field("trade_date").type)
    days = [as_of - pd.Timedelta(days=k) for k in range(11)]
    vals = [d if typ.startswith(("timestamp", "date")) else (int(d.strftime("%Y%m%d")) if typ.startswith("int")
                                                            else d.strftime("%Y%m%d")) for d in days]
    df = dset.to_table(columns=["ts_code", "trade_date", "close", "total_mv"],
                       filter=pds.field("trade_date").isin(vals)).to_pandas()
    df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str).str.replace("-", "").str.slice(0, 8))
    return df.sort_values("trade_date").groupby("ts_code").last()


def c5_flags(lake: Path, as_of, prices: pd.DataFrame) -> set[str]:
    """as_of 当天命中 C1–C4 任一条的股票。prices：v6 价格长表（ts_code、trade_date、close）。"""
    d = pd.Timestamp(as_of)
    n = _st_names(Path(lake))
    st = set(n.loc[(n["s"] <= d) & (n["e"] >= d), "ts_code"])
    db = _daily_basic_on(Path(lake), d)
    low = set(db.index[db["close"] < PRICE_MIN]) | set(db.index[db["total_mv"] < MV_MIN_WAN])
    px = prices[["ts_code", "trade_date", "close"]].copy()
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px[(px["trade_date"] <= d) & (px["trade_date"] > d - pd.Timedelta(days=120))]
    has = px.pivot_table(index="trade_date", columns="ts_code", values="close").sort_index().notna()
    h = has.to_numpy()
    run = np.zeros(h.shape[1])
    cum = np.zeros(h.shape[1])
    flag = np.zeros(h.shape[1], dtype=bool)
    for i in range(h.shape[0]):
        run = (run + 1) * h[i]
        cum = cum + h[i]
        flag = h[i] & (run < RESUME_TD) & (cum > run)
    resumed = set(has.columns[flag]) if len(has) and has.index[-1] == d else set()
    return st | low | resumed


def adv20_on(prices: pd.DataFrame, as_of) -> pd.Series:
    d = pd.Timestamp(as_of)
    px = prices[["ts_code", "trade_date", "amount"]].copy()
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px[(px["trade_date"] <= d) & (px["trade_date"] > d - pd.Timedelta(days=60))]
    w = px.pivot_table(index="trade_date", columns="ts_code", values="amount").sort_index()
    return (w.tail(20).mean() * 1000).where(w.tail(20).count() >= 10)


def apply(spec: dict, base: pd.DataFrame, *, lake: Path, prices: pd.DataFrame,
          domains: dict[str, pd.DataFrame] | None = None) -> tuple[pd.DataFrame, dict]:
    """base：参考臂在各前瞻网格日的 (as_of, ticker, score)。返回 (挑战臂的分数表, 每个决策日的说明)。"""
    out, notes = [], {}
    for d, g in base.groupby("as_of"):
        key = str(pd.Timestamp(d).date())
        if spec["transform"] == "c5_filter":
            fl = c5_flags(lake, d, prices)
            keep = ~g["ticker"].isin(fl)
            out.append(g[keep])
            notes[key] = {"flagged_in_scores": int((~keep).sum())}
        elif spec["transform"] == "quota_extended":
            p = spec["params"]
            dom = (domains or {}).get(key)
            if dom is None:
                raise ValueError(f"no F2 domains for {key}")
            pool = set(dom.index[dom["domain"] == "research_pool"])
            adv = adv20_on(prices, d)
            gp = g[g["ticker"].isin(pool)].nlargest(p["n_pool"], "score")
            ext = g[~g["ticker"].isin(pool) & g["ticker"].map(adv).ge(p["adv_floor_rmb"]).fillna(False).to_numpy()]
            ge = ext.nlargest(p["n_ext"], "score")
            out.append(pd.concat([gp, ge]))
            notes[key] = {"n_pool": int(len(gp)), "n_ext": int(len(ge)), "ext_eligible": int(len(ext))}
        else:
            raise ValueError(spec["transform"])
    return (pd.concat(out, ignore_index=True) if out else base.iloc[:0]), notes
