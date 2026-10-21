"""T-039：时点版（PIT）基本面因子 —— 只在评估日上算全截面，与冻结构建器同一套计算，只换财报输入.

与 `docs/work/evidence/T-030/build_daily_fnd.py` 的 one_day 逐行对应：daily_basic@(≤ as_of 的最后交易日) + 财报
→ compute_all_fundamental_factors + fcf_yield_cf → 截面 1%/99% 缩尾（在当日全部 daily_basic 股票上）。

两种模式（财报以外的输入都用冻结时刻的快照：财报表 `data/lake/_backup_20261019/`（10-11 08:45 写入，冻结面板
10-13 构建时就是这一份），daily_basic `data/lake/_backup_20261020/daily_basic.parquet`（10-10 22:41））：
  lake  冻结构建器的取数：财报 ann_date ≤ as_of，f_ann_date 设成 ann_date。用来复现冻结面板（闸门 G2）。
  pit   T-039「PIT 规则 v1」：财报来自 `data/lake_pit/`（接口返回的全部行）
        - 三大表：f_ann_date ≤ as_of 的行里，同一 (ts_code, end_date) 取 f_ann_date 最大的，并列按整行内容取第一行；
        - fina_indicator：每行的可用日 = 与同期三大表对账（debt_to_assets ↔ 100·负债/资产，netprofit_margin ↔
          100·净利润/营收，容差 0.01）对上的报表行里最早的 f_ann_date，对不上用 ann_date；可用日 ≤ as_of 的行里，
          同一期按 (ann_date, 整行内容) 排序取第一行。
输出 <data_root>/data/research/pit_audit/fnd_evaldates/<mode>/<YYYY-MM>.parquet（index as_of × ticker，列同冻结面板）；
已存在的月份跳过，从不覆盖。

用法：PYTHONPATH=. python docs/work/evidence/T-039/build_fnd_pit.py <data_root> --mode lake|pit [--workers 2] [--months 2022-02,...]
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import scripts.survivorship.p63_2_build_factors as p63
from quantmind.features.fundamental import compute_all_fundamental_factors

FROZEN_FIN = "data/lake/_backup_20261019"
FROZEN_DB = "data/lake/_backup_20261020/daily_basic.parquet"
EVAL_ROWS = Path(__file__).resolve().parents[1] / "T-030/p6/frozen_eval_rows_20d.parquet"   # 63d 的 189 天是它的子集
STMT = {"income": "inc", "balancesheet": "bs", "cashflow": "cf"}
STR_COLS = ("ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type", "end_type", "update_flag")
G: dict = {}


def content_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """「整行内容」：除 _row、_fetched_at 外的字段按固定顺序，空值为空串、数值用 repr(float)，以 | 连接。"""
    parts = []
    for c in cols:
        s = df[c]
        if c in STR_COLS:
            parts.append(s.astype(object).where(s.notna(), "").astype(str))
        else:
            parts.append(s.astype("float64").map(lambda x: "" if np.isnan(x) else repr(float(x))))
    out = parts[0]
    for p in parts[1:]:
        out = out + "|" + p
    return out


def _fields(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not c.startswith("_")]


def fina_avail(fi: pd.DataFrame, inc: pd.DataFrame, bs: pd.DataFrame) -> pd.Series:
    """fina 每行的可用日：对上的同期报表行里最早的 f_ann_date；对不上用 ann_date。"""
    f = fi[["ts_code", "end_date", "ann_date", "netprofit_margin", "debt_to_assets"]].reset_index(names="_i")
    i = inc.assign(npm=100 * inc.n_income / inc.revenue)[["ts_code", "end_date", "f_ann_date", "npm"]]
    b = bs.assign(dta=100 * bs.total_liab / bs.total_assets)[["ts_code", "end_date", "f_ann_date", "dta"]]
    mi = f.merge(i, on=["ts_code", "end_date"])
    mi = mi[(mi.npm - mi.netprofit_margin).abs() < 0.01]
    mb = f.merge(b, on=["ts_code", "end_date"])
    mb = mb[(mb.dta - mb.debt_to_assets).abs() < 0.01]
    m = pd.concat([mi[["_i", "f_ann_date"]], mb[["_i", "f_ann_date"]]]).groupby("_i").f_ann_date.min()
    borrowed = m.reindex(fi.index)
    later = borrowed.notna() & (borrowed > fi["ann_date"])     # 可用日不早于这一行自己的 ann_date
    return fi["ann_date"].where(~later, borrowed)


def setup(root: Path, mode: str) -> None:
    db_path = root / FROZEN_DB
    td = pd.DatetimeIndex(np.unique(pq.read_table(db_path, columns=["trade_date"]).column(0).to_numpy()))
    dates = pd.DatetimeIndex(sorted(pd.read_parquet(EVAL_ROWS, columns=["as_of"]).as_of.unique()))
    ltd = {d: td[td <= d][-1] for d in dates}
    db = pq.read_table(db_path, filters=[("trade_date", "in", sorted(set(ltd.values())))]).to_pandas()
    if "ticker" not in db.columns:
        db["ticker"] = db["ts_code"].astype(str)
    G.update(db=db, ltd=ltd, dates=dates, mode=mode)

    if mode == "lake":
        fin = root / FROZEN_FIN
        fi = pd.read_parquet(fin / "fina_indicator_full.parquet")
        fi["ticker"] = fi["ts_code"].astype(str)
        fi["report_date"] = fi["end_date"]
        G["fi"] = fi
        for name, k in STMT.items():
            d = pd.read_parquet(fin / f"{name}.parquet")
            d["ticker"] = d["ts_code"].astype(str)
            d["report_date"] = d["end_date"]
            d["f_ann_date"] = d["ann_date"]
            G[k] = d
        return

    pit = root / "data/lake_pit"
    raw = {}
    for name in STMT:
        d = pd.read_parquet(pit / f"{name}.parquet")
        d["f_ann_date"] = d["f_ann_date"].where(d["f_ann_date"].notna(), d["ann_date"])
        d["_key"] = content_key(d, _fields(d))
        raw[name] = d
        d = d.sort_values(["ts_code", "end_date", "f_ann_date", "_key"], ascending=[True, True, False, True],
                          kind="mergesort").reset_index(drop=True)
        d["ticker"] = d["ts_code"].astype(str)
        d["report_date"] = d["end_date"]
        G[STMT[name]] = d
    fi = pd.read_parquet(pit / "fina_indicator.parquet")
    fi = fi[fi["ann_date"].notna()].reset_index(drop=True)
    fi["_key"] = content_key(fi, _fields(fi))
    fi["_avail"] = fina_avail(fi, raw["income"], raw["balancesheet"])
    fi = fi.sort_values(["ts_code", "end_date", "ann_date", "_key"], kind="mergesort").reset_index(drop=True)
    fi["ticker"] = fi["ts_code"].astype(str)
    fi["report_date"] = fi["end_date"]
    G["fi"] = fi


def snapshot(asof_str: str) -> dict:
    fi, inc, bs, cf = G["fi"], G["inc"], G["bs"], G["cf"]
    if G["mode"] == "lake":            # 与 build_daily_fnd.one_day 相同
        return {"financial_indicators": fi[fi["ann_date"] <= asof_str],
                "financials_income": inc[inc["ann_date"] <= asof_str],
                "financials_balance_sheet": bs[bs["ann_date"] <= asof_str],
                "financials_cashflow": cf[cf["ann_date"] <= asof_str]}

    def stmt(d):                       # 已按 (ts, end, f_ann 降序, 内容) 排好：过滤后每期第一行
        return d[d["f_ann_date"] <= asof_str].drop_duplicates(["ts_code", "end_date"], keep="first")
    f = fi[fi["_avail"] <= asof_str].drop_duplicates(["ts_code", "end_date"], keep="first")
    return {"financial_indicators": f, "financials_income": stmt(inc), "financials_balance_sheet": stmt(bs),
            "financials_cashflow": stmt(cf)}


def one_day(as_of: pd.Timestamp) -> pd.DataFrame:
    db = G["db"]
    asof_str = as_of.strftime("%Y%m%d")
    db_a = db[db.trade_date == G["ltd"][as_of]]
    snap = {"daily_basic": db_a, **snapshot(asof_str)}
    fac = compute_all_fundamental_factors(snap, as_of.date())
    cfa = snap["financials_cashflow"].sort_values(["ticker", "report_date"], ascending=[True, False])
    cfa = cfa.drop_duplicates("ticker", keep="first").set_index("ticker")
    mv = db_a.set_index("ticker")["total_mv"]
    num = cfa.get("n_cashflow_act", pd.Series(dtype=float)) - cfa.get("c_paid_invest", 0.0)
    common = num.index.intersection(mv.index)
    fac["fcf_yield_cf"] = (num.loc[common] / mv.loc[common].replace(0, np.nan))
    fac = fac.apply(p63._winsor, axis=0)
    fac["as_of"] = as_of
    return fac.reset_index().rename(columns={"index": "ticker"})


def do_month(args: tuple[str, str]) -> str:
    ym, out_dir = args
    out = Path(out_dir) / f"{ym}.parquet"
    if out.exists():
        return f"{ym} exists"
    t0 = time.time()
    df = pd.concat([one_day(d) for d in G["dates"] if d.strftime("%Y-%m") == ym], ignore_index=True)
    df["ticker"] = df["ticker"].astype(str)
    df = df.set_index(["as_of", "ticker"]).sort_index()
    tmp = out.with_suffix(".tmp")
    df.to_parquet(tmp)
    os.replace(tmp, out)
    return f"{ym} {df.shape} {time.time() - t0:.0f}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--mode", choices=["lake", "pit"], required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--months", default="", help="只算这些月份（逗号分隔），试跑用")
    a = ap.parse_args()
    t0 = time.time()
    setup(a.root, a.mode)
    out_dir = a.root / "data/research/pit_audit/fnd_evaldates" / a.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    months = sorted({d.strftime("%Y-%m") for d in G["dates"]})
    if a.months:
        months = [m for m in months if m in set(a.months.split(","))]
    todo = [m for m in months if not (out_dir / f"{m}.parquet").exists()]
    print(f"[fnd_{a.mode}] setup {time.time() - t0:.0f}s; {len(G['dates'])} 个评估日，{len(todo)} 个月待算", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for msg in ex.map(do_month, [(m, str(out_dir)) for m in todo]):
            print(f"  {msg} (total {time.time() - t0:.0f}s)", flush=True)
    print(f"[fnd_{a.mode}] done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
