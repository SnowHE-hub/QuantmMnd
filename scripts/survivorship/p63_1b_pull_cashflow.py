"""P63-1b：补拉 cashflow（留现金流因子的底：FCF yield 等）.

= p63_1 同套（token 防泄漏 + 断点续传 + 按 ts_code），仅 cashflow 一表。
保留 ann_date（PIT）。输出 data/lake/cashflow.parquet。独立 checkpoint。
FCF yield ≈ (n_cashflow_act − c_paid_invest) / total_mv（total_mv 走 daily_basic）。
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import scripts.backfill_tushare as bf
bf._load_token_into_env()
from quantmind.data import tushare_provider as tp
try:
    from loguru import logger as _lg; _lg.remove()
except Exception:
    pass
try:
    from quantmind.utils.silence_provider_logging import silence_provider_logging
    silence_provider_logging()
except Exception:
    pass

LAKE = REPO / "data" / "lake"
CKPT = LAKE / "_checkpoints" / "cashflow.json"
CKPT.parent.mkdir(parents=True, exist_ok=True)
START, END, SLEEP = "20190101", "20261231", 0.4
FIELDS = "ts_code,ann_date,end_date,n_cashflow_act,c_paid_invest,c_pay_acq_const_fiolta,free_cashflow"


def _retry(ts, attempts=4):
    for i in range(attempts):
        try:
            r = tp._call("cashflow", ts_code=ts, start_date=START, end_date=END, fields=FIELDS)
            return r if r is not None else pd.DataFrame()
        except Exception:
            time.sleep(1.5 * (i + 1))
    return pd.DataFrame()


def main():
    sb = pd.read_parquet(LAKE / "stock_basic_full.parquet")
    tickers = sorted(sb["ts_code"].astype(str).unique())
    ck = json.loads(CKPT.read_text()) if CKPT.exists() else {"done": []}
    done = set(ck.get("done", []))
    todo = [t for t in tickers if t not in done]
    print(f"[P63-1b cashflow] universe={len(tickers)} done={len(done)} todo={len(todo)}", flush=True)
    acc = []
    p = LAKE / "cashflow.parquet"
    if p.exists():
        acc.append(pd.read_parquet(p))
    t0 = time.time(); n_fail = 0
    for i, ts in enumerate(todo):
        df = _retry(ts)
        if df is not None and len(df):
            acc.append(df); done.add(ts)
        else:
            n_fail += 1
        time.sleep(SLEEP)
        if (i + 1) % 100 == 0 or i + 1 == len(todo):
            out = pd.concat(acc, ignore_index=True).drop_duplicates(subset=["ts_code", "end_date", "ann_date"])
            out.to_parquet(p, index=False); acc = [out]
            CKPT.write_text(json.dumps({"done": sorted(done), "updated": time.strftime("%FT%T")},
                                       ensure_ascii=False))
            print(f"  {i+1}/{len(todo)} done={len(done)} fail={n_fail} ({time.time()-t0:.0f}s)", flush=True)
    out = pd.read_parquet(p)
    print(f"[P63-1b] 完成 done={len(done)}/{len(tickers)} fail={n_fail} "
          f"cashflow rows={len(out):,} tickers={out.ts_code.nunique()}", flush=True)


if __name__ == "__main__":
    main()
