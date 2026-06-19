"""P63-1c：补拉扩池 —— fina_indicator 全字段 + income + balancesheet（按 ts_code，PIT 保留 ann_date）.

解锁缺失因子：
  fina_full → equity_multiplier(assets_to_eqt) / operating_profit_yoy(op_yoy) /
              quarterly_revenue_yoy(q_sales_yoy) / earnings_accel_q(q_op_qoq) /
              revenue_accel_q(q_sales_yoy) / fcf_yield(fcff) + roe/roa_yearly
  income     → accruals(n_income_attr_p) / ocf_to_revenue(total_revenue)
  balancesheet → accruals(total_assets)
用法：--table {fina_full,income,balancesheet}（各自独立 checkpoint，可并发跑——不同 API 不抢限额）。
token 走 backfill 同套 + loguru remove，绝不打印。输出 data/lake/{table}.parquet。
"""
from __future__ import annotations
import argparse, json, sys, time
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
START, END, SLEEP = "20190101", "20261231", 0.4

SPEC = {
    "fina_full": ("fina_indicator", "fina_indicator_full",
                  "ts_code,ann_date,end_date,roe,roa,roe_yearly,roa_yearly,grossprofit_margin,"
                  "netprofit_margin,assets_turn,current_ratio,quick_ratio,debt_to_assets,"
                  "assets_to_eqt,ebit_ps,bps,eps,ocfps,fcff,or_yoy,tr_yoy,netprofit_yoy,"
                  "dt_netprofit_yoy,roe_yoy,op_yoy,q_sales_yoy,q_op_qoq,q_gr_yoy,q_profit_yoy"),
    "income": ("income", "income",
               "ts_code,ann_date,end_date,total_revenue,revenue,n_income,n_income_attr_p,operate_profit"),
    "balancesheet": ("balancesheet", "balancesheet",
                     "ts_code,ann_date,end_date,total_assets,total_liab,total_hldr_eqy_exc_min_int,"
                     "total_cur_assets,total_cur_liab"),
}


def _retry(api, ts, fields, attempts=4):
    for i in range(attempts):
        try:
            r = tp._call(api, ts_code=ts, start_date=START, end_date=END, fields=fields)
            return r if r is not None else pd.DataFrame()
        except Exception:
            time.sleep(1.5 * (i + 1))
    return pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", choices=list(SPEC), required=True)
    a = ap.parse_args()
    api, out_name, fields = SPEC[a.table]
    out_path = LAKE / f"{out_name}.parquet"
    ckpt = LAKE / "_checkpoints" / f"{out_name}.json"
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    sb = pd.read_parquet(LAKE / "stock_basic_full.parquet")
    tickers = sorted(sb["ts_code"].astype(str).unique())
    ck = json.loads(ckpt.read_text()) if ckpt.exists() else {"done": []}
    done = set(ck.get("done", []))
    todo = [t for t in tickers if t not in done]
    print(f"[{a.table}] api={api} universe={len(tickers)} done={len(done)} todo={len(todo)}", flush=True)
    acc = []
    if out_path.exists():
        acc.append(pd.read_parquet(out_path))
    t0 = time.time(); n_fail = 0
    for i, ts in enumerate(todo):
        df = _retry(api, ts, fields)
        if df is not None and len(df):
            acc.append(df); done.add(ts)
        else:
            n_fail += 1
        time.sleep(SLEEP)
        if (i + 1) % 100 == 0 or i + 1 == len(todo):
            out = pd.concat(acc, ignore_index=True).drop_duplicates(subset=["ts_code", "end_date", "ann_date"])
            out.to_parquet(out_path, index=False); acc = [out]
            ckpt.write_text(json.dumps({"done": sorted(done), "updated": time.strftime("%FT%T")}, ensure_ascii=False))
            print(f"  {i+1}/{len(todo)} done={len(done)} fail={n_fail} ({time.time()-t0:.0f}s)", flush=True)
    out = pd.read_parquet(out_path)
    print(f"[{a.table}] 完成 done={len(done)}/{len(tickers)} fail={n_fail} "
          f"rows={len(out):,} tickers={out.ts_code.nunique()}", flush=True)


if __name__ == "__main__":
    main()
