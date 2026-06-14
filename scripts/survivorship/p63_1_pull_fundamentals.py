"""P63-1：v6 全市场财报 PIT 拉数（quantmind env）.

拉 fina_indicator / income / balancesheet / cashflow（按 ts_code，5741 票含退市）。
**保留 ann_date（实际公告日）与 end_date（报告期）两列** —— PIT 对齐在因子构建时用 ann_date，
绝不用 end_date（A 股财报最易错处）。这里只负责把原始数据（含 ann_date）落到 lake。
token 走 backfill 同一套（api_key.txt 标签 TUSHARE_TOKEN）+ loguru remove，绝不打印/落日志。
断点续传（checkpoint 记 done ts_code）。输出 data/lake/{fina_indicator,income,balancesheet,cashflow}.parquet。
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
CKPT = LAKE / "_checkpoints" / "fundamentals.json"
CKPT.parent.mkdir(parents=True, exist_ok=True)
START, END = "20190101", "20261231"
SLEEP = 0.5

TABLES = {
    "fina_indicator": "ts_code,ann_date,end_date,roe,roa,grossprofit_margin,netprofit_margin,"
                      "assets_turn,current_ratio,quick_ratio,debt_to_assets,ebit_ps,bps,eps,"
                      "or_yoy,netprofit_yoy,roe_yoy",
    "income": "ts_code,ann_date,end_date,total_revenue,revenue,n_income,n_income_attr_p,operate_profit",
    "balancesheet": "ts_code,ann_date,end_date,total_assets,total_liab,total_hldr_eqy_exc_min_int,"
                    "total_cur_assets,total_cur_liab",
    "cashflow": "ts_code,ann_date,end_date,n_cashflow_act,c_paid_invest",
}


def _retry(api, ts, fields, attempts=4):
    for i in range(attempts):
        try:
            r = tp._call(api, ts_code=ts, start_date=START, end_date=END, fields=fields)
            return r if r is not None else pd.DataFrame()
        except Exception as e:
            time.sleep(1.5 * (i + 1))
            if i == attempts - 1:
                return pd.DataFrame()
    return pd.DataFrame()


def main():
    sb = pd.read_parquet(LAKE / "stock_basic_full.parquet")
    tickers = sorted(sb["ts_code"].astype(str).unique())
    ck = json.loads(CKPT.read_text()) if CKPT.exists() else {"done": []}
    done = set(ck.get("done", []))
    todo = [t for t in tickers if t not in done]
    print(f"[P63-1] universe={len(tickers)} done={len(done)} todo={len(todo)} "
          f"(~{len(todo)*4*SLEEP/60:.0f}min)", flush=True)

    acc = {k: [] for k in TABLES}
    # 续传时先载入已存的 lake 表，避免覆盖
    for k in TABLES:
        p = LAKE / f"{k}.parquet"
        if p.exists():
            acc[k].append(pd.read_parquet(p))

    t0 = time.time(); n_fail = 0
    for i, ts in enumerate(todo):
        got = False
        for api, fields in TABLES.items():
            df = _retry(api, ts, fields)
            if df is not None and len(df):
                acc[api].append(df); got = True
            time.sleep(SLEEP)
        if got:
            done.add(ts)
        else:
            n_fail += 1
        if (i + 1) % 100 == 0 or i + 1 == len(todo):
            for k in TABLES:
                if acc[k]:
                    out = pd.concat(acc[k], ignore_index=True).drop_duplicates(
                        subset=["ts_code", "end_date", "ann_date"])
                    out.to_parquet(LAKE / f"{k}.parquet", index=False)
                    acc[k] = [out]   # 压实，省内存
            CKPT.write_text(json.dumps({"done": sorted(done), "updated": time.strftime("%FT%T"),
                                        "n_universe": len(tickers)}, ensure_ascii=False))
            print(f"  {i+1}/{len(todo)} done={len(done)} fail={n_fail} ({time.time()-t0:.0f}s)", flush=True)

    print(f"[P63-1] 完成 done={len(done)}/{len(tickers)} fail={n_fail}", flush=True)
    for k in TABLES:
        p = LAKE / f"{k}.parquet"
        if p.exists():
            d = pd.read_parquet(p, columns=["ts_code", "end_date"])
            print(f"  {k}: rows={len(d):,} tickers={d.ts_code.nunique()}", flush=True)


if __name__ == "__main__":
    main()
