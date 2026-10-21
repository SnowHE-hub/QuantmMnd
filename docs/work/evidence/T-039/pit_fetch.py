"""T-039：抽样拉取带版本字段的财报（只读研究，写新路径，不碰 lake）.

对 lake 财报表的股票池随机抽 N 只（固定种子），按 lake 构建时的同一参数（ts_code、start_date=20190101、
end_date=20261231，按公告日过滤）调用官方 Tushare 的 income / balancesheet / cashflow / fina_indicator，
字段 = lake 的字段 + 版本字段（f_ann_date、update_flag、report_type、comp_type、end_type；fina_indicator 只有 update_flag）。
不去重，接口返回的每一行原样保存，另加 _row（接口返回顺序）。

输出 <data_root>/data/research/pit_audit/raw/<api>/<ts_code>.parquet（空返回也写 0 行文件）、
sample.json（抽样参数与名单）、_provenance.json（每次运行的时间与参数）。已存在的文件跳过（断点续跑）。
token 走 backfill_tushare 同一套（api_key.txt，官方端点），loguru 全部移除，不回显不落盘。

用法：PYTHONPATH=. python docs/work/evidence/T-039/pit_fetch.py --data-root /home/lenovo/projects/quantmind [--n 300] [--rate-interval 1.0] [--workers 3]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

START, END = "20190101", "20261231"   # 与 p63_1b / p63_1c 相同
VER = "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,end_type,update_flag"
SPEC = {
    "income": VER + ",total_revenue,revenue,n_income,n_income_attr_p,operate_profit",
    "balancesheet": VER + ",total_assets,total_liab,total_hldr_eqy_exc_min_int,total_cur_assets,total_cur_liab",
    "cashflow": VER + ",n_cashflow_act,c_paid_invest,c_pay_acq_const_fiolta,free_cashflow",
    "fina_indicator": ("ts_code,ann_date,end_date,update_flag,roe,roa,roe_yearly,roa_yearly,grossprofit_margin,"
                       "netprofit_margin,assets_turn,current_ratio,quick_ratio,debt_to_assets,assets_to_eqt,ebit_ps,"
                       "bps,eps,ocfps,fcff,or_yoy,tr_yoy,netprofit_yoy,dt_netprofit_yoy,roe_yoy,op_yoy,q_sales_yoy,"
                       "q_op_qoq,q_gr_yoy,q_profit_yoy"),
}


def draw_sample(lake: Path, n: int, seed: int) -> tuple[list[str], int]:
    pool = sorted(set(pd.read_parquet(lake / "income.parquet", columns=["ts_code"])["ts_code"].astype(str)))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(pool, size=n, replace=False).tolist()), len(pool)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--rate-interval", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tickers", nargs="*", help="指定名单（不抽样），写到 --tag 子目录，例如 T-038 点名的案例")
    ap.add_argument("--tag", default="", help="与 --tickers 一起用：输出到 pit_audit/<tag>/")
    a = ap.parse_args()
    root = a.data_root.resolve()
    out = root / "data/research/pit_audit"
    if a.tickers:
        if not a.tag:
            raise SystemExit("--tickers 需要 --tag，避免混进随机样本")
        out = out / a.tag
    out.mkdir(parents=True, exist_ok=True)

    sp = out / "sample.json"
    if a.tickers:
        sample = sorted(set(a.tickers))
        sp.write_text(json.dumps({"tickers": sample, "note": "named cases, not a random sample"}, indent=1))
    elif sp.exists():
        sample = json.loads(sp.read_text())["tickers"]
    else:
        sample, pool = draw_sample(root / "data/lake", a.n, a.seed)
        sp.write_text(json.dumps({"seed": a.seed, "n": a.n, "pool": "data/lake/income.parquet ts_code",
                                  "pool_size": pool, "tickers": sample}, ensure_ascii=False, indent=1))

    logging.disable(logging.INFO)
    import scripts.backfill_tushare as bf
    bf.REPO_ROOT = str(root)
    bf._load_token_into_env()
    from quantmind.data import tushare_provider as tp
    try:
        from loguru import logger as _lg
        _lg.remove()
    except Exception:  # noqa: BLE001
        pass
    tp._MIN_INTERVAL = max(tp._MIN_INTERVAL, a.rate_interval)

    started = datetime.now().isoformat(timespec="seconds")
    t0, n_call, fails = time.time(), 0, []

    def one_ticker(ts: str) -> tuple[int, list[dict]]:
        calls, errs = 0, []
        for api, fields in SPEC.items():
            p = out / "raw" / api / f"{ts}.parquet"
            if p.exists():
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            try:
                df = tp._call(api, ts_code=ts, start_date=START, end_date=END, fields=fields)
            except Exception as e:  # noqa: BLE001
                errs.append({"ts_code": ts, "api": api, "error": repr(e)[:160]})
                continue
            calls += 1
            df = (df if df is not None else pd.DataFrame(columns=fields.split(","))).reset_index(drop=True)
            df["_row"] = np.arange(len(df))
            df["_fetched_at"] = datetime.now().isoformat(timespec="seconds")
            tmp = p.with_name(p.name + ".tmp")
            df.to_parquet(tmp, index=False)
            tmp.replace(p)
        return calls, errs

    # 延迟主要在网络上（单次 2–20 秒），多线程只为重叠等待；全局最小间隔由 provider 的锁保证
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        for i, (calls, errs) in enumerate(ex.map(one_ticker, sample), 1):
            n_call += calls
            fails += errs
            if i % 25 == 0 or i == len(sample):
                print(f"  {i}/{len(sample)} calls={n_call} fails={len(fails)} ({time.time() - t0:.0f}s)", flush=True)

    prov = out / "_provenance.json"
    runs = json.loads(prov.read_text())["runs"] if prov.exists() else []
    runs.append({"started": started, "ended": datetime.now().isoformat(timespec="seconds"), "calls": n_call,
                 "fails": fails, "params": {"start_date": START, "end_date": END, "fields": SPEC,
                                            "rate_interval": tp._MIN_INTERVAL, "workers": a.workers,
                                            "endpoint": "tushare official"}})
    prov.write_text(json.dumps({"runs": runs}, ensure_ascii=False, indent=1))
    missing = [(ts, api) for ts in sample for api in SPEC if not (out / "raw" / api / f"{ts}.parquet").exists()]
    print(f"[pit_fetch] done calls={n_call} fails={len(fails)} missing={len(missing)}", flush=True)
    return 0 if not missing else 3


if __name__ == "__main__":
    raise SystemExit(main())
