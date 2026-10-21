"""scripts/data_pipeline/fetch_fin_versions.py — 带版本字段的财报表全量拉取（T-039）.

lake 的 income / balancesheet / cashflow / fina_indicator_full 拉取时没要 f_ann_date、update_flag，并按
(ts_code, end_date, ann_date) 去重，同一期的原始版和更正版只留了接口先返回的一行（T-039）。本脚本按 lake 的同一参数
（ts_code、start_date=20190101、end_date=20261231，按公告日过滤）重新拉取，加版本字段，**接口返回的每一行原样保存**，
写到新路径 ``<data_root>/data/lake_pit/``，不碰 lake。

- 股票池：lake 四张表出现过的股票 ∪ ``data/lake/stock_basic_full.parquet``（6 月那次拉取的股票池），写进 universe.json
- 每只股票每个接口一个文件 ``raw/<api>/<ts_code>.parquet``（含接口返回顺序 _row 与拉取时间 _fetched_at）；文件在即完成，断点续跑
- ``--seed-from DIR``：先把 DIR/raw/<api>/<ts>.parquet（同一字段集拉过的，如 T-039 抽样）复制进来，来源记进 _provenance.json
- 空返回：lake 里这只股票这张表没有行 → 直接记为空（写 0 行文件）；lake 里有行 → 先不写，留到重试轮，
  最后一轮仍为空才写 0 行文件并记进 _provenance.json 的 empty_but_lake_has_rows
- 异常不写文件，重试轮再拉（默认 2 轮，间隔 --retry-wait 秒）
- 限速：provider 全局最小间隔（--rate-interval，默认 2 秒，即 ≤30 次/分钟）；多线程只为重叠网络等待
- ``--consolidate``：把 raw 合成 ``<api>.parquet``（全部行，不去重）和 manifest.json（行数、股票数、sha256）
token 走 backfill_tushare 同一套（api_key.txt，官方端点），loguru 全部移除，不回显不落盘。

用法：python scripts/data_pipeline/fetch_fin_versions.py --data-root /home/lenovo/projects/quantmind \
          [--seed-from data/research/pit_audit] [--rate-interval 2.0] [--workers 3]
      python scripts/data_pipeline/fetch_fin_versions.py --data-root ... --consolidate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

START, END = "20190101", "20261231"   # 与 p63_1b / p63_1c 相同
VER = "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,end_type,update_flag"
# 与 docs/work/evidence/T-039/pit_fetch.py 相同的字段集（--seed-from 复用那次的文件）
SPEC = {
    "income": VER + ",total_revenue,revenue,n_income,n_income_attr_p,operate_profit",
    "balancesheet": VER + ",total_assets,total_liab,total_hldr_eqy_exc_min_int,total_cur_assets,total_cur_liab",
    "cashflow": VER + ",n_cashflow_act,c_paid_invest,c_pay_acq_const_fiolta,free_cashflow",
    "fina_indicator": ("ts_code,ann_date,end_date,update_flag,roe,roa,roe_yearly,roa_yearly,grossprofit_margin,"
                       "netprofit_margin,assets_turn,current_ratio,quick_ratio,debt_to_assets,assets_to_eqt,ebit_ps,"
                       "bps,eps,ocfps,fcff,or_yoy,tr_yoy,netprofit_yoy,dt_netprofit_yoy,roe_yoy,op_yoy,q_sales_yoy,"
                       "q_op_qoq,q_gr_yoy,q_profit_yoy"),
}
LAKE = {"income": "income", "balancesheet": "balancesheet", "cashflow": "cashflow",
        "fina_indicator": "fina_indicator_full"}


def universe(lake: Path) -> tuple[list[str], dict]:
    per = {t: set(pd.read_parquet(lake / f"{t}.parquet", columns=["ts_code"])["ts_code"].astype(str))
           for t in LAKE.values()}
    sb = set(pd.read_parquet(lake / "stock_basic_full.parquet", columns=["ts_code"])["ts_code"].astype(str))
    u = sorted(set().union(*per.values()) | sb)
    return u, {"lake_tables": {t: len(s) for t, s in per.items()}, "stock_basic_full": len(sb), "union": len(u)}


def lake_has_rows(lake: Path) -> dict[str, set[str]]:
    return {api: set(pd.read_parquet(lake / f"{t}.parquet", columns=["ts_code"])["ts_code"].astype(str))
            for api, t in LAKE.items()}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def seed(out: Path, src: Path) -> dict:
    n = {}
    for api, fields in SPEC.items():
        k = 0
        for f in sorted((src / "raw" / api).glob("*.parquet")):
            dst = out / "raw" / api / f.name
            if dst.exists():
                continue
            cols = {c for c in pd.read_parquet(f).columns if not c.startswith("_")}
            if cols != set(fields.split(",")):
                raise SystemExit(f"种子文件字段与本脚本不同：{f}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            k += 1
        n[api] = k
    return n


def consolidate(out: Path) -> dict:
    man = {"built": datetime.now().isoformat(timespec="seconds"), "tables": {}}
    for api, fields in SPEC.items():
        files = sorted((out / "raw" / api).glob("*.parquet"))
        parts = [pd.read_parquet(f) for f in files]
        df = pd.concat([p for p in parts if len(p)], ignore_index=True)
        for c in ("ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type", "end_type",
                  "update_flag"):
            if c in df.columns:
                df[c] = df[c].astype(object).where(df[c].notna(), None)
        for c in fields.split(","):
            if c not in ("ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type", "end_type",
                         "update_flag"):
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
        df = df.sort_values(["ts_code", "_row"], kind="stable").reset_index(drop=True)
        p = out / f"{api}.parquet"
        tmp = p.with_name(p.name + ".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(p)
        man["tables"][api] = {"files": len(files), "rows": int(len(df)), "tickers": int(df.ts_code.nunique()),
                              "empty_files": int(sum(len(x) == 0 for x in parts)),
                              "fetched_at": [str(df._fetched_at.min()), str(df._fetched_at.max())],
                              "sha256": sha256(p)}
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1))
    return man


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", default="data/lake_pit")
    ap.add_argument("--seed-from", type=Path)
    ap.add_argument("--rate-interval", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--retry-wait", type=float, default=120.0)
    ap.add_argument("--max-tickers", type=int, default=0, help="只处理前 N 只（试跑）")
    ap.add_argument("--consolidate", action="store_true")
    a = ap.parse_args()
    root = a.data_root.resolve()
    lake = root / "data/lake"
    out = root / a.out
    out.mkdir(parents=True, exist_ok=True)
    if a.consolidate:
        print(json.dumps(consolidate(out), ensure_ascii=False, indent=1))
        return 0

    up = out / "universe.json"
    if up.exists():
        tickers = json.loads(up.read_text())["tickers"]
    else:
        tickers, counts = universe(lake)
        up.write_text(json.dumps({"built": datetime.now().isoformat(timespec="seconds"), **counts,
                                  "tickers": tickers}, ensure_ascii=False, indent=1))
    if a.max_tickers:
        tickers = tickers[:a.max_tickers]
    has = lake_has_rows(lake)
    run = {"started": datetime.now().isoformat(timespec="seconds"),
           "params": {"start_date": START, "end_date": END, "fields": SPEC, "rate_interval": a.rate_interval,
                      "workers": a.workers, "endpoint": "tushare official"}}
    if a.seed_from:
        src = a.seed_from if a.seed_from.is_absolute() else root / a.seed_from
        run["seeded"] = {"from": str(src), "copied": seed(out, src)}

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

    lock = threading.Lock()
    stat = {"calls": 0, "written": 0, "empty": 0, "empty_suspect": 0, "failed": 0}
    fails: list[dict] = []
    suspects: dict[tuple[str, str], int] = {}

    def todo_of(ts: str) -> list[str]:
        return [api for api in SPEC if not (out / "raw" / api / f"{ts}.parquet").exists()]

    def write(api: str, ts: str, df: pd.DataFrame) -> None:
        p = out / "raw" / api / f"{ts}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        df = df.reset_index(drop=True)
        df["_row"] = np.arange(len(df))
        df["_fetched_at"] = datetime.now().isoformat(timespec="seconds")
        tmp = p.with_name(p.name + ".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(p)

    def one(ts: str, last_pass: bool) -> None:
        for api in todo_of(ts):
            try:
                df = tp._call(api, ts_code=ts, start_date=START, end_date=END, fields=SPEC[api])
            except Exception as e:  # noqa: BLE001
                with lock:
                    stat["calls"] += 1
                    stat["failed"] += 1
                    fails.append({"ts_code": ts, "api": api, "error": repr(e)[:160],
                                  "at": datetime.now().isoformat(timespec="seconds")})
                continue
            df = df if df is not None else pd.DataFrame(columns=SPEC[api].split(","))
            with lock:
                stat["calls"] += 1
            if len(df):
                write(api, ts, df)
                with lock:
                    stat["written"] += 1
            elif ts not in has[api] or last_pass:
                write(api, ts, pd.DataFrame(columns=SPEC[api].split(",")))
                with lock:
                    stat["empty"] += 1
                    if ts in has[api]:
                        suspects[(api, ts)] = suspects.get((api, ts), 0) + 1
            else:
                with lock:
                    stat["empty_suspect"] += 1
                    suspects[(api, ts)] = suspects.get((api, ts), 0) + 1

    t0 = time.time()

    def one_pass(label: str, todo: list[str], last_pass: bool) -> None:
        print(f"[{label}] {len(todo)} 只待拉；最小间隔 {tp._MIN_INTERVAL}s，{a.workers} 线程", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
            for i, _ in enumerate(ex.map(lambda t: one(t, last_pass), todo), 1):
                if i % 100 == 0 or i == len(todo):
                    with lock:
                        s = dict(stat)
                    rate = s["calls"] / max(time.time() - t0, 1) * 60
                    print(f"  [{label}] {i}/{len(todo)} {s} {rate:.1f} 次/分钟 ({time.time() - t0:.0f}s)", flush=True)

    todo = [t for t in tickers if todo_of(t)]
    one_pass("main", todo, last_pass=a.retries == 0)
    for r in range(1, a.retries + 1):
        todo = [t for t in tickers if todo_of(t)]
        if not todo:
            break
        print(f"[retry{r}] 等待 {a.retry_wait:.0f}s", flush=True)
        time.sleep(a.retry_wait)
        one_pass(f"retry{r}", todo, last_pass=r == a.retries)

    missing = [(t, api) for t in tickers for api in todo_of(t)]
    run.update({"ended": datetime.now().isoformat(timespec="seconds"), "stat": stat, "fails": fails[-200:],
                "n_fails": len(fails), "missing": len(missing),
                "empty_but_lake_has_rows": sorted(f"{api}:{ts}" for (api, ts) in suspects
                                                  if (out / "raw" / api / f"{ts}.parquet").exists())})
    prov = out / "_provenance.json"
    runs = json.loads(prov.read_text())["runs"] if prov.exists() else []
    runs.append(run)
    prov.write_text(json.dumps({"runs": runs}, ensure_ascii=False, indent=1))
    print(f"[fetch_fin_versions] 结束 {stat} missing={len(missing)} ({time.time() - t0:.0f}s)", flush=True)
    return 0 if not missing else 3


if __name__ == "__main__":
    raise SystemExit(main())
