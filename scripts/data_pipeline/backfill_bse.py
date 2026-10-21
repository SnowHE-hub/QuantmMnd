"""scripts/data_pipeline/backfill_bse.py — 北交所的股票基础表与财报回补（T-035；北交所作为扩展研究臂的数据准备）.

为什么写到单独目录 data/lake/bse/：
- 冻结的 fnd 构建器会在全市场截面上做 1%/99% 缩尾。如果把北交所财报并进现有的 income / balancesheet 等表，
  所有股票的财务特征都会改变，参考臂就不再是原样复现了。
- 所以北交所数据只给扩展研究臂用，与参考臂的输入完全隔离。

内容：
- stock_basic（exchange = BSE，L / D / P）→ stock_basic_bse_<END>.parquet；
- 每只北交所股票的 income / balancesheet / cashflow / fina_indicator（字段与 update_fundamentals_latest 相同）
  → <表>.parquet。已存在时只追加新主键，旧行不动。

token 走 backfill_tushare 同一套，不回显、不落盘。
用法：python scripts/data_pipeline/backfill_bse.py --data-root <root> --end 2026-10-20 [--since 2018-01-01]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.data_pipeline.update_fundamentals_latest import KEY, SPEC, clip_ann  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=REPO)
    ap.add_argument("--end", required=True)
    ap.add_argument("--since", default="2018-01-01")
    a = ap.parse_args(argv)
    import scripts.backfill_tushare as bf
    bf.REPO_ROOT = str(a.data_root)
    bf._load_token_into_env()
    from quantmind.data import tushare_provider as tp
    try:
        from loguru import logger as _lg
        _lg.remove()
    except Exception:  # noqa: BLE001
        pass
    out = a.data_root / "data/lake/bse"
    out.mkdir(parents=True, exist_ok=True)
    end, since = pd.Timestamp(a.end).strftime("%Y%m%d"), pd.Timestamp(a.since).strftime("%Y%m%d")
    sbp = out / f"stock_basic_bse_{end}.parquet"
    if sbp.exists():
        sb = pd.read_parquet(sbp)
    else:
        parts = []
        for st in ("L", "D", "P"):
            df = tp._call("stock_basic", exchange="BSE", list_status=st,
                          fields="ts_code,symbol,name,area,industry,market,list_status,list_date,delist_date")
            if df is not None and len(df):
                parts.append(df)
        sb = pd.concat(parts, ignore_index=True).drop_duplicates("ts_code")
        sb.to_parquet(sbp, index=False)
    tickers = sorted(sb["ts_code"].astype(str))
    print(f"[bse] stock_basic {len(tickers)} tickers -> {sbp}", flush=True)
    ck = out / f"_checkpoint_{end}.json"
    done = set(json.loads(ck.read_text())["done"]) if ck.exists() else set()
    buf: dict[str, list[pd.DataFrame]] = {t: [] for t, _ in SPEC.values()}

    def flush() -> None:
        for table, parts in buf.items():
            if not parts:
                continue
            p = out / f"{table}.parquet"
            new = pd.concat(parts, ignore_index=True)
            if p.exists():
                old = pd.read_parquet(p)
                k_old = set(map(tuple, old[KEY].astype(str).to_numpy()))
                new = new[[tuple(r) not in k_old for r in new[KEY].astype(str).to_numpy()]]
                new = pd.concat([old, new], ignore_index=True)
            tmp = p.with_suffix(".tmp")
            new.to_parquet(tmp, index=False)
            tmp.replace(p)
            parts.clear()
        ck.write_text(json.dumps({"done": sorted(done)}))

    t0 = time.time()
    for i, t in enumerate(tickers, 1):
        if t in done:
            continue
        for api, (table, fields) in SPEC.items():
            df = tp._call(api, ts_code=t, start_date=since, end_date=end, fields=fields)
            df = clip_ann(df, end)
            if len(df):
                buf[table].append(df)
        done.add(t)
        if i % 50 == 0 or i == len(tickers):
            flush()
            print(f"  {i}/{len(tickers)} ({time.time() - t0:.0f}s)", flush=True)
    flush()
    print("[bse] done", {t: len(pd.read_parquet(out / f"{t}.parquet")) for t, _ in SPEC.values()
                         if (out / f"{t}.parquet").exists()}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
