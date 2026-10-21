"""scripts/data_pipeline/fetch_namechange_full.py — 按股票拉名称变更的全部历史（T-038 X5）.

为什么：lake 的 namechange 表按公告月份从 2018 年起下载，2018 年以前开始、之后才改名的名称段（例如"华新水泥"1994–2025）
不在表里，规则链接器因此认不出很多旧名（X5 开发集里 31 个链接错误几乎都是这个原因）。
做法：每只股票（含北交所、已退市）调一次 namechange(ts_code=...)，取全部名称段；写到新文件，不覆盖原表。
输出：data/lake/events/namechange_full_<日期>.parquet，检查点 data/lake/events/_checkpoints/namechange_full.json。
用法：python scripts/data_pipeline/fetch_namechange_full.py --data-root <root> [--interval 0.4]
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--interval", type=float, default=0.4)
    a = ap.parse_args()
    import scripts.backfill_tushare as bf
    bf.REPO_ROOT = str(a.data_root)
    bf._load_token_into_env()
    from quantmind.data import tushare_provider as tp
    lake = a.data_root / "data/lake"
    sb = pd.read_parquet(sorted(lake.glob("stock_basic_full_merged_2*.parquet"))[-1], columns=["ts_code"])
    bse = sorted((lake / "bse").glob("stock_basic_bse_*.parquet"))
    codes = sorted(set(sb["ts_code"].astype(str)) | (set(pd.read_parquet(bse[-1])["ts_code"]) if bse else set()))
    ck_path = lake / "events/_checkpoints/namechange_full.json"
    ck = json.loads(ck_path.read_text()) if ck_path.exists() else {"done": {}}
    parts_dir = lake / "events/_parts/namechange_full"
    parts_dir.mkdir(parents=True, exist_ok=True)
    todo = [c for c in codes if c not in ck["done"]]
    print(f"[namechange_full] {len(codes)} codes, {len(todo)} to fetch", flush=True)
    buf, t0 = [], time.time()
    for i, c in enumerate(todo, 1):
        for attempt in range(4):
            try:
                df = tp._call("namechange", ts_code=c,
                              fields="ts_code,name,start_date,end_date,ann_date,change_reason")
                break
            except Exception as e:  # noqa: BLE001  频率超限或网络抖动：等一会儿再试
                if attempt == 3:
                    print(f"  {c} failed: {str(e)[:100]}", flush=True)
                    df = None
                time.sleep(20 * (attempt + 1))
        if df is not None:
            if len(df):
                buf.append(df)
            ck["done"][c] = int(len(df))
        time.sleep(a.interval)
        if i % 500 == 0 or i == len(todo):
            if buf:
                pd.concat(buf, ignore_index=True).to_parquet(parts_dir / f"part_{len(list(parts_dir.glob('*.parquet'))):04d}.parquet")
                buf = []
            tmp = ck_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(ck))
            tmp.replace(ck_path)
            print(f"  {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    allp = sorted(parts_dir.glob("*.parquet"))
    if allp:
        out = pd.concat([pd.read_parquet(p) for p in allp], ignore_index=True).drop_duplicates()
        dst = lake / f"events/namechange_full_{time.strftime('%Y%m%d')}.parquet"
        out.to_parquet(dst)
        print(f"[namechange_full] {len(out)} rows, {out['ts_code'].nunique()} codes -> {dst}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
