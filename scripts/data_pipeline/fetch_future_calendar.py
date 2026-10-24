"""scripts/data_pipeline/fetch_future_calendar.py — 取上交所交易日历到明年年底，另存 data/lake/trade_cal_sse_future.parquet.

`trade_cal_sse.parquet` 只到数据截止日（每日运行按它决定补哪些交易日，不能混入未来日期），正式日报算"下一个调仓日"
需要未来的交易日，所以单独存一份。交易所的休市安排一般在上一年年底公布，次年的日历在年底前可能不全，结果里写明覆盖到哪天。
用法：PYTHONPATH=. python scripts/data_pipeline/fetch_future_calendar.py --data-root /home/lenovo/projects/quantmind
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    a = ap.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import scripts.backfill_tushare as bf          # 与 update_lake_latest 相同的取 token 方式（import 时已屏蔽日志）
    bf.REPO_ROOT = str(a.data_root)                 # token 文件在数据所在的主 checkout（同 update_lake_latest.set_data_root）
    bf._load_token_into_env()
    from quantmind.data import tushare_provider as tp
    end = f"{pd.Timestamp.today().year + 1}1231"
    raw = tp._call("trade_cal", exchange="SSE", start_date="20190101", end_date=end, is_open="1")
    cal = pd.DataFrame({"cal_date": sorted(pd.to_datetime(raw["cal_date"].astype(str)).unique())})
    if cal.empty:
        raise SystemExit("trade_cal 返回空")
    out = a.data_root / "data/lake/trade_cal_sse_future.parquet"
    tmp = out.with_suffix(".tmp.parquet")
    cal.to_parquet(tmp)
    tmp.replace(out)
    print(f"{len(cal)} trading days, {cal['cal_date'].min().date()} -> {cal['cal_date'].max().date()}")


if __name__ == "__main__":
    main()
