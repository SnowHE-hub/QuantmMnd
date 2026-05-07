"""探针脚本：测试 Tushare 当前积分等级下可用的接口。

运行：python scripts/probe_tushare.py
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv

load_dotenv()
import tushare as ts  # type: ignore[import-untyped]  # noqa: E402  # 必须先 load_dotenv

ts.set_token(os.getenv("TUSHARE_TOKEN", ""))
pro = ts.pro_api()


def _try(name: str, points: int, fn: Callable[[], Any]) -> None:
    t0 = time.monotonic()
    try:
        df = fn()
        elapsed = time.monotonic() - t0
        if df is None or df.empty:
            print(f"  {name:<20}{points:<8}空      {elapsed:.1f}s")
            return
        date_cols = [
            c for c in df.columns
            if any(k in c.lower() for k in ("ann_date", "trade_date", "end_date", "actual_date"))
        ][:5]
        print(f"  {name:<20}{points:<8}OK      {elapsed:.1f}s  shape={df.shape}  关键日期: {date_cols}")
    except Exception as e:  # noqa: BLE001
        elapsed = time.monotonic() - t0
        print(f"  {name:<20}{points:<8}FAIL    {elapsed:.1f}s  {str(e)[:80]}")
    time.sleep(0.3)


def main() -> None:
    print(f"{'接口':<22}{'门槛':<8}{'状态':<8}耗时   信息")
    print("=" * 100)

    # 100 分
    _try("stock_basic", 100, lambda: pro.stock_basic(list_status="L", fields="ts_code,name"))
    _try("trade_cal", 100, lambda: pro.trade_cal(start_date="20240101", end_date="20240105"))
    # 120 分
    _try("daily", 120, lambda: pro.daily(ts_code="000001.SZ", start_date="20240101", end_date="20240105"))
    # 600 分
    _try("daily_basic", 600, lambda: pro.daily_basic(ts_code="000001.SZ", trade_date="20240105"))
    _try("income", 600, lambda: pro.income(ts_code="000001.SZ", start_date="20230101", end_date="20231231"))
    _try("balancesheet", 600, lambda: pro.balancesheet(ts_code="000001.SZ", start_date="20230101", end_date="20231231"))
    _try("cashflow", 600, lambda: pro.cashflow(ts_code="000001.SZ", start_date="20230101", end_date="20231231"))
    _try("disclosure_date", 500, lambda: pro.disclosure_date(end_date="20231231"))
    _try("stk_holdernumber", 600, lambda: pro.stk_holdernumber(
        ts_code="000001.SZ", start_date="20230101", end_date="20231231"
    ))
    # 2000 分
    _try("fina_indicator", 2000, lambda: pro.fina_indicator(
        ts_code="000001.SZ", start_date="20230101", end_date="20231231"
    ))
    _try("fina_audit", 2000, lambda: pro.fina_audit(ts_code="000001.SZ"))
    _try("index_weight", 2000, lambda: pro.index_weight(
        index_code="000300.SH", start_date="20230101", end_date="20230131"
    ))
    _try("index_basic", 2000, lambda: pro.index_basic(market="SSE"))
    _try("index_daily", 2000, lambda: pro.index_daily(
        ts_code="000300.SH", start_date="20240101", end_date="20240131"
    ))
    _try("moneyflow_hsgt", 2000, lambda: pro.moneyflow_hsgt(start_date="20240101", end_date="20240105"))
    _try("hsgt_top10", 2000, lambda: pro.hsgt_top10(trade_date="20240105"))
    _try("top_list", 2000, lambda: pro.top_list(trade_date="20240105"))
    _try("adj_factor", 2000, lambda: pro.adj_factor(
        ts_code="000001.SZ", start_date="20240101", end_date="20240105"
    ))
    _try("suspend_d", 2000, lambda: pro.suspend_d(
        suspend_type="S", start_date="20240101", end_date="20240131"
    ))
    _try("namechange", 2000, lambda: pro.namechange(ts_code="000001.SZ"))


if __name__ == "__main__":
    main()
