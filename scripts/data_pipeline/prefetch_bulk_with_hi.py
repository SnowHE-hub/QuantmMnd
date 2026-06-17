#!/usr/bin/env python3
"""Token B 高频代理 Sidecar — 按交易日 bulk 预取全市场宽表.

目的
----
当前 build_snapshot 对 hk_hold / margin_detail / daily_basic 是
「逐 ticker × lookback 天」串行调用，1374 只 × 各步耗时 ~1–3 小时/期。
本脚本改为「每个交易日调用一次全市场接口」——只需 ~1500 次请求就能
覆盖 2019-2024 全部日期，并落盘到 data/raw/bulk_v1/。

之后 build_snapshot Q2..Q20 可直接从 bulk 表切片，跳过重复拉取。

用法（不影响正在跑的主进程）
----
    source scripts/data_pipeline/setup_api_config.sh
    nohup python -u scripts/data_pipeline/prefetch_bulk_with_hi.py \
        > logs/prefetch_hi.log 2>&1 &
    disown
    tail -f logs/prefetch_hi.log

可选参数
--------
    --tables        : 逗号分隔要拉的表，默认 hk_hold,margin_detail,daily_basic
    --start / --end : 日期范围，默认 20190101 ~ 20241231
    --sleep         : 每次请求后等待秒（默认 0.36s，180次/分保守值）
    --dry-run       : 只打印计划，不实际调用
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# ── 复用既有 .env 机制读取凭证 + 屏蔽 provider 日志（防 token 外泄）──────────────
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass
try:
    from quantmind.utils.silence_provider_logging import silence_provider_logging
    silence_provider_logging()
except Exception:
    pass
OUT_DIR = ROOT / "data" / "raw" / "bulk_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 高频代理限速：官方说 180 次/分钟，超频触发 ~6 分钟冷却
# 安全间隔：60/180 ≈ 0.333s → 取 0.36s 留缓冲
DEFAULT_SLEEP = 0.36
# 遇到限频错误时的强制等待（6 min + 30s 缓冲）
RATE_LIMIT_BACKOFF = 390


def _make_api(token: str, url: str | None = None, timeout: int = 120):
    """直接用指定 token 创建 DataApi 实例，绕过 get_token() 的环境变量优先级。

    tushare 1.4.29 的 get_token() 先读 TUSHARE_TOKEN 环境变量，导致
    ts.set_token(token_b) + ts.pro_api() 仍然拿到 token_a。
    直接实例化 DataApi 可以规避这个问题。
    """
    from tushare.pro.client import DataApi  # type: ignore[import-untyped]

    api = DataApi(token=token, timeout=timeout)
    if url is not None:
        api._DataApi__http_url = url  # type: ignore[attr-defined]
    return api


def _probe_proxy(token_b: str, url_b: str) -> bool:
    """测试代理是否当前可用。返回 True 表示代理正常。"""
    try:
        pro = _make_api(token_b, url=url_b, timeout=10)
        df = pro.stock_basic(list_status="L", fields="ts_code", limit=1)
        return df is not None and len(df) > 0
    except Exception:
        return False


def _init_pro():
    """初始化双客户端：Token A（官方，用于 trade_cal）和 Token B（高频代理 / 回退官方）。
    返回 (pro_a, pro_b)。
    """
    token_a = os.environ.get(
        "TUSHARE_TOKEN",
        "",
    ).strip()
    token_b = os.environ.get(
        "TUSHARE_TOKEN_HI",
        "",
    ).strip()
    url_b = os.environ.get("TUSHARE_HI_URL", "http://tsy.xiaodefa.cn")

    # Token A — 官方，用于 trade_cal 等基础接口
    pro_a = _make_api(token_a, timeout=120)
    print("[init] Token A（官方）已加载", flush=True)  # 不打印 token 前缀（泄漏根因）

    # Token B — 先探测代理，可用则走代理（180 req/min），否则走官方
    # 全市场 daily_basic 单次数据量大，设 90s 超时
    proxy_ok = _probe_proxy(token_b, url_b)
    if proxy_ok:
        pro_b = _make_api(token_b, url=url_b, timeout=90)
        print(f"[init] Token B ✅ 代理可用 → {url_b}", flush=True)  # 不打印 token 前缀
    else:
        pro_b = _make_api(token_b, timeout=90)
        print(
            f"[init] Token B ⚠️  代理不可用（冷却中/故障），退回官方 tushare  prefix={token_b[:8]}...",
            flush=True,
        )

    return pro_a, pro_b


def _safe_call(fn, label: str, attempts: int = 4, sleep_between: float = 0.36):
    """调用 tushare API，遇到限频直接睡 RATE_LIMIT_BACKOFF 秒。"""
    for k in range(attempts):
        try:
            result = fn()
            return result if result is not None else pd.DataFrame()
        except Exception as e:
            msg = str(e).lower()
            is_rate = any(w in msg for w in ("频率", "rate", "limit", "too many", "超限"))
            is_timeout = any(w in msg for w in ("timeout", "timed out", "read timeout"))
            is_token = any(w in msg for w in ("token", "权限", "积分", "invalid"))

            if is_token:
                # 代理可能触发冷却后返回 token 错误；记录但不终止，继续下一天
                print(
                    f"[{label}] Token/权限错误（可能代理冷却）: {e}",
                    flush=True,
                )
                break

            if is_rate:
                print(
                    f"[{label}] 触发限频冷却，等待 {RATE_LIMIT_BACKOFF}s ... ({e})",
                    flush=True,
                )
                time.sleep(RATE_LIMIT_BACKOFF)
                continue

            if is_timeout and k < attempts - 1:
                wait = 15 * (k + 1)
                print(f"[{label}] 超时 attempt {k+1}/{attempts}，等待 {wait}s", flush=True)
                time.sleep(wait)
                continue

            print(f"[{label}] 非限频错误: {e}", flush=True)
            break

    return pd.DataFrame()


def _get_trading_dates(pro_a, start: str, end: str) -> list[str]:
    """用 Token A（官方）获取交易日历——proxy 不一定支持 trade_cal。"""
    print(f"[trade_cal] 用 Token A 获取 {start}~{end} 交易日历...", flush=True)
    df = _safe_call(
        lambda: pro_a.trade_cal(
            exchange="SSE",
            start_date=start,
            end_date=end,
            is_open="1",
            fields="cal_date",
        ),
        label="trade_cal",
    )
    if df.empty:
        print("[trade_cal] 返回空，检查 Token A 是否有效", flush=True)
        return []
    today = pd.Timestamp.today().strftime("%Y%m%d")
    dates = sorted(df["cal_date"].tolist())
    dates = [d for d in dates if d <= today]
    print(f"[trade_cal] 共 {len(dates)} 个交易日", flush=True)
    return dates


# --------------------------------------------------------------------------
# 各表的 bulk-by-date 拉取函数
# --------------------------------------------------------------------------

_TABLE_DEFS: dict[str, dict] = {
    "daily_basic": {
        "api": "daily_basic",
        "desc": "全市场 daily_basic（PE/PB/市值/换手）",
        "call": lambda pro, d: pro.daily_basic(trade_date=d),
    },
    "hk_hold": {
        "api": "hk_hold",
        "desc": "全市场港股通持股明细",
        "call": lambda pro, d: pro.hk_hold(trade_date=d),
    },
    "margin_detail": {
        "api": "margin_detail",
        "desc": "全市场融资融券明细",
        "call": lambda pro, d: pro.margin_detail(trade_date=d),
    },
}


def _done_years_file(table: str) -> Path:
    return OUT_DIR / f"{table}_done_years.txt"


def _done_dates_file(table: str) -> Path:
    return OUT_DIR / f"{table}_done_dates.txt"


def _load_done_dates(table: str) -> set[str]:
    p = _done_dates_file(table)
    return set(p.read_text().splitlines()) if p.exists() else set()


def _save_done_dates(table: str, done: set[str]):
    _done_dates_file(table).write_text("\n".join(sorted(done)))


def _year_parquet(table: str, year: str) -> Path:
    return OUT_DIR / f"{table}_{year}.parquet"


def _flush_year(table: str, year: str, rows: list[pd.DataFrame]):
    if not rows:
        return
    chunk = pd.concat(rows, ignore_index=True)
    out = _year_parquet(table, year)
    if out.exists():
        try:
            old = pd.read_parquet(out)
            # 合并时去重（日期+代码）
            date_col = "trade_date" if "trade_date" in chunk.columns else None
            code_col = next((c for c in ("ts_code", "ticker") if c in chunk.columns), None)
            chunk = pd.concat([old, chunk], ignore_index=True)
            if date_col and code_col:
                chunk = chunk.drop_duplicates(subset=[date_col, code_col], keep="last")
        except Exception:
            pass
    chunk.to_parquet(out, index=False, compression="snappy")
    print(f"  [flush] {out.name}: {len(chunk):,} 行", flush=True)


def run_table(
    pro_b,
    table: str,
    all_dates: list[str],
    sleep: float,
    dry_run: bool = False,
):
    defn = _TABLE_DEFS[table]
    print(f"\n=== {table} ({defn['desc']}) ===", flush=True)

    done = _load_done_dates(table)
    todo = [d for d in all_dates if d not in done]
    print(f"  总 {len(all_dates)} 天，已完成 {len(done)} 天，剩余 {len(todo)} 天", flush=True)

    if not todo:
        print("  全部已完成，跳过", flush=True)
        return

    est_min = len(todo) * (sleep + 0.3) / 60
    print(f"  预计耗时：~{est_min:.0f} 分钟", flush=True)

    if dry_run:
        print("  [dry-run] 跳过实际调用", flush=True)
        return

    # 按年分批，每年结束写一次 parquet
    years = sorted({d[:4] for d in todo})
    for year in years:
        year_dates = [d for d in todo if d.startswith(year)]
        year_rows: list[pd.DataFrame] = []
        success = skip = 0

        for i, dt in enumerate(year_dates):
            df = _safe_call(
                lambda d=dt: defn["call"](pro_b, d),
                label=f"{table} {dt}",
            )
            time.sleep(sleep)

            if not df.empty:
                year_rows.append(df)
                done.add(dt)
                success += 1
            else:
                skip += 1

            if (i + 1) % 100 == 0 or i + 1 == len(year_dates):
                print(
                    f"  {year}: {i+1}/{len(year_dates)} | 成功 {success} | 空/失败 {skip}",
                    flush=True,
                )
                _flush_year(table, year, year_rows)
                _save_done_dates(table, done)
                year_rows.clear()

        # 年尾确保写盘
        _flush_year(table, year, year_rows)
        _save_done_dates(table, done)

    print(f"  {table} 完成：成功 {len(done)} 天", flush=True)


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Token B sidecar：按交易日 bulk 预取全市场宽表"
    )
    parser.add_argument(
        "--tables",
        default="hk_hold,margin_detail,daily_basic",
        help="逗号分隔的表名，可选: hk_hold,margin_detail,daily_basic",
    )
    parser.add_argument("--start", default="20190101", help="起始日期 YYYYMMDD")
    parser.add_argument("--end", default="20241231", help="结束日期 YYYYMMDD")
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP,
        help=f"每次请求间隔秒（默认 {DEFAULT_SLEEP}，≈180次/分）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只计划，不实际调用",
    )
    args = parser.parse_args()

    tables = [t.strip() for t in args.tables.split(",") if t.strip()]
    unknown = [t for t in tables if t not in _TABLE_DEFS]
    if unknown:
        print(f"[ERROR] 未知表名: {unknown}，可选: {list(_TABLE_DEFS.keys())}")
        sys.exit(1)

    print(f"[prefetch] tables={tables}  {args.start}~{args.end}  sleep={args.sleep}s")
    print(f"[prefetch] 输出目录: {OUT_DIR}")

    if args.dry_run:
        print("[dry-run] 仅计划模式，不初始化 API")
        pro_a = pro_b = None
    else:
        pro_a, pro_b = _init_pro()

    # 交易日历用 Token A（官方，proxy 不一定支持 trade_cal）
    if pro_a is not None:
        all_dates = _get_trading_dates(pro_a, args.start, args.end)
    else:
        # 干跑：粗估 2019-2024 约 1448 个交易日
        all_dates = [f"placeholder_{i}" for i in range(1448)]

    if not all_dates:
        print("[ERROR] 交易日历为空，检查 Token A 是否有效", flush=True)
        sys.exit(1)

    for table in tables:
        run_table(pro_b, table, all_dates, args.sleep, dry_run=args.dry_run)

    print("\n[prefetch] 全部完成！", flush=True)
    print(f"  落盘目录: {OUT_DIR}", flush=True)
    print("  后续可用 scripts/data_pipeline/merge_bulk_to_snapshot.py 将 bulk 切片注入各 snapshot 目录")


if __name__ == "__main__":
    main()
