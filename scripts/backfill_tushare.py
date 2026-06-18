"""scripts/backfill_tushare.py — Tushare 日频回补（按探针 §6 策略）.

只补三项（其余序列由快照拼接已足，见 data_sufficiency_audit §4.2）:
    daily_basic   全程 2019-01-01 → 2026-05-31（按交易日历，全市场一调/日）
    margin_detail 仅 2019-01-01 → 2019-11-30 头部
    hk_hold       仅 2019-01-01 → 2019-11-30 头部
                  （**不**尝试 2024-09→2026 日频：交易所 2024-08-18 已停发，源端无数据）

特性:
    - 断点续传：data/lake/_checkpoints/<series>.json 记已完成 trade_date，断哪续哪
    - 幂等：write_lake 按 (ts_code, trade_date) 去重 keep='last'
    - 按月增量落盘：每完成一个自然月一批 flush（崩溃最多丢在途一月）
    - 退避：复用 provider _call 的指数退避(2/4/8s)+超时(15/30s)
    - 限速：运行期设 _MIN_INTERVAL=0.5s（≤120 调/分钟），保守低于 2000 积分配额

安全:
    token 从 api_key.txt 读取（标签 TUSHARE_TOKEN），全程不回显/不落盘/不入日志；
    logging.disable(INFO) 屏蔽 provider 启动时的 token[:8] 日志行。

用法:
    python scripts/backfill_tushare.py                      # 三项全补
    python scripts/backfill_tushare.py --series daily_basic # 只补某项
    python scripts/backfill_tushare.py --status            # 看进度不调用 API
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from datetime import datetime

# --- 屏蔽日志：防止 provider 的 "token[:8]..." 行泄漏 ---
# 关键：provider 用的是 loguru（非 stdlib logging），logging.disable 拦不住它，
# 必须 logger.remove() 移除 loguru 的所有 sink（stderr + 文件）。两者都做。
logging.disable(logging.INFO)
try:
    from loguru import logger as _loguru_logger

    _loguru_logger.remove()  # 移除全部 loguru sink，杜绝 token[:8] 落屏/落盘
except Exception:  # noqa: BLE001
    pass

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from quantmind.data import lake  # noqa: E402
from quantmind.data.base import normalize_ticker  # noqa: E402

# 回补窗口（含端点；以交易日历裁剪）
WINDOWS = {
    "daily_basic": ("2019-01-01", "2026-05-31"),
    "margin": ("2019-01-01", "2019-11-30"),
    "hk_hold": ("2019-01-01", "2019-11-30"),
}
# series -> tushare api 名
API_NAME = {"daily_basic": "daily_basic", "margin": "margin_detail", "hk_hold": "hk_hold"}


# <private> ---- token 读取：不回显/不落盘/不入日志 ----
def _load_token_into_env() -> None:
    """从 api_key.txt 读取 TUSHARE_TOKEN 注入环境变量，不回显。"""
    path = os.path.join(REPO_ROOT, "api_key.txt")
    tok = None
    for line in open(path, encoding="utf-8").read().splitlines():
        line = line.strip()
        if "：" in line:
            label, value = line.split("：", 1)
            if label.strip() == "TUSHARE_TOKEN":
                tok = value.strip()
    if not tok:
        raise SystemExit("TUSHARE_TOKEN not found in api_key.txt")
    os.environ.pop("TUSHARE_HI_URL", None)  # 走官方端点
    os.environ["TUSHARE_TOKEN"] = tok
    tok = None  # 丢弃本地引用
# </private>


def _target_trade_dates(series: str) -> list[str]:
    """该序列回补目标 trade_date 列表（YYYYMMDD），按交易日历裁剪。"""
    start, end = WINDOWS[series]
    cal = lake.load_trading_calendar()
    sel = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
    return [d.strftime("%Y%m%d") for d in sel]


def _safe_ticker(ts_code: str) -> str:
    """normalize_ticker 对港股(.HK)等会抛 ValueError；失败时回退原值，绝不中断回补。"""
    try:
        return normalize_ticker(ts_code)
    except Exception:  # noqa: BLE001
        return str(ts_code)


def _normalize_fetched(series: str, df: pd.DataFrame) -> pd.DataFrame:
    """把 raw API 返回对齐到湖表 schema（保留原生列 + 补 ticker / 重命名 hk_hold）。"""
    df = df.copy()
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str))
    if series == "hk_hold":
        # raw: code,trade_date,ts_code,name,vol,ratio,exchange → 对齐快照 hold_vol/hold_ratio
        ren = {}
        if "vol" in df.columns and "hold_vol" not in df.columns:
            ren["vol"] = "hold_vol"
        if "ratio" in df.columns and "hold_ratio" not in df.columns:
            ren["ratio"] = "hold_ratio"
        if ren:
            df = df.rename(columns=ren)
        # 仅保留 A 股北向持股（.SH/.SZ），与快照建的 hk_hold（0 条 .HK）schema 一致；
        # .HK 是南向港股通持港股，与 north_hold_* 因子无关。
        if "ts_code" in df.columns:
            df = df[~df["ts_code"].astype(str).str.endswith(".HK")].reset_index(drop=True)
    if "ts_code" in df.columns and "ticker" not in df.columns:
        df["ticker"] = df["ts_code"].apply(_safe_ticker)
    return df


def backfill_series(series: str, max_failures_abort: int = 50) -> dict:
    """回补单序列：断点续传 + 按月落盘 + 退避。"""
    from quantmind.data import tushare_provider as tp

    api = API_NAME[series]
    targets = _target_trade_dates(series)
    done = lake.load_checkpoint(series)
    todo = [d for d in targets if d not in done]

    print(
        f"[{series}] api={api} 目标={len(targets)} 已完成={len(done)} 待补={len(todo)} "
        f"窗口={WINDOWS[series][0]}~{WINDOWS[series][1]}"
    )
    if not todo:
        print(f"[{series}] 已全部完成，跳过。")
        return {"series": series, "done": len(done), "new": 0}

    buffer: list[pd.DataFrame] = []
    cur_month: str | None = None
    n_new = 0
    n_rows = 0
    failures: list[dict] = []
    t0 = time.perf_counter()

    def flush():
        nonlocal buffer
        if buffer:
            batch = pd.concat(buffer, ignore_index=True, sort=False)
            total = lake.write_lake(series, batch)
            lake.save_checkpoint(
                series, done,
                extra={"window": WINDOWS[series], "lake_rows": total,
                       "api": api, "failures": failures[-20:]},
            )
            buffer = []
            return total
        return None

    for i, d in enumerate(todo, 1):
        month = d[:6]
        if cur_month is None:
            cur_month = month
        if month != cur_month:
            total = flush()
            elapsed = time.perf_counter() - t0
            print(f"  [{series}] flush月={cur_month} 累计新增日={n_new} 行={n_rows} "
                  f"湖表={total} 用时={elapsed:.0f}s")
            cur_month = month
        try:
            raw = tp._call(api, trade_date=d)
            if raw is not None and len(raw):
                norm = _normalize_fetched(series, raw)
                buffer.append(norm)
                n_rows += len(norm)
            # 即使当日 0 行也标记完成（避免反复重试空交易日）
            done.add(d)
            n_new += 1
        except Exception as e:  # noqa: BLE001
            failures.append({"date": d, "error": repr(e)[:160]})
            print(f"  [{series}] FAIL {d}: {repr(e)[:120]}")
            if len(failures) >= max_failures_abort:
                print(f"  [{series}] 失败累计 {len(failures)} ≥ {max_failures_abort}，中止本序列。")
                break

    total = flush()
    elapsed = time.perf_counter() - t0
    print(f"[{series}] 完成：新增日={n_new} 新增行={n_rows} 湖表行={total} "
          f"失败={len(failures)} 用时={elapsed:.0f}s ({elapsed/60:.1f}min)")
    return {"series": series, "new": n_new, "rows": n_rows, "lake_rows": total,
            "failures": len(failures), "elapsed_s": round(elapsed, 1)}


def print_status() -> None:
    for s in WINDOWS:
        targets = _target_trade_dates(s)
        done = lake.load_checkpoint(s)
        cov = lake.coverage(s)
        pct = round(100.0 * len(done) / len(targets), 1) if targets else 0.0
        print(f"[{s}] 目标={len(targets)} 已完成={len(done)} ({pct}%) "
              f"湖表行={cov.get('rows','-')} 覆盖={cov.get('start','-')}→{cov.get('end','-')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="*", default=list(WINDOWS.keys()),
                    choices=list(WINDOWS.keys()))
    ap.add_argument("--status", action="store_true", help="仅看进度，不调用 API")
    ap.add_argument("--rate-interval", type=float, default=0.5,
                    help="最小调用间隔秒（默认 0.5 = ≤120 调/分钟）")
    args = ap.parse_args()

    if args.status:
        print_status()
        return 0

    _load_token_into_env()
    # 限速 ≤120 调/分钟：运行期提升 provider 的最小调用间隔
    from quantmind.data import tushare_provider as tp

    # 关键：import quantmind.data.tushare_provider 会在导入时重新注册 4 个 loguru sink
    # （含文件 sink logs/quantmind_*.log）。必须在 import **之后**、任何 _call **之前**
    # 再次 logger.remove()，否则 _get_tushare_pro 的 token[:8] 行会写进文件 sink。
    try:
        from loguru import logger as _lg
        _lg.remove()
    except Exception:  # noqa: BLE001
        pass

    tp._MIN_INTERVAL = max(tp._MIN_INTERVAL, args.rate_interval)
    print(f"=== 回补开始 {datetime.now().isoformat(timespec='seconds')} "
          f"min_interval={tp._MIN_INTERVAL}s ===")

    # 顺序：先快的头部，再长任务 daily_basic
    order = sorted(args.series, key=lambda s: 0 if s != "daily_basic" else 1)
    results = []
    for s in order:
        results.append(backfill_series(s))

    print("\n=== 回补汇总 ===")
    for r in results:
        print(" ", r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
