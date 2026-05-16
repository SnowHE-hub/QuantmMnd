#!/usr/bin/env python3
"""Step 3: 下载每日指标（daily_basic）— 按交易日批量拉取.

每次调用返回"当日全市场"数据，再过滤 Alpha Universe，效率极高。
快照日期：公历季末（3/31、6/30、9/30、12/31）先经 **SSE trade_cal** 落在「不超过季末的最后一个交易日」，
再按该交易日请求 daily_basic（避免写死日历日落在周末/节假日）。

预计时间：28 次 × ~1s = <1 分钟
使用：高频 API（每次只需 1 个请求就能覆盖全市场）

字段（用于漏斗 Layer1/4 + ValuationAgent）：
  pe_ttm, pb, ps_ttm, dv_ratio, total_mv, circ_mv, turnover_rate,
  free_share_ratio, close, pre_close

输出：data/snapshots/{date}/daily_basic.parquet（已有的补全）
      data/alpha_universe/alpha_daily_basic_combined.parquet（合并版）
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
UNIVERSE_TXT = ROOT / "data" / "alpha_universe" / "alpha_universe.txt"
SNAPSHOTS_DIR = ROOT / "data" / "snapshots"


def _init_pro_hi():
    import tushare as ts
    token = os.environ.get("TUSHARE_TOKEN_HI", "").strip()
    if not token:
        raise SystemExit("请设置 TUSHARE_TOKEN_HI")
    ts.set_token(token)
    pro = ts.pro_api(timeout=120)
    url = os.environ.get("TUSHARE_HI_URL", "http://tsy.xiaodefa.cn")
    pro._DataApi__http_url = url
    print(f"[API] 高频 → {url}")
    return pro


def _nominal_quarter_ends(start_year: int = 2019, end_year: int = 2026) -> list[str]:
    dates = []
    for y in range(start_year, end_year + 1):
        for m, d in [(3, 31), (6, 30), (9, 30), (12, 31)]:
            dates.append(f"{y}{m:02d}{d:02d}")
    today = pd.Timestamp.today().strftime("%Y%m%d")
    return [d for d in dates if d <= today]


def _sse_open_days(pro, start_yyyymmdd: str, end_yyyymmdd: str) -> list[str]:
    df = pro.trade_cal(
        exchange="SSE",
        start_date=start_yyyymmdd,
        end_date=end_yyyymmdd,
        is_open="1",
        fields="cal_date",
    )
    if df is None or df.empty:
        return []
    return sorted(df["cal_date"].astype(str).str.strip().tolist())


def quarter_end_trade_dates(pro, start_year: int = 2019, end_year: int = 2026) -> list[tuple[str, str]]:
    """每个公历季末对应 SSE 最后交易日。(日历季末 yyyymmdd, 实际交易日 yyyymmdd)。

    同一交易日若被重复映射（极少见）只拉取一次。
    """
    nominal_list = _nominal_quarter_ends(start_year, end_year)
    if not nominal_list:
        return []

    buf_days = 120  # 春节等长假：季末前足够回溯
    start_buf = (pd.to_datetime(min(nominal_list), format="%Y%m%d") - pd.Timedelta(days=buf_days)).strftime("%Y%m%d")
    end_buf = pd.Timestamp.today().strftime("%Y%m%d")
    open_days = _sse_open_days(pro, start_buf, end_buf)
    if not open_days:
        raise SystemExit("trade_cal 返回空，无法解析季末交易日（检查网络 / TUSHARE_TOKEN_HI）")

    out: list[tuple[str, str]] = []
    seen_trade: set[str] = set()
    for qe in nominal_list:
        cand = [d for d in open_days if d <= qe]
        if not cand:
            print(f"  [日历季末 {qe}] 此前无交易日（跳过）")
            continue
        td = max(cand)
        if td in seen_trade:
            print(f"  [日历季末 {qe}] → 交易日 {td}（本期已拉取，跳过）")
            continue
        seen_trade.add(td)
        out.append((qe, td))
    return out


def main():
    if not UNIVERSE_TXT.exists():
        raise SystemExit("先运行 step1")

    universe = set(UNIVERSE_TXT.read_text().splitlines())

    pro = _init_pro_hi()
    time.sleep(0.4)

    pairs = quarter_end_trade_dates(pro)
    print(f"[Step3] daily_basic：SSE 对齐季末交易日共 {len(pairs)} 期，Universe {len(universe)} 只")

    SLEEP = 0.4
    all_parts: list[pd.DataFrame] = []
    fields = ("ts_code,trade_date,pe_ttm,pb,ps_ttm,dv_ratio,total_mv,circ_mv,"
              "turnover_rate,free_share_ratio,close,pre_close")

    for i, (qe_nominal, dt) in enumerate(pairs):
        tag = f"{dt}" + ("" if dt == qe_nominal else f" ←季末{qe_nominal}")
        try:
            df = pro.daily_basic(trade_date=dt, fields=fields)
            if df is None or df.empty:
                print(f"  [{tag}] daily_basic 仍为空（跳过）")
                time.sleep(SLEEP)
                continue
            # 过滤 Alpha Universe
            df_filtered = df[df["ts_code"].isin(universe)].copy()

            # 保存到对应快照目录（按 API 返回的交易日）
            snap_dir = SNAPSHOTS_DIR / pd.Timestamp(dt).strftime("%Y-%m-%d")
            snap_dir.mkdir(parents=True, exist_ok=True)
            out = snap_dir / "daily_basic.parquet"
            df_filtered.to_parquet(out, index=False)
            all_parts.append(df_filtered)
            print(f"  [{tag}] {len(df_filtered)} 只 → {out.name}")
        except Exception as e:
            print(f"  [{tag}] 失败: {e}")
        time.sleep(SLEEP)
        if (i + 1) % 10 == 0:
            print(f"  进度 {i+1}/{len(pairs)}")

    if all_parts:
        combined = pd.concat(all_parts, ignore_index=True).drop_duplicates(["ts_code", "trade_date"])
        out_combined = ROOT / "data" / "alpha_universe" / "alpha_daily_basic_combined.parquet"
        combined.to_parquet(out_combined, index=False, compression="snappy")
        print(f"\n[Done] 合并 daily_basic: {out_combined}")
        print(f"  {len(combined):,} 行 | {combined['ts_code'].nunique()} 只 | {combined['trade_date'].nunique()} 期")
    return 0


if __name__ == "__main__":
    sys.exit(main())
