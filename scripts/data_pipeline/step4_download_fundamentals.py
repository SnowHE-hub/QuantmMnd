#!/usr/bin/env python3
"""Step 4: 下载财务基本面（fina_indicator / income / balancesheet / cashflow）.

使用 自己的 2000积分 API（长期有效，频率较低但稳定）。
按股票循环，每只 4 次请求（4张报表），合计约 4000 次。

频率：2000积分约 50-100次/分，保守 sleep 0.6s
预计时间：1000 只 × 4 × 0.6s ÷ 60 ≈ 40 分钟（可断点续传，分多次运行）

字段覆盖：
  fina_indicator: ROE/ROA/毛利率/净利率/资产周转率/流动比率...
  income: 营收/净利润/营收增速
  balancesheet: 总资产/总负债/负债率
  cashflow: 经营现金流

输出：data/alpha_universe/fundamentals/{ts_code}.parquet（每股一文件）
      data/alpha_universe/fina_indicator_combined.parquet（合并）
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
UNIVERSE_TXT = ROOT / "data" / "alpha_universe" / "alpha_universe.txt"
FUNDA_DIR = ROOT / "data" / "alpha_universe" / "fundamentals"
FUNDA_DIR.mkdir(parents=True, exist_ok=True)
DONE_FILE = ROOT / "data" / "alpha_universe" / "funda_done.txt"
FAILED_FILE = ROOT / "data" / "alpha_universe" / "funda_failed.txt"


def _init_pro_own():
    """自己的 2000 积分 API（长期有效，用官方地址）。"""
    import tushare as ts
    token = os.environ.get(
        "TUSHARE_TOKEN",
        "64a18c359c1d28fab92fed6bebd1f1662cc6e34872ad9ee643b55f56",
    ).strip()
    ts.set_token(token)
    print("[API] 官方 2000积分")
    return ts.pro_api(timeout=120)


def _retry(fn, label: str, attempts: int = 4):
    for i in range(attempts):
        try:
            r = fn()
            return r if r is not None else pd.DataFrame()
        except Exception as e:
            print(f"    [{label}] {i+1}/{attempts}: {str(e)[:60]}")
            time.sleep(1.5 * (i + 1))
    return pd.DataFrame()


def download_one(pro, ticker: str, start: str = "20190101", end: str = "20261231") -> dict[str, pd.DataFrame]:
    SLEEP = float(os.environ.get("QUANTMIND_TUSHARE_FUNDA_SLEEP", "0.6"))
    result = {}

    # fina_indicator: ROE/ROA/毛利率等
    fields_fi = ("ts_code,ann_date,end_date,roe,roa,grossprofit_margin,"
                 "netprofit_margin,assets_turn,current_ratio,quick_ratio,"
                 "debt_to_assets,ebit_ps,bps,eps")
    df_fi = _retry(
        lambda t=ticker: pro.fina_indicator(
            ts_code=t, start_date=start, end_date=end, fields=fields_fi
        ),
        label=f"fina_indicator {ticker}",
    )
    result["fina_indicator"] = df_fi
    time.sleep(SLEEP)

    # income: 营收/净利润
    fields_inc = "ts_code,ann_date,end_date,total_revenue,n_income,n_income_attr_p,revenue"
    df_inc = _retry(
        lambda t=ticker: pro.income(
            ts_code=t, start_date=start, end_date=end, fields=fields_inc
        ),
        label=f"income {ticker}",
    )
    result["income"] = df_inc
    time.sleep(SLEEP)

    # balancesheet: 总资产/负债
    fields_bs = "ts_code,ann_date,end_date,total_assets,total_liab,total_hldr_eqy_exc_min_int"
    df_bs = _retry(
        lambda t=ticker: pro.balancesheet(
            ts_code=t, start_date=start, end_date=end, fields=fields_bs
        ),
        label=f"balancesheet {ticker}",
    )
    result["balancesheet"] = df_bs
    time.sleep(SLEEP)

    # cashflow: 经营现金流
    fields_cf = "ts_code,ann_date,end_date,n_cashflow_act"
    df_cf = _retry(
        lambda t=ticker: pro.cashflow(
            ts_code=t, start_date=start, end_date=end, fields=fields_cf
        ),
        label=f"cashflow {ticker}",
    )
    result["cashflow"] = df_cf
    time.sleep(SLEEP)

    return result


def main():
    if not UNIVERSE_TXT.exists():
        raise SystemExit("先运行 step1")

    tickers = [t.strip() for t in UNIVERSE_TXT.read_text().splitlines() if t.strip()]

    # 断点续传
    done: set[str] = set()
    if DONE_FILE.exists():
        done = set(DONE_FILE.read_text().splitlines())
    todo = [t for t in tickers if t not in done]
    print(f"[Step4] 财务数据：共 {len(tickers)} 只，已完成 {len(done)}，待下载 {len(todo)}")
    print(f"  预计 {len(todo) * 4 * 0.6 / 60:.0f}~{len(todo) * 4 * 0.8 / 60:.0f} 分钟")
    print(f"  支持中断后重跑（断点续传）")

    pro = _init_pro_own()
    failed: list[str] = []
    fi_all: list[pd.DataFrame] = []

    for i, ticker in enumerate(todo):
        try:
            data = download_one(pro, ticker)

            # 合并三大表 + 保存
            frames = {k: v for k, v in data.items() if not v.empty}
            if not frames:
                failed.append(ticker)
                continue

            # 将所有报表合并到一个 parquet（按 end_date 对齐）
            base = None
            for k, df in frames.items():
                if df.empty:
                    continue
                df = df.rename(columns={c: f"{c}" for c in df.columns})
                if base is None:
                    base = df
                else:
                    merge_keys = [c for c in ["ts_code", "end_date"] if c in df.columns and c in base.columns]
                    if merge_keys:
                        base = base.merge(df, on=merge_keys, how="outer", suffixes=("", f"_{k}"))

            if base is not None and not base.empty:
                out = FUNDA_DIR / f"{ticker}.parquet"
                base.to_parquet(out, index=False)

            if not data["fina_indicator"].empty:
                data["fina_indicator"]["ts_code"] = ticker
                fi_all.append(data["fina_indicator"])

            done.add(ticker)

            if (i + 1) % 50 == 0 or i + 1 == len(todo):
                print(f"  进度 {i+1}/{len(todo)} | 成功 {len(done)-len(done & set(failed))} | 失败 {len(failed)}")
                DONE_FILE.write_text("\n".join(sorted(done)), encoding="utf-8")
                # 定期写合并文件
                if fi_all:
                    fi_combined = pd.concat(fi_all, ignore_index=True).drop_duplicates(
                        subset=["ts_code", "end_date"] if "end_date" in fi_all[0].columns else ["ts_code"]
                    )
                    fi_combined.to_parquet(
                        ROOT / "data" / "alpha_universe" / "fina_indicator_combined.parquet",
                        index=False,
                        compression="snappy",
                    )

        except Exception as e:
            print(f"  [ERR] {ticker}: {e}")
            failed.append(ticker)

    # 最终保存
    DONE_FILE.write_text("\n".join(sorted(done)), encoding="utf-8")
    if failed:
        FAILED_FILE.write_text("\n".join(sorted(set(failed))), encoding="utf-8")
    if fi_all:
        fi_combined = pd.concat(fi_all, ignore_index=True)
        out_fi = ROOT / "data" / "alpha_universe" / "fina_indicator_combined.parquet"
        fi_combined.to_parquet(out_fi, index=False, compression="snappy")
        print(f"\n[Done] fina_indicator: {out_fi} ({fi_combined['ts_code'].nunique()} 只)")
    print(f"[Done] Step4 完成 | 成功 {len(done)} | 失败 {len(failed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
