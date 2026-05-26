"""scripts/build_position_nav.py — 30 日模拟盘持仓 NAV 曲线 + 止损模拟.

从 30 日 JSON 选股记录和历史价格数据中，还原每批次持仓的逐日净值曲线，
并模拟固定止损 / 追踪止损的实际效果。

Usage
-----
python scripts/build_position_nav.py --help
python scripts/build_position_nav.py --entry-date all --stop-loss -0.15 --trailing True
python scripts/build_position_nav.py --entry-date 20251009 --stop-loss -0.10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── 路径常量 ──────────────────────────────────────────────────────────────────
DAILY_DIR    = ROOT / "data" / "sim30d" / "daily"
POSITIONS_PQ = ROOT / "data" / "sim30d" / "positions.parquet"
RETURNS_PQ   = ROOT / "data" / "sim30d" / "stock_returns.parquet"
PRICES_PQ    = ROOT / "data" / "alpha_universe" / "alpha_prices_wide.parquet"
DEFAULT_OUT  = ROOT / "data" / "sim30d" / "nav"

# 插值锚点（交易日数 → return 列名），顺序影响插值
_RETURN_CHECKPOINTS = [(5, "return_1w"), (10, "return_2w"), (21, "return_21d"), (63, "return_3m")]


# ── 核心函数 ──────────────────────────────────────────────────────────────────

def build_nav_for_batch(
    entry_date: str,
    stock_list: list[str],
    prices_df: pd.DataFrame,
    stop_loss: float = -0.15,
    trailing: bool = False,
    holding_days: int = 63,
) -> dict:
    """还原一批持仓（entry_date 买入）的逐日 NAV 与止损模拟.

    Parameters
    ----------
    entry_date : str
        建仓日期，格式 'YYYYMMDD'（如 '20251009'）。
    stock_list : list[str]
        该批次最终选出的股票列表（ts_code）。
    prices_df : pd.DataFrame
        宽表价格（或标准化净值），index=DatetimeIndex，columns=ticker。
        对于无日频价格的股票，应提前用插值填充（见 _make_synthetic_prices）。
    stop_loss : float
        止损阈值，负数，如 -0.15 表示亏损 15% 触发。
    trailing : bool
        True = 追踪止损（从持仓期最高点回撤超阈值）；
        False = 固定止损（相对建仓价格亏损超阈值）。
    holding_days : int
        最大持仓交易日数，默认 63（约 3 个月）。

    Returns
    -------
    dict  见模块文档字符串中的结构说明。
    """
    entry_ts = pd.Timestamp(entry_date)

    # ── 取建仓日后的价格切片 ────────────────────────────────────────────────
    if prices_df.empty or not hasattr(prices_df.index, "min"):
        return _empty_result(entry_date, stock_list)

    if not isinstance(prices_df.index, pd.DatetimeIndex):
        return _empty_result(entry_date, stock_list)

    if entry_ts not in prices_df.index:
        # 找最近的下一个可用日期
        later = prices_df.index[prices_df.index >= entry_ts]
        if len(later) == 0:
            return _empty_result(entry_date, stock_list)
        entry_ts = later[0]

    price_window = prices_df.loc[entry_ts:].iloc[:holding_days + 1]
    if len(price_window) < 2:
        return _empty_result(entry_date, stock_list)

    trading_dates = price_window.index.tolist()

    # ── 逐股计算 NAV ─────────────────────────────────────────────────────────
    stocks_result: dict[str, dict] = {}
    all_nav_series: list[pd.Series] = []           # 等权组合用
    all_nav_with_sl: list[pd.Series] = []          # 止损版组合用
    stopped_stocks: list[str] = []

    for ticker in stock_list:
        if ticker not in price_window.columns:
            continue                                # 无价格数据，跳过

        px = price_window[ticker].dropna()
        if len(px) < 2:
            continue

        # 标准化：第一个有效价格 = 1.0
        entry_px = px.iloc[0]
        if entry_px == 0 or np.isnan(entry_px):
            continue
        nav = (px / entry_px).rename(ticker)

        # ── 固定止损 / 追踪止损 ───────────────────────────────────────────
        stop_triggered = False
        stop_date      = None
        stop_price     = None
        running_max    = 1.0                        # 追踪止损用

        nav_with_sl = nav.copy().astype(float)

        for i, (dt, val) in enumerate(nav.items()):
            if i == 0:
                running_max = val
                continue

            running_max = max(running_max, val)

            if trailing:
                drawdown = val / running_max - 1.0  # 从峰值回撤
            else:
                drawdown = val - 1.0                # 相对入场亏损

            if drawdown < stop_loss and not stop_triggered:
                stop_triggered = True
                stop_date      = dt.strftime("%Y%m%d") if hasattr(dt, "strftime") else str(dt)
                stop_price     = val
                # 止损后冻结 NAV 于止损当日水平
                nav_with_sl.loc[dt:] = val
                stopped_stocks.append(ticker)
                break

        # ── 统计 ──────────────────────────────────────────────────────────
        final_nav_vals  = nav.values
        nav_arr         = final_nav_vals
        peak            = np.maximum.accumulate(nav_arr)
        dd_arr          = nav_arr / peak - 1.0
        max_drawdown    = float(dd_arr.min())
        holding_actual  = int((~nav.isna()).sum()) - 1

        # 如果触发止损，实际持仓天数 = 到止损日的天数
        if stop_triggered and stop_date is not None:
            stop_ts = pd.Timestamp(stop_date)
            holding_actual = sum(1 for d in nav.index if d <= stop_ts) - 1

        stocks_result[ticker] = {
            "daily_returns":       list(zip(
                [d.strftime("%Y%m%d") for d in nav.index],
                nav.pct_change().fillna(0).round(6).tolist(),
            )),
            "cumulative_nav":      list(zip(
                [d.strftime("%Y%m%d") for d in nav.index],
                nav.round(6).tolist(),
            )),
            "stop_loss_triggered": stop_triggered,
            "stop_loss_date":      stop_date,
            "stop_loss_nav":       float(stop_price) if stop_price is not None else None,
            "max_drawdown":        round(max_drawdown, 6),
            "final_return":        round(float(nav.iloc[-1]) - 1.0, 6),
            "holding_days_actual": holding_actual,
        }

        # 对齐到统一 DatetimeIndex（用于等权组合计算）
        nav_aligned    = nav.reindex(trading_dates)
        sl_aligned     = nav_with_sl.reindex(trading_dates)
        all_nav_series.append(nav_aligned)
        all_nav_with_sl.append(sl_aligned)

    # ── 等权组合 NAV ─────────────────────────────────────────────────────────
    if not all_nav_series:
        return _empty_result(entry_date, stock_list)

    port_nav    = pd.concat(all_nav_series, axis=1).mean(axis=1)
    port_sl_nav = pd.concat(all_nav_with_sl, axis=1).mean(axis=1)

    def _portfolio_stats(nav_series: pd.Series) -> tuple[float, float]:
        arr  = nav_series.ffill().values
        peak = np.maximum.accumulate(arr)
        dd   = arr / peak - 1.0
        return float(arr[-1]) - 1.0, float(dd.min())

    port_ret,    port_mdd    = _portfolio_stats(port_nav)
    port_sl_ret, port_sl_mdd = _portfolio_stats(port_sl_nav)

    def _fmt_nav(series: pd.Series) -> list[list]:
        return [
            [d.strftime("%Y%m%d"), round(float(v), 6)]
            for d, v in series.items()
            if not np.isnan(v)
        ]

    return {
        "entry_date":    entry_date,
        "stocks":        stocks_result,
        "portfolio_nav": _fmt_nav(port_nav),
        "portfolio_final_return": round(port_ret, 6),
        "portfolio_max_drawdown": round(port_mdd, 6),
        "with_stop_loss": {
            "portfolio_nav":          _fmt_nav(port_sl_nav),
            "portfolio_final_return": round(port_sl_ret, 6),
            "portfolio_max_drawdown": round(port_sl_mdd, 6),
            "stopped_stocks":         stopped_stocks,
            "improvement_vs_no_stop": round(port_sl_ret - port_ret, 6),
        },
        "stop_loss":  stop_loss,
        "trailing":   trailing,
        "n_stocks":   len(stocks_result),
    }


def _empty_result(entry_date: str, stock_list: list[str]) -> dict:
    """无可用价格时的空结果占位."""
    return {
        "entry_date": entry_date,
        "stocks":     {t: {} for t in stock_list},
        "portfolio_nav": [],
        "portfolio_final_return": float("nan"),
        "portfolio_max_drawdown": float("nan"),
        "with_stop_loss": {
            "portfolio_nav": [],
            "portfolio_final_return": float("nan"),
            "portfolio_max_drawdown": float("nan"),
            "stopped_stocks": [],
            "improvement_vs_no_stop": float("nan"),
        },
        "stop_loss": -0.15,
        "trailing":  False,
        "n_stocks":  0,
    }


# ── 价格数据加载 ──────────────────────────────────────────────────────────────

def load_prices(returns_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """加载 alpha_prices_wide + 合成无价格股票的插值净值.

    对于在 alpha_prices_wide 中没有数据的股票，
    使用 stock_returns 的周期收益率（1w/2w/21d/3m）做线性插值，
    合成为伪日频净值序列（标准化到 entry=1.0 不适合这里，保留为伪价格）。

    Returns
    -------
    pd.DataFrame  宽表：index=DatetimeIndex, columns=ts_code
    """
    prices = pd.read_parquet(PRICES_PQ)

    if returns_df is None:
        returns_df = pd.read_parquet(RETURNS_PQ)

    # 生成合成价格并追加到 prices
    synthetic = _make_synthetic_prices(prices, returns_df)
    if not synthetic.empty:
        prices = pd.concat([prices, synthetic], axis=1)

    return prices


def _make_synthetic_prices(
    prices: pd.DataFrame,
    returns_df: pd.DataFrame,
) -> pd.DataFrame:
    """为无日频价格的股票生成线性插值伪价格序列.

    每只股票以建仓日 = 1.0 为基准，在 5/10/21/63 交易日锚点用
    已知累积收益率做 numpy 线性插值，生成每日数值。
    """
    # 仅处理 in_final 的股票
    in_final_col = "in_final" if "in_final" in returns_df.columns else None
    if in_final_col:
        final = returns_df[returns_df[in_final_col] == True].copy()
    else:
        final = returns_df.copy()

    # 哪些股票缺少日频价格
    missing = set(final["ticker"].unique()) - set(prices.columns)
    if not missing:
        return pd.DataFrame()

    prices_idx = prices.index  # reference DatetimeIndex (no duplicates)

    synth_cols: dict[str, pd.Series] = {}

    for ticker in missing:
        rows = final[final["ticker"] == ticker].sort_values("date")
        if rows.empty:
            continue

        # 取第一个建仓日（多批次取最早一次，避免重复日期冲突）
        row = rows.iloc[0]
        entry_ts = pd.Timestamp(row["date"])

        # 锚点：(交易日偏移, 累积净值)
        anchors_d = [0] + [cp[0] for cp in _RETURN_CHECKPOINTS]
        anchors_v = [1.0] + [
            1.0 + float(row.get(col, 0) or 0)
            for _, col in _RETURN_CHECKPOINTS
        ]

        # 生成该建仓日之后 63 个交易日的伪价格（对齐到 prices 的日期）
        later_dates = prices_idx[prices_idx >= entry_ts][:64]
        if len(later_dates) < 2:
            continue

        x_new  = np.arange(len(later_dates))
        interp = np.interp(x_new, anchors_d, anchors_v)

        synth_cols[ticker] = pd.Series(interp, index=later_dates, name=ticker)

    if not synth_cols:
        return pd.DataFrame()

    # 合并所有合成列，对齐到 prices_idx（只保留原有交易日）
    result = pd.concat(synth_cols.values(), axis=1)
    result = result.reindex(prices_idx)   # 对齐到已知无重复的 prices 索引
    return result


# ── 批量处理 ──────────────────────────────────────────────────────────────────

def process_all_batches(
    stop_loss: float = -0.15,
    trailing: bool   = False,
    out_dir: Path    = DEFAULT_OUT,
) -> list[dict]:
    """处理全部 30 个交易日批次，输出 nav/{date}.json."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载价格（含合成）
    returns_df = pd.read_parquet(RETURNS_PQ)
    prices_df  = load_prices(returns_df)

    # 加载每日 JSON
    daily_files = sorted(DAILY_DIR.glob("*.json"))
    if not daily_files:
        print("[build_position_nav] 未找到 data/sim30d/daily/*.json")
        return []

    summary_rows = []
    for fp in daily_files:
        with fp.open(encoding="utf-8") as fh:
            day = json.load(fh)

        entry_date = str(day["date"])
        picks = [s["ticker"] for s in day.get("system3_final_list", [])]

        if not picks:
            continue

        result = build_nav_for_batch(
            entry_date   = entry_date,
            stock_list   = picks,
            prices_df    = prices_df,
            stop_loss    = stop_loss,
            trailing     = trailing,
            holding_days = 63,
        )

        # 写 JSON
        out_path = out_dir / f"{entry_date}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2, default=str)

        # 打印摘要
        n_stopped   = len(result["with_stop_loss"]["stopped_stocks"])
        port_ret    = result["portfolio_final_return"]
        sl_ret      = result["with_stop_loss"]["portfolio_final_return"]
        improved    = sl_ret > port_ret

        port_str = f"{port_ret*100:+.1f}%" if not np.isnan(port_ret) else "N/A"
        sl_str   = f"{sl_ret*100:+.1f}%"  if not np.isnan(sl_ret)   else "N/A"
        diff_str = f"{(sl_ret-port_ret)*100:+.1f}%" if not (np.isnan(port_ret) or np.isnan(sl_ret)) else "N/A"

        print(
            f"  {entry_date}: 止损触发 {n_stopped:2d} 只，"
            f"不止损 {port_str}，有止损 {sl_str}，"
            f"止损{'改善了' if improved else '没改善'} {diff_str}"
        )

        summary_rows.append({
            "entry_date":      entry_date,
            "n_stocks":        result["n_stocks"],
            "n_stopped":       n_stopped,
            "port_ret":        port_ret,
            "port_mdd":        result["portfolio_max_drawdown"],
            "sl_ret":          sl_ret,
            "sl_mdd":          result["with_stop_loss"]["portfolio_max_drawdown"],
            "improvement":     sl_ret - port_ret if not np.isnan(sl_ret) and not np.isnan(port_ret) else float("nan"),
        })

    # 打印总体统计
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        print()
        print("── 汇总（全部批次）──")
        print(f"  批次总数:             {len(df)}")
        most_stopped = df.loc[df["n_stopped"].idxmax()]
        print(f"  止损触发最多批次:     {most_stopped['entry_date']}（{int(most_stopped['n_stopped'])} 只）")
        print(f"  不止损平均收益:       {df['port_ret'].mean()*100:+.2f}%")
        print(f"  不止损平均 MaxDD:     {df['port_mdd'].mean()*100:.2f}%")
        print(f"  有止损平均收益:       {df['sl_ret'].mean()*100:+.2f}%")
        print(f"  有止损平均 MaxDD:     {df['sl_mdd'].mean()*100:.2f}%")
        n_improved = (df["improvement"] > 0).sum()
        print(f"  止损改善收益的批次:   {n_improved}/{len(df)}")

    return summary_rows


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="构建 30 日模拟盘持仓逐日 NAV + 止损模拟",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--entry-date",
        default="all",
        help="建仓日期（YYYYMMDD 格式）或 'all' 处理全部批次",
    )
    p.add_argument(
        "--stop-loss",
        type=float,
        default=-0.15,
        dest="stop_loss",
        help="止损阈值（负数），如 -0.15 表示亏损 15%% 触发",
    )
    p.add_argument(
        "--trailing",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=False,
        help="是否追踪止损（从最高点回撤超阈值），默认 False",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="输出目录",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"[build_position_nav] 止损={args.stop_loss*100:.0f}%，"
          f"追踪止损={args.trailing}，输出={args.out}")

    if args.entry_date == "all":
        process_all_batches(
            stop_loss = args.stop_loss,
            trailing  = args.trailing,
            out_dir   = args.out,
        )
    else:
        returns_df = pd.read_parquet(RETURNS_PQ)
        prices_df  = load_prices(returns_df)

        daily_path = DAILY_DIR / f"{args.entry_date}.json"
        if not daily_path.exists():
            print(f"[ERROR] 找不到 {daily_path}")
            sys.exit(1)

        with daily_path.open(encoding="utf-8") as fh:
            day = json.load(fh)

        picks = [s["ticker"] for s in day.get("system3_final_list", [])]
        result = build_nav_for_batch(
            entry_date   = args.entry_date,
            stock_list   = picks,
            prices_df    = prices_df,
            stop_loss    = args.stop_loss,
            trailing     = args.trailing,
            holding_days = 63,
        )

        args.out.mkdir(parents=True, exist_ok=True)
        out_path = args.out / f"{args.entry_date}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2, default=str)

        n_stopped = len(result["with_stop_loss"]["stopped_stocks"])
        print(f"  {args.entry_date}: 止损触发 {n_stopped} 只，"
              f"组合收益 {result['portfolio_final_return']*100:+.1f}%，"
              f"止损版 {result['with_stop_loss']['portfolio_final_return']*100:+.1f}%")
        print(f"  已写入 → {out_path}")


if __name__ == "__main__":
    main()
