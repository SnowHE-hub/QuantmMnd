"""paper_trading_sim.py — 模拟盘回测与实时跟踪.

功能
====
1. 历史回测模式（--mode backtest）：
   从 data/recommendations/ 加载所有历史推荐，
   对每个推荐日模拟等权买入 Top-N，
   计算 1w/2w/21d/3m 多期收益，对比沪深300/中证500 基准

2. 前向模拟模式（--mode forward）：
   从指定开始日期起，每个交易日运行选股三系统，
   记录建仓意向，到期后结算，累积模拟净值曲线

3. 系统评估（--mode evaluate）：
   基于已积累的模拟盘数据，计算系统能力指标，
   输出优化建议（信号强度、因子权重调整方向）

输出
====
  data/paper_trading/
    positions.parquet     所有模拟持仓记录
    performance.json      多期绩效摘要
    nav_curve.parquet     净值曲线（forward 模式）
    optimization.json     优化信号（evaluate 模式）

用法
====
  # 历史回测（2025Q4 全部推荐日）
  python scripts/paper_trading_sim.py --mode backtest \\
      --start 2025-09-30 --end 2026-05-16 \\
      --recs data/recommendations/ \\
      --prices data/raw/alpha_prices_panel.parquet \\
      --benchmark data/raw/benchmark_prices.parquet

  # 评估并输出优化建议
  python scripts/paper_trading_sim.py --mode evaluate \\
      --positions data/paper_trading/positions.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)  # type: ignore[assignment]

# 多期持有天数（交易日数）
HORIZONS = {"1w": 5, "2w": 10, "21d": 21, "3m": 63}

# 基准代码（用于价格面板中的列名）
BENCH_CSI300 = "000300.SH"
BENCH_CSI500 = "000905.SH"


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载工具
# ─────────────────────────────────────────────────────────────────────────────

def load_price_pivot(prices_path: Path) -> pd.DataFrame:
    """加载价格面板，pivot 为 (trade_date × ts_code) 宽表."""
    df = pd.read_parquet(prices_path, columns=["ts_code", "trade_date", "close"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    wide = df.pivot_table(index="trade_date", columns="ts_code", values="close", aggfunc="last")
    wide = wide.sort_index()
    logger.info(f"价格宽表: {wide.shape}（{wide.index[0].date()} → {wide.index[-1].date()}）")
    return wide


def load_benchmark(bench_path: Path) -> pd.DataFrame:
    """加载基准指数收盘价."""
    df = pd.read_parquet(bench_path)
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index("trade_date").sort_index()
    return df


def scan_recommendations(recs_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    """扫描推荐目录，返回 [as_of, ticker, score, rank] 列表."""
    records = []
    # 优先使用顶层 YYYY-MM-DD.json；若无则用 YYYY-MM-DD/top10.json
    # 通过 as_of 去重，避免两种格式重复计入同一日期
    seen_dates: set = set()
    json_files = sorted(recs_dir.glob("*.json"))
    subdir_files = [d / "top10.json" for d in sorted(recs_dir.iterdir())
                    if d.is_dir() and (d / "top10.json").exists()]
    all_files = list(json_files) + subdir_files

    for fpath in all_files:
        try:
            payload = json.loads(fpath.read_text(encoding="utf-8"))
            as_of_str = payload.get("as_of") or payload.get("date")
            if not as_of_str:
                continue
            as_of = pd.Timestamp(as_of_str)
            if not (start <= as_of <= end):
                continue
            if as_of in seen_dates:
                continue
            seen_dates.add(as_of)
            top10 = payload.get("top10", [])
            for item in top10:
                records.append({
                    "as_of": as_of,
                    "ticker": item["ticker"],
                    "predicted_rank": item.get("lgbm_rank", item.get("rank", 999)),
                    "predicted_score": item.get("lgbm_score", 0.0),
                })
        except Exception as e:
            logger.warning(f"解析 {fpath} 失败: {e}")
            continue

    logger.info(f"扫描推荐记录: {len(records)} 条（{len(set(r['as_of'] for r in records))} 个日期）")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 收益计算
# ─────────────────────────────────────────────────────────────────────────────

def get_next_trading_date(price_index: pd.DatetimeIndex, as_of: pd.Timestamp) -> Optional[pd.Timestamp]:
    """获取 as_of 之后的第一个交易日（买入日）."""
    future = price_index[price_index > as_of]
    return future[0] if len(future) > 0 else None


def calc_return(
    price_wide: pd.DataFrame,
    ticker: str,
    entry_date: pd.Timestamp,
    holding_days: int,
) -> Optional[float]:
    """计算单只股票 N 交易日后的持有收益率."""
    if ticker not in price_wide.columns:
        return None
    prices = price_wide[ticker].dropna()
    idx = prices.index
    if entry_date not in idx:
        # 找最近的交易日
        avail = idx[idx >= entry_date]
        if len(avail) == 0:
            return None
        entry_date = avail[0]
    entry_pos = idx.get_loc(entry_date)
    exit_pos = entry_pos + holding_days
    if exit_pos >= len(idx):
        return None
    entry_price = prices.iloc[entry_pos]
    exit_price = prices.iloc[exit_pos]
    if entry_price == 0 or pd.isna(entry_price) or pd.isna(exit_price):
        return None
    return (exit_price - entry_price) / entry_price


# ─────────────────────────────────────────────────────────────────────────────
# 回测模式
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    recs_dir: Path,
    prices_path: Path,
    bench_path: Optional[Path],
    start: pd.Timestamp,
    end: pd.Timestamp,
    top_n: int,
    out_dir: Path,
) -> pd.DataFrame:
    """运行历史回测，返回持仓明细 DataFrame."""
    price_wide = load_price_pivot(prices_path)
    bench_wide: Optional[pd.DataFrame] = None
    if bench_path and bench_path.exists():
        bench_wide = load_benchmark(bench_path)

    recs = scan_recommendations(recs_dir, start, end)
    if not recs:
        logger.error("没有找到任何推荐记录，请检查 --recs 路径和日期范围")
        return pd.DataFrame()

    rows = []
    dates = sorted(set(r["as_of"] for r in recs))
    logger.info(f"回测日期: {[d.date() for d in dates]}")

    for as_of in dates:
        period_recs = [r for r in recs if r["as_of"] == as_of]
        period_recs = sorted(period_recs, key=lambda x: x["predicted_rank"])[:top_n]

        entry_date = get_next_trading_date(price_wide.index, as_of)
        if entry_date is None:
            logger.warning(f"  {as_of.date()}: 找不到买入日，跳过")
            continue

        # 计算各期基准收益
        bench_returns: dict[str, Optional[float]] = {}
        for name, hdays in HORIZONS.items():
            for bench_col in [BENCH_CSI300, BENCH_CSI500]:
                key = f"bench_{bench_col}_{name}"
                if bench_wide is not None and bench_col in bench_wide.columns:
                    bench_returns[key] = calc_return(bench_wide, bench_col, entry_date, hdays)
                elif bench_wide is None and bench_col in price_wide.columns:
                    bench_returns[key] = calc_return(price_wide, bench_col, entry_date, hdays)
                else:
                    bench_returns[key] = None

        for rec in period_recs:
            ticker = rec["ticker"]
            row: dict = {
                "as_of": as_of,
                "entry_date": entry_date,
                "ticker": ticker,
                "predicted_rank": rec["predicted_rank"],
                "predicted_score": rec["predicted_score"],
            }
            for name, hdays in HORIZONS.items():
                ret = calc_return(price_wide, ticker, entry_date, hdays)
                row[f"return_{name}"] = ret
                # 超额收益（相对于 CSI300）
                bench_key = f"bench_{BENCH_CSI300}_{name}"
                bench_ret = bench_returns.get(bench_key)
                if ret is not None and bench_ret is not None:
                    row[f"excess_{name}_vs_300"] = ret - bench_ret
                else:
                    row[f"excess_{name}_vs_300"] = None
            rows.append(row)
        logger.info(
            f"  {as_of.date()} → 买入日 {entry_date.date()}，"
            f"持仓 {len(period_recs)} 只"
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # 添加等权组合标识（同一日期所有持仓为一个组合）
    df["portfolio_date"] = df["as_of"].dt.strftime("%Y-%m-%d")

    out_dir.mkdir(parents=True, exist_ok=True)
    pos_path = out_dir / "positions.parquet"
    df.to_parquet(pos_path, index=False)
    logger.info(f"持仓明细已保存: {pos_path}（{len(df)} 条）")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 绩效统计
# ─────────────────────────────────────────────────────────────────────────────

def portfolio_stats(group: pd.Series) -> dict:
    """计算等权组合单期统计."""
    valid = group.dropna()
    if len(valid) == 0:
        return {"mean": None, "median": None, "win_rate": None, "n": 0}
    return {
        "mean": float(valid.mean()),
        "median": float(valid.median()),
        "win_rate": float((valid > 0).mean()),
        "n": len(valid),
        "std": float(valid.std()),
        "sharpe_annualized": float(valid.mean() / valid.std() * (252 ** 0.5)) if valid.std() > 0 else None,
    }


def calc_ic(df: pd.DataFrame, ret_col: str) -> Optional[float]:
    """计算预测分 vs 实际收益的 Spearman IC."""
    sub = df[["predicted_score", ret_col]].dropna()
    if len(sub) < 5:
        return None
    ic, _ = stats.spearmanr(sub["predicted_score"], sub[ret_col])
    return float(ic)


def calc_rank_ic(df: pd.DataFrame, ret_col: str) -> Optional[float]:
    """计算预测排名 vs 实际收益的 Spearman IC（排名越小越好，取负）."""
    sub = df[["predicted_rank", ret_col]].dropna()
    if len(sub) < 5:
        return None
    ic, _ = stats.spearmanr(-sub["predicted_rank"], sub[ret_col])
    return float(ic)


def generate_performance_report(df: pd.DataFrame, out_dir: Path) -> dict:
    """生成多维度绩效报告."""
    report: dict = {
        "generated_at": datetime.now().isoformat(),
        "total_positions": len(df),
        "n_dates": df["as_of"].nunique(),
        "date_range": {
            "start": str(df["as_of"].min().date()),
            "end": str(df["as_of"].max().date()),
        },
        "horizons": {},
        "per_date": {},
        "signals": {},
    }

    # 多期汇总
    for name in HORIZONS:
        ret_col = f"return_{name}"
        excess_col = f"excess_{name}_vs_300"
        if ret_col not in df.columns:
            continue

        stats_all = portfolio_stats(df[ret_col])
        excess_stats = portfolio_stats(df[excess_col]) if excess_col in df.columns else {}
        ic = calc_ic(df, ret_col)
        rank_ic = calc_rank_ic(df, ret_col)

        report["horizons"][name] = {
            **stats_all,
            "ic_spearman": ic,
            "rank_ic": rank_ic,
            "excess_return_vs_csi300": excess_stats,
        }

    # 按日期分解
    for as_of, grp in df.groupby("as_of"):
        date_str = str(as_of.date())
        report["per_date"][date_str] = {}
        for name in HORIZONS:
            ret_col = f"return_{name}"
            if ret_col in grp.columns:
                report["per_date"][date_str][name] = portfolio_stats(grp[ret_col])

    # 系统能力信号
    ret_3m = df["return_3m"].dropna()
    signals: dict = {}

    # 信号1：整体选股 IC
    ic_3m = calc_ic(df, "return_3m")
    if ic_3m is not None:
        if ic_3m > 0.05:
            signals["alpha_quality"] = {"status": "GOOD", "ic_3m": ic_3m, "msg": "3m IC > 0.05，选股质量良好"}
        elif ic_3m > 0:
            signals["alpha_quality"] = {"status": "WEAK", "ic_3m": ic_3m, "msg": "3m IC > 0 但偏低，建议增大样本"}
        else:
            signals["alpha_quality"] = {"status": "CRITICAL", "ic_3m": ic_3m, "msg": "3m IC <= 0，需检查模型"}

    # 信号2：短期 vs 长期 IC 衰减
    ic_1w = calc_ic(df, "return_1w")
    if ic_1w is not None and ic_3m is not None:
        signals["ic_decay_pattern"] = {
            "ic_1w": ic_1w, "ic_3m": ic_3m,
            "msg": "短期 IC 高于长期" if ic_1w > ic_3m else "长期 IC 高于短期（价值类信号）",
        }

    # 信号3：胜率分析
    win_3m = float((ret_3m > 0).mean()) if len(ret_3m) > 0 else None
    if win_3m is not None:
        signals["win_rate_3m"] = {
            "value": win_3m,
            "status": "OK" if win_3m >= 0.5 else "BELOW_50PCT",
            "msg": f"3m 胜率 {win_3m:.1%}，{'有效' if win_3m >= 0.5 else '偏低（尾部收益驱动）'}",
        }

    # 信号4：尾部收益依赖（mean vs median 偏差）
    mean_3m = float(ret_3m.mean()) if len(ret_3m) > 0 else None
    median_3m = float(ret_3m.median()) if len(ret_3m) > 0 else None
    if mean_3m is not None and median_3m is not None and median_3m != 0:
        tail_ratio = abs(mean_3m - median_3m) / (abs(median_3m) + 1e-6)
        signals["tail_dependency"] = {
            "mean_3m": mean_3m, "median_3m": median_3m,
            "tail_ratio": round(tail_ratio, 2),
            "status": "HIGH_TAIL" if tail_ratio > 2 else "NORMAL",
            "msg": "收益高度依赖尾部牛股，需增加分散度" if tail_ratio > 2 else "收益分布较均匀",
        }

    report["signals"] = signals

    perf_path = out_dir / "performance.json"
    perf_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"绩效报告已保存: {perf_path}")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# 净值曲线（等权组合 NAV）
# ─────────────────────────────────────────────────────────────────────────────

def build_nav_curve(df: pd.DataFrame, horizon: str, out_dir: Path) -> pd.DataFrame:
    """基于等权组合构建 NAV 曲线（每期再投资）."""
    ret_col = f"return_{horizon}"
    if ret_col not in df.columns:
        return pd.DataFrame()

    nav_rows = []
    nav = 1.0
    for as_of in sorted(df["as_of"].unique()):
        grp = df[df["as_of"] == as_of][ret_col].dropna()
        if len(grp) == 0:
            continue
        period_ret = float(grp.mean())
        nav = nav * (1 + period_ret)
        nav_rows.append({"as_of": as_of, "nav": nav, "period_return": period_ret})

    nav_df = pd.DataFrame(nav_rows)
    if not nav_df.empty:
        nav_path = out_dir / f"nav_curve_{horizon}.parquet"
        nav_df.to_parquet(nav_path, index=False)
        total_ret = nav - 1
        n_periods = len(nav_df)
        logger.info(
            f"NAV 曲线 [{horizon}]: {n_periods} 期，"
            f"累计收益 {total_ret:+.2%}，最终 NAV={nav:.4f}"
        )
    return nav_df


# ─────────────────────────────────────────────────────────────────────────────
# 优化评估模式
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluate(positions_path: Path, out_dir: Path) -> dict:
    """基于已有模拟盘数据输出优化建议."""
    df = pd.read_parquet(positions_path)
    logger.info(f"加载持仓记录: {len(df)} 条，{df['as_of'].nunique()} 个日期")

    out_dir.mkdir(parents=True, exist_ok=True)
    report = generate_performance_report(df, out_dir)

    # 构建优化建议
    recommendations: list[dict] = []
    signals = report.get("signals", {})

    alpha_q = signals.get("alpha_quality", {})
    if alpha_q.get("status") == "CRITICAL":
        recommendations.append({
            "priority": "HIGH",
            "action": "RETRAIN_LGBM",
            "reason": f"3m IC={alpha_q.get('ic_3m', 0):.4f}，需重训 LGBM 或调整特征集",
        })
    elif alpha_q.get("status") == "WEAK":
        recommendations.append({
            "priority": "MEDIUM",
            "action": "EXPAND_SAMPLE",
            "reason": "3m IC 为正但统计检验力不足，积累更多推荐日期后再评估",
        })

    tail = signals.get("tail_dependency", {})
    if tail.get("status") == "HIGH_TAIL":
        recommendations.append({
            "priority": "MEDIUM",
            "action": "INCREASE_TOP_N",
            "reason": f"尾部依赖 ratio={tail.get('tail_ratio', 0):.1f}，建议扩大持仓数至 15-20 只增加分散",
        })

    win = signals.get("win_rate_3m", {})
    if win.get("status") == "BELOW_50PCT":
        recommendations.append({
            "priority": "LOW",
            "action": "ADD_STOP_LOSS",
            "reason": f"3m 胜率{win.get('value', 0):.1%} < 50%，考虑加入止损条件（如跌破-15%平仓）",
        })

    decay = signals.get("ic_decay_pattern", {})
    if decay.get("ic_1w", 0) and decay.get("ic_3m", 0):
        if decay["ic_1w"] > decay["ic_3m"]:
            recommendations.append({
                "priority": "LOW",
                "action": "SHORTEN_HORIZON",
                "reason": "短期 IC 高于长期 IC，系统偏向动量/技术类信号，建议缩短持仓期",
            })

    opt_report = {
        "generated_at": datetime.now().isoformat(),
        "n_positions": len(df),
        "n_dates": df["as_of"].nunique(),
        "key_metrics": {
            horizon: {
                "ic": report["horizons"].get(horizon, {}).get("ic_spearman"),
                "mean_return": report["horizons"].get(horizon, {}).get("mean"),
                "win_rate": report["horizons"].get(horizon, {}).get("win_rate"),
            }
            for horizon in HORIZONS if horizon in report.get("horizons", {})
        },
        "signals": signals,
        "optimization_recommendations": recommendations,
        "next_review_date": "下一个季末面板更新后（预计 2026Q3，约 2026-10-31）重新评估",
    }

    opt_path = out_dir / "optimization.json"
    opt_path.write_text(json.dumps(opt_report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"优化报告已保存: {opt_path}")
    return opt_report


# ─────────────────────────────────────────────────────────────────────────────
# 打印摘要
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(report: dict) -> None:
    print("\n" + "=" * 60)
    print("  模拟盘绩效摘要")
    print("=" * 60)
    dr = report.get("date_range", {})
    print(f"  日期范围: {dr.get('start')} → {dr.get('end')}")
    print(f"  总持仓条数: {report.get('total_positions')}，推荐期数: {report.get('n_dates')}")
    print()
    print(f"  {'期限':<8} {'均值收益':>10} {'中位收益':>10} {'胜率':>8} {'IC(Spearman)':>14}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*14}")
    for name in ["1w", "2w", "21d", "3m"]:
        h = report.get("horizons", {}).get(name, {})
        mean_r = h.get("mean")
        med_r = h.get("median")
        win_r = h.get("win_rate")
        ic = h.get("ic_spearman")
        print(
            f"  {name:<8} "
            f"{f'{mean_r:+.2%}' if mean_r is not None else 'N/A':>10} "
            f"{f'{med_r:+.2%}' if med_r is not None else 'N/A':>10} "
            f"{f'{win_r:.1%}' if win_r is not None else 'N/A':>8} "
            f"{f'{ic:+.4f}' if ic is not None else 'N/A':>14}"
        )

    signals = report.get("signals", {})
    if signals:
        print()
        print("  系统信号：")
        for key, sig in signals.items():
            status = sig.get("status", "")
            msg = sig.get("msg", "")
            icon = "✅" if "GOOD" in status or "OK" in status or "NORMAL" in status else (
                "⚠️" if "WEAK" in status else "🔴"
            )
            print(f"    {icon} {key}: {msg}")

    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["backtest", "evaluate"], default="backtest",
                   help="运行模式（backtest=历史回测，evaluate=优化评估）")
    p.add_argument("--start", type=pd.Timestamp, default=pd.Timestamp("2025-09-30"),
                   help="回测开始日期（含）")
    p.add_argument("--end", type=pd.Timestamp, default=pd.Timestamp.today(),
                   help="回测结束日期（含）")
    p.add_argument("--recs", type=Path, default=Path("data/recommendations/"),
                   help="推荐记录目录")
    p.add_argument("--prices", type=Path, default=Path("data/raw/alpha_prices_panel.parquet"),
                   help="价格面板（ts_code/trade_date/close）")
    p.add_argument("--benchmark", type=Path, default=None,
                   help="基准指数价格面板（可选）")
    p.add_argument("--positions", type=Path, default=Path("data/paper_trading/positions.parquet"),
                   help="已有持仓文件（evaluate 模式使用）")
    p.add_argument("--out", type=Path, default=Path("data/paper_trading/"),
                   help="输出目录")
    p.add_argument("--top-n", type=int, default=10,
                   help="每期持仓数量")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.mode == "backtest":
        logger.info(f"运行历史回测：{args.start.date()} → {args.end.date()}")
        df = run_backtest(
            recs_dir=args.recs,
            prices_path=args.prices,
            bench_path=args.benchmark,
            start=args.start,
            end=args.end,
            top_n=args.top_n,
            out_dir=args.out,
        )
        if df.empty:
            logger.error("回测失败：无有效数据")
            return
        report = generate_performance_report(df, args.out)
        for horizon in HORIZONS:
            build_nav_curve(df, horizon, args.out)
        print_summary(report)

    elif args.mode == "evaluate":
        logger.info(f"运行优化评估：{args.positions}")
        if not args.positions.exists():
            logger.error(f"持仓文件不存在: {args.positions}，请先运行 --mode backtest")
            return
        opt = run_evaluate(args.positions, args.out)
        print("\n优化建议：")
        for rec in opt.get("optimization_recommendations", []):
            print(f"  [{rec['priority']}] {rec['action']}: {rec['reason']}")
        print()


if __name__ == "__main__":
    main()
