"""scripts/validate_strategies.py — 策略历史回测验证系统.

对每个 InvestmentStrategy 的 BUY 信号做历史走查验证，
评估胜率、期望收益等指标，给出 ACCEPTABLE / WATCHLIST / AVOID 三档判断。

用法：
    python scripts/validate_strategies.py \
        --strategies-json reports/investment_pipeline/2024-12-31/strategies.json \
        --panel data/panel/test.parquet data/panel/val.parquet \
        --output reports/investment_pipeline/2024-12-31/validations.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_PRICE_FILE = _ROOT / "data" / "prices" / "csi300_daily_adj_close.parquet"
_PRICE_FILE_ALPHA = _ROOT / "data" / "raw" / "alpha_prices_panel.parquet"
_PRICE_FILE_DAILY = _ROOT / "data" / "raw" / "daily_prices_panel.parquet"


def _load_wide_price_df() -> "pd.DataFrame":
    """加载复权收盘价宽表（index=trade_date，columns=tickers）。

    优先 alpha_prices_panel（长表 pivot），退回 CSI300 宽表。
    """
    import pandas as _pd
    import pyarrow.parquet as _pq

    for longp in [_PRICE_FILE_ALPHA, _PRICE_FILE_DAILY]:
        if not longp.is_file():
            continue
        try:
            schema_names = _pq.ParquetFile(str(longp)).schema_arrow.names
            price_col = "adj_close" if "adj_close" in schema_names else "close"
            df = _pd.read_parquet(longp, columns=["trade_date", "ts_code", price_col])
            wide = df.pivot_table(index="trade_date", columns="ts_code", values=price_col, aggfunc="last")
            wide.index = _pd.to_datetime(wide.index)
            wide.columns.name = None
            return wide
        except Exception:  # noqa: BLE001
            pass
    if _PRICE_FILE.is_file():
        return _pd.read_parquet(_PRICE_FILE)
    return _pd.DataFrame()


# ── 数据类 ────────────────────────────────────────────────────────────────────

@dataclass
class StrategyValidationResult:
    ticker: str
    rating: str              # 来自 StrategyAgent
    validation_status: str   # ACCEPTABLE / WATCHLIST / AVOID

    # 历史验证数据
    n_signals: int
    win_rate: float
    avg_return: float
    max_loss: float
    hit_stop: float
    expected_value: float

    # 最终建议
    final_recommendation: str
    action: str              # "建议买入" / "等待机会" / "暂时回避"
    position_size: str       # 来自 StrategyAgent

    # 附加元数据
    stop_loss_pct: float = 0.05
    signal_dates: list[str] = None

    def __post_init__(self):
        if self.signal_dates is None:
            self.signal_dates = []


# ── 核心验证函数 ──────────────────────────────────────────────────────────────

def validate_strategy(
    strategy: dict[str, Any],
    price_df: pd.DataFrame,
    panel_df: pd.DataFrame | None = None,
    top_pct: float = 0.30,
    lookback_quarters: int = 8,
) -> StrategyValidationResult:
    """对单只股票策略做历史回测验证.

    Args:
        strategy: InvestmentStrategy 序列化的 dict
        price_df: 复权日线收盘价面板（index=trade_date，columns=tickers）
        panel_df: 因子面板（MultiIndex: as_of × ticker），含模型预测得分或特征
        top_pct: 定义"强信号"的百分位阈值（默认 Top 30%）
        lookback_quarters: 回溯季度数

    Returns:
        StrategyValidationResult
    """
    ticker = strategy["ticker"]
    as_of_str = strategy["as_of"]
    rating = strategy.get("rating", "观察")
    position_size = strategy.get("position_size", "不建仓")

    # 止损比例从策略参数反推
    stop_loss_price = strategy.get("stop_loss_price", 0.0)
    current_price = _get_current_price(ticker, as_of_str, price_df)
    if current_price and current_price > 0 and stop_loss_price > 0:
        stop_loss_pct = max(0.03, 1 - stop_loss_price / current_price)
    else:
        stop_loss_pct = 0.08  # 默认 8%

    # ── 若无面板数据，用价格面板做简单动量信号 ──────────────────────────────
    signals: list[tuple[str, float, float]] = []  # (date, entry_price, exit_price)

    if panel_df is not None and ticker in panel_df.index.get_level_values("ticker"):
        signals = _extract_signals_from_panel(
            ticker, as_of_str, panel_df, price_df, top_pct, lookback_quarters
        )
    else:
        # 降级：用价格面板做简单回测（每季度末动量信号）
        signals = _extract_signals_from_price(
            ticker, as_of_str, price_df, lookback_quarters
        )

    # ── 统计 ─────────────────────────────────────────────────────────────────
    if len(signals) == 0:
        return _no_data_result(ticker, rating, position_size, stop_loss_pct)

    returns = [
        (exit_p / entry_p - 1) if entry_p > 0 else 0.0
        for (_, entry_p, exit_p) in signals
    ]

    win_rate = sum(1 for r in returns if r > 0) / len(returns)
    avg_return = float(np.mean(returns))
    max_loss = float(min(returns)) if returns else 0.0
    hit_stop = sum(1 for r in returns if r < -stop_loss_pct) / len(returns)

    avg_loss = float(np.mean([r for r in returns if r < 0])) if any(r < 0 for r in returns) else 0.0
    expected_value = avg_return * win_rate + avg_loss * (1 - win_rate)

    signal_dates = [d for (d, _, _) in signals]

    # ── 可接受性判断 ──────────────────────────────────────────────────────────
    validation_status = _determine_status(
        expected_value, win_rate, hit_stop, rating
    )

    action, recommendation = _generate_recommendation(
        ticker, validation_status, rating, win_rate, avg_return, expected_value
    )

    return StrategyValidationResult(
        ticker=ticker,
        rating=rating,
        validation_status=validation_status,
        n_signals=len(signals),
        win_rate=round(win_rate, 3),
        avg_return=round(avg_return, 4),
        max_loss=round(max_loss, 4),
        hit_stop=round(hit_stop, 3),
        expected_value=round(expected_value, 4),
        final_recommendation=recommendation,
        action=action,
        position_size=position_size,
        stop_loss_pct=round(stop_loss_pct, 3),
        signal_dates=signal_dates,
    )


def _extract_signals_from_panel(
    ticker: str,
    as_of_str: str,
    panel_df: pd.DataFrame,
    price_df: pd.DataFrame,
    top_pct: float,
    lookback_quarters: int,
) -> list[tuple[str, float, float]]:
    """从因子面板提取历史信号（Top 30% 截面排名）."""
    as_of_cutoff = pd.Timestamp(as_of_str)
    ticker_panel = panel_df.xs(ticker, level="ticker", drop_level=False) if ticker in panel_df.index.get_level_values("ticker") else None
    if ticker_panel is None or len(ticker_panel) == 0:
        return []

    # 获取 as_of 之前的所有季度截面
    all_dates = panel_df.index.get_level_values("as_of").unique()
    hist_dates = sorted([d for d in all_dates if d < as_of_cutoff])[-lookback_quarters:]

    signals = []
    score_col = "forward_return_21d" if "forward_return_21d" in panel_df.columns else panel_df.columns[0]

    # 使用 momentum_6m 或 roe_ttm 近似打分（若无模型得分）
    rank_col = "momentum_6m" if "momentum_6m" in panel_df.columns else score_col

    for dt in hist_dates:
        try:
            cross_section = panel_df.xs(dt, level="as_of")
            if ticker not in cross_section.index or rank_col not in cross_section.columns:
                continue
            ticker_rank = cross_section[rank_col].rank(pct=True)
            if ticker not in ticker_rank.index:
                continue
            rank_pct = ticker_rank[ticker]
            # Top 30% 为强信号（高值 = 高排名）
            if rank_pct >= (1 - top_pct):
                entry_p = _get_price_on(ticker, dt, price_df, offset=1)
                exit_p = _get_price_on(ticker, dt, price_df, offset=22)
                if entry_p and exit_p and entry_p > 0:
                    signals.append((dt.strftime("%Y-%m-%d"), entry_p, exit_p))
        except Exception as e:
            logger.debug(f"[validate] {ticker} {dt}: {e}")

    return signals


def _extract_signals_from_price(
    ticker: str,
    as_of_str: str,
    price_df: pd.DataFrame,
    lookback_quarters: int,
) -> list[tuple[str, float, float]]:
    """降级策略：用价格动量生成历史信号（每季度末检查过去3月动量）."""
    if ticker not in price_df.columns:
        return []

    series = price_df[ticker].dropna()
    as_of_cutoff = pd.Timestamp(as_of_str)
    series = series[series.index < as_of_cutoff]
    if len(series) < 130:
        return []

    signals = []
    # 每约 63 个交易日生成一个截面
    step = 63
    n_periods = min(lookback_quarters * 2, len(series) // step)

    for i in range(n_periods):
        end_idx = len(series) - 1 - i * step
        if end_idx < 65:
            break
        start_idx = end_idx - 63
        ret_63 = series.iloc[end_idx] / series.iloc[start_idx] - 1
        # 动量 > 5% 视为信号
        if ret_63 > 0.05:
            dt = series.index[end_idx]
            entry_p = float(series.iloc[end_idx])
            exit_idx = min(end_idx + 21, len(series) - 1)
            exit_p = float(series.iloc[exit_idx])
            signals.append((dt.strftime("%Y-%m-%d"), entry_p, exit_p))

    return signals[:lookback_quarters * 2]


def _get_price_on(
    ticker: str,
    dt: pd.Timestamp,
    price_df: pd.DataFrame,
    offset: int = 0,
) -> float | None:
    """获取指定日期附近（+offset个交易日）的价格."""
    if ticker not in price_df.columns:
        return None
    series = price_df[ticker].dropna()
    future = series[series.index >= dt]
    if len(future) <= offset:
        return None
    return float(future.iloc[min(offset, len(future) - 1)])


def _get_current_price(ticker: str, as_of_str: str, price_df: pd.DataFrame) -> float | None:
    if ticker not in price_df.columns:
        return None
    series = price_df[ticker].dropna()
    cutoff = pd.Timestamp(as_of_str)
    series = series[series.index <= cutoff]
    return float(series.iloc[-1]) if len(series) > 0 else None


def _determine_status(
    expected_value: float,
    win_rate: float,
    hit_stop: float,
    rating: str,
) -> str:
    if rating == "回避":
        return "AVOID"
    if expected_value < -0.005 or hit_stop > 0.5:
        return "AVOID"
    if expected_value > 0.005 and win_rate > 0.45 and hit_stop < 0.35:
        return "ACCEPTABLE"
    if expected_value > 0 and win_rate > 0.4:
        return "WATCHLIST"
    return "AVOID"


def _generate_recommendation(
    ticker: str,
    status: str,
    rating: str,
    win_rate: float,
    avg_return: float,
    expected_value: float,
) -> tuple[str, str]:
    """生成操作建议和推荐文字."""
    if status == "ACCEPTABLE":
        action = "建议买入"
        rec = (
            f"{ticker} 历史信号胜率{win_rate:.0%}，期望月收益{avg_return:+.1%}，"
            f"验证通过。评级：{rating}，建议适当配置。"
        )
    elif status == "WATCHLIST":
        action = "等待机会"
        rec = (
            f"{ticker} 历史信号期望值{expected_value:+.2%}，处于观察区间。"
            f"评级：{rating}，建议持续跟踪，等待更明确信号。"
        )
    else:
        action = "暂时回避"
        rec = (
            f"{ticker} 历史回测显示风险较大（期望值{expected_value:+.2%}）。"
            f"评级：{rating}，建议回避或等待基本面改善。"
        )
    return action, rec


def _no_data_result(
    ticker: str, rating: str, position_size: str, stop_loss_pct: float
) -> StrategyValidationResult:
    return StrategyValidationResult(
        ticker=ticker,
        rating=rating,
        validation_status="WATCHLIST",
        n_signals=0,
        win_rate=0.0,
        avg_return=0.0,
        max_loss=0.0,
        hit_stop=0.0,
        expected_value=0.0,
        final_recommendation=f"{ticker} 历史信号数据不足，无法验证，进入观察名单。",
        action="等待机会",
        position_size=position_size,
        stop_loss_pct=stop_loss_pct,
    )


# ── 批量验证 ──────────────────────────────────────────────────────────────────

def batch_validate(
    strategies: list[dict[str, Any]],
    price_df: pd.DataFrame,
    panel_df: pd.DataFrame | None,
    only_buy_signals: bool = True,
) -> list[StrategyValidationResult]:
    """批量验证策略列表."""
    results = []
    for strat in strategies:
        rating = strat.get("rating", "观察")
        if only_buy_signals and rating not in ["积极关注", "谨慎关注"]:
            # 直接标记为 AVOID
            result = StrategyValidationResult(
                ticker=strat["ticker"],
                rating=rating,
                validation_status="AVOID",
                n_signals=0,
                win_rate=0.0,
                avg_return=0.0,
                max_loss=0.0,
                hit_stop=0.0,
                expected_value=0.0,
                final_recommendation=f"{strat['ticker']} 评级为{rating}，不进入回测验证。",
                action="暂时回避",
                position_size=strat.get("position_size", "不建仓"),
            )
        else:
            result = validate_strategy(strat, price_df, panel_df)
        results.append(result)
        logger.info(
            f"[validate] {strat['ticker']} → {result.validation_status} "
            f"(win_rate={result.win_rate:.0%}, EV={result.expected_value:+.2%})"
        )
    return results


# ── CLI 入口 ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="策略历史回测验证系统")
    p.add_argument("--strategies-json", type=Path, required=True, help="策略 JSON 文件")
    p.add_argument(
        "--panel",
        type=Path,
        nargs="+",
        default=[_ROOT / "data/panel/test.parquet", _ROOT / "data/panel/val.parquet"],
        help="因子面板 parquet 文件",
    )
    p.add_argument("--output", type=Path, default=None, help="验证结果 JSON 输出路径")
    p.add_argument("--top-pct", type=float, default=0.30, help="信号强度阈值（Top N%）")
    p.add_argument("--lookback-quarters", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 读取策略
    strategies_data = json.loads(args.strategies_json.read_text(encoding="utf-8"))
    if isinstance(strategies_data, dict):
        strategies = strategies_data.get("strategies", [strategies_data])
    else:
        strategies = strategies_data
    logger.info(f"[validate] 加载 {len(strategies)} 个策略")

    # 读取价格数据（优先 alpha 长表，pivot 成宽表）
    price_df = _load_wide_price_df()
    logger.info(f"[validate] 价格面板: {price_df.shape}")

    # 读取因子面板
    panel_df = None
    if args.panel:
        panels = []
        for p in args.panel:
            if p.exists():
                panels.append(pd.read_parquet(p))
        if panels:
            panel_df = pd.concat(panels).sort_index()
            logger.info(f"[validate] 因子面板: {panel_df.shape}")

    # 批量验证
    results = batch_validate(strategies, price_df, panel_df)

    # 输出
    output_data = [asdict(r) for r in results]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"[validate] 结果写入 → {args.output}")

    # 打印汇总
    acceptable = [r for r in results if r.validation_status == "ACCEPTABLE"]
    watchlist = [r for r in results if r.validation_status == "WATCHLIST"]
    avoid = [r for r in results if r.validation_status == "AVOID"]
    print(f"\n验证汇总：ACCEPTABLE={len(acceptable)}, WATCHLIST={len(watchlist)}, AVOID={len(avoid)}")
    for r in acceptable:
        print(f"  ✅ {r.ticker}: {r.action} | 胜率={r.win_rate:.0%} | EV={r.expected_value:+.2%}")


if __name__ == "__main__":
    main()
