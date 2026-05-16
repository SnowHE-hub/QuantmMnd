"""quantmind.agents.tools.analysis_tools — 量化分析计算工具.

纯函数，不调用 LLM。LLM 只负责解读结果，不参与计算。
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

__all__ = [
    "compute_financial_ratios",
    "compute_dcf_valuation",
    "compute_comparable_multiples",
    "compute_technical_indicators",
    "run_factor_screening",
]


def compute_financial_ratios(financials: dict[str, Any]) -> dict[str, float]:
    """从财务数据 dict 计算常用财务比率.

    Args:
        financials: fetch_financials_pit 的返回值（flat key 格式）

    Returns:
        dict: roe, roa, gross_margin, net_margin, current_ratio, debt_to_equity,
              asset_turnover, revenue_growth, earnings_growth 等
    """
    def _get(key_candidates: list[str]) -> float | None:
        for k in key_candidates:
            for fk, v in financials.items():
                if fk.endswith(k) and v is not None:
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        pass
        return None

    revenue       = _get(["revenue", "total_revenue", "n_income_attr_p"])
    net_income    = _get(["n_income_attr_p", "net_income", "n_profit"])
    total_assets  = _get(["total_assets", "total_asset"])
    total_equity  = _get(["total_hldr_eqy_exc_min_int", "total_equity", "equity"])
    total_liab    = _get(["total_liab", "total_liabilities"])
    current_assets = _get(["total_cur_assets", "current_assets"])
    current_liab  = _get(["total_cur_liab", "current_liabilities"])
    gross_profit  = _get(["gross_profit", "grossprofit"])
    operate_cf    = _get(["n_cashflow_act", "operating_cashflow"])

    ratios: dict[str, float] = {}

    if net_income and total_equity and total_equity != 0:
        ratios["roe"] = round(net_income / total_equity, 4)

    if net_income and total_assets and total_assets != 0:
        ratios["roa"] = round(net_income / total_assets, 4)

    if gross_profit and revenue and revenue != 0:
        ratios["gross_margin"] = round(gross_profit / revenue, 4)

    if net_income and revenue and revenue != 0:
        ratios["net_margin"] = round(net_income / revenue, 4)

    if current_assets and current_liab and current_liab != 0:
        ratios["current_ratio"] = round(current_assets / current_liab, 2)

    if total_liab and total_equity and total_equity != 0:
        ratios["debt_to_equity"] = round(total_liab / total_equity, 2)

    if revenue and total_assets and total_assets != 0:
        ratios["asset_turnover"] = round(revenue / total_assets, 2)

    if operate_cf and net_income and net_income != 0:
        ratios["cfo_to_net_income"] = round(operate_cf / net_income, 2)

    return ratios


def compute_dcf_valuation(
    financials: dict[str, Any],
    wacc: float = 0.10,
    growth: float = 0.05,
    years: int = 5,
) -> dict[str, Any]:
    """简单 DCF 估值（基于自由现金流外推）.

    Args:
        financials: 财务数据 dict
        wacc:       加权平均资本成本（默认 10%）
        growth:     预测期年均增长率（默认 5%）
        years:      预测期（年）

    Returns:
        dict: intrinsic_value, terminal_value, pv_fcf_list,
              fcf_base, assumptions
    """
    def _get_fcf() -> float | None:
        for k, v in financials.items():
            if "cashflow" in k and ("n_cashflow_act" in k or "free_cash" in k):
                try:
                    return float(v)
                except (ValueError, TypeError):
                    pass
        return None

    def _get_shares() -> float | None:
        for k, v in financials.items():
            if "share_bonus_liab" in k or "total_share" in k:
                try:
                    return float(v) * 1e4  # Tushare 单位：万股
                except (ValueError, TypeError):
                    pass
        return None

    fcf_base = _get_fcf()
    shares = _get_shares()

    if fcf_base is None or fcf_base <= 0:
        return {
            "intrinsic_value": None,
            "error": "无法获取自由现金流数据",
            "assumptions": {"wacc": wacc, "growth": growth, "years": years},
        }

    # 预测未来 FCF
    pv_fcfs = []
    for t in range(1, years + 1):
        fcf_t = fcf_base * (1 + growth) ** t
        pv_t = fcf_t / (1 + wacc) ** t
        pv_fcfs.append(round(pv_t, 0))

    # 终值（Gordon Growth Model，永续增长率取 growth/2）
    terminal_growth = min(growth / 2, 0.03)
    terminal_fcf = fcf_base * (1 + growth) ** years * (1 + terminal_growth)
    terminal_value = terminal_fcf / (wacc - terminal_growth)
    pv_terminal = terminal_value / (1 + wacc) ** years

    total_value = sum(pv_fcfs) + pv_terminal

    result: dict[str, Any] = {
        "fcf_base": round(fcf_base, 0),
        "pv_fcf_list": pv_fcfs,
        "terminal_value": round(pv_terminal, 0),
        "intrinsic_value": round(total_value, 0),
        "assumptions": {
            "wacc": wacc,
            "growth": growth,
            "terminal_growth": terminal_growth,
            "years": years,
        },
    }

    if shares and shares > 0:
        result["per_share_value"] = round(total_value / shares, 2)

    return result


def compute_comparable_multiples(
    target: dict[str, Any],
    peers: list[dict[str, Any]],
) -> dict[str, Any]:
    """基于同行可比估值分析.

    Args:
        target: fetch_stock_basics 的目标公司数据
        peers:  list of fetch_stock_basics 结果

    Returns:
        dict: target_pe, peer_median_pe, peer_avg_pe,
              target_pb, peer_median_pb, peer_avg_pb,
              premium_discount_pe (%)
    """
    def _collect(items: list[dict], key: str) -> list[float]:
        return [float(x[key]) for x in items if x.get(key) is not None]

    target_pe = target.get("pe_ttm")
    target_pb = target.get("pb")

    peer_pes = _collect(peers, "pe_ttm")
    peer_pbs = _collect(peers, "pb")

    result: dict[str, Any] = {
        "target_pe": target_pe,
        "target_pb": target_pb,
        "n_peers": len(peers),
    }

    if peer_pes:
        result["peer_median_pe"] = round(float(np.median(peer_pes)), 1)
        result["peer_avg_pe"] = round(float(np.mean(peer_pes)), 1)
        if target_pe and result["peer_median_pe"] != 0:
            result["premium_discount_pe"] = round(
                (target_pe / result["peer_median_pe"] - 1) * 100, 1
            )

    if peer_pbs:
        result["peer_median_pb"] = round(float(np.median(peer_pbs)), 2)
        result["peer_avg_pb"] = round(float(np.mean(peer_pbs)), 2)
        if target_pb and result["peer_median_pb"] != 0:
            result["premium_discount_pb"] = round(
                (target_pb / result["peer_median_pb"] - 1) * 100, 1
            )

    return result


def compute_technical_indicators(prices: pd.DataFrame) -> dict[str, float]:
    """计算常用技术指标.

    Args:
        prices: DataFrame with columns close, high, low, volume

    Returns:
        dict: ma5, ma20, ma60, rsi14, macd, macd_signal, macd_hist,
              bb_upper, bb_lower, bb_width, atr14, volume_ma20
    """
    if prices.empty or "close" not in prices.columns:
        return {}

    close = prices["close"].astype(float)
    result: dict[str, float] = {}

    def _safe_last(series: pd.Series) -> float | None:
        if len(series) == 0:
            return None
        v = series.iloc[-1]
        return None if (isinstance(v, float) and math.isnan(v)) else float(v)

    # 移动平均
    for period in [5, 20, 60]:
        if len(close) >= period:
            result[f"ma{period}"] = round(float(close.rolling(period).mean().iloc[-1]), 2)

    # RSI-14
    if len(close) >= 15:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, float("nan"))
        rsi = 100 - 100 / (1 + rs)
        result["rsi14"] = round(float(rsi.iloc[-1]), 1)

    # MACD (12,26,9)
    if len(close) >= 35:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        hist = macd_line - signal_line
        result["macd"] = round(float(macd_line.iloc[-1]), 4)
        result["macd_signal"] = round(float(signal_line.iloc[-1]), 4)
        result["macd_hist"] = round(float(hist.iloc[-1]), 4)

    # Bollinger Bands (20, 2σ)
    if len(close) >= 20:
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        result["bb_upper"] = round(float((ma20 + 2 * std20).iloc[-1]), 2)
        result["bb_lower"] = round(float((ma20 - 2 * std20).iloc[-1]), 2)
        if result.get("bb_upper") and result.get("bb_lower"):
            result["bb_width"] = round(
                (result["bb_upper"] - result["bb_lower"]) / float(ma20.iloc[-1]), 4
            )

    # ATR-14
    if len(prices) >= 15 and "high" in prices.columns and "low" in prices.columns:
        high = prices["high"].astype(float)
        low = prices["low"].astype(float)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        result["atr14"] = round(float(tr.rolling(14).mean().iloc[-1]), 2)

    # 成交量均值
    if "volume" in prices.columns and len(prices) >= 20:
        vol = prices["volume"].astype(float)
        result["volume_ma20"] = round(float(vol.rolling(20).mean().iloc[-1]), 0)
        result["volume_ratio"] = round(float(vol.iloc[-1] / result["volume_ma20"]), 2)

    return result


def run_factor_screening(
    criteria: dict[str, Any],
    as_of: date,
) -> list[str]:
    """按因子条件筛选股票.

    Args:
        criteria: {"pe_ttm_max": 30, "roe_min": 0.10, ...}
        as_of:    数据截止日期

    Returns:
        list of ticker strings
    """
    try:
        from quantmind.data.snapshot import load_snapshot
        snap = load_snapshot(as_of)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"run_factor_screening: snapshot load failed: {e}")
        return []

    daily_basic: pd.DataFrame | None = snap.get("daily_basic")
    if daily_basic is None or daily_basic.empty:
        return []

    # 取每只股票最近一天
    if "trade_date" in daily_basic.columns and "ts_code" in daily_basic.columns:
        daily_basic = daily_basic.sort_values("trade_date").groupby("ts_code").tail(1)

    mask = pd.Series([True] * len(daily_basic), index=daily_basic.index)

    for key, val in criteria.items():
        if key.endswith("_max") and key[:-4] in daily_basic.columns:
            col = key[:-4]
            mask &= pd.to_numeric(daily_basic[col], errors="coerce") <= val
        elif key.endswith("_min") and key[:-4] in daily_basic.columns:
            col = key[:-4]
            mask &= pd.to_numeric(daily_basic[col], errors="coerce") >= val

    result_df = daily_basic[mask]
    if "ts_code" in result_df.columns:
        return result_df["ts_code"].tolist()
    return []
