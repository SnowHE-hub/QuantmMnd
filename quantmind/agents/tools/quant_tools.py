"""quantmind.agents.tools.quant_tools — 量化信号工具.

读取 LightGBM 模型预测分数和 LLM Reranker 结论，供 TechnicalAgent 使用。
"""

from __future__ import annotations

import warnings
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

__all__ = [
    "get_factor_signal",
    "get_llm_rerank_thesis",
    "get_backtest_performance",
]

# 模型路径（从项目根目录相对路径）
_MODEL_PATH = Path("models/lgbm_ranker.pkl")
_RERANK_CACHE_DIR = Path("data/rerank_cache")


def get_factor_signal(ticker: str, as_of: date) -> dict[str, Any]:
    """获取 LightGBM 因子模型的预测信号和 SHAP 解释.

    Args:
        ticker: 股票代码
        as_of:  数据截止日期

    Returns:
        dict: score, rank, percentile, shap_values (factor contributions),
              top_positive_factors, top_negative_factors
    """
    result: dict[str, Any] = {
        "ticker": ticker,
        "as_of": str(as_of),
        "score": None,
        "rank": None,
        "percentile": None,
        "shap_values": {},
        "top_positive_factors": [],
        "top_negative_factors": [],
    }

    # 尝试从 snapshot 中读取因子数据并用已训练模型预测
    try:
        from quantmind.data.snapshot import load_snapshot
        snap = load_snapshot(as_of)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"get_factor_signal: snapshot load failed: {e}")
        return result

    # 读取 financial_indicators
    indicators: pd.DataFrame | None = snap.get("financial_indicators")
    daily_basic: pd.DataFrame | None = snap.get("daily_basic")

    if indicators is None and daily_basic is None:
        return result

    # 构造特征行
    ts_code = ticker
    feature_cols = ["pe_ttm", "pb", "roe_ttm", "accruals",
                    "distance_to_52w_high", "momentum_6m", "volatility_3m"]

    row_data: dict[str, float] = {}
    if daily_basic is not None and "ts_code" in daily_basic.columns:
        sub = daily_basic[daily_basic["ts_code"] == ts_code]
        if "trade_date" in daily_basic.columns:
            sub = sub[pd.to_datetime(sub["trade_date"]) <= pd.Timestamp(as_of)]
            sub = sub.sort_values("trade_date").tail(1)
        if len(sub) > 0:
            r = sub.iloc[0]
            for col in ["pe_ttm", "pb"]:
                if col in r:
                    try:
                        row_data[col] = float(r[col])
                    except (ValueError, TypeError):
                        pass

    if indicators is not None and "ts_code" in indicators.columns:
        sub = indicators[indicators["ts_code"] == ts_code]
        if "end_date" in indicators.columns:
            sub = sub[pd.to_datetime(sub["end_date"]) <= pd.Timestamp(as_of)]
            sub = sub.sort_values("end_date").tail(1)
        if len(sub) > 0:
            r = sub.iloc[0]
            for col in ["roe_ttm", "accruals"]:
                if col in r:
                    try:
                        row_data[col] = float(r[col])
                    except (ValueError, TypeError):
                        pass

    if not row_data:
        return result

    # 尝试加载已保存模型
    try:
        import pickle
        model_path = Path.cwd() / _MODEL_PATH
        if not model_path.exists():
            return result
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        available_cols = [c for c in feature_cols if c in row_data]
        if len(available_cols) < 2:
            return result

        X = np.array([[row_data.get(c, 0.0) for c in available_cols]], dtype=np.float32)
        score = float(model.predict(X)[0])
        result["score"] = round(score, 4)

        # SHAP 解释
        try:
            shap_dict = model.explain(X, tickers=[ticker])
            shap_vals = shap_dict.get(ticker, {})
            result["shap_values"] = shap_vals
            sorted_shap = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)
            result["top_positive_factors"] = [k for k, v in sorted_shap if v > 0][:3]
            result["top_negative_factors"] = [k for k, v in sorted_shap if v < 0][:3]
        except Exception:  # noqa: BLE001
            pass

    except Exception as e:  # noqa: BLE001
        logger.debug(f"get_factor_signal: model inference failed: {e}")

    return result


def get_llm_rerank_thesis(ticker: str, as_of: date) -> dict[str, Any]:
    """获取 LLM Reranker 对该股票的投资观点（从缓存读取）.

    Args:
        ticker: 股票代码
        as_of:  数据截止日期

    Returns:
        dict: rank, reason, portfolio_thesis, risk_warnings, is_fallback
    """
    result: dict[str, Any] = {
        "ticker": ticker,
        "as_of": str(as_of),
        "rank": None,
        "reason": "",
        "portfolio_thesis": "",
        "risk_warnings": [],
        "is_fallback": False,
        "found": False,
    }

    # 从缓存目录读取最近一期结果
    cache_dir = Path.cwd() / _RERANK_CACHE_DIR
    if not cache_dir.exists():
        return result

    import json
    # 找最近的缓存文件（as_of 当天或之前）
    cache_files = sorted(cache_dir.glob(f"rerank_*.json"), reverse=True)
    for cf in cache_files:
        try:
            data = json.loads(cf.read_text(encoding="utf-8"))
            results_list = data.get("results", data) if isinstance(data, dict) else data
            for item in results_list:
                if item.get("ticker") == ticker:
                    result.update({
                        "rank": item.get("rank"),
                        "reason": item.get("reason", ""),
                        "portfolio_thesis": item.get("portfolio_thesis", ""),
                        "risk_warnings": item.get("risk_warnings", []),
                        "is_fallback": item.get("is_fallback", False),
                        "found": True,
                    })
                    return result
        except Exception:  # noqa: BLE001
            continue

    return result


def get_backtest_performance(strategy_id: str) -> dict[str, Any]:
    """获取回测绩效（stub，Phase 6 补全）.

    Args:
        strategy_id: 策略标识符

    Returns:
        dict: annual_return, sharpe_ratio, max_drawdown, win_rate (stub zeros)
    """
    logger.debug(f"get_backtest_performance('{strategy_id}') — Phase 6 stub，返回空")
    return {
        "strategy_id": strategy_id,
        "annual_return": None,
        "sharpe_ratio": None,
        "max_drawdown": None,
        "win_rate": None,
        "note": "Phase 6 回测引擎尚未实现",
    }
