"""run_30day_sim.py — 30 交易日全 A 股三系统模拟盘.

从 2025-10-09 起连续 30 个交易日，每日走完：
  系统1 筛选: 全A股 5500+ → Layer1-6 过滤 → 15 只候选
  系统2 分析: 为 15 只生成估值/动量/质量/技术四维评分 + 投资评级
  系统3 回测: 历史胜率验证 + 风险评估 → 最终可投名单 + 持仓策略
  收益跟踪: 记录每日投资组合的 1w/2w/21d/3m 实际收益

输出:
  data/sim30d/
    raw/           Tushare 原始数据缓存（只拉一次）
    daily/         每日三系统输出（YYYYMMDD.json）
    positions.parquet  30日全量持仓明细
    summary.json       综合绩效与优化报告

用法:
  # 步骤1：拉取数据（需联网，约10-15分钟）
  TUSHARE_TOKEN=xxx python scripts/run_30day_sim.py --step fetch

  # 步骤2：运行模拟（不需联网，基于缓存数据）
  python scripts/run_30day_sim.py --step simulate

  # 步骤3：评估与优化建议
  python scripts/run_30day_sim.py --step evaluate

  # 一键执行全流程
  TUSHARE_TOKEN=xxx python scripts/run_30day_sim.py --step all
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    logger = logging.getLogger(__name__)

# ─── 全局配置 ────────────────────────────────────────────────────────────────

SIM_DIR = _ROOT / "data" / "sim30d"
RAW_DIR = SIM_DIR / "raw"
DAILY_DIR = SIM_DIR / "daily"

START_DATE = "20251009"       # 第 1 个交易日（10月1日国庆，10月8日无交易，实际10月9日）
END_DATE   = "20251119"       # 第 30 个交易日
LOOKBACK_START = "20250701"   # 技术因子计算回溯起点（60+ 交易日）
RETURN_END = "20260511"       # 持仓收益计算截止日（价格数据最后日期）

TOP_FUNNEL  = 15              # 筛选系统最终输出
FINAL_TOP_N = 10              # 每日最终可投名单

CSI300 = "000300.SH"
CSI500 = "000905.SH"

# v4 模型特征（按数据可得性分组）
V4_FEATURES = [
    "pe_ttm", "pb", "ps_ttm", "book_to_market", "earnings_yield",
    "dividend_yield_ttm", "log_market_cap", "roe_approx",
    "momentum_1m", "momentum_3m", "momentum_6m", "reversal_1w",
    "rsi_14", "bollinger_position", "price_to_52w_low",
    "turnover_3m_avg", "volume_spike_5_30", "turnover_rate_quantile",
    "turnover_acceleration", "amplitude_quantile", "volume_price_corr_20d",
    "beta_60d", "beta_252d",
    "relative_strength_vs_csi300_60d", "relative_strength_vs_csi500_60d",
    "relative_strength_vs_csi300_120d",
    "log_circ_market_cap", "list_age_years", "free_float_ratio",
    # 以下北向/融资数据缺失，填 0
    "north_hold_ratio", "north_hold_amount", "north_hold_amount_change_20d",
    "north_hold_ratio_change_60d", "margin_buy_intensity",
    "margin_buy_amount_20d", "margin_balance_change_20d", "margin_short_ratio",
    "accruals", "ocf_to_revenue_ttm", "earnings_accel_q",
]


# ─── 数据获取模块 ─────────────────────────────────────────────────────────────

class TushareLoader:
    """Tushare 数据拉取器（带本地缓存）."""

    def __init__(self, token: str):
        import tushare as ts
        ts.set_token(token)
        self._pro = ts.pro_api()
        RAW_DIR.mkdir(parents=True, exist_ok=True)

    def _call(self, func, **kwargs):
        """带重试的 API 调用."""
        for attempt in range(3):
            try:
                df = func(**kwargs)
                time.sleep(0.35)   # 避免频率限制
                return df
            except Exception as e:
                logger.warning(f"API 失败 (attempt {attempt+1}): {e}")
                time.sleep(2)
        return pd.DataFrame()

    # ── 股票基础信息 ──────────────────────────────────────────────────────────

    def fetch_stock_basic(self) -> pd.DataFrame:
        cache = RAW_DIR / "stock_basic.parquet"
        if cache.exists():
            return pd.read_parquet(cache)
        logger.info("拉取全A股基础信息...")
        df = self._call(self._pro.stock_basic, exchange="", list_status="L",
                        fields="ts_code,name,list_date,industry,market,area")
        if not df.empty:
            df.to_parquet(cache, index=False)
            logger.info(f"  已保存 {len(df)} 只股票基础信息")
        return df

    # ── 交易日历 ─────────────────────────────────────────────────────────────

    def fetch_trade_cal(self) -> List[str]:
        cache = RAW_DIR / "trade_cal_30d.json"
        if cache.exists():
            return json.loads(cache.read_text())
        logger.info("拉取交易日历...")
        df = self._call(self._pro.trade_cal, exchange="SSE",
                        start_date=START_DATE, end_date=END_DATE, is_open="1")
        days = sorted(df["cal_date"].tolist()) if not df.empty else []
        cache.write_text(json.dumps(days))
        logger.info(f"  {len(days)} 个交易日: {days[0]} → {days[-1]}")
        return days

    # ── 基准指数日线 ──────────────────────────────────────────────────────────

    def fetch_index_daily(self, ts_code: str) -> pd.DataFrame:
        code_safe = ts_code.replace(".", "_")
        cache = RAW_DIR / f"index_{code_safe}.parquet"
        if cache.exists():
            return pd.read_parquet(cache)
        logger.info(f"拉取指数 {ts_code} 日线...")
        df = self._call(self._pro.index_daily, ts_code=ts_code,
                        start_date=LOOKBACK_START, end_date=RETURN_END,
                        fields="trade_date,open,high,low,close,vol,pct_chg")
        if not df.empty:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.sort_values("trade_date")
            df.to_parquet(cache, index=False)
        return df

    # ── 全市场日线价格（按交易日批量） ───────────────────────────────────────

    def fetch_all_daily_prices(self) -> pd.DataFrame:
        """拉取回溯期 + 模拟期的全部日线数据（按日缓存）."""
        cache = RAW_DIR / "prices_all.parquet"
        if cache.exists():
            return pd.read_parquet(cache)

        logger.info("拉取全市场日线价格（2025-07 至 2025-11-19）...")

        # 先获取日历：回溯 60 个交易日 + 模拟 30 个交易日 + 3 个月后价格
        cal_df = self._call(self._pro.trade_cal, exchange="SSE",
                            start_date=LOOKBACK_START, end_date=RETURN_END, is_open="1")
        all_dates = sorted(cal_df["cal_date"].tolist()) if not cal_df.empty else []
        logger.info(f"  共 {len(all_dates)} 个交易日需拉取")

        all_dfs = []
        for i, date in enumerate(all_dates):
            day_cache = RAW_DIR / f"day_{date}.parquet"
            if day_cache.exists():
                all_dfs.append(pd.read_parquet(day_cache))
                continue
            df = self._call(self._pro.daily, trade_date=date,
                            fields="ts_code,trade_date,open,high,low,close,pre_close,vol,amount")
            if not df.empty:
                df.to_parquet(day_cache, index=False)
                all_dfs.append(df)
            if (i + 1) % 20 == 0:
                logger.info(f"  价格进度: {i+1}/{len(all_dates)}")

        if all_dfs:
            df_all = pd.concat(all_dfs, ignore_index=True)
            df_all["trade_date"] = pd.to_datetime(df_all["trade_date"])
            df_all.to_parquet(cache, index=False)
            logger.info(f"  价格汇总: {df_all.shape}")
            return df_all
        return pd.DataFrame()

    # ── 日级估值数据（PE/PB/换手率等）───────────────────────────────────────

    def fetch_daily_basic(self, trade_days: List[str]) -> pd.DataFrame:
        """拉取模拟期 30 日的 PE/PB/市值/换手率."""
        cache = RAW_DIR / "daily_basic_30d.parquet"
        if cache.exists():
            return pd.read_parquet(cache)
        logger.info("拉取日级估值数据（30个交易日）...")
        dfs = []
        for i, date in enumerate(trade_days):
            day_cache = RAW_DIR / f"basic_{date}.parquet"
            if day_cache.exists():
                dfs.append(pd.read_parquet(day_cache))
                continue
            df = self._call(self._pro.daily_basic, trade_date=date,
                            fields="ts_code,pe_ttm,pb,ps_ttm,dv_ttm,total_mv,circ_mv,turnover_rate_f,free_share_ratio")
            if not df.empty:
                df["trade_date"] = date
                df.to_parquet(day_cache, index=False)
                dfs.append(df)
            if (i + 1) % 10 == 0:
                logger.info(f"  估值进度: {i+1}/{len(trade_days)}")
        if dfs:
            merged = pd.concat(dfs, ignore_index=True)
            merged.to_parquet(cache, index=False)
            return merged
        return pd.DataFrame()


# ─── 技术指标计算模块 ──────────────────────────────────────────────────────────

class FeatureBuilder:
    """从价格序列计算技术/动量因子."""

    def __init__(self, prices: pd.DataFrame, csi300: pd.DataFrame, csi500: pd.DataFrame):
        # prices: (ts_code, trade_date, open, high, low, close, vol, amount)
        self.prices = prices.copy()
        self.prices["trade_date"] = pd.to_datetime(self.prices["trade_date"])
        self.csi300 = csi300.set_index("trade_date")["close"] if "close" in csi300 else pd.Series()
        self.csi500 = csi500.set_index("trade_date")["close"] if "close" in csi500 else pd.Series()
        # 预计算宽表（pivot: 日期 × 股票）
        self._close_wide: Optional[pd.DataFrame] = None
        self._vol_wide: Optional[pd.DataFrame] = None
        self._high_wide: Optional[pd.DataFrame] = None
        self._low_wide: Optional[pd.DataFrame] = None
        self._amount_wide: Optional[pd.DataFrame] = None

    def _get_wide(self, col: str) -> pd.DataFrame:
        attr = f"_{col}_wide"
        if getattr(self, attr, None) is None:
            wide = self.prices.pivot_table(index="trade_date", columns="ts_code",
                                            values=col, aggfunc="last")
            setattr(self, attr, wide.sort_index())
        return getattr(self, attr)

    def compute_as_of(self, as_of: pd.Timestamp) -> pd.DataFrame:
        """
        给定一个截面日期，计算所有股票的技术因子。
        返回 DataFrame，index=ts_code，columns=factor_names
        """
        close = self._get_wide("close")
        vol   = self._get_wide("vol")
        high  = self._get_wide("high")
        low   = self._get_wide("low")

        # 截至 as_of 的历史序列
        hist = close[close.index <= as_of]
        if len(hist) < 21:
            return pd.DataFrame()

        # 最后收盘价
        last_close = hist.iloc[-1]

        results: Dict[str, pd.Series] = {}

        # 动量因子
        def safe_mom(n: int) -> pd.Series:
            if len(hist) < n + 1:
                return pd.Series(dtype=float)
            return (hist.iloc[-1] / hist.iloc[-n - 1] - 1).replace([np.inf, -np.inf], np.nan)

        results["momentum_1m"]  = safe_mom(21)
        results["momentum_3m"]  = safe_mom(63)
        results["momentum_6m"]  = safe_mom(126)
        results["reversal_1w"]  = safe_mom(5)

        # 52周低点距离
        hist_252 = hist.tail(252)
        results["price_to_52w_low"] = (last_close / hist_252.min() - 1).replace([np.inf, -np.inf], np.nan)

        # RSI-14
        delta = hist.diff().tail(15)
        gain = delta.clip(lower=0).mean()
        loss = (-delta.clip(upper=0)).mean()
        rs = gain / (loss + 1e-9)
        results["rsi_14"] = (100 - 100 / (1 + rs)).iloc[-1] / 100  # 归一化到 0-1

        # 布林带位置
        hist20 = hist.tail(20)
        ma20 = hist20.mean()
        std20 = hist20.std()
        results["bollinger_position"] = ((last_close - (ma20 - 2 * std20)) /
                                          (4 * std20 + 1e-9)).clip(0, 1)

        # 换手率相关
        vol_hist = vol[vol.index <= as_of]
        if len(vol_hist) >= 60:
            results["turnover_3m_avg"] = vol_hist.tail(60).mean()
            # 量比 (spike)
            vol5 = vol_hist.tail(5).mean()
            vol30 = vol_hist.tail(30).mean()
            results["volume_spike_5_30"] = (vol5 / (vol30 + 1e-9)).replace([np.inf, -np.inf], np.nan)
            # 量价相关
            close20 = hist.tail(20)
            vol20 = vol_hist.tail(20)
            if len(close20) == len(vol20) and len(close20) == 20:
                corr = close20.corrwith(vol20)
                results["volume_price_corr_20d"] = corr
            # 换手率分位（近20日换手相对60日的分位）
            turn20 = vol_hist.tail(20).mean()
            turn60 = vol_hist.tail(60).quantile(0.5)
            results["turnover_rate_quantile"] = (turn20 / (turn60 + 1e-9)).clip(0, 3) / 3
            # 换手率加速度
            turn5 = vol_hist.tail(5).mean()
            turn20_prev = vol_hist.iloc[-25:-5].mean() if len(vol_hist) >= 25 else vol_hist.tail(20).mean()
            results["turnover_acceleration"] = (turn5 / (turn20_prev + 1e-9)).replace([np.inf,-np.inf], np.nan)

        # 振幅分位
        high_hist = high[high.index <= as_of]
        low_hist  = low[low.index <= as_of]
        if len(high_hist) >= 20:
            amp20 = ((high_hist.tail(20) - low_hist.tail(20)) / (low_hist.tail(20) + 1e-9)).mean()
            amp60 = ((high_hist.tail(60) - low_hist.tail(60)) / (low_hist.tail(60) + 1e-9)).mean() if len(high_hist) >= 60 else amp20
            results["amplitude_quantile"] = (amp20 / (amp60 + 1e-9)).clip(0, 3) / 3

        # Beta 相对 CSI300
        if len(self.csi300) > 0:
            mkt = self.csi300[self.csi300.index <= as_of]
            stock_ret = hist.pct_change().tail(60)
            mkt_ret = mkt.pct_change().tail(60)
            common_idx = stock_ret.index.intersection(mkt_ret.index)
            if len(common_idx) >= 20:
                mkt_r = mkt_ret.loc[common_idx]
                mkt_var = mkt_r.var()
                if mkt_var > 0:
                    beta_60 = stock_ret.loc[common_idx].apply(
                        lambda s: s.cov(mkt_r) / mkt_var if s.notna().sum() >= 10 else np.nan)
                    results["beta_60d"] = beta_60
            stock_ret252 = hist.pct_change().tail(252)
            mkt_ret252 = mkt.pct_change().tail(252)
            common252 = stock_ret252.index.intersection(mkt_ret252.index)
            if len(common252) >= 60:
                mkt_r252 = mkt_ret252.loc[common252]
                mkt_var252 = mkt_r252.var()
                if mkt_var252 > 0:
                    results["beta_252d"] = stock_ret252.loc[common252].apply(
                        lambda s: s.cov(mkt_r252) / mkt_var252 if s.notna().sum() >= 30 else np.nan)

            # 相对强度
            mkt_close = self.csi300[self.csi300.index <= as_of]
            if len(mkt_close) >= 60:
                rs_60 = (last_close / hist.iloc[-60] - 1) - float(mkt_close.iloc[-1] / mkt_close.iloc[-60] - 1)
                results["relative_strength_vs_csi300_60d"] = rs_60
            if len(mkt_close) >= 120:
                rs_120 = (last_close / hist.iloc[-120] - 1) - float(mkt_close.iloc[-1] / mkt_close.iloc[-120] - 1)
                results["relative_strength_vs_csi300_120d"] = rs_120

        if len(self.csi500) > 0:
            mkt500 = self.csi500[self.csi500.index <= as_of]
            if len(mkt500) >= 60:
                rs500 = (last_close / hist.iloc[-60] - 1) - float(mkt500.iloc[-1] / mkt500.iloc[-60] - 1)
                results["relative_strength_vs_csi500_60d"] = rs500

        # 组合结果：pd.DataFrame(dict of Series) → index=ts_code, cols=factors
        factor_df = pd.DataFrame(results)
        factor_df.index.name = "ts_code"
        return factor_df

    def build_factor_matrix(self, as_of: pd.Timestamp,
                             basic_df: pd.DataFrame) -> pd.DataFrame:
        """
        合并技术因子 + 估值因子，返回完整因子矩阵.
        basic_df: daily_basic for as_of date
        """
        tech = self.compute_as_of(as_of)
        if tech.empty:
            return pd.DataFrame()

        # 估值因子（来自 daily_basic）
        basic = basic_df.copy()
        basic = basic.set_index("ts_code")
        basic["book_to_market"]    = 1 / basic["pb"].replace(0, np.nan)
        basic["earnings_yield"]    = 1 / basic["pe_ttm"].replace(0, np.nan)
        basic["dividend_yield_ttm"] = basic["dv_ttm"] / 100
        basic["log_market_cap"]    = np.log(basic["total_mv"].replace(0, np.nan) + 1)
        basic["log_circ_market_cap"] = np.log(basic["circ_mv"].replace(0, np.nan) + 1)
        basic["roe_approx"]        = (basic["pb"] / basic["pe_ttm"].replace(0, np.nan)).clip(-5, 5)
        basic["free_float_ratio"]  = basic.get("free_share_ratio", pd.Series(dtype=float))

        fundamental_cols = ["pe_ttm", "pb", "ps_ttm", "book_to_market", "earnings_yield",
                             "dividend_yield_ttm", "log_market_cap", "log_circ_market_cap",
                             "roe_approx", "free_float_ratio", "turnover_rate_f"]

        # 合并
        merged = tech.join(basic[fundamental_cols], how="left")

        # 填充北向/融资数据（无此来源，填0）
        for col in ["north_hold_ratio", "north_hold_amount", "north_hold_amount_change_20d",
                    "north_hold_ratio_change_60d", "margin_buy_intensity",
                    "margin_buy_amount_20d", "margin_balance_change_20d", "margin_short_ratio",
                    "accruals", "ocf_to_revenue_ttm", "earnings_accel_q"]:
            if col not in merged.columns:
                merged[col] = 0.0

        # 上市天数（从 stock_basic 推算）
        if "list_age_years" not in merged.columns:
            merged["list_age_years"] = 5.0  # 默认值

        return merged


# ─── 三系统实现 ───────────────────────────────────────────────────────────────

class SelectionSystem:
    """系统1：6层筛选漏斗，全A→15只候选."""

    def __init__(self, stock_basic: pd.DataFrame):
        self.stock_basic = stock_basic.set_index("ts_code")

    def run(self, as_of: pd.Timestamp, factor_df: pd.DataFrame,
            model, top_n: int = 15) -> pd.DataFrame:
        df = factor_df.copy()
        df.index.name = "ts_code"
        n_start = len(df)

        # Layer 1: 基础质量（排除 ST / 上市<180天 / 极小市值）
        # 排除 ST（名称含 ST）
        if "name" in self.stock_basic.columns:
            st_codes = self.stock_basic[
                self.stock_basic["name"].str.contains("ST|退", na=False)
            ].index
            df = df[~df.index.isin(st_codes)]
        # 排除新股（上市<180天）
        if "list_date" in self.stock_basic.columns:
            list_dates = pd.to_datetime(
                self.stock_basic["list_date"], format="%Y%m%d", errors="coerce")
            new_ipo = list_dates[list_dates >= as_of - pd.Timedelta(days=180)].index
            df = df[~df.index.isin(new_ipo)]
        # 排除市值<30亿（Tushare total_mv 单位为万元，30亿=300000万元，log≈12.61）
        if "log_market_cap" in df.columns:
            df = df[df["log_market_cap"].fillna(0) >= math.log(30 * 1e4)]
        logger.info(f"  Layer1 基础质量: {n_start} → {len(df)}")

        # Layer 2: 流动性（换手率 > 0 分位的 30%，即相对充裕）
        if "turnover_rate_f" in df.columns:
            turnover_thresh = df["turnover_rate_f"].quantile(0.30)
            df = df[df["turnover_rate_f"].fillna(0) >= turnover_thresh]
        elif "turnover_3m_avg" in df.columns:
            t_thresh = df["turnover_3m_avg"].quantile(0.30)
            df = df[df["turnover_3m_avg"].fillna(0) >= t_thresh]
        logger.info(f"  Layer2 流动性:   → {len(df)}")

        # Layer 3: 趋势（近期没有极端大跌，动量不是最差尾部）
        if "reversal_1w" in df.columns:
            df = df[df["reversal_1w"].fillna(-1) > -0.15]   # 近1周不超过-15%
        if "momentum_1m" in df.columns:
            mom_thresh = df["momentum_1m"].quantile(0.10)    # 排除最差10%
            df = df[df["momentum_1m"].fillna(mom_thresh) >= mom_thresh]
        logger.info(f"  Layer3 趋势:     → {len(df)}")

        # Layer 4: 基本面（PE 合理区间、ROE 为正）
        if "pe_ttm" in df.columns:
            df = df[(df["pe_ttm"].fillna(-1) > 0) & (df["pe_ttm"].fillna(999) < 150)]
        if "pb" in df.columns:
            df = df[df["pb"].fillna(-1) > 0]
        if "roe_approx" in df.columns:
            df = df[df["roe_approx"].fillna(-1) > 0]
        logger.info(f"  Layer4 基本面:   → {len(df)}")

        # Layer 5: LGBM 打分 → Top 50
        # 获取模型期望的特征列表（严格按顺序）
        model_feats = None
        for attr in ["_feature_names", "feature_names_", "feature_name_"]:
            v = getattr(model, attr, None)
            if v is not None:
                model_feats = list(v)
                break
        if model_feats is None:
            try:
                model_feats = model._model.feature_name()
            except Exception:
                pass
        if model_feats is None:
            model_feats = V4_FEATURES  # 降级回预设列表

        # 构建严格对齐的特征矩阵（缺失列填 0）
        X_df = pd.DataFrame(index=df.index)
        for feat in model_feats:
            if feat in df.columns:
                X_df[feat] = df[feat].fillna(0.0)
            else:
                X_df[feat] = 0.0
        X = X_df.to_numpy(dtype=np.float32)

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores_raw = model.predict(X)
            scores = pd.Series(scores_raw, index=df.index, name="lgbm_score_raw")
            scores_pct = scores.rank(pct=True).rename("lgbm_score")
            df = df.join(scores).join(scores_pct)
            df = df.sort_values("lgbm_score", ascending=False).head(50)
        except Exception as e:
            logger.warning(f"  LGBM 打分失败: {e}，使用综合得分排序")
            comp = pd.Series(0.0, index=df.index)
            for col, sign in [("momentum_3m", 1), ("earnings_yield", 1),
                               ("relative_strength_vs_csi300_60d", 1), ("rsi_14", -1)]:
                if col in df.columns:
                    comp += sign * df[col].rank(pct=True).fillna(0.5)
            df["lgbm_score"] = comp.rank(pct=True)
            df["lgbm_score_raw"] = comp
            df = df.sort_values("lgbm_score", ascending=False).head(50)
        logger.info(f"  Layer5 LGBM:     → {len(df)}")

        # Layer 6: 行业分散 → Top N（每个行业最多 3 只）
        if "industry" in self.stock_basic.columns:
            df["industry"] = self.stock_basic["industry"].reindex(df.index)
            selected = []
            industry_count: Dict[str, int] = {}
            for ticker, row in df.iterrows():
                ind = row.get("industry", "其他") or "其他"
                if industry_count.get(ind, 0) < 3:
                    selected.append(ticker)
                    industry_count[ind] = industry_count.get(ind, 0) + 1
                if len(selected) >= top_n:
                    break
            df = df.loc[selected]
        else:
            df = df.head(top_n)

        logger.info(f"  Layer6 分散:     → {len(df)} 只候选")
        return df


class AnalysisSystem:
    """系统2：四维评分 + 投资评级."""

    WEIGHTS = {"value": 0.30, "momentum": 0.25, "quality": 0.25, "technical": 0.20}

    def analyze(self, candidates: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
        df = candidates.copy()

        # 估值得分（0-100）：earnings_yield + book_to_market 高 = 便宜
        val_factors = ["earnings_yield", "book_to_market", "dividend_yield_ttm"]
        df["value_score"] = self._factor_score(df, val_factors, direction=1)

        # 动量得分：短中期动量综合
        mom_factors = ["momentum_1m", "momentum_3m", "relative_strength_vs_csi300_60d"]
        df["momentum_score"] = self._factor_score(df, mom_factors, direction=1)

        # 质量得分：ROE/流动性
        qual_factors = ["roe_approx", "free_float_ratio"]
        df["quality_score"] = self._factor_score(df, qual_factors, direction=1)

        # 技术得分：RSI + 布林带位置 + 价格距52周低点
        tech_factors_pos = ["price_to_52w_low"]
        tech_factors_neg = ["rsi_14", "bollinger_position"]  # 不过热为佳
        df["technical_score"] = (
            self._factor_score(df, tech_factors_pos, direction=1) * 0.5 +
            self._factor_score(df, tech_factors_neg, direction=-1) * 0.5
        )

        # 综合得分
        df["composite_score"] = (
            df["value_score"]     * self.WEIGHTS["value"] +
            df["momentum_score"]  * self.WEIGHTS["momentum"] +
            df["quality_score"]   * self.WEIGHTS["quality"] +
            df["technical_score"] * self.WEIGHTS["technical"]
        )

        # 投资评级
        df["rating"] = pd.cut(df["composite_score"],
                               bins=[0, 40, 55, 70, 100],
                               labels=["观望", "持有", "买入", "强烈买入"],
                               include_lowest=True)

        # 建议持仓期：高动量 → 短期；高估值低动量 → 长期
        df["suggested_horizon"] = df.apply(self._suggest_horizon, axis=1)

        return df.sort_values("composite_score", ascending=False)

    @staticmethod
    def _factor_score(df: pd.DataFrame, factors: list, direction: int = 1) -> pd.Series:
        scores = []
        for f in factors:
            if f in df.columns:
                rank = df[f].rank(pct=True) * 100
                scores.append(rank * direction if direction == 1 else 100 - rank)
        if not scores:
            return pd.Series(50.0, index=df.index)
        return pd.concat(scores, axis=1).mean(axis=1)

    @staticmethod
    def _suggest_horizon(row) -> str:
        mom_score = row.get("momentum_score", 50)
        val_score = row.get("value_score", 50)
        if mom_score >= 70:
            return "1w"       # 强动量，短持
        elif mom_score >= 55 and val_score < 50:
            return "2w"       # 中动量，偏短
        elif val_score >= 65:
            return "3m"       # 深度价值，长持
        else:
            return "21d"      # 中性，标准持仓期


class BacktestSystem:
    """系统3：历史胜率验证 + 风险评估 → 最终可投名单."""

    def __init__(self, prices: pd.DataFrame):
        self._close = prices.pivot_table(
            index="trade_date", columns="ts_code", values="close", aggfunc="last").sort_index()

    def validate(self, candidates: pd.DataFrame, as_of: pd.Timestamp,
                 lookback_days: int = 60) -> pd.DataFrame:
        """
        对候选股票做历史回测验证：计算过去 lookback_days 的持有表现.
        """
        hist = self._close[self._close.index <= as_of].tail(lookback_days + 1)
        if len(hist) < 21:
            candidates["hist_win_rate"] = 0.5
            candidates["hist_sharpe"]   = 0.0
            candidates["hist_maxdd"]    = -0.2
            candidates["risk_level"]    = "中"
            candidates["investable"]    = True
            return candidates

        results = []
        for ticker in candidates.index:
            if ticker not in hist.columns:
                results.append({"ts_code": ticker, "hist_win_rate": 0.5,
                                 "hist_sharpe": 0.0, "hist_maxdd": -0.2})
                continue
            prices = hist[ticker].dropna()
            if len(prices) < 10:
                results.append({"ts_code": ticker, "hist_win_rate": 0.5,
                                 "hist_sharpe": 0.0, "hist_maxdd": -0.2})
                continue
            rets = prices.pct_change().dropna()
            # 历史胜率（5日滚动收益为正的比例）
            roll5 = prices.pct_change(5).dropna()
            win_rate = float((roll5 > 0).mean())
            # Sharpe（日收益率年化）
            sharpe = float(rets.mean() / (rets.std() + 1e-9) * math.sqrt(252))
            # 最大回撤
            roll_max = prices.cummax()
            dd = (prices / roll_max - 1)
            max_dd = float(dd.min())
            results.append({"ts_code": ticker, "hist_win_rate": win_rate,
                             "hist_sharpe": sharpe, "hist_maxdd": max_dd})

        bt_df = pd.DataFrame(results)
        if not bt_df.empty and "ts_code" in bt_df.columns:
            bt_df = bt_df.set_index("ts_code")
        out = candidates.join(bt_df, how="left") if not bt_df.empty else candidates.copy()

        # 风险等级
        def risk_level(row):
            if row.get("hist_maxdd", -1) < -0.30 or row.get("hist_sharpe", 0) < -0.5:
                return "高"
            elif row.get("hist_maxdd", -1) < -0.20 or row.get("hist_sharpe", 0) < 0:
                return "中"
            return "低"

        out["risk_level"] = out.apply(risk_level, axis=1)
        # 可投资判断：历史胜率 > 40%，最大回撤 > -35%，风险非极高
        out["investable"] = (
            (out["hist_win_rate"].fillna(0.4) >= 0.40) &
            (out["hist_maxdd"].fillna(-0.5) > -0.40) &
            (out["risk_level"] != "高")
        )
        logger.info(f"  回测验证: {out['investable'].sum()} / {len(out)} 只通过")
        return out.sort_values("composite_score", ascending=False)


# ─── 持仓收益计算 ─────────────────────────────────────────────────────────────

def calc_horizon_returns(
    final_picks: pd.DataFrame,
    as_of: pd.Timestamp,
    prices: pd.DataFrame,
    horizons: Dict[str, int],
) -> Dict[str, float]:
    """计算等权组合各持仓期收益."""
    close = prices.pivot_table(
        index="trade_date", columns="ts_code", values="close", aggfunc="last").sort_index()

    # 买入日 = as_of 之后第一个交易日
    future = close.index[close.index > as_of]
    if len(future) == 0:
        return {}
    entry_date = future[0]
    tickers = final_picks.index.tolist()

    results = {}
    for hz_name, h_days in horizons.items():
        exit_future = close.index[close.index > entry_date]
        if len(exit_future) < h_days:
            results[hz_name] = None
            continue
        exit_date = exit_future[h_days - 1]

        rets = []
        for ticker in tickers:
            if ticker not in close.columns:
                continue
            try:
                entry_p = close.loc[entry_date, ticker]
                exit_p  = close.loc[exit_date, ticker]
                if pd.notna(entry_p) and pd.notna(exit_p) and entry_p > 0:
                    rets.append((exit_p - entry_p) / entry_p)
            except Exception:
                pass

        if rets:
            results[hz_name] = {
                "mean":    float(np.mean(rets)),
                "median":  float(np.median(rets)),
                "win_rate": float(np.mean([r > 0 for r in rets])),
                "max":     float(np.max(rets)),
                "min":     float(np.min(rets)),
                "n":       len(rets),
                "entry_date": str(entry_date.date()),
                "exit_date":  str(exit_date.date()),
            }
        else:
            results[hz_name] = None

    return results


# ─── 主模拟流程 ───────────────────────────────────────────────────────────────

def step_fetch(token: str) -> None:
    """步骤1: 拉取并缓存所有原始数据."""
    logger.info("=" * 60)
    logger.info("步骤1：数据拉取（Tushare）")
    loader = TushareLoader(token)

    loader.fetch_stock_basic()
    days_30 = loader.fetch_trade_cal()
    loader.fetch_index_daily(CSI300)
    loader.fetch_index_daily(CSI500)
    loader.fetch_all_daily_prices()
    loader.fetch_daily_basic(days_30)

    logger.info("数据拉取完成！所有文件已缓存至 data/sim30d/raw/")


def step_simulate(model_path: Path) -> pd.DataFrame:
    """步骤2: 逐日运行三系统，生成每日持仓."""
    logger.info("=" * 60)
    logger.info("步骤2：30日三系统模拟")
    DAILY_DIR.mkdir(parents=True, exist_ok=True)

    # 加载缓存数据
    stock_basic = pd.read_parquet(RAW_DIR / "stock_basic.parquet")
    trade_days  = json.loads((RAW_DIR / "trade_cal_30d.json").read_text())
    csi300_df   = pd.read_parquet(RAW_DIR / f"index_{CSI300.replace('.','_')}.parquet")
    csi500_df   = pd.read_parquet(RAW_DIR / f"index_{CSI500.replace('.','_')}.parquet")

    # 价格数据
    prices_all = pd.read_parquet(RAW_DIR / "prices_all.parquet")
    prices_all["trade_date"] = pd.to_datetime(prices_all["trade_date"])

    # 估值数据
    basic_30d = pd.read_parquet(RAW_DIR / "daily_basic_30d.parquet")

    # 加载 LGBM 模型
    import pickle
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"模型加载: {model_path}")

    # 初始化特征构建器
    feat_builder = FeatureBuilder(prices_all, csi300_df, csi500_df)
    selector     = SelectionSystem(stock_basic)
    analyzer     = AnalysisSystem()
    backtester   = BacktestSystem(prices_all)

    HORIZONS = {"1w": 5, "2w": 10, "21d": 21, "3m": 63}

    all_rows = []

    for day_idx, date_str in enumerate(trade_days):
        as_of = pd.Timestamp(date_str)
        logger.info(f"\n{'='*50}")
        logger.info(f"Day{day_idx+1:02d} | {date_str}")

        daily_cache = DAILY_DIR / f"{date_str}.json"
        if daily_cache.exists():
            logger.info("  (已有缓存，跳过)")
            payload = json.loads(daily_cache.read_text())
            # 仍然补充 returns（若缺失）
            if "returns" not in payload:
                pass
        else:
            # 当日估值数据
            today_basic = basic_30d[basic_30d["trade_date"] == date_str].copy()
            if today_basic.empty:
                logger.warning(f"  {date_str} 无估值数据，跳过")
                continue

            # === 系统1: 筛选 ===
            logger.info("  [系统1] 构建因子矩阵...")
            factor_df = feat_builder.build_factor_matrix(as_of, today_basic)
            if factor_df.empty:
                logger.warning(f"  {date_str} 因子矩阵为空，跳过")
                continue

            logger.info(f"  [系统1] 运行筛选漏斗（{len(factor_df)} 只 → {TOP_FUNNEL} 只）...")
            candidates = selector.run(as_of, factor_df, model, top_n=TOP_FUNNEL)
            if candidates.empty:
                logger.warning(f"  {date_str} 筛选结果为空，跳过")
                continue

            # === 系统2: 分析 ===
            logger.info("  [系统2] 四维因子分析...")
            analyzed = analyzer.analyze(candidates, as_of)

            # === 系统3: 回测验证 ===
            logger.info("  [系统3] 历史胜率验证...")
            validated = backtester.validate(analyzed, as_of)
            final_list = validated[validated["investable"]].head(FINAL_TOP_N)
            if final_list.empty:
                logger.warning(f"  {date_str} 回测验证后无可投股票，使用 Top-5 候选")
                final_list = validated.head(5)

            # === 持仓收益计算 ===
            logger.info(f"  [收益] 计算 {len(final_list)} 只股票多期收益...")
            returns = calc_horizon_returns(final_list, as_of, prices_all, HORIZONS)

            # 输出
            def safe_val(v):
                if isinstance(v, (np.integer,)): return int(v)
                if isinstance(v, (np.floating,)): return None if np.isnan(v) else float(v)
                if isinstance(v, (bool, np.bool_)): return bool(v)
                if pd.isna(v) if not isinstance(v, (list, dict, str)) else False: return None
                return v

            payload = {
                "date": date_str,
                "day_index": day_idx + 1,
                "system1_candidates": [
                    {
                        "rank": i + 1,
                        "ticker": t,
                        "name": stock_basic.set_index("ts_code")["name"].get(t, ""),
                        "industry": stock_basic.set_index("ts_code")["industry"].get(t, ""),
                        "lgbm_score": safe_val(row.get("lgbm_score")),
                        "lgbm_score_raw": safe_val(row.get("lgbm_score_raw")),
                    }
                    for i, (t, row) in enumerate(candidates.iterrows())
                ],
                "system2_analysis": {
                    t: {
                        "composite_score": safe_val(row.get("composite_score")),
                        "value_score": safe_val(row.get("value_score")),
                        "momentum_score": safe_val(row.get("momentum_score")),
                        "quality_score": safe_val(row.get("quality_score")),
                        "technical_score": safe_val(row.get("technical_score")),
                        "rating": str(row.get("rating", "")),
                        "suggested_horizon": str(row.get("suggested_horizon", "21d")),
                    }
                    for t, row in analyzed.iterrows()
                },
                "system3_final_list": [
                    {
                        "rank": i + 1,
                        "ticker": t,
                        "name": stock_basic.set_index("ts_code")["name"].get(t, ""),
                        "industry": stock_basic.set_index("ts_code")["industry"].get(t, ""),
                        "composite_score": safe_val(row.get("composite_score")),
                        "rating": str(row.get("rating", "")),
                        "suggested_horizon": str(row.get("suggested_horizon", "21d")),
                        "hist_win_rate": safe_val(row.get("hist_win_rate")),
                        "hist_sharpe": safe_val(row.get("hist_sharpe")),
                        "hist_maxdd": safe_val(row.get("hist_maxdd")),
                        "risk_level": str(row.get("risk_level", "中")),
                        "investable": bool(row.get("investable", True)),
                    }
                    for i, (t, row) in enumerate(final_list.iterrows())
                ],
                "returns": returns,
            }
            daily_cache.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"  ✅ Day{day_idx+1:02d} 完成，最终 {len(final_list)} 只，已写入 {daily_cache.name}")

        # 收集持仓行（用于汇总）
        for item in payload.get("system3_final_list", []):
            for hz, ret_data in payload.get("returns", {}).items():
                if ret_data is None:
                    continue
                # per-stock returns 需要 positions-level 数据，这里记录 portfolio-level
            row_base = {
                "date": date_str,
                "day_index": payload["day_index"],
                "ticker": item["ticker"],
                "name": item.get("name", ""),
                "industry": item.get("industry", ""),
                "rank": item["rank"],
                "composite_score": item.get("composite_score"),
                "rating": item.get("rating", ""),
                "suggested_horizon": item.get("suggested_horizon", "21d"),
                "hist_win_rate": item.get("hist_win_rate"),
                "hist_sharpe": item.get("hist_sharpe"),
                "hist_maxdd": item.get("hist_maxdd"),
                "risk_level": item.get("risk_level", "中"),
            }
            # 附上组合级别的各期收益（所有该期股票共享同一组合均值）
            for hz in ["1w", "2w", "21d", "3m"]:
                r = payload.get("returns", {}).get(hz)
                row_base[f"portfolio_{hz}_mean"] = r["mean"] if r else None
                row_base[f"portfolio_{hz}_win_rate"] = r["win_rate"] if r else None
            all_rows.append(row_base)

    positions = pd.DataFrame(all_rows)
    pos_path = SIM_DIR / "positions.parquet"
    positions.to_parquet(pos_path, index=False)
    logger.info(f"\n持仓明细已保存: {pos_path}（{len(positions)} 行）")
    return positions


def step_evaluate(positions: pd.DataFrame) -> dict:
    """步骤3: 综合评估与优化建议."""
    logger.info("\n" + "=" * 60)
    logger.info("步骤3：30日模拟盘综合评估")

    # 按日期汇总组合绩效
    daily_perf = []
    for date, grp in positions.groupby("date"):
        row = {"date": date, "n_stocks": len(grp), "day": grp["day_index"].iloc[0]}
        for hz in ["1w", "2w", "21d", "3m"]:
            col = f"portfolio_{hz}_mean"
            vals = grp[col].dropna()
            row[f"{hz}_return"] = float(vals.mean()) if len(vals) > 0 else None
        daily_perf.append(row)
    perf_df = pd.DataFrame(daily_perf).sort_values("date")

    # 多期汇总统计
    report: dict = {
        "generated_at": datetime.now().isoformat(),
        "simulation_period": f"{positions['date'].min()} → {positions['date'].max()}",
        "n_trading_days": positions["date"].nunique(),
        "total_positions": len(positions),
        "avg_stocks_per_day": float(positions.groupby("date").size().mean()),
        "horizons": {},
        "system_evaluation": {},
        "factor_insights": {},
        "optimization": {},
    }

    print("\n" + "=" * 65)
    print("  QuantMind 30日模拟盘综合报告")
    print("=" * 65)
    print(f"  模拟期: {report['simulation_period']}")
    print(f"  交易日数: {report['n_trading_days']}  日均持仓: {report['avg_stocks_per_day']:.1f} 只")
    print()

    print(f"  {'日期':<12} {'1w':>8} {'2w':>8} {'21d':>8} {'3m':>9}  持仓数")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*9}  -----")
    for _, row in perf_df.iterrows():
        def fmt(v): return "%+.2f%%" % (v*100) if v is not None else "  N/A "
        print("  %-12s %8s %8s %8s %9s  %d" % (
            row["date"], fmt(row.get("1w_return")),
            fmt(row.get("2w_return")), fmt(row.get("21d_return")),
            fmt(row.get("3m_return")), row["n_stocks"]))

    print()
    print(f"  {'期限':<8} {'均值':>9} {'中位':>9} {'胜率':>8} {'最大':>9} {'最小':>9}  {'IR':>6}")
    print(f"  {'-'*8} {'-'*9} {'-'*9} {'-'*8} {'-'*9} {'-'*9}  {'-'*6}")
    for hz in ["1w", "2w", "21d", "3m"]:
        col = f"{hz}_return"
        vals = perf_df[col].dropna()
        if len(vals) == 0:
            continue
        mean_r  = float(vals.mean())
        med_r   = float(vals.median())
        win_r   = float((vals > 0).mean())
        max_r   = float(vals.max())
        min_r   = float(vals.min())
        std_r   = float(vals.std())
        ir      = mean_r / std_r if std_r > 0 else 0
        report["horizons"][hz] = {
            "mean": mean_r, "median": med_r, "win_rate": win_r,
            "max": max_r, "min": min_r, "std": std_r, "ir": ir, "n": len(vals)
        }
        print("  %-8s %+8.2f%% %+8.2f%% %7.1f%% %+8.2f%% %+8.2f%%  %+.3f" % (
            hz, mean_r*100, med_r*100, win_r*100, max_r*100, min_r*100, ir))

    # 三系统能力评估
    print()
    print("  【三系统能力评估】")

    # 系统1: 漏斗有效性
    if "hist_win_rate" in positions.columns:
        avg_win = positions["hist_win_rate"].mean()
        print(f"  系统1 (筛选): 候选股历史胜率均值={avg_win:.1%}")
        report["system_evaluation"]["s1_avg_hist_win_rate"] = float(avg_win)

    # 系统2: 综合得分与实际收益相关性
    if "composite_score" in positions.columns and "portfolio_21d_mean" in positions.columns:
        sub = positions[["composite_score", "portfolio_21d_mean"]].dropna()
        if len(sub) >= 10:
            ic, p = stats.spearmanr(sub["composite_score"], sub["portfolio_21d_mean"])
            print(f"  系统2 (分析): 综合得分→21d收益 IC={ic:+.4f} (p={p:.3f})")
            report["system_evaluation"]["s2_composite_vs_21d_ic"] = float(ic)

    # 系统3: 回测验证有效性
    if "investable" in positions.columns:
        investable_ratio = positions.groupby("date").apply(
            lambda g: g["investable"].mean() if "investable" in g else 0.5).mean()
    if "hist_sharpe" in positions.columns:
        avg_sharpe = positions["hist_sharpe"].mean()
        print(f"  系统3 (回测): 候选股历史 Sharpe 均值={avg_sharpe:.3f}")
        report["system_evaluation"]["s3_avg_hist_sharpe"] = float(avg_sharpe)

    # 评级分布
    if "rating" in positions.columns:
        rating_dist = positions["rating"].value_counts(normalize=True)
        print(f"  评级分布: {dict(rating_dist.round(3))}")

    # 行业集中度
    if "industry" in positions.columns:
        top_inds = positions["industry"].value_counts().head(5)
        print(f"  行业 Top5: {dict(top_inds)}")

    # 优化建议
    print()
    print("  【优化建议】")
    opt_recs = []

    best_hz = max(report["horizons"].items(), key=lambda x: x[1]["ir"], default=(None, {}))[0]
    if best_hz:
        best_ir = report["horizons"][best_hz]["ir"]
        best_wr = report["horizons"][best_hz]["win_rate"]
        print(f"  1. 最优持仓期: {best_hz}（IR={best_ir:+.3f}, 胜率={best_wr:.1%}）")
        opt_recs.append({"priority": "HIGH", "action": f"SET_HORIZON_{best_hz}", "detail": f"IR={best_ir:.3f}"})

    for hz, v in report["horizons"].items():
        if v.get("ir", 0) < 0:
            print(f"  2. {hz} 期 IR={v['ir']:+.3f} < 0，建议避免该持仓期或优化因子权重")
            opt_recs.append({"priority": "MEDIUM", "action": "REVIEW_FACTORS",
                             "detail": f"{hz} IC为负，检查当前市场 regime"})

    if report["horizons"].get("21d", {}).get("win_rate", 0) < 0.5:
        print("  3. 21d 胜率 < 50%：建议加入止损机制（-10% 止损线）")
        opt_recs.append({"priority": "MEDIUM", "action": "ADD_STOP_LOSS", "detail": "threshold=-10%"})

    print()
    print("  【下阶段行动计划】")
    print("  1. 对比每日筛选结果与实际市场表现，识别筛选层的误杀率")
    print("  2. 统计系统2评级 vs 实际收益的准确率，优化权重配置")
    print("  3. 根据最优持仓期更新 daily_update.py 策略参数")
    print("  4. 在 2026Q2 面板更新后，将 30 日结果纳入 realized_pnl 重训 meta-learner")
    print("  5. 若 3m IC < 0 持续 2 期，触发 RETRAIN_LGBM 信号")

    report["optimization"] = {
        "recommendations": opt_recs,
        "next_actions": [
            "更新 strategy_config.json 的最优持仓期",
            "收集 30 日实际收益，扩充 realized_pnl 至 120+ 条",
            "重训 meta-learner（样本量翻倍后 R² 预期提升）",
        ]
    }

    print("=" * 65 + "\n")

    summary_path = SIM_DIR / "summary.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"综合报告已保存: {summary_path}")
    return report


# ─── CLI 入口 ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--step", choices=["fetch", "simulate", "evaluate", "all"],
                   default="all", help="执行步骤")
    p.add_argument("--model", type=Path,
                   default=Path("models/lgbm_v6_alpha.pkl"),
                   help="LGBM 冠军模型路径")
    p.add_argument("--token", type=str,
                   default=os.environ.get("TUSHARE_TOKEN", ""),
                   help="Tushare API Token")
    return p.parse_args()


def main():
    args = parse_args()
    SIM_DIR.mkdir(parents=True, exist_ok=True)

    if args.step in ("fetch", "all"):
        if not args.token:
            logger.error("请设置 TUSHARE_TOKEN 环境变量或使用 --token 参数")
            sys.exit(1)
        step_fetch(args.token)

    if args.step in ("simulate", "all"):
        positions = step_simulate(args.model)
    else:
        pos_path = SIM_DIR / "positions.parquet"
        if pos_path.exists():
            positions = pd.read_parquet(pos_path)
        else:
            logger.error(f"positions.parquet 不存在，请先运行 --step simulate")
            sys.exit(1)

    if args.step in ("evaluate", "all"):
        step_evaluate(positions)


if __name__ == "__main__":
    main()
