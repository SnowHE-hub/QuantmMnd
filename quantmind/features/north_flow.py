"""quantmind/features/north_flow.py

北向资金流向因子：
  1. north_net_buy_5d  — 个股维度：月度北向持仓比例变动（百分点）
     数据来源：hk_hold（月度快照），每月末的持仓变化
     = ratio_curr_month - ratio_prev_month
     正值 = 北向增持，负值 = 北向减持；未被北向持仓的股票填 0

  2. north_flow_market_5d — 市场维度：沪深港通日频净流入5日均值（百万元）
     数据来源：moneyflow_hsgt（日频），north_money 日差分
     用途：HMM Regime 辅助校正信号

数据说明
--------
* hk_hold API 仅提供月度快照（每月末一次，非日频），因此 north_net_buy_5d
  实际为 month-over-month 持仓比例变化，命名遵循功能规格。
* moneyflow_hsgt.north_money 单位：百万元（M yuan），为累计余额。
  日频净流入 = north_money(t) - north_money(t-1)。
  50亿元 = 5000 M yuan（HMM 辅助信号阈值）。

Workflow
--------
1. fetch_hk_hold_monthly()  → data/text/north_flow.parquet（增量缓存）
2. build_north_flow_factor() → MultiIndex(ts_code, trade_date) Series
3. fetch_market_north_flow() → data/text/market_north_flow.parquet（增量缓存）
4. build_market_flow_factor() → Index(trade_date) Series
5. compute_north_ic()        → {ic, p, n, valid}（接口与 text_sentiment 一致）
6. apply_north_flow_regime_correction() → HMM regime 辅助校正
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

_ROOT           = Path(__file__).resolve().parents[2]
_TEXT_DIR       = _ROOT / "data" / "text"
_FLOW_CACHE     = _TEXT_DIR / "north_flow.parquet"
_MKT_FLOW_CACHE = _TEXT_DIR / "market_north_flow.parquet"
_RETURNS_PATH   = _ROOT / "data" / "sim30d" / "stock_returns.parquet"

# 公开因子名列表
NORTH_FLOW_FACTORS: list[str] = ["north_net_buy_5d"]

# HMM 辅助信号阈值（百万元，M yuan）
# 50亿元 = 5000 M yuan
_FLOW_BULL_THRESHOLD: float = 5000.0
_FLOW_BEAR_THRESHOLD: float = -5000.0

# A 股交易所代码（过滤 HK 股）
_ASHARE_EXCHANGES: frozenset[str] = frozenset({"SH", "SZ"})


# ─── 工具 ──────────────────────────────────────────────────────────────────────

def _load_token() -> str:
    """从 .env 或环境变量读取 Tushare token."""
    env_path = _ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("TUSHARE_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("TUSHARE_TOKEN", "")


def _get_pro():
    """初始化并返回 Tushare Pro API 对象."""
    import tushare as ts
    token = _load_token()
    if not token:
        raise RuntimeError(
            "Tushare token 未配置，请在项目根 .env 中写入 TUSHARE_TOKEN=xxxx"
        )
    return ts.pro_api(token)


# ─── Step 1a: 个股北向持仓（hk_hold）──────────────────────────────────────────

def fetch_hk_hold_monthly(
    start_date: str = "20240101",
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """拉取 hk_hold 月度快照并缓存到 north_flow.parquet。

    hk_hold API 每次只能查一个 trade_date，按月末逐月拉取。
    仅保留 A 股（exchange ∈ {SH, SZ}）。
    有缓存则增量更新，不全量重拉。

    Parameters
    ----------
    start_date : str  'YYYYMMDD'
    end_date   : str  'YYYYMMDD'，默认为今天
    use_cache  : bool  False 则忽略缓存强制重拉

    Returns
    -------
    pd.DataFrame  columns: ts_code, trade_date, vol, ratio, exchange
    """
    _TEXT_DIR.mkdir(parents=True, exist_ok=True)

    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y%m%d")

    end_dt = pd.to_datetime(end_date)

    if use_cache and _FLOW_CACHE.exists():
        cached = pd.read_parquet(_FLOW_CACHE)
        cached["trade_date"] = pd.to_datetime(cached["trade_date"])
        last_cached = cached["trade_date"].max()

        # 增量起点：上次缓存末月的下一个月初
        next_start_dt = (last_cached + pd.DateOffset(months=1)).replace(day=1)
        if next_start_dt > end_dt:
            logger.info(f"hk_hold 缓存最新至 {last_cached.date()}，无需增量更新")
            return cached

        logger.info(f"hk_hold 增量更新：从 {next_start_dt.date()}")
        new_data = _fetch_hk_hold_range(next_start_dt.strftime("%Y%m%d"), end_date)
        if new_data.empty:
            return cached
        combined = pd.concat([cached, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=["ts_code", "trade_date"])
        combined = combined.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        combined.to_parquet(_FLOW_CACHE, index=False)
        return combined

    df = _fetch_hk_hold_range(start_date, end_date)
    if not df.empty:
        df.to_parquet(_FLOW_CACHE, index=False)
    return df


def _fetch_hk_hold_range(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """按月末逐月拉取 hk_hold，仅保留 A 股（SH/SZ）。"""
    pro = _get_pro()

    # 生成月末日期序列（pandas ME 频率）
    month_ends = pd.date_range(
        start=pd.to_datetime(start_date).to_period("M").to_timestamp("M"),
        end=pd.to_datetime(end_date),
        freq="ME",
    )

    chunks: list[pd.DataFrame] = []
    for dt in month_ends:
        trade_date = dt.strftime("%Y%m%d")
        try:
            df = pro.hk_hold(trade_date=trade_date)
            time.sleep(0.5)          # 速率限制
        except Exception as exc:
            logger.warning(f"hk_hold {trade_date} 拉取失败: {exc}")
            continue

        if df is None or df.empty:
            logger.debug(f"hk_hold {trade_date}: API 返回空")
            continue

        # 过滤 A 股
        df = df[df["exchange"].isin(_ASHARE_EXCHANGES)].copy()
        if df.empty:
            logger.debug(f"hk_hold {trade_date}: 无 A 股记录")
            continue

        df["trade_date"] = pd.to_datetime(trade_date)
        df["vol"]   = pd.to_numeric(df["vol"],   errors="coerce")
        df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce")
        chunks.append(df[["ts_code", "trade_date", "vol", "ratio", "exchange"]])
        logger.info(f"hk_hold {trade_date}: {len(df)} 只 A 股")

    if not chunks:
        return pd.DataFrame(
            columns=["ts_code", "trade_date", "vol", "ratio", "exchange"]
        )

    result = pd.concat(chunks, ignore_index=True)
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    return result


# ─── Step 2: 个股因子构造 ──────────────────────────────────────────────────────

def build_north_flow_factor(
    start_date: str = "2024-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.Series:
    """构造 north_net_buy_5d 个股北向资金流向因子。

    north_net_buy_5d = ratio_curr_month_end - ratio_prev_month_end （百分点）
    = 北向持仓比例的月度变化，正值=增持，负值=减持。
    未被北向持仓的股票（不在 hk_hold 中）填 0（中性）。

    数据频率：月度快照（hk_hold API 限制），不支持日频。

    Parameters
    ----------
    start_date : str  'YYYY-MM-DD' 格式
    end_date   : str  'YYYY-MM-DD'，None = 今天
    use_cache  : bool  有缓存则增量更新

    Returns
    -------
    pd.Series
        MultiIndex(ts_code, trade_date)，name='north_net_buy_5d'
        每个 (ts_code, trade_date) 对应一个月末截面的持仓变动比例。
    """
    start_ts = pd.to_datetime(start_date).strftime("%Y%m%d")
    end_ts = (
        pd.Timestamp.today() if end_date is None else pd.to_datetime(end_date)
    ).strftime("%Y%m%d")

    # 多拉 2 个月以便计算第一个月的差值
    start_prev = (
        pd.to_datetime(start_date) - pd.DateOffset(months=2)
    ).strftime("%Y%m%d")

    hk_df = fetch_hk_hold_monthly(start_prev, end_ts, use_cache=use_cache)

    if hk_df.empty:
        logger.warning("hk_hold 返回空数据，返回空因子")
        return pd.Series(dtype=float, name="north_net_buy_5d")

    hk_df = hk_df.sort_values(["ts_code", "trade_date"]).copy()

    # 月度持仓比例差值
    hk_df["prev_ratio"] = hk_df.groupby("ts_code")["ratio"].shift(1)
    hk_df["north_net_buy"] = (
        hk_df["ratio"].fillna(0.0) - hk_df["prev_ratio"].fillna(0.0)
    )

    # 仅保留有上期数据的行
    hk_df = hk_df.dropna(subset=["prev_ratio"])

    # 过滤到目标日期范围
    start_filter = pd.to_datetime(start_date)
    end_filter = (
        pd.Timestamp.today() if end_date is None else pd.to_datetime(end_date)
    )
    hk_df = hk_df[
        (hk_df["trade_date"] >= start_filter)
        & (hk_df["trade_date"] <= end_filter)
    ]

    if hk_df.empty:
        logger.warning("过滤后因子数据为空")
        return pd.Series(dtype=float, name="north_net_buy_5d")

    factor = (
        hk_df.set_index(["ts_code", "trade_date"])["north_net_buy"]
        .fillna(0.0)
        .rename("north_net_buy_5d")
    )
    logger.info(
        f"north_net_buy_5d 因子：{factor.index.get_level_values('ts_code').nunique()} 只股票"
        f"，{factor.index.get_level_values('trade_date').nunique()} 个月末截面"
    )
    return factor


# ─── Step 1b: 市场级北向流入（moneyflow_hsgt）────────────────────────────────

def fetch_market_north_flow(
    start_date: str = "20240101",
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """拉取 moneyflow_hsgt 并缓存到 market_north_flow.parquet。

    日频净流入 = north_money(t) - north_money(t-1)（百万元）。
    有缓存则增量更新，不全量重拉。

    Returns
    -------
    pd.DataFrame  columns: trade_date(datetime), north_money(float), daily_net_flow(float)
    """
    _TEXT_DIR.mkdir(parents=True, exist_ok=True)

    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y%m%d")

    if use_cache and _MKT_FLOW_CACHE.exists():
        cached = pd.read_parquet(_MKT_FLOW_CACHE)
        cached["trade_date"] = pd.to_datetime(cached["trade_date"])
        last_cached = cached["trade_date"].max()

        next_day = (last_cached + pd.Timedelta(days=1)).strftime("%Y%m%d")
        if next_day > end_date:
            logger.info(f"market_north_flow 缓存最新至 {last_cached.date()}")
            return cached

        logger.info(f"market_north_flow 增量更新：从 {next_day}")
        new_raw = _fetch_moneyflow_hsgt_raw(next_day, end_date)
        if new_raw.empty:
            return cached

        # 接上缓存末行的 north_money 计算首日 diff
        last_north_money = float(cached["north_money"].iloc[-1])
        combined = pd.concat([cached, new_raw], ignore_index=True)
        combined = combined.drop_duplicates(subset=["trade_date"])
        combined = combined.sort_values("trade_date").reset_index(drop=True)
        combined["daily_net_flow"] = combined["north_money"].diff()
        combined.to_parquet(_MKT_FLOW_CACHE, index=False)
        return combined

    df = _fetch_moneyflow_hsgt_raw(start_date, end_date)
    if not df.empty:
        df.to_parquet(_MKT_FLOW_CACHE, index=False)
    return df


def _fetch_moneyflow_hsgt_raw(start_date: str, end_date: str) -> pd.DataFrame:
    """拉取 moneyflow_hsgt 原始数据并计算日频净流入."""
    pro = _get_pro()
    try:
        df = pro.moneyflow_hsgt(start_date=start_date, end_date=end_date)
    except Exception as exc:
        logger.warning(f"moneyflow_hsgt 拉取失败: {exc}")
        return pd.DataFrame(columns=["trade_date", "north_money", "daily_net_flow"])

    if df is None or df.empty:
        return pd.DataFrame(columns=["trade_date", "north_money", "daily_net_flow"])

    df = df[["trade_date", "north_money"]].copy()
    df["trade_date"]   = pd.to_datetime(df["trade_date"])
    df["north_money"]  = pd.to_numeric(df["north_money"], errors="coerce")
    df = df.sort_values("trade_date").reset_index(drop=True)
    df["daily_net_flow"] = df["north_money"].diff()
    return df


# ─── Step 3: 市场因子构造 ──────────────────────────────────────────────────────

def build_market_flow_factor(
    start_date: str = "2024-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
    window: int = 5,
) -> pd.Series:
    """构造 north_flow_market_5d 市场级北向资金5日均值因子。

    = 近 window 日北向日频净流入均值（百万元，M yuan）
    正值 = 净流入，负值 = 净流出。
    50亿元 ≈ 5000 M yuan（HMM 辅助信号阈值）。

    Parameters
    ----------
    start_date : str  'YYYY-MM-DD'
    end_date   : str  'YYYY-MM-DD'，None = 今天
    use_cache  : bool
    window     : int  滚动窗口天数，默认 5

    Returns
    -------
    pd.Series  Index(trade_date)，name='north_flow_market_5d'
    """
    start_ts = pd.to_datetime(start_date).strftime("%Y%m%d")
    end_ts = (
        pd.Timestamp.today() if end_date is None else pd.to_datetime(end_date)
    ).strftime("%Y%m%d")

    df = fetch_market_north_flow(start_ts, end_ts, use_cache=use_cache)

    if df.empty:
        return pd.Series(dtype=float, name="north_flow_market_5d")

    df = df.set_index("trade_date").sort_index()
    market_flow = (
        df["daily_net_flow"]
        .rolling(window, min_periods=1)
        .mean()
        .rename("north_flow_market_5d")
        .dropna()
    )
    return market_flow


def get_latest_market_north_flow(
    use_cache: bool = True,
    lookback_days: int = 30,
) -> float:
    """返回最近 5 日北向净流入均值（M yuan）供 HMM 辅助信号使用。

    失败时返回 0.0（中性，不干预 HMM 判断）。
    """
    end_ts = pd.Timestamp.today().strftime("%Y%m%d")
    start_ts = (
        pd.Timestamp.today() - pd.Timedelta(days=lookback_days)
    ).strftime("%Y%m%d")

    try:
        market_flow = build_market_flow_factor(
            start_date=start_ts, end_date=end_ts, use_cache=use_cache
        )
        if market_flow.empty:
            return 0.0
        return float(market_flow.iloc[-1])
    except Exception as exc:
        logger.warning(f"get_latest_market_north_flow 失败: {exc}")
        return 0.0


# ─── Step 4: IC 验证 ──────────────────────────────────────────────────────────

def compute_north_ic(
    factor: pd.Series,
    returns: Optional[pd.DataFrame] = None,
    returns_path: Path = _RETURNS_PATH,
) -> dict:
    """Spearman IC(north_net_buy_5d, return_3m)。

    接口与 text_sentiment.compute_ic 和 analyst_revision.compute_analyst_ic 保持一致。

    对齐策略：按 ts_code 聚合时序均值，再与收益数据按 ts_code join。

    Parameters
    ----------
    factor       : pd.Series  MultiIndex(ts_code, trade_date)
    returns      : pd.DataFrame  可选，直接传入（跳过读文件）
    returns_path : Path  returns 文件路径（当 returns=None 时读取）

    Returns
    -------
    dict: {ic, p, n, valid}
        ic    : Spearman 相关系数
        p     : p 值
        n     : 有效配对样本数（按 ts_code 去重后）
        valid : |IC| > 0.03 且 p < 0.1
    """
    _EMPTY = {"ic": float("nan"), "p": float("nan"), "n": 0, "valid": False}

    if factor.empty:
        return _EMPTY

    if returns is None:
        rp = Path(returns_path)
        if not rp.exists():
            logger.warning(f"收益数据不存在: {returns_path}")
            return _EMPTY
        returns = pd.read_parquet(rp)

    if "return_3m" not in returns.columns:
        logger.warning(f"return_3m 列不存在，可用列: {returns.columns.tolist()}")
        return _EMPTY

    # 因子：按 ts_code 取时序均值
    factor_df = factor.rename("factor").reset_index()
    if "ts_code" not in factor_df.columns:
        # MultiIndex 展开后列名可能不同
        cols = list(factor_df.columns)
        factor_df.columns = ["ts_code", "trade_date", "factor"]
    factor_mean = factor_df.groupby("ts_code")["factor"].mean()

    # 收益：按 ts_code/ticker 取 return_3m 均值
    ret = returns.copy()
    if "ticker" in ret.columns and "ts_code" not in ret.columns:
        ret = ret.rename(columns={"ticker": "ts_code"})
    if "ts_code" not in ret.columns:
        logger.warning("returns 缺少 ts_code/ticker 列")
        return _EMPTY

    ret_mean = (
        ret[["ts_code", "return_3m"]]
        .dropna()
        .groupby("ts_code")["return_3m"]
        .mean()
    )

    merged = (
        factor_mean.to_frame("factor")
        .join(ret_mean.rename("return_3m"), how="inner")
        .dropna()
    )
    n = len(merged)

    if n < 30:
        logger.warning(f"样本量 {n} < 30，IC 不可靠（北向覆盖股票数不足）")
        return {**_EMPTY, "n": n}

    ic, p_val = stats.spearmanr(merged["factor"], merged["return_3m"])
    valid = (abs(ic) > 0.03) and (p_val < 0.1)
    result = {
        "ic":    float(ic),
        "p":     float(p_val),
        "n":     int(n),
        "valid": bool(valid),
    }
    logger.info(
        f"IC(north_net_buy_5d, return_3m) = {ic:+.4f}  "
        f"p={p_val:.4f}  n={n}  "
        f"→ {'有效信号 ✅' if valid else '信号弱 ⚠️'}"
    )
    return result


# ─── Step 5: HMM 辅助校正 ─────────────────────────────────────────────────────

def apply_north_flow_regime_correction(
    regime: str,
    market_flow_5d: Optional[float] = None,
) -> str:
    """根据北向资金净流入辅助校正 HMM Regime 判断。

    规则（用户规格 Step 5）：
    - 近5日均净流入 > 5000 M yuan（50亿元）且 regime ≠ 'bear' → 'bull'
    - 近5日均净流入 < -5000 M yuan（-50亿元）且 regime ≠ 'bull' → 'bear'
    - 其余区间：维持 HMM 原始判断

    Parameters
    ----------
    regime         : HMM 原始 regime，'bull' / 'neutral' / 'bear'
    market_flow_5d : 近5日北向净流入均值（M yuan）；None 则实时获取

    Returns
    -------
    str: 校正后的 regime（'bull' / 'neutral' / 'bear'）
    """
    if market_flow_5d is None:
        market_flow_5d = get_latest_market_north_flow()

    if market_flow_5d > _FLOW_BULL_THRESHOLD and regime != "bear":
        logger.debug(
            f"north_flow 辅助信号：flow={market_flow_5d:.0f}M > 5000M "
            f"→ bull（原 regime={regime}）"
        )
        return "bull"
    elif market_flow_5d < _FLOW_BEAR_THRESHOLD and regime != "bull":
        logger.debug(
            f"north_flow 辅助信号：flow={market_flow_5d:.0f}M < -5000M "
            f"→ bear（原 regime={regime}）"
        )
        return "bear"
    return regime


def discretize_market_flow(flow_5d: float) -> str:
    """将市场北向净流入离散化为 Low / Mid / High 三档。

    - High : flow > 5000 M yuan (50亿元，强净流入)
    - Low  : flow < -5000 M yuan (-50亿元，强净流出)
    - Mid  : 其余

    Parameters
    ----------
    flow_5d : float  近5日均净流入（M yuan）

    Returns
    -------
    str: 'Low' | 'Mid' | 'High'
    """
    if flow_5d > _FLOW_BULL_THRESHOLD:
        return "High"
    elif flow_5d < _FLOW_BEAR_THRESHOLD:
        return "Low"
    return "Mid"
