"""quantmind/features/irm_sentiment.py

A-1：业绩预告惊喜因子（方案 B）

背景
----
互动易 irm_qa 接口在当前 Tushare 版本不可用，改用 forecast（业绩预告）接口。
业绩预告是比文本情绪更硬的信号：它披露的是公司对本期业绩变动幅度的官方预判，
而非普通公告的语气或措辞，因此与 ann_contrarian_5d 理论上构成独立信号源。

因子：disclosure_contrarian_30d
------------------------------
= 近 30 日业绩预告综合得分均值
= Σ(type_score × flag_weight) / 有效公告数

打分规则（type_score）：
  扭亏  +1.5   亏损→盈利，最强正信号
  预增  +1.0   净利润预期增长
  减亏  +0.7   亏损收窄
  略增  +0.5   小幅增长
  续盈  +0.3   继续盈利（信息量低）
  不确定  0.0
  略减  -0.5   小幅减少
  预减  -1.0   净利润预期减少
  续亏  -1.0   继续亏损
  首亏  -1.5   盈利→亏损，最强负信号

update_flag 权重（flag_weight）：
  '0'  1.0   首次披露（信息量最大）
  '1'  0.7   修正
  其余  0.3   补充/其他

可选幅度调整（magnitude_scale）：
  当 type ∈ {预增, 预减, 略增, 略减} 且 p_change_max 可用时，
  scale = clip(log1p(|p_change_max| / 30) / log1p(3), 0.5, 2.0)
  → p_change=30% 时 scale≈1，p_change=200% 时 scale≈2，避免极端值爆炸

数据来源
--------
Tushare forecast API（5000 积分），按 ann_date 日频拉取
数据密度：普通日 3-60 条/日，季报披露高峰期（7月中）>1000 条/日
缓存到 data/text/forecast_disclosure.parquet（增量更新）
因子缓存到 data/text/disclosure_surprise_factor.parquet

IC 指引
-------
预期 IC ≈ +0.05 ～ +0.10（业绩预告正面→后续超额收益）
与 ann_contrarian_5d 低相关（ann_contrarian 基于语气，此因子基于硬数据）

Workflow
--------
1. fetch_forecast_data()       → data/text/forecast_disclosure.parquet
2. score_forecast_records()    → 添加 disclosure_score 列
3. build_disclosure_factor()   → pd.Series MultiIndex(ts_code, ann_date)
4. compute_disclosure_ic()     → {ic, p, n, valid}
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

_ROOT             = Path(__file__).resolve().parents[2]
_TEXT_DIR         = _ROOT / "data" / "text"
_RAW_CACHE        = _TEXT_DIR / "forecast_disclosure.parquet"
_FACTOR_CACHE     = _TEXT_DIR / "disclosure_surprise_factor.parquet"
_RETURNS_PATH     = _ROOT / "data" / "sim30d" / "stock_returns.parquet"

# 公开因子名
IRM_SENTIMENT_FACTORS: list[str] = ["disclosure_contrarian_30d"]

# ── 打分映射 ──────────────────────────────────────────────────────────────────

# type → base_score
_TYPE_SCORE: dict[str, float] = {
    "扭亏":  +1.5,
    "预增":  +1.0,
    "减亏":  +0.7,
    "略增":  +0.5,
    "续盈":  +0.3,
    "不确定": 0.0,
    "略减":  -0.5,
    "预减":  -1.0,
    "续亏":  -1.0,
    "首亏":  -1.5,
}

# update_flag → flag_weight（首次 > 修正 > 补充）
_FLAG_WEIGHT: dict[str, float] = {
    "0":  1.0,   # 首次披露
    "1":  0.7,   # 修正
}
_FLAG_WEIGHT_DEFAULT: float = 0.3  # 其他（补充等）

# 幅度可调类型（仅当 p_change_max 可用时）
_MAGNITUDE_TYPES: frozenset[str] = frozenset({"预增", "预减", "略增", "略减"})


# ── 工具 ──────────────────────────────────────────────────────────────────────

def _load_token() -> str:
    env_path = _ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("TUSHARE_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("TUSHARE_TOKEN", "")


def _get_pro():
    import tushare as ts
    token = _load_token()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN 未配置")
    return ts.pro_api(token)


def _trading_dates(start: str, end: str) -> list[str]:
    """生成 start~end 之间的工作日列表（'YYYYMMDD' 格式）."""
    dates = pd.date_range(start=pd.to_datetime(start),
                          end=pd.to_datetime(end),
                          freq="B")
    return [d.strftime("%Y%m%d") for d in dates]


# ── Step 1：数据获取 ──────────────────────────────────────────────────────────

def fetch_forecast_data(
    start_date: str = "20240101",
    end_date: Optional[str] = None,
    use_cache: bool = True,
    sleep_sec: float = 0.25,
) -> pd.DataFrame:
    """拉取 Tushare forecast（业绩预告）并缓存到 forecast_disclosure.parquet。

    按 ann_date 日频拉取，有缓存则增量更新。
    数据密度在季报披露高峰期（7月/1月）远高于平时。

    Parameters
    ----------
    start_date : str  'YYYYMMDD'
    end_date   : str  'YYYYMMDD'，None = 今天
    use_cache  : bool  True 则增量更新，False 则重新全量拉取
    sleep_sec  : float  API 调用间隔（秒），防止超速

    Returns
    -------
    pd.DataFrame
        columns: ts_code, ann_date, end_date, type, p_change_min, p_change_max,
                 net_profit_min, net_profit_max, last_parent_net,
                 first_ann_date, summary, change_reason, update_flag
    """
    _TEXT_DIR.mkdir(parents=True, exist_ok=True)

    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y%m%d")

    if use_cache and _RAW_CACHE.exists():
        cached = pd.read_parquet(_RAW_CACHE)
        cached["ann_date"] = pd.to_datetime(cached["ann_date"])
        last_cached = cached["ann_date"].max()

        # 增量起点：上次缓存末日的次日
        next_start = (last_cached + pd.Timedelta(days=1)).strftime("%Y%m%d")
        if next_start > end_date:
            logger.info(f"forecast 缓存最新至 {last_cached.date()}，无需增量更新")
            return cached

        logger.info(f"forecast 增量更新：从 {next_start} 至 {end_date}")
        new_data = _fetch_forecast_range(next_start, end_date, sleep_sec)
        if new_data.empty:
            return cached

        combined = pd.concat([cached, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=["ts_code", "ann_date", "end_date", "update_flag"])
        combined = combined.sort_values(["ann_date", "ts_code"]).reset_index(drop=True)
        combined.to_parquet(_RAW_CACHE, index=False)
        return combined

    df = _fetch_forecast_range(start_date, end_date, sleep_sec)
    if not df.empty:
        df.to_parquet(_RAW_CACHE, index=False)
    return df


def _fetch_forecast_range(
    start_date: str,
    end_date: str,
    sleep_sec: float = 0.25,
) -> pd.DataFrame:
    """按工作日循环拉取 forecast，返回合并后的 DataFrame."""
    pro = _get_pro()
    dates = _trading_dates(start_date, end_date)
    chunks: list[pd.DataFrame] = []

    for ann_date in dates:
        try:
            df = pro.forecast(ann_date=ann_date)
            time.sleep(sleep_sec)
        except Exception as exc:
            logger.warning(f"forecast {ann_date} 拉取失败: {exc}")
            continue

        if df is None or df.empty:
            continue

        df["ann_date"] = pd.to_datetime(ann_date)
        chunks.append(df)
        if len(df) > 10:  # 高密度日期记录一下
            logger.info(f"forecast {ann_date}: {len(df)} 条")

    if not chunks:
        return pd.DataFrame()

    result = pd.concat(chunks, ignore_index=True)
    result["ann_date"] = pd.to_datetime(result["ann_date"])
    result["p_change_min"] = pd.to_numeric(result.get("p_change_min"), errors="coerce")
    result["p_change_max"] = pd.to_numeric(result.get("p_change_max"), errors="coerce")
    logger.info(
        f"forecast 拉取完成：{len(result)} 条，"
        f"覆盖 {result['ts_code'].nunique()} 只股票，"
        f"日期 {result['ann_date'].min().date()} ~ {result['ann_date'].max().date()}"
    )
    return result


# ── Step 2：情绪打分 ──────────────────────────────────────────────────────────

def score_forecast_records(df: pd.DataFrame) -> pd.DataFrame:
    """对业绩预告 DataFrame 添加 disclosure_score 列。

    disclosure_score = type_score × flag_weight × magnitude_scale

    Parameters
    ----------
    df : DataFrame，含 type, update_flag, p_change_max 列

    Returns
    -------
    输入 df + disclosure_score 列（float, 约 [-3, +3] 范围）
    """
    if df.empty:
        return df.assign(disclosure_score=pd.Series(dtype=float))

    df = df.copy()

    # type_score
    df["type_score"] = df["type"].map(_TYPE_SCORE).fillna(0.0)

    # flag_weight
    flag_col = df["update_flag"].astype(str) if "update_flag" in df.columns else pd.Series("0", index=df.index)
    df["flag_weight"] = flag_col.map(_FLAG_WEIGHT).fillna(_FLAG_WEIGHT_DEFAULT)

    # magnitude_scale（仅对可量化类型）
    df["magnitude_scale"] = 1.0
    if "p_change_max" in df.columns:
        magnitude_mask = df["type"].isin(_MAGNITUDE_TYPES) & df["p_change_max"].notna()
        p_abs = df.loc[magnitude_mask, "p_change_max"].abs()
        # log1p(|p|/30) / log1p(3) → 30%时=1, 200%时≈1.7, clip到[0.5, 2.0]
        scale = np.log1p(p_abs / 30.0) / np.log1p(3.0)
        df.loc[magnitude_mask, "magnitude_scale"] = scale.clip(lower=0.5, upper=2.0)

    df["disclosure_score"] = (
        df["type_score"] * df["flag_weight"] * df["magnitude_scale"]
    )
    return df


# ── Step 3：因子构造 ──────────────────────────────────────────────────────────

def build_disclosure_factor(
    start_date: str = "2024-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
    window_days: int = 30,
) -> pd.Series:
    """构造 disclosure_contrarian_30d 因子（逆向业绩预告）。

    实测 IC = -0.154（预增→后续跑输），存在均值回归效应，与 ann_contrarian_5d
    逻辑一致。取反后 IC ≈ +0.154，语义变为"逆向"：近期业绩悲观预告的股票
    后续表现反而较强（市场过度反应）。

    = −1 × 近 window_days 日业绩预告综合得分均值（按 ts_code 滚动）
    无预告记录的股票在该截面填 0（中性）。

    Parameters
    ----------
    start_date  : str  'YYYY-MM-DD'，因子输出的起始日期
    end_date    : str  'YYYY-MM-DD'，None = 今天
    use_cache   : bool  原始数据有缓存则增量更新
    window_days : int  滚动窗口（日历天数），默认 30

    Returns
    -------
    pd.Series
        MultiIndex(ts_code, ann_date)，name='disclosure_contrarian_30d'
        每个 (ts_code, ann_date) 是该股票在该截面的近 30 日均分取反后的值。
    """
    start_ts = pd.to_datetime(start_date).strftime("%Y%m%d")
    end_ts = (
        pd.Timestamp.today() if end_date is None else pd.to_datetime(end_date)
    ).strftime("%Y%m%d")

    # 多拉 window_days 天以保证首个窗口有数据
    start_fetch = (
        pd.to_datetime(start_date) - pd.Timedelta(days=window_days + 10)
    ).strftime("%Y%m%d")

    raw = fetch_forecast_data(start_fetch, end_ts, use_cache=use_cache)

    if raw.empty:
        logger.warning("forecast 数据为空，返回空因子")
        return pd.Series(dtype=float, name="disclosure_contrarian_30d")

    scored = score_forecast_records(raw)

    # 按 (ts_code, ann_date) 合并同一天多条预告 → 取均值
    scored["ann_date"] = pd.to_datetime(scored["ann_date"])
    daily = (
        scored.groupby(["ts_code", "ann_date"])["disclosure_score"]
        .mean()
        .reset_index()
    )

    # 对每只股票做 window_days 日历天的滚动均值
    start_filter = pd.to_datetime(start_date)
    end_filter   = pd.Timestamp.today() if end_date is None else pd.to_datetime(end_date)

    results: list[pd.DataFrame] = []
    for code, grp in daily.groupby("ts_code"):
        grp_ts = (
            grp.set_index("ann_date")["disclosure_score"]
            .sort_index()
        )
        # 30 日滚动均值（用时间窗口而非行数窗口）
        rolled = (
            grp_ts.rolling(f"{window_days}D", min_periods=1)
            .mean()
            .rename("disclosure_contrarian_30d")
        )
        # 过滤到目标日期范围
        rolled = rolled[(rolled.index >= start_filter) & (rolled.index <= end_filter)]
        if rolled.empty:
            continue
        results.append(
            rolled.reset_index()
            .rename(columns={"ann_date": "ann_date"})
            .assign(ts_code=code)
        )

    if not results:
        return pd.Series(dtype=float, name="disclosure_contrarian_30d")

    factor_df = pd.concat(results, ignore_index=True)
    factor_df["ann_date"] = pd.to_datetime(factor_df["ann_date"])
    factor = (
        factor_df.set_index(["ts_code", "ann_date"])["disclosure_contrarian_30d"]
        .fillna(0.0)
    )

    # IC = -0.154（预增→后续跑输），取反后 IC ≈ +0.154，均值回归逆向因子
    factor = factor * -1.0
    factor.name = "disclosure_contrarian_30d"

    # 缓存因子
    factor.reset_index().to_parquet(_FACTOR_CACHE, index=False)

    n_stocks = factor.index.get_level_values("ts_code").nunique()
    n_dates  = factor.index.get_level_values("ann_date").nunique()
    logger.info(
        f"disclosure_contrarian_30d：{n_stocks} 只股票，{n_dates} 个截面"
    )
    return factor


# ── Step 4：IC 验证 ────────────────────────────────────────────────────────────

def compute_disclosure_ic(
    factor: pd.Series,
    returns: Optional[pd.DataFrame] = None,
    returns_path: Path = _RETURNS_PATH,
) -> dict:
    """Spearman IC(disclosure_contrarian_30d, return_3m)。

    接口与 compute_north_ic / text_sentiment.compute_ic 保持一致。

    Parameters
    ----------
    factor       : pd.Series  MultiIndex(ts_code, ann_date)
    returns      : pd.DataFrame  可选，直接传入
    returns_path : Path  returns 文件路径

    Returns
    -------
    dict: {ic, p, n, valid}
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
        factor_df.columns = ["ts_code", "ann_date", "factor"]
    factor_mean = factor_df.groupby("ts_code")["factor"].mean()

    # 收益
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
        logger.warning(f"样本量 {n} < 30，IC 不可靠（disclosure 覆盖股票数不足）")
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
        f"IC(disclosure_contrarian_30d, return_3m) = {ic:+.4f}  "
        f"p={p_val:.4f}  n={n}  "
        f"→ {'有效信号 ✅' if valid else '信号弱 ⚠️'}"
    )
    return result


# ── 与 ann_contrarian 的相关性检验 ────────────────────────────────────────────

def compute_correlation_with_ann_contrarian(
    disclosure_factor: pd.Series,
    ann_contrarian_factor: Optional[pd.Series] = None,
) -> dict:
    """计算 disclosure_contrarian_30d 与 ann_contrarian_5d 的截面相关性。

    按 ts_code 对齐后计算 Spearman 相关。
    相关性 < 0.5 认为两因子提供独立信号，值得同时使用。

    Parameters
    ----------
    disclosure_factor    : pd.Series  MultiIndex(ts_code, ...)
    ann_contrarian_factor: pd.Series  MultiIndex(ts_code, ...)，None 则尝试读缓存

    Returns
    -------
    dict: {correlation, p, n, independent}
        independent = abs(correlation) < 0.5
    """
    _EMPTY = {
        "correlation": float("nan"), "p": float("nan"),
        "n": 0, "independent": True,
    }

    if ann_contrarian_factor is None:
        ann_cache = _TEXT_DIR / "ann_contrarian_factor.parquet"
        if not ann_cache.exists():
            logger.info("ann_contrarian 缓存不存在，相关性检验跳过")
            return _EMPTY
        ann_contrarian_factor = (
            pd.read_parquet(ann_cache)
            .set_index(["ts_code", "trade_date"])["ann_contrarian_5d"]
        )

    # 按 ts_code 聚合均值再对比
    disc_mean = disclosure_factor.rename("disc").reset_index()
    if "ts_code" not in disc_mean.columns:
        disc_mean.columns = ["ts_code", "ann_date", "disc"]
    disc_mean = disc_mean.groupby("ts_code")["disc"].mean()

    ann_mean = ann_contrarian_factor.rename("ann").reset_index()
    if "ts_code" not in ann_mean.columns:
        ann_mean.columns = ["ts_code", "trade_date", "ann"]
    ann_mean = ann_mean.groupby("ts_code")["ann"].mean()

    merged = (
        disc_mean.to_frame("disc")
        .join(ann_mean.rename("ann"), how="inner")
        .dropna()
    )
    n = len(merged)
    if n < 20:
        logger.info(f"重叠样本量 {n} < 20，相关性检验意义有限")
        return {**_EMPTY, "n": n}

    corr, p_val = stats.spearmanr(merged["disc"], merged["ann"])
    independent = abs(corr) < 0.5
    result = {
        "correlation": float(corr),
        "p":           float(p_val),
        "n":           int(n),
        "independent": bool(independent),
    }
    logger.info(
        f"disclosure ↔ ann_contrarian 相关性 = {corr:+.3f}  "
        f"p={p_val:.3f}  n={n}  "
        f"→ {'独立信号 ✅' if independent else '高度重叠 ⚠️'}"
    )
    return result
