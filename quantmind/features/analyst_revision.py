"""quantmind/features/analyst_revision.py

分析师评级修正因子：基于 Tushare report_rc（卖方评级变动）数据。

Workflow
--------
1. 拉取 Tushare report_rc → data/text/analyst_reports.parquet（缓存，有则跳过）
   速率限制：1 次/分钟，按月循环拉取，每两次请求之间 sleep(61s)
2. 评级数字化：强烈推荐/买入 → 5，推荐/增持 → 4，中性/持有 → 3，
              减持/卖出 → 2，强烈卖出 → 1，未知 → NaN
3. analyst_revision_30d = 近 30 日均评级 − 前 30 日均评级
   正值 → 评级上调（机构积极转向），负值 → 评级下调
4. IC 验证：Spearman(analyst_revision_30d, return_3m) 按 ts_code 聚合
   |IC| > 0.03 且 p < 0.1 视为"有信号"

API 说明
--------
report_rc 字段：ts_code, report_date, rating（无 target_price，token 层级限制）
target_price_revision_30d 因子暂不实现。

Backtest-Expert 指引（参考 ~/.claude/skills/claude-trading-skills）
----------------------------------------------------------------------
* 样本量至少 30 个配对（否则不报 IC）
* 因子窗口选定后不回看调参（防止多重检验过拟合）
* IC 验证仅作去留决策，不做阈值搜索
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

_ROOT          = Path(__file__).resolve().parents[2]
_TEXT_DIR      = _ROOT / "data" / "text"
_ANALYST_CACHE = _TEXT_DIR / "analyst_reports.parquet"
_RETURNS_PATH  = _ROOT / "data" / "sim30d" / "stock_returns.parquet"

# 公开特征名
# analyst_revision_30d：近 30 日均评级 − 前 30 日均评级（评级升降幅度）
ANALYST_REVISION_FACTORS: list[str] = ["analyst_revision_30d"]

# 评级映射表（中文 → 1-5 数值）
_RATING_MAP: dict[str, float] = {
    # 强买
    "强烈推荐": 5.0,
    "强推":     5.0,
    "买入":     5.0,
    # 买
    "推荐":     4.0,
    "增持":     4.0,
    "审慎推荐": 4.0,
    "谨慎推荐": 4.0,
    "优于大市": 4.0,
    "跑赢行业": 4.0,
    "超配":     4.0,
    # 中性
    "中性":     3.0,
    "持有":     3.0,
    "观望":     3.0,
    "与大市同步": 3.0,
    "同步大市": 3.0,
    "标配":     3.0,
    "中立":     3.0,
    # 减持
    "减持":     2.0,
    "低于大市": 2.0,
    "跑输行业": 2.0,
    "欠配":     2.0,
    "回避":     2.0,
    # 强卖
    "卖出":     1.0,
    "强卖":     1.0,
    "强烈卖出": 1.0,
}


# ─── 工具 ─────────────────────────────────────────────────────────────────────

def _load_token() -> str:
    """从 .env 或环境变量读取 Tushare token（不硬编码）."""
    env_path = _ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("TUSHARE_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("TUSHARE_TOKEN", "")


def _digitize_rating(rating_str: Optional[str]) -> float:
    """将中文评级字符串转换为 1-5 数值；未知 → NaN.

    Parameters
    ----------
    rating_str : str or None  评级字符串，如 "增持"、"买入"

    Returns
    -------
    float  1.0–5.0，或 NaN（未知评级）
    """
    if not isinstance(rating_str, str) or not rating_str.strip():
        return float("nan")
    key = rating_str.strip()
    return _RATING_MAP.get(key, float("nan"))


# ─── Step 1: 数据获取 ─────────────────────────────────────────────────────────

def fetch_analyst_reports(
    start_date: str = "20240101",
    end_date: Optional[str] = None,
    force: bool = False,
) -> pd.DataFrame:
    """拉取 Tushare report_rc 分析师评级变动数据；有缓存则直接返回。

    Parameters
    ----------
    start_date : str  'YYYYMMDD'
    end_date   : str  'YYYYMMDD'，默认为今天
    force      : bool  True 则忽略缓存强制重新拉取

    Returns
    -------
    pd.DataFrame  columns: ts_code, report_date, rating, rating_score
    """
    _TEXT_DIR.mkdir(parents=True, exist_ok=True)

    if _ANALYST_CACHE.exists() and not force:
        logger.info(f"分析师评级缓存命中，跳过拉取：{_ANALYST_CACHE}")
        df = pd.read_parquet(_ANALYST_CACHE)
        df["report_date"] = pd.to_datetime(df["report_date"])
        return df

    import tushare as ts

    token = _load_token()
    if not token:
        raise RuntimeError(
            "Tushare token 未配置，请在项目根 .env 中写入 TUSHARE_TOKEN=xxxx"
        )
    pro = ts.pro_api(token)

    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y%m%d")

    # 按月循环拉取（rate limit: 1 次/分钟）
    month_starts = pd.date_range(start=start_date, end=end_date, freq="MS")
    chunks: list[pd.DataFrame] = []

    for i, dt in enumerate(month_starts):
        s = dt.strftime("%Y%m%d")
        e = min(
            (dt + pd.offsets.MonthEnd(0)).strftime("%Y%m%d"),
            end_date,
        )
        try:
            chunk = pro.report_rc(
                start_date=s, end_date=e,
                fields="ts_code,report_date,rating",
            )
            if chunk is not None and not chunk.empty:
                chunks.append(chunk)
                logger.info(f"  report_rc {s}~{e}: {len(chunk):,} 条")
            else:
                logger.info(f"  report_rc {s}~{e}: 0 条")
        except Exception as exc:
            logger.warning(f"  report_rc {s}~{e} 拉取失败: {exc}")

        # 速率限制：每两次请求之间等待 61 秒（1 次/分钟）
        if i < len(month_starts) - 1:
            logger.info("  等待 61 秒（API 速率限制 1 次/分钟）...")
            time.sleep(61)

    if not chunks:
        logger.warning("report_rc 未返回任何数据，返回空 DataFrame")
        empty = pd.DataFrame(columns=["ts_code", "report_date", "rating", "rating_score"])
        empty.to_parquet(_ANALYST_CACHE, index=False)
        return empty

    df = (
        pd.concat(chunks, ignore_index=True)
        .drop_duplicates(subset=["ts_code", "report_date", "rating"])
        .sort_values(["ts_code", "report_date"])
        .reset_index(drop=True)
    )
    df["report_date"] = pd.to_datetime(df["report_date"])
    # 数字化评级
    df["rating_score"] = df["rating"].apply(_digitize_rating)

    df.to_parquet(_ANALYST_CACHE, index=False)
    logger.info(f"分析师评级缓存至 {_ANALYST_CACHE}，共 {len(df):,} 条")
    return df


# ─── Step 2: 因子构造 ─────────────────────────────────────────────────────────

def build_analyst_revision_factor(
    df: pd.DataFrame,
    window: int = 30,
) -> pd.DataFrame:
    """构造 analyst_revision_30d 因子。

    计算逻辑
    --------
    对每只股票：
    * 近 window 日均评级  = 最近 window 天内所有报告的 rating_score 均值
    * 前 window 日均评级  = window+1 ~ 2*window 天前所有报告的 rating_score 均值
    * analyst_revision_30d = 近期均值 − 前期均值
    * 若近期无报告 → NaN；若前期无报告 → 前期均值视为 3.0（中性基准）

    输出
    ----
    pd.DataFrame  columns: ts_code, report_date, analyst_revision_{window}d
    只包含每只股票最新一条记录（截面因子），便于与 panel 对齐。

    Parameters
    ----------
    df     : fetch_analyst_reports() 返回的 DataFrame
    window : int  滚动窗口天数，默认 30

    Returns
    -------
    pd.DataFrame  columns: [ts_code, report_date, analyst_revision_30d]
    """
    col = f"analyst_revision_{window}d"

    if df.empty or "rating_score" not in df.columns:
        return pd.DataFrame(columns=["ts_code", "report_date", col])

    df = df.copy()
    df["report_date"] = pd.to_datetime(df["report_date"])
    df = df.dropna(subset=["rating_score"]).sort_values(["ts_code", "report_date"])

    records: list[dict] = []
    for ts_code, grp in df.groupby("ts_code"):
        grp = grp.sort_values("report_date")
        ref_date = grp["report_date"].max()

        recent_mask = grp["report_date"] > (ref_date - pd.Timedelta(days=window))
        prev_mask   = (
            (grp["report_date"] <= (ref_date - pd.Timedelta(days=window))) &
            (grp["report_date"] >  (ref_date - pd.Timedelta(days=2 * window)))
        )

        recent_scores = grp.loc[recent_mask, "rating_score"].dropna()
        prev_scores   = grp.loc[prev_mask,   "rating_score"].dropna()

        if recent_scores.empty:
            # 最近 window 天无报告 → 无法计算
            continue

        recent_mean = recent_scores.mean()
        prev_mean   = prev_scores.mean() if not prev_scores.empty else 3.0  # 中性基准

        records.append({
            "ts_code":    ts_code,
            "report_date": ref_date,
            col:          float(recent_mean - prev_mean),
        })

    if not records:
        return pd.DataFrame(columns=["ts_code", "report_date", col])

    result = pd.DataFrame(records)
    result["report_date"] = pd.to_datetime(result["report_date"])
    return result.reset_index(drop=True)


# ─── Step 3: IC 验证 ─────────────────────────────────────────────────────────

def compute_analyst_ic(
    factor_df: pd.DataFrame,
    returns_path: Path = _RETURNS_PATH,
    window: int = 30,
) -> dict:
    """Spearman IC(analyst_revision_30d, return_3m).

    对齐策略
    --------
    factor_df 含 ts_code + analyst_revision_30d（截面），
    stock_returns 含 ts_code + return_3m（时序）。
    按 ts_code 取 return_3m 均值，join，再算 Spearman。

    Returns
    -------
    dict: ic, p_value, n, valid (|IC|>0.03 且 p<0.1)
    """
    _EMPTY = {"ic": float("nan"), "p_value": float("nan"), "n": 0, "valid": False}
    col = f"analyst_revision_{window}d"

    if factor_df.empty or col not in factor_df.columns:
        return _EMPTY
    if not Path(returns_path).exists():
        logger.warning(f"收益数据不存在：{returns_path}")
        return _EMPTY

    ret = pd.read_parquet(returns_path)
    if "return_3m" not in ret.columns:
        logger.warning(f"return_3m 列不存在，可用列: {ret.columns.tolist()}")
        return _EMPTY

    # 因子按 ts_code 聚合（截面已是每股一行，但以防万一）
    factor_mean = (
        factor_df[["ts_code", col]]
        .dropna()
        .groupby("ts_code")[col]
        .mean()
    )

    # 收益按 ticker/ts_code 取均值
    ret_col = ret.rename(columns={"ticker": "ts_code"}) if "ticker" in ret.columns else ret
    if "ts_code" not in ret_col.columns:
        logger.warning("stock_returns 缺少 ticker/ts_code 列")
        return _EMPTY
    ret_mean = (
        ret_col[["ts_code", "return_3m"]]
        .dropna()
        .groupby("ts_code")["return_3m"]
        .mean()
    )

    merged = (
        factor_mean.to_frame(col)
        .join(ret_mean.rename("return_3m"), how="inner")
        .dropna()
    )
    n = len(merged)

    if n < 30:
        logger.warning(f"样本量 {n} < 30，IC 不可靠")
        return {**_EMPTY, "n": n}

    ic, p_val = stats.spearmanr(merged[col], merged["return_3m"])
    valid = (abs(ic) > 0.03) and (p_val < 0.1)
    result = {
        "ic":      float(ic),
        "p_value": float(p_val),
        "n":       int(n),
        "valid":   bool(valid),
    }
    logger.info(
        f"IC(analyst_revision_{window}d, return_3m) = {ic:+.4f}  "
        f"p={p_val:.4f}  n={n}  "
        f"→ {'有信号 ✓' if valid else '信号不足 ✗'}"
    )
    return result


# ─── 端到端 pipeline ──────────────────────────────────────────────────────────

def run_analyst_pipeline(
    start_date: str = "20240101",
    force_fetch: bool = False,
    window: int = 30,
) -> tuple[pd.DataFrame, dict]:
    """端到端：数据获取 → 评级数字化 → 因子构造 → IC 验证.

    Returns
    -------
    factor_df  : pd.DataFrame  columns: [ts_code, report_date, analyst_revision_30d]
    ic_result  : dict          {ic, p_value, n, valid}
    """
    _TEXT_DIR.mkdir(parents=True, exist_ok=True)

    reports_df = fetch_analyst_reports(start_date=start_date, force=force_fetch)
    factor_df  = build_analyst_revision_factor(reports_df, window=window)
    ic_result  = compute_analyst_ic(factor_df, window=window)
    return factor_df, ic_result
