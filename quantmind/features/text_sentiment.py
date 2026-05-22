"""quantmind/features/text_sentiment.py

文本情绪因子：基于 Tushare 公告数据 + 预训练中文金融情绪模型。

Workflow
--------
1. 拉取 Tushare anns_d 公告摘要 → data/text/announcements.parquet（缓存，有则跳过）
2. 用金融情绪 BERT 打分：positive_prob - negative_prob ∈ [-1, 1]
   优先级：IDEA-CCNL → hw2942 → 词典降级
3. 按 (ts_code, trade_date) 滚动 5 日均值 → ann_sentiment_5d
4. 输出 MultiIndex(ts_code, trade_date) → ann_sentiment_5d；缺失填 0
5. IC 验证：Spearman(ann_sentiment_5d, return_3m)
   |IC| > 0.03 且 p < 0.1 视为"有信号"

Backtest-Expert 指引（参考 ~/.claude/skills/claude-trading-skills）
----------------------------------------------------------------------
* 样本量至少 30 个配对（否则不报 IC）
* 打分模型确定后不回看调参（防止多重检验过拟合）
* IC 验证仅作去留决策，不做阈值搜索
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

_ROOT      = Path(__file__).resolve().parents[2]
_TEXT_DIR  = _ROOT / "data" / "text"
_ANN_CACHE = _TEXT_DIR / "announcements.parquet"
_FACTOR_CACHE = _TEXT_DIR / "ann_sentiment_factor.parquet"
_RETURNS_PATH = _ROOT / "data" / "sim30d" / "stock_returns.parquet"

# Transformer 模型候选（按优先级）
_CANDIDATE_MODELS = [
    "IDEA-CCNL/Erlangshen-RoBERTa-110M-Sentiment",
    "hw2942/bert-base-chinese-finetuning-financial-news-sentiment-v2",
]

# 词典降级
_POS_WORDS: frozenset[str] = frozenset({
    "增长", "上涨", "盈利", "增收", "超预期", "回购", "分红", "中标",
    "突破", "增持", "优异", "创新高", "签约", "获批", "领先",
})
_NEG_WORDS: frozenset[str] = frozenset({
    "亏损", "下滑", "减少", "减持", "风险", "诉讼", "处罚", "暴跌",
    "违规", "负面", "违约", "退市警示", "问询", "立案",
})

# 公开特征名
TEXT_SENTIMENT_FACTORS = ["ann_sentiment_5d"]


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


# ─── Step 1: 数据获取 ─────────────────────────────────────────────────────────

def fetch_announcements(
    start_date: str = "20240101",
    end_date: Optional[str] = None,
    force: bool = False,
) -> pd.DataFrame:
    """拉取 Tushare anns_d 公告数据；有缓存则直接返回缓存。

    Parameters
    ----------
    start_date : str  'YYYYMMDD'
    end_date   : str  'YYYYMMDD'，默认为今天
    force      : bool  True 则忽略缓存强制重新拉取

    Returns
    -------
    pd.DataFrame  columns: ts_code, ann_date, title, content
    """
    _TEXT_DIR.mkdir(parents=True, exist_ok=True)

    if _ANN_CACHE.exists() and not force:
        logger.info(f"公告缓存命中，跳过拉取：{_ANN_CACHE}")
        df = pd.read_parquet(_ANN_CACHE)
        df["ann_date"] = pd.to_datetime(df["ann_date"])
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

    # 按月循环拉取（单次 API 有行数限制）
    month_starts = pd.date_range(start=start_date, end=end_date, freq="MS")
    chunks: list[pd.DataFrame] = []

    for dt in month_starts:
        s = dt.strftime("%Y%m%d")
        e = min(
            (dt + pd.offsets.MonthEnd(0)).strftime("%Y%m%d"),
            end_date,
        )
        try:
            chunk = pro.anns_d(
                start_date=s, end_date=e,
                fields="ts_code,ann_date,title,content",
            )
            if chunk is not None and not chunk.empty:
                chunks.append(chunk)
                logger.info(f"  anns_d {s}~{e}: {len(chunk):,} 条")
        except Exception as exc:
            logger.warning(f"  anns_d {s}~{e} 拉取失败: {exc}")

    if not chunks:
        logger.warning("anns_d 未返回任何数据，返回空 DataFrame")
        empty = pd.DataFrame(columns=["ts_code", "ann_date", "title", "content"])
        empty.to_parquet(_ANN_CACHE, index=False)
        return empty

    df = (
        pd.concat(chunks, ignore_index=True)
        .drop_duplicates(subset=["ts_code", "ann_date", "title"])
        .sort_values(["ts_code", "ann_date"])
        .reset_index(drop=True)
    )
    df["ann_date"] = pd.to_datetime(df["ann_date"])
    df.to_parquet(_ANN_CACHE, index=False)
    logger.info(f"公告数据缓存至 {_ANN_CACHE}，共 {len(df):,} 条")
    return df


# ─── Step 2: 情绪打分 ─────────────────────────────────────────────────────────

def _load_sentiment_pipeline():
    """按优先级加载 transformers pipeline；全部失败则返回 (None, 'dict')."""
    for model_name in _CANDIDATE_MODELS:
        try:
            from transformers import pipeline as hf_pipeline
            pipe = hf_pipeline(
                "text-classification",
                model=model_name,
                return_all_scores=True,
                truncation=True,
                max_length=512,
            )
            logger.info(f"情绪模型加载成功：{model_name}")
            return pipe, model_name
        except Exception as exc:
            logger.warning(f"模型 {model_name} 加载失败: {exc}")
    logger.warning("所有 transformers 模型不可用，降级为词典法")
    return None, "dict"


def _score_dict(text: str) -> float:
    """词典法：(正面词数 - 负面词数) / (正面词数 + 负面词数 + 1e-9)."""
    if not isinstance(text, str) or not text:
        return 0.0
    pos = sum(1 for w in _POS_WORDS if w in text)
    neg = sum(1 for w in _NEG_WORDS if w in text)
    return (pos - neg) / (pos + neg + 1e-9)


def _results_to_score(results: list) -> float:
    """transformers all_scores 输出 → positive_prob - negative_prob."""
    lookup: dict[str, float] = {}
    for r in results:
        label = r["label"].lower()
        if any(k in label for k in ("pos", "positive", "1")):
            lookup["pos"] = r["score"]
        elif any(k in label for k in ("neg", "negative", "0")):
            lookup["neg"] = r["score"]
    pos = lookup.get("pos", 0.5)
    neg = lookup.get("neg", 0.5)
    return float(np.clip(pos - neg, -1.0, 1.0))


def score_announcements(
    df: pd.DataFrame,
    batch_size: int = 64,
    _pipe_override=None,       # 供测试注入 mock pipeline
) -> pd.DataFrame:
    """对公告 DataFrame 添加 sentiment_score 列 ∈ [-1, 1].

    Parameters
    ----------
    df            : 含 ts_code, ann_date, title, content 的 DataFrame
    batch_size    : transformer batch 推理大小
    _pipe_override: 测试用，直接传入已构建好的 pipeline（跳过模型加载）

    Returns
    -------
    输入 df + sentiment_score 列
    """
    if df.empty:
        return df.assign(sentiment_score=pd.Series(dtype=float), model_used="none")

    df = df.copy()
    texts = df.apply(
        lambda r: (
            str(r.get("title") or "")
            + " "
            + str(r.get("content") or "")[:200]
        ).strip(),
        axis=1,
    ).tolist()

    if _pipe_override is not None:
        pipe, model_name = _pipe_override, "mock"
    else:
        pipe, model_name = _load_sentiment_pipeline()

    scores: list[float] = []

    if pipe is not None:
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            try:
                raw = pipe(batch)
                for item in raw:
                    scores.append(_results_to_score(item))
            except Exception as exc:
                logger.warning(f"batch {i // batch_size} 推理失败，降级词典: {exc}")
                scores.extend(_score_dict(t) for t in batch)
    else:
        scores = [_score_dict(t) for t in texts]

    df["sentiment_score"] = [float(np.clip(s, -1.0, 1.0)) for s in scores]
    df["model_used"] = model_name
    return df


# ─── Step 3: 因子构造 ─────────────────────────────────────────────────────────

def build_ann_sentiment_factor(
    ann_df: pd.DataFrame,
    trading_cal: Optional[pd.DatetimeIndex] = None,
) -> pd.Series:
    """构造 ann_sentiment_5d 因子。

    聚合规则
    --------
    * 同一 (ts_code, trade_date) 多条公告 → 情绪均值
    * 每只股票做 5 日滚动均值（min_periods=1）
    * 缺失日期（无公告）填 0（中性）

    Returns
    -------
    pd.Series  MultiIndex(ts_code, trade_date), name='ann_sentiment_5d'
    """
    if ann_df.empty or "sentiment_score" not in ann_df.columns:
        return pd.Series(dtype=float, name="ann_sentiment_5d")

    df = ann_df.copy()
    df["trade_date"] = pd.to_datetime(df["ann_date"])

    # 同股同日多条公告 → 均值
    daily = (
        df.groupby(["ts_code", "trade_date"])["sentiment_score"]
        .mean()
        .reset_index()
    )

    results: list[pd.DataFrame] = []
    for code, grp in daily.groupby("ts_code"):
        grp_ts = grp.set_index("trade_date")["sentiment_score"].sort_index()

        # 对齐交易日历（如果提供）
        # 用整个 trading_cal（调用方负责控制输出范围）；
        # 早于第一条公告的日期也填 0，晚于最后公告的日期同样填 0。
        if trading_cal is not None:
            cal_slice = trading_cal[trading_cal >= trading_cal.min()]  # 即全日历
            grp_ts = grp_ts.reindex(trading_cal).fillna(0.0)

        # 5 日滚动均值，缺失填 0
        rolled = grp_ts.fillna(0.0).rolling(5, min_periods=1).mean()
        rolled.name = "ann_sentiment_5d"
        results.append(
            rolled.reset_index().rename(columns={"index": "trade_date"})
            .assign(ts_code=code)
        )

    if not results:
        return pd.Series(dtype=float, name="ann_sentiment_5d")

    factor_df = pd.concat(results, ignore_index=True)
    factor_df["trade_date"] = pd.to_datetime(factor_df["trade_date"])
    factor_series = (
        factor_df.set_index(["ts_code", "trade_date"])["ann_sentiment_5d"]
        .fillna(0.0)
    )
    return factor_series


# ─── Step 4: IC 验证 ─────────────────────────────────────────────────────────

def compute_ic(
    factor: pd.Series,
    returns_path: Path = _RETURNS_PATH,
) -> dict:
    """Spearman IC(ann_sentiment_5d, return_3m).

    stock_returns.parquet 格式：columns = date, ticker, ..., return_3m

    Returns
    -------
    dict: ic, p_value, n, valid (|IC|>0.03 且 p<0.1)
    """
    _EMPTY = {"ic": float("nan"), "p_value": float("nan"), "n": 0, "valid": False}

    if factor.empty:
        return _EMPTY
    if not Path(returns_path).exists():
        logger.warning(f"收益数据不存在：{returns_path}")
        return _EMPTY

    ret = pd.read_parquet(returns_path)
    if "return_3m" not in ret.columns:
        logger.warning(f"return_3m 列不存在，可用列: {ret.columns.tolist()}")
        return _EMPTY

    # stock_returns 用 ticker / date；factor 用 ts_code / trade_date
    ret = ret[["ticker", "date", "return_3m"]].dropna()
    ret["trade_date"] = pd.to_datetime(ret["date"])
    ret = ret.rename(columns={"ticker": "ts_code"}).set_index(["ts_code", "trade_date"])

    merged = (
        factor.rename("factor")
        .to_frame()
        .join(ret["return_3m"], how="inner")
        .dropna()
    )
    n = len(merged)

    if n < 30:
        logger.warning(f"样本量 {n} < 30，IC 不可靠")
        return {**_EMPTY, "n": n}

    ic, p_val = stats.spearmanr(merged["factor"], merged["return_3m"])
    valid = (abs(ic) > 0.03) and (p_val < 0.1)
    result = {
        "ic": float(ic),
        "p_value": float(p_val),
        "n": int(n),
        "valid": bool(valid),
    }
    logger.info(
        f"IC(ann_sentiment_5d, return_3m) = {ic:+.4f}  "
        f"p={p_val:.4f}  n={n}  "
        f"→ {'有信号 ✓' if valid else '信号不足 ✗'}"
    )
    return result


# ─── 端到端 pipeline ──────────────────────────────────────────────────────────

def run_full_pipeline(
    start_date: str = "20240101",
    force_fetch: bool = False,
) -> tuple[pd.Series, dict]:
    """端到端：数据获取 → 打分 → 因子构造 → IC 验证.

    Returns
    -------
    factor     : pd.Series  ann_sentiment_5d  MultiIndex(ts_code, trade_date)
    ic_result  : dict       {ic, p_value, n, valid}
    """
    _TEXT_DIR.mkdir(parents=True, exist_ok=True)

    ann_df  = fetch_announcements(start_date=start_date, force=force_fetch)
    scored  = score_announcements(ann_df)
    factor  = build_ann_sentiment_factor(scored)

    if not factor.empty:
        factor.reset_index().to_parquet(_FACTOR_CACHE, index=False)
        logger.info(f"因子已保存：{_FACTOR_CACHE}，shape={factor.shape}")

    ic_result = compute_ic(factor)
    return factor, ic_result
