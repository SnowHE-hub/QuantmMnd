#!/usr/bin/env python
"""训练 SentimentAgent bert_v3：正负种子语句 → 中心向量（TF-IDF+jieba 默认，可选本地 ST）."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from loguru import logger

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from quantmind.agents.investment_agents.sentiment_tokenizer import jieba_cut_analyzer

POSITIVE_CORPUS = [
    "业绩超预期",
    "净利润大幅增长",
    "营收增速创新高",
    "盈利能力持续改善",
    "在手订单饱满",
    "分红预案慷慨",
    "回购彰显信心",
    "产能利用率提升",
    "毛利率环比上行",
    "现金流充裕稳健",
    "海外业务扩张顺利",
    "新产品放量快速增长",
    "费用率显著优化",
    "市场份额持续提升",
    "政策红利逐步兑现",
    "战略合作深化落地",
    "项目如期投产见效",
    "核心客户粘性增强",
    "资产负债表持续修复",
    "全年指引上调明朗",
    "盈利预测上调",
    "分析师一致看好",
    "订单能见度大幅提升",
    "研发成果转化显著",
    "成本下行增厚利润",
    "量价齐升驱动增长",
    "提价传导顺畅",
    "减值压力显著减轻",
    "景气度延续高位",
    "监管风险可控局面清晰",
]

NEGATIVE_CORPUS = [
    "业绩亏损",
    "净利润大幅下滑",
    "营收不及预期",
    "毛利率承压走弱",
    "现金流恶化紧张",
    "存货减值风险上升",
    "应收账款回收困难",
    "下调全年指引",
    "监管处罚",
    "立案调查",
    "高管被查",
    "核心资产出售",
    "大客户流失",
    "价格战拖累盈利",
    "产能过剩利用率下行",
    "债务违约风险抬升",
    "兑付压力骤增",
    "商誉减值巨额计提",
    "诉讼缠身赔偿不确定",
    "环保安全事故影响生产",
    "行业景气拐点向下",
    "政策收紧冲击主业",
    "募投项目延期",
    "股权激励费用拖累",
    "汇兑损失扩大",
    "原材料暴涨侵蚀利润",
    "停工停产整顿",
    "评级下调至卖出",
    "审计出具非标意见",
]


def _train_tfidf_bundle(
    pos_corpus: list[str],
    neg_corpus: list[str],
    output_path: Path,
) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(
        analyzer=jieba_cut_analyzer,
        max_features=12000,
        min_df=1,
        sublinear_tf=True,
    )
    corpus = pos_corpus + neg_corpus
    vec.fit(corpus)
    pos_mat = vec.transform(pos_corpus).toarray().astype(np.float32)
    neg_mat = vec.transform(neg_corpus).toarray().astype(np.float32)
    pos_center = pos_mat.mean(axis=0)
    neg_center = neg_mat.mean(axis=0)
    bundle = {
        "type": "tfidf_cosine",
        "vectorizer": vec,
        "pos_center": pos_center,
        "neg_center": neg_center,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(bundle, f)
    return {
        "method": "tfidf_cosine",
        "pos_center_dim": int(pos_center.shape[0]),
        "n_positive_seed": len(pos_corpus),
        "n_negative_seed": len(neg_corpus),
        "note": "默认离线：jieba + TF-IDF 中心向量；未下载任何预训练权重",
    }


def _train_st_bundle_if_local(
    pos_corpus: list[str],
    neg_corpus: list[str],
    local_model_path: Path,
    output_path: Path,
) -> dict:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(local_model_path))
    pos_emb = model.encode(pos_corpus, convert_to_numpy=True, show_progress_bar=False)
    neg_emb = model.encode(neg_corpus, convert_to_numpy=True, show_progress_bar=False)
    pos_center = pos_emb.mean(axis=0).astype(np.float32)
    neg_center = neg_emb.mean(axis=0).astype(np.float32)
    bundle = {
        "type": "sentence_transformer",
        "local_model_path": str(local_model_path.resolve()),
        "pos_center": pos_center,
        "neg_center": neg_center,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(bundle, f)
    return {
        "method": "sentence_transformer",
        "pos_center_dim": int(pos_center.shape[0]),
        "n_positive_seed": len(pos_corpus),
        "n_negative_seed": len(neg_corpus),
        "note": f"本地 SentenceTransformer：{local_model_path}",
    }


def _eval_kb_cosine(bundle: dict, kb_parquet: Path | None) -> None:
    """可选：对 KB 导出新闻做快速一致性检查（不参与训练拟合）."""
    if kb_parquet is None or not kb_parquet.exists():
        return
    try:
        import pandas as pd

        df = pd.read_parquet(kb_parquet).head(200)
        texts = df.get("text", df.get("title", pd.Series(dtype=str))).dropna().astype(str).tolist()
        if not texts:
            return
        # 这里只做占位评估：避免引入额外依赖逻辑
        logger.info(f"[sentiment_v3] KB 抽样 {len(texts)} 条用于人工检视（未写入 metrics 数值）")
    except Exception as exc:
        logger.warning(f"[sentiment_v3] KB 评估跳过: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train SentimentAgent bert_v3 bundle")
    ap.add_argument(
        "--output-model",
        type=Path,
        default=_ROOT / "models" / "agents" / "sentiment_bert_v3.pkl",
    )
    ap.add_argument(
        "--metrics-json",
        type=Path,
        default=_ROOT / "reports" / "model_training" / "sentiment_bert_v3_metrics.json",
    )
    ap.add_argument("--kb-news-parquet", type=Path, default=None, help="可选 KB 新闻 parquet")
    args = ap.parse_args()

    st_path = os.environ.get("QUANTMIND_ST_MODEL_PATH", "").strip()

    if st_path:
        metrics = _train_st_bundle_if_local(
            POSITIVE_CORPUS,
            NEGATIVE_CORPUS,
            Path(st_path),
            args.output_model,
        )
    else:
        metrics = _train_tfidf_bundle(POSITIVE_CORPUS, NEGATIVE_CORPUS, args.output_model)

    _eval_kb_cosine({}, args.kb_news_parquet)

    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[sentiment_v3] bundle saved → {args.output_model}")


if __name__ == "__main__":
    main()
