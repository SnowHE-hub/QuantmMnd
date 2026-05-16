"""训练完成后注册升级的 Agent 模型版本为 active.

用法:
    python scripts/register_upgraded_agents.py --quality --sentiment
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from loguru import logger


def register_quality_lgbm_v2(model_path: str = "models/agents/quality_lgbm_v2.pkl") -> None:
    """注册 QualityAgent quality_lgbm_v2 为 active。"""
    from quantmind.agents.investment_agents.agent_registry import AgentModelRecord, AgentModelRegistry

    path = _ROOT / model_path
    if not path.exists():
        logger.error(f"模型文件不存在: {path}，请先运行 train_quality_agent_v2.py")
        return

    import pickle
    from datetime import datetime

    with open(path, "rb") as f:
        bundle = pickle.load(f)
    metrics = bundle.get("metrics", {})

    reg = AgentModelRegistry()
    record = AgentModelRecord(
        agent_name="QualityAgent",
        model_version="quality_lgbm_v2",
        model_type="ml",
        model_path=str(path),
        created_at=datetime.now().strftime("%Y-%m-%d"),
        performance={
            "accuracy": round(metrics.get("accuracy", 0), 4),
            "auc": round(metrics.get("auc", 0), 4),
            "ic_mean": round(metrics.get("ic_mean", 0), 4),
            "ic_ir": round(metrics.get("ic_ir", 0), 3),
            "n_features": len(bundle.get("feature_names", [])),
            "n_periods": bundle.get("n_periods", 0),
        },
        is_active=True,
        upgrade_notes="IC加权自监督质量标签训练，20财务因子，top/bottom 30%二分类",
    )
    reg.register(record)
    reg.set_active("QualityAgent", "quality_lgbm_v2")
    logger.info("QualityAgent → quality_lgbm_v2 已注册为 active ✓")


def register_sentiment_finbert_v4(model_path: str = "models/agents/sentiment_bert_v3.pkl") -> None:
    """注册 SentimentAgent finbert_llm_v4 为 active（复用 bert_v3 bundle 作为降级）。"""
    from quantmind.agents.investment_agents.agent_registry import AgentModelRecord, AgentModelRegistry

    path = _ROOT / model_path
    if not path.exists():
        logger.error(f"bert_v3 bundle 不存在: {path}")
        return

    from datetime import datetime

    reg = AgentModelRegistry()
    record = AgentModelRecord(
        agent_name="SentimentAgent",
        model_version="finbert_llm_v4",
        model_type="ml",
        model_path=str(path),  # bert_v3 bundle 用作 FinBERT 降级兜底
        created_at=datetime.now().strftime("%Y-%m-%d"),
        performance={
            "pipeline": "finbert+llm",
            "finbert_model": "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment",
            "fallback": "bert_v3",
            "description": "FinBERT批量扫描 + LLM深度合成，降级路径: bert_v3 TF-IDF余弦",
        },
        is_active=True,
        upgrade_notes="双级流水线：FinBERT批量打分选TopK → LLM情感深度合成",
    )
    reg.register(record)
    reg.set_active("SentimentAgent", "finbert_llm_v4")
    logger.info("SentimentAgent → finbert_llm_v4 已注册为 active ✓")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", action="store_true", help="注册 QualityAgent v2")
    parser.add_argument("--sentiment", action="store_true", help="注册 SentimentAgent finbert_llm_v4")
    parser.add_argument("--all", action="store_true", help="注册所有")
    args = parser.parse_args()

    if args.all or args.quality:
        register_quality_lgbm_v2()
    if args.all or args.sentiment:
        register_sentiment_finbert_v4()

    if not (args.quality or args.sentiment or args.all):
        parser.print_help()


if __name__ == "__main__":
    main()
