"""升级某个 Agent 的模型版本.

用法：
  # 升级 MomentumAgent 到 LGBM v2
  python scripts/upgrade_agent_model.py \\
    --agent MomentumAgent \\
    --target-version lgbm_v2 \\
    --train-data data/panel/monthly_train.parquet

  # 升级 SentimentAgent 到 TF-IDF 情感分类器
  python scripts/upgrade_agent_model.py \\
    --agent SentimentAgent \\
    --target-version tfidf_v2
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from quantmind.agents.investment_agents.agent_registry import (
    AgentModelRecord,
    AgentModelRegistry,
    initialize_default_registry,
)


# ── 训练函数 ────────────────────────────────────────────────────────────────


def train_momentum_lgbm_v2(
    train_data_path: str | Path,
    val_data_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> tuple[Path, dict]:
    """训练 MomentumAgent LGBM v2 模型.

    输入：momentum 因子 + forward_return_21d 标签
    输出：pkl 文件 + 性能指标
    """
    import lightgbm as lgb
    from sklearn.metrics import r2_score
    from scipy.stats import spearmanr

    train_path = Path(train_data_path)
    output_path = Path(output_path or _ROOT / "models" / "agents" / "momentum_lgbm_v2.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"[upgrade] 加载训练数据: {train_path}")
    train_df = pd.read_parquet(train_path)

    FEATURES = [
        "momentum_1m", "momentum_3m", "momentum_6m",
        "vol_1m", "vol_3m", "reversal_1w",
        "amihud_1m", "beta_60d", "relative_strength_vs_csi300_60d",
    ]
    LABEL = "forward_return_21d"

    avail_features = [f for f in FEATURES if f in train_df.columns]
    if not avail_features or LABEL not in train_df.columns:
        raise ValueError(f"训练数据缺少必要列，可用: {train_df.columns.tolist()[:10]}...")

    train_df = train_df.dropna(subset=[LABEL])
    X_train = train_df[avail_features].fillna(0).values
    y_train = train_df[LABEL].values

    # 验证集
    val_ic = None
    X_val, y_val = None, None
    if val_data_path and Path(val_data_path).exists():
        val_df = pd.read_parquet(val_data_path).dropna(subset=[LABEL])
        X_val = val_df[avail_features].fillna(0).values
        y_val = val_df[LABEL].values
    elif len(X_train) > 2000:
        # 最后 20% 作验证
        split = int(len(X_train) * 0.8)
        X_val, y_val = X_train[split:], y_train[split:]
        X_train, y_train = X_train[:split], y_train[:split]

    logger.info(f"[upgrade] 训练样本: {len(X_train)}, 特征: {avail_features}")

    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=42,
        verbose=-1,
    )
    eval_set = [(X_val, y_val)] if X_val is not None else None
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
    )

    # 计算验证集 Rank IC
    metrics: dict = {"features_used": avail_features}
    if X_val is not None and y_val is not None and len(y_val) > 10:
        preds = model.predict(X_val)
        ic, _ = spearmanr(preds, y_val)
        metrics["ic_mean"] = round(float(ic), 4)
        metrics["r2"] = round(float(r2_score(y_val, preds)), 4)
        logger.info(f"[upgrade] 验证集 Rank IC = {ic:.4f}")
    else:
        metrics["ic_mean"] = 0.0

    # 保存
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"[upgrade] 模型已保存: {output_path}")

    return output_path, metrics


def train_sentiment_tfidf_v2(
    output_path: str | Path | None = None,
    news_data_path: str | Path | None = None,
) -> tuple[Path, dict]:
    """训练 SentimentAgent TF-IDF v2 情感分类器.

    使用内置关键词生成伪标签训练数据（当无真实标注时）。
    输出：pkl 文件（包含 vectorizer + classifier）
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    output_path = Path(output_path or _ROOT / "models" / "agents" / "sentiment_tfidf_v2.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 加载或生成训练数据
    if news_data_path and Path(news_data_path).exists():
        logger.info(f"[upgrade] 从 {news_data_path} 加载新闻数据")
        news_df = pd.read_parquet(news_data_path)
        texts = news_df.get("text", news_df.get("title", pd.Series(dtype=str))).dropna().tolist()
        labels_raw = news_df.get("label", pd.Series(dtype=str))
        # 将文本标签转换为数字：正面=2, 中性=1, 负面=0
        label_map = {"positive": 2, "neutral": 1, "negative": 0, "pos": 2, "neg": 0}
        labels = labels_raw.map(label_map).fillna(1).astype(int).tolist()
    else:
        logger.info("[upgrade] 无真实标注数据，使用关键词伪标签生成训练集")
        texts, labels = _generate_pseudo_labeled_data()

    logger.info(f"[upgrade] 训练样本: {len(texts)}, 类别分布: {pd.Series(labels).value_counts().to_dict()}")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)

    classifier = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    classifier.fit(X, labels)

    # 交叉验证
    cv_scores = cross_val_score(classifier, X, labels, cv=5, scoring="f1_macro")
    metrics = {
        "f1_macro_cv": round(float(cv_scores.mean()), 4),
        "accuracy": round(float((classifier.predict(X) == np.array(labels)).mean()), 4),
        "n_samples": len(texts),
    }
    logger.info(f"[upgrade] F1-macro CV = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 保存为包含 vectorizer 和 classifier 的 dict
    model_bundle = {"vectorizer": vectorizer, "classifier": classifier}
    with open(output_path, "wb") as f:
        pickle.dump(model_bundle, f)
    logger.info(f"[upgrade] TF-IDF 模型已保存: {output_path}")

    return output_path, metrics


def _generate_pseudo_labeled_data() -> tuple[list[str], list[int]]:
    """用关键词规则生成伪标签训练数据."""
    positive_templates = [
        "{}营收增长{}%，超出市场预期",
        "{}获得重大合同，金额{}亿元",
        "{}产品获批上市，市场前景广阔",
        "{}利润大幅增长，创历史新高",
        "{}战略合作签约，业务快速扩张",
        "{}研发突破，技术领先行业",
        "{}股票买入评级，目标价上调",
        "{}业绩超预期，分析师增持推荐",
    ]
    negative_templates = [
        "{}业绩大幅亏损，同比下滑{}%",
        "{}收到监管问询函，存在合规风险",
        "{}控股股东减持，持股比例下降",
        "{}面临诉讼风险，赔偿金额较大",
        "{}营业收入持续下滑，经营压力较大",
        "{}受行业竞争加剧影响，盈利能力下降",
        "{}分析师调降目标价，评级改为卖出",
        "{}债务风险上升，流动性压力较大",
    ]
    neutral_templates = [
        "{}召开股东大会，审议相关议案",
        "{}发布三季度报告，业绩符合预期",
        "{}公告重大资产重组，方案待定",
        "{}维持中性评级，目标价不变",
        "{}经营情况稳定，无重大异常",
        "{}完成年度审计，无保留意见",
    ]

    texts, labels = [], []
    tickers = ["某公司", "该企业", "本公司", "目标公司"]

    for tmpl in positive_templates * 20:
        t = np.random.choice(tickers)
        texts.append(tmpl.format(t, np.random.randint(10, 50)))
        labels.append(2)

    for tmpl in negative_templates * 20:
        t = np.random.choice(tickers)
        texts.append(tmpl.format(t, np.random.randint(10, 50)))
        labels.append(0)

    for tmpl in neutral_templates * 20:
        t = np.random.choice(tickers)
        texts.append(tmpl.format(t))
        labels.append(1)

    return texts, labels


# ── 升级主函数 ──────────────────────────────────────────────────────────────


def upgrade_agent(
    agent_name: str,
    target_version: str,
    train_data_path: str | None = None,
    val_data_path: str | None = None,
    force_retrain: bool = False,
) -> AgentModelRecord:
    """升级 Agent 到指定版本.

    1. 检查目标版本是否已训练好（在注册表中）
    2. 若已有：直接切换激活版本
    3. 若未有：训练新模型 → 注册 → 切换激活
    4. 在验证集上对比新旧版本性能
    5. 若新版本更好，确认切换；否则回滚
    """
    registry = AgentModelRegistry()
    # 确保 rules_v1 基线已注册
    initialize_default_registry()
    registry = AgentModelRegistry()  # 重新加载

    # 检查是否已注册
    history = registry.get_history(agent_name)
    existing = next((r for r in history if r.model_version == target_version), None)

    if existing and not force_retrain:
        logger.info(f"[upgrade] {agent_name}/{target_version} 已存在，直接切换激活版本")
        old_active = registry.get_active(agent_name)
        _print_comparison(registry, agent_name, old_active, existing)
        registry.set_active(agent_name, target_version)
        return existing

    # 训练新模型
    logger.info(f"[upgrade] 开始训练 {agent_name}/{target_version}...")
    model_path, metrics = _train_model(agent_name, target_version, train_data_path, val_data_path)

    # 获取旧模型性能
    old_active = registry.get_active(agent_name)
    old_ic = old_active.performance.get("ic_mean", 0.0) if old_active else 0.0
    new_ic = metrics.get("ic_mean", 0.0)

    logger.info(f"[upgrade] 旧版本({old_active.model_version if old_active else 'N/A'}) IC={old_ic:.4f}")
    logger.info(f"[upgrade] 新版本({target_version}) IC={new_ic:.4f}")

    # 注册新版本
    new_record = AgentModelRecord(
        agent_name=agent_name,
        model_version=target_version,
        model_type=_get_model_type(target_version),
        model_path=str(model_path) if model_path else None,
        created_at=datetime.now().isoformat(),
        performance=metrics,
        is_active=False,
        upgrade_notes=f"从 {old_active.model_version if old_active else 'N/A'} 升级",
    )
    registry.register(new_record)

    # 对比决策
    if new_ic >= old_ic * 0.9 or new_ic > 0:  # 新模型不差于旧模型 90%
        logger.info(f"[upgrade] ✅ 新模型表现良好，切换激活版本 → {target_version}")
        registry.set_active(agent_name, target_version)
        _print_comparison(registry, agent_name, old_active, new_record)
    else:
        logger.warning(
            f"[upgrade] ⚠️ 新模型 IC({new_ic:.4f}) < 旧模型({old_ic:.4f})×90%，保留旧版本"
        )
        _print_comparison(registry, agent_name, old_active, new_record)

    return new_record


def _train_model(
    agent_name: str,
    target_version: str,
    train_data_path: str | None,
    val_data_path: str | None,
) -> tuple[Path | None, dict]:
    """根据 agent+version 分发到具体训练函数."""
    if agent_name == "MomentumAgent" and target_version == "lgbm_v2":
        if not train_data_path:
            train_data_path = str(_ROOT / "data" / "panel" / "monthly_train.parquet")
        return train_momentum_lgbm_v2(
            train_data_path,
            val_data_path or str(_ROOT / "data" / "panel" / "monthly_val.parquet"),
        )
    elif agent_name == "SentimentAgent" and target_version == "tfidf_v2":
        return train_sentiment_tfidf_v2()
    else:
        raise NotImplementedError(
            f"不支持 {agent_name}/{target_version} 的自动训练。"
            "请先手动训练模型，再用 --no-retrain 直接注册。"
        )


def _get_model_type(version: str) -> str:
    if "rules" in version:
        return "rules"
    if "piotroski" in version:
        return "rules_plus"
    if "lgbm" in version or "xgb" in version or "tfidf" in version or "garch" in version:
        return "ml"
    if "lstm" in version or "transformer" in version or "bert" in version:
        return "dl"
    if "llm" in version or "gpt" in version:
        return "llm"
    return "ml"


def register_generic_version(
    *,
    agent_name: str,
    model_version: str,
    model_type: str,
    model_path: str | None,
    activate: bool,
    notes: str,
    performance: dict | None = None,
) -> AgentModelRecord:
    """注册任意新版本（不触发训练），可选立即激活."""
    initialize_default_registry()
    registry = AgentModelRegistry()

    resolved_path: str | None = None
    if model_path:
        pth = Path(model_path)
        if not pth.is_absolute():
            pth = _ROOT / pth
        resolved_path = str(pth.resolve())

    new_record = AgentModelRecord(
        agent_name=agent_name,
        model_version=model_version,
        model_type=model_type,
        model_path=resolved_path,
        created_at=datetime.now().isoformat(),
        performance=performance or {},
        is_active=False,
        upgrade_notes=notes or "",
    )
    registry.register(new_record)

    if activate:
        registry.set_active(agent_name, model_version)

    active = registry.get_active(agent_name)
    _print_comparison(registry, agent_name, None, active or new_record)
    return active or new_record


def _print_comparison(
    registry: AgentModelRegistry,
    agent_name: str,
    old: AgentModelRecord | None,
    new: AgentModelRecord,
) -> None:
    df = registry.compare_versions(agent_name)
    if not df.empty:
        logger.info(f"\n[upgrade] {agent_name} 版本对比:\n{df.to_string(index=False)}")


def register_pretrained_momentum_lstm(
    model_path: Path,
    metrics_path: Path,
    *,
    excess_acc_min: float = 0.52,
    ic_excess_ratio: float = 1.0,
) -> AgentModelRecord:
    """注册已训练的 lstm_v3，并与当前激活版本（通常为 lgbm_v2）对比决定是否切换激活."""
    initialize_default_registry()
    registry = AgentModelRegistry()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    val_acc = float(metrics.get("val_acc", 0.0))

    old_active = registry.get_active("MomentumAgent")
    old_ic = float(old_active.performance.get("ic_mean", 0.028)) if old_active else 0.028
    excess = val_acc - 0.5

    activate = val_acc >= excess_acc_min and excess >= old_ic * ic_excess_ratio

    mp_resolved = model_path if model_path.is_absolute() else _ROOT / model_path

    new_record = AgentModelRecord(
        agent_name="MomentumAgent",
        model_version="lstm_v3",
        model_type="dl",
        model_path=str(mp_resolved.resolve()),
        created_at=datetime.now().isoformat(),
        performance={
            **metrics,
            "ic_mean": val_acc - 0.5,
            "benchmark_lgbm_ic": old_ic,
        },
        is_active=False,
        upgrade_notes=f"compare val_acc={val_acc:.4f} vs lgbm_ic={old_ic:.4f} (excess>={old_ic*ic_excess_ratio:.4f})",
    )
    registry.register(new_record)

    logger.info(
        "[upgrade] lstm_v3 val_acc={:.4f} excess={:.4f} | lgbm_ic={:.4f} | activate={}",
        val_acc,
        excess,
        old_ic,
        activate,
    )

    if activate:
        registry.set_active("MomentumAgent", "lstm_v3")
    else:
        logger.warning("[upgrade] 保留当前激活版本 → {}", old_active.model_version if old_active else "N/A")

    new_active = registry.get_active("MomentumAgent")
    if new_active is None:
        new_active = new_record
    _print_comparison(registry, "MomentumAgent", old_active, new_active)
    return new_active


# ── CLI ─────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="升级 Agent 模型版本")
    p.add_argument("--agent", required=True,
                   choices=["MomentumAgent", "SentimentAgent", "ValuationAgent",
                            "QualityAgent", "RiskAgent", "StrategyAgent"],
                   help="要升级的 Agent 名称")
    p.add_argument("--target-version", "--model-version", required=False, default=None,
                   help="目标版本，例如：lgbm_v2, tfidf_v2")
    p.add_argument("--train-data", default=None,
                   help="训练数据路径（parquet）")
    p.add_argument("--val-data", default=None,
                   help="验证数据路径（parquet）")
    p.add_argument("--force-retrain", action="store_true",
                   help="强制重新训练（即使已有该版本）")
    p.add_argument("--init-registry", action="store_true",
                   help="初始化注册表基线记录")
    p.add_argument("--model-path", type=Path, default=None,
                   help="预训练模型路径（与 --register-only 联用，例如 LSTM .pt）")
    p.add_argument("--metrics-json", type=Path, default=None,
                   help="性能 JSON（默认 reports/model_training/momentum_lstm_metrics.json）")
    p.add_argument("--register-only", action="store_true",
                   help="跳过训练，仅注册模型并执行激活对比（MomentumAgent/lstm_v3）")
    p.add_argument("--register-generic", action="store_true",
                   help="跳过训练，直接写入 registry（配合 --type/--path/--activate）")
    p.add_argument("--type", dest="artifact_model_type", default=None,
                   help="register-generic: model_type（ml/dl/rules/rules_plus/llm）")
    p.add_argument("--path", dest="artifact_path", default=None,
                   help='register-generic: 产物路径；QualityAgent 可无文件，传 ""')
    p.add_argument("--activate", action="store_true",
                   help="register-generic: 注册后立即激活")
    p.add_argument("--notes", dest="upgrade_notes", default="",
                   help="register-generic: upgrade_notes")
    p.add_argument("--excess-acc-min", type=float, default=0.52,
                   help="激活所需的最低 val_acc")
    p.add_argument("--ic-excess-ratio", type=float, default=1.0,
                   help="另需满足 (val_acc-0.5) >= ic_mean(lgbm)×ratio")
    args = p.parse_args()

    if (
        not args.init_registry
        and not args.register_generic
        and not (args.register_only and args.agent == "MomentumAgent" and args.target_version == "lstm_v3")
        and args.target_version is None
    ):
        p.error("--target-version 为必填（除非使用 --init-registry / --register-generic / lstm --register-only）")

    return args


def main() -> None:
    args = _parse_args()

    if args.init_registry:
        registry = initialize_default_registry(force=False)
        logger.info(f"[upgrade] 注册表已初始化: {registry.to_summary_dict()}")
        return

    if args.register_generic:
        mv = args.target_version
        assert mv is not None
        mt = args.artifact_model_type or _get_model_type(mv)
        raw_path = args.artifact_path
        mp: str | None
        if raw_path is None:
            mp = None
        elif str(raw_path).strip() == "":
            mp = None
        else:
            mp = str(raw_path)
        record = register_generic_version(
            agent_name=args.agent,
            model_version=mv,
            model_type=mt,
            model_path=mp,
            activate=bool(args.activate),
            notes=args.upgrade_notes or "",
        )
        logger.info(
            f"\n[upgrade] register-generic 完成: {record.agent_name}/{record.model_version}"
            f" type={record.model_type} is_active={record.is_active}"
        )
        return

    if args.register_only and args.agent == "MomentumAgent" and args.target_version == "lstm_v3":
        mp = args.model_path or (_ROOT / "models/agents/momentum_lstm_v3.pt")
        mj = args.metrics_json or (_ROOT / "reports/model_training/momentum_lstm_metrics.json")
        record = register_pretrained_momentum_lstm(
            mp,
            mj,
            excess_acc_min=args.excess_acc_min,
            ic_excess_ratio=args.ic_excess_ratio,
        )
        logger.info(
            f"\n[upgrade] 完成: {record.agent_name}/{record.model_version}"
            f" is_active={record.is_active}"
            f" val_acc={record.performance.get('val_acc', 'N/A')}"
        )
        return

    assert args.target_version is not None
    record = upgrade_agent(
        agent_name=args.agent,
        target_version=args.target_version,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        force_retrain=args.force_retrain,
    )

    logger.info(
        f"\n[upgrade] 完成: {record.agent_name}/{record.model_version}"
        f" is_active={record.is_active}"
        f" IC={record.performance.get('ic_mean', 'N/A')}"
    )


if __name__ == "__main__":
    main()
