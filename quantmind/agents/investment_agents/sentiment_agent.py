"""quantmind.agents.investment_agents.sentiment_agent — 情绪分析 Agent.

模型版本层级（finbert_llm_v4 为最新）：
  finbert_llm_v4: FinBERT 快速全扫描 → LLM 深度合成
  bert_v3:        TF-IDF 语义中心向量
  tfidf_v2:       TF-IDF + 分类器
  rules_v1:       关键词规则（兜底）
"""

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from quantmind.agents.investment_agents.base_agent import AgentSignal, BaseInvestmentAgent

# ── FinBERT 单例缓存（避免重复加载，首次约 10-30s）────────────────────────
_FINBERT_PIPELINE = None
_FINBERT_LOAD_ATTEMPTED = False
_FINBERT_MODEL_NAME = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"


def _get_finbert_pipeline():
    """惰性加载 FinBERT pipeline（全局单例）。不可用时静默返回 None。"""
    global _FINBERT_PIPELINE, _FINBERT_LOAD_ATTEMPTED
    if _FINBERT_LOAD_ATTEMPTED:
        return _FINBERT_PIPELINE
    _FINBERT_LOAD_ATTEMPTED = True
    try:
        from transformers import pipeline as hf_pipeline

        model_name = os.environ.get("QUANTMIND_FINBERT_MODEL", _FINBERT_MODEL_NAME)
        device = 0 if _has_gpu() else -1  # 0=GPU, -1=CPU
        logger.info(f"[SentimentAgent] 加载 FinBERT: {model_name}（device={device}）")
        _FINBERT_PIPELINE = hf_pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=device,
            truncation=True,
            max_length=512,
        )
        logger.info("[SentimentAgent] FinBERT 加载完成 ✓")
    except Exception as e:
        logger.warning(f"[SentimentAgent] FinBERT 不可用（{e}），将使用 bert_v3 fallback")
        _FINBERT_PIPELINE = None
    return _FINBERT_PIPELINE


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_POSITIVE_WORDS = ["增长", "突破", "创新高", "超预期", "扩张", "获批", "上涨", "利好", "战略", "签约"]
_NEGATIVE_WORDS = ["亏损", "下滑", "问询", "监管", "减持", "诉讼", "下跌", "风险", "处罚", "违规"]
_BULLISH_WORDS = ["买入", "强烈推荐", "强推", "增持", "看好"]
_NEUTRAL_WORDS = ["中性", "持有", "观望"]
_BEARISH_WORDS = ["卖出", "减持", "回避", "看空", "做空"]


class SentimentAgent(BaseInvestmentAgent):
    """情绪分析 Agent — 规则层 + 可选 LLM + TF-IDF v2 + bert_v3（离线向量中心）."""

    def __init__(
        self,
        ticker: str,
        as_of: str,
        context: dict,
        provider: str = "none",
        model: str = "qwen-plus",
        model_version: str = "active",
    ) -> None:
        super().__init__(ticker, as_of, context, model_version=model_version)
        self.provider = provider
        self.model = model

    def _load_model(self, record):  # type: ignore[override]
        """bert_v3 使用 pickle bundle；其余走基类逻辑."""
        from quantmind.agents.investment_agents.agent_registry import AgentModelRecord

        if not isinstance(record, AgentModelRecord):
            return super()._load_model(record)
        if record.model_version == "bert_v3" and record.model_path:
            path = Path(str(record.model_path))
            if not path.is_absolute():
                path = _ROOT / path
            if path.exists():
                with open(path, "rb") as f:
                    return pickle.load(f)
            logger.warning(f"[SentimentAgent] bert_v3 文件不存在: {path}")
        return super()._load_model(record)

    def analyze(self) -> AgentSignal:
        active_version = (
            self._model_record.model_version if self._model_record else "rules_v1"
        )

        # ── finbert_llm_v4：两级流水线（FinBERT 扫描 + LLM 合成）────────────
        if active_version == "finbert_llm_v4":
            sig = self._analyze_finbert_llm_v4()
            if sig is not None:
                return sig
            # FinBERT 不可用时自动降级到 bert_v3

        if active_version in ("finbert_llm_v4", "bert_v3") and isinstance(self._ml_model, dict):
            sig = self._analyze_bert_v3()
            if sig is not None:
                return sig

        if active_version == "tfidf_v2" and self._ml_model is not None:
            return self._analyze_tfidf()

        return self._analyze_rules()

    def _analyze_bert_v3(self) -> AgentSignal | None:
        bundle = self._ml_model
        evidence: dict = {
            "method": "bert_v3",
            "encoder": bundle.get("type", ""),
        }
        warnings: list[str] = []

        news_items = self.context.get("news_context", [])
        report_items = self.context.get("report_context", [])

        texts: list[str] = []
        for item in news_items[:12]:
            if isinstance(item, dict):
                t = item.get("text", "") or item.get("title", "")
            else:
                t = str(item)
            if str(t).strip():
                texts.append(str(t)[:400])
        for item in report_items[:6]:
            if isinstance(item, dict):
                t = item.get("text", "") or item.get("title", "")
            else:
                t = str(item)
            if str(t).strip():
                texts.append(str(t)[:400])

        # 兼容直接在 context 注入列表的场景（测试 / 工具）
        for extra_key in ("news_texts", "report_texts"):
            raw = self.context.get(extra_key)
            if isinstance(raw, list):
                texts.extend(str(x)[:400] for x in raw if str(x).strip())

        if not texts:
            evidence["note"] = "无文本，降级规则"
            return self._analyze_rules()

        try:
            score = float(self._cosine_score(bundle, texts))
            score = self._clamp(score)
            evidence["news_count"] = len(texts)

            if score > 0.2:
                summary = "语义中心判定情绪偏正面"
            elif score < -0.2:
                summary = "语义中心判定情绪偏负面"
            else:
                summary = "语义中心判定情绪中性"

            confidence = min(0.88, 0.55 + min(len(texts), 8) * 0.03)
            return AgentSignal(
                agent_name="SentimentAgent",
                ticker=self.ticker,
                signal=score,
                confidence=confidence,
                summary=summary[:50],
                evidence=evidence,
                warnings=warnings,
            )
        except Exception as exc:
            logger.warning(f"[SentimentAgent/bert_v3] 失败: {exc}，降级规则")
            return None

    def _cosine_score(self, bundle: dict, texts: list[str]) -> float:
        """cos(text, pos) - cos(text, neg)，裁剪到 [-1,1]."""
        pos = np.asarray(bundle["pos_center"], dtype=np.float64).ravel()
        neg = np.asarray(bundle["neg_center"], dtype=np.float64).ravel()

        btype = bundle.get("type")

        if btype == "tfidf_cosine":
            vec = bundle["vectorizer"]
            mat = vec.transform(texts).toarray().astype(np.float64)
        elif btype == "sentence_transformer":
            local = bundle.get("local_model_path") or os.environ.get("QUANTMIND_ST_MODEL_PATH", "")
            if not local:
                raise RuntimeError("缺少 QUANTMIND_ST_MODEL_PATH / bundle.local_model_path，无法加载 ST")
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(str(local))
            mat = model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype(np.float64)
        else:
            raise ValueError(f"未知 bundle.type={btype}")

        v = mat.mean(axis=0)

        def cos(a: np.ndarray, b: np.ndarray) -> float:
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
            return float(np.dot(a, b) / denom)

        raw = cos(v, pos) - cos(v, neg)
        return float(max(-1.0, min(1.0, raw)))

    # ─────────────────────────────────────────────────────────────────────────
    # finbert_llm_v4：两级流水线
    # ─────────────────────────────────────────────────────────────────────────

    def _analyze_finbert_llm_v4(self) -> Optional[AgentSignal]:
        """
        Level-1：FinBERT 快速对全部文档打分，选出信号最强的 Top-K 篇。
        Level-2：qwen-plus 对 Top-K 篇做有推理的深度合成。
        最终分数 = 0.4 × finbert_avg + 0.6 × llm_score（无 LLM 时全用 FinBERT）。
        """
        pipe = _get_finbert_pipeline()
        if pipe is None:
            return None  # 降级到 bert_v3

        texts = self._collect_texts(max_news=20, max_reports=8)
        if not texts:
            return self._analyze_rules()

        evidence: dict = {"method": "finbert_llm_v4"}
        warnings: list[str] = []

        # ── Level-1：FinBERT 批量打分 ──────────────────────────────────────
        try:
            results = pipe(texts, batch_size=8)
        except Exception as e:
            logger.warning(f"[SentimentAgent/finbert] 推理失败: {e}")
            return None

        label_map = {
            "正面": 1.0, "POSITIVE": 1.0, "POS": 1.0, "LABEL_2": 1.0,
            "负面": -1.0, "NEGATIVE": -1.0, "NEG": -1.0, "LABEL_0": -1.0,
            "中性": 0.0, "NEUTRAL": 0.0, "NEU": 0.0, "LABEL_1": 0.0,
        }
        scored: list[tuple[float, str]] = []  # (score, text)
        for res, text in zip(results, texts):
            label = str(res.get("label", "")).upper().strip()
            score_raw = label_map.get(label, label_map.get(res.get("label", ""), 0.0))
            confidence = float(res.get("score", 0.5))
            # 将置信度加权：(pos/neg) × confidence，中性权重低
            weighted = score_raw * confidence if score_raw != 0 else 0.0
            scored.append((weighted, text))

        finbert_scores = [s for s, _ in scored]
        finbert_avg = float(np.mean(finbert_scores)) if finbert_scores else 0.0
        finbert_avg = self._clamp(finbert_avg * 1.5)  # 适当放大

        evidence["finbert_avg"] = round(finbert_avg, 3)
        evidence["doc_count"] = len(texts)
        evidence["pos_count"] = sum(1 for s, _ in scored if s > 0.1)
        evidence["neg_count"] = sum(1 for s, _ in scored if s < -0.1)

        # ── Level-2：选 Top-K 最强信号文档 → LLM 深度合成 ───────────────────
        top_docs = sorted(scored, key=lambda x: abs(x[0]), reverse=True)[:5]
        top_texts = [t for _, t in top_docs]

        llm_score, llm_reason = self._get_llm_synthesis(top_texts)

        if llm_score is not None:
            final_score = self._clamp(0.4 * finbert_avg + 0.6 * llm_score)
            evidence["llm_score"] = round(llm_score, 3)
            evidence["llm_reason"] = llm_reason or ""
            confidence = min(0.92, 0.65 + min(len(texts), 10) * 0.02)
            summary = llm_reason[:50] if llm_reason else (
                "FinBERT+LLM情绪正面" if final_score > 0.2 else
                "FinBERT+LLM情绪负面" if final_score < -0.2 else
                "FinBERT+LLM情绪中性"
            )
        else:
            final_score = finbert_avg
            confidence = min(0.82, 0.55 + min(len(texts), 10) * 0.02)
            summary = (
                "FinBERT情绪偏正面" if final_score > 0.2 else
                "FinBERT情绪偏负面" if final_score < -0.2 else
                "FinBERT情绪中性"
            )

        if evidence["neg_count"] >= 3:
            warnings.append(f"发现{evidence['neg_count']}篇负面文档")

        return AgentSignal(
            agent_name="SentimentAgent",
            ticker=self.ticker,
            signal=final_score,
            confidence=confidence,
            summary=summary[:50],
            evidence=evidence,
            warnings=warnings,
        )

    def _collect_texts(self, max_news: int = 15, max_reports: int = 6) -> list[str]:
        """从 context 提取文本列表，去空去重。"""
        texts: list[str] = []
        for key, limit in [("news_context", max_news), ("report_context", max_reports)]:
            for item in self.context.get(key, [])[:limit]:
                t = (item.get("text", "") or item.get("title", "")) if isinstance(item, dict) else str(item)
                t = str(t).strip()[:400]
                if t:
                    texts.append(t)
        for extra_key in ("news_texts", "report_texts"):
            for x in self.context.get(extra_key, []):
                t = str(x).strip()[:400]
                if t:
                    texts.append(t)
        # 去重（保序）
        seen: set[str] = set()
        deduped = []
        for t in texts:
            if t[:80] not in seen:
                seen.add(t[:80])
                deduped.append(t)
        return deduped

    def _get_llm_synthesis(self, top_texts: list[str]) -> tuple[Optional[float], Optional[str]]:
        """用 LLM 对 Top-K 文档做深度情感合成，返回 (score, reason)。"""
        if self.provider == "none" or not top_texts:
            return None, None
        try:
            from quantmind.agents.llm_client import build_client

            client = build_client(provider=self.provider, model=self.model)
            combined = "\n---\n".join(top_texts[:5])
            prompt = (
                f"你是一名专业的A股量化分析师。请分析以下关于股票 {self.ticker} 的新闻/公告，"
                f"判断对该股票的情感倾向。\n\n"
                f"文档内容：\n{combined[:1500]}\n\n"
                f"请返回JSON格式：\n"
                f'{{\"score\": <-1到1的小数>, \"reason\": \"<50字内的核心原因>\"}}\n'
                f"score含义：-1=极度利空，0=中性，1=极度利好。只返回JSON，不要其他内容。"
            )
            resp = client.complete(prompt)
            if resp:
                # 提取 JSON
                json_match = re.search(r'\{[^{}]*"score"\s*:\s*(-?\d+\.?\d*)[^{}]*\}', resp, re.DOTALL)
                if json_match:
                    import json
                    try:
                        data = json.loads(json_match.group())
                        score = self._clamp(float(data.get("score", 0)))
                        reason = str(data.get("reason", ""))[:100]
                        return score, reason
                    except Exception:
                        pass
                # fallback：只提取数字
                num_match = re.search(r"-?\d+\.?\d*", resp.strip())
                if num_match:
                    return self._clamp(float(num_match.group())), None
        except Exception as e:
            logger.debug(f"[SentimentAgent] LLM synthesis 失败: {e}")
        return None, None

    def _analyze_rules(self) -> AgentSignal:
        evidence: dict = {}
        warnings: list[str] = []

        news_items = self.context.get("news_context", [])
        report_items = self.context.get("report_context", [])

        news_texts = []
        for item in news_items[:5]:
            if isinstance(item, dict):
                text = item.get("text", "") or item.get("title", "")
            else:
                text = str(item)
            news_texts.append(text[:200])

        report_text = " ".join(
            (item.get("text", "") if isinstance(item, dict) else str(item))[:300]
            for item in report_items[:3]
        )

        evidence["news_count"] = len(news_texts)
        evidence["report_count"] = len(report_items)
        evidence["model_version"] = "rules_v1"

        rule_signal = 0.0
        all_news_text = " ".join(news_texts)

        pos_hits = [w for w in _POSITIVE_WORDS if w in all_news_text]
        neg_hits = [w for w in _NEGATIVE_WORDS if w in all_news_text]

        rule_signal += len(pos_hits) * 0.1
        rule_signal -= len(neg_hits) * 0.1

        if pos_hits:
            evidence["positive_keywords"] = pos_hits[:5]
        if neg_hits:
            evidence["negative_keywords"] = neg_hits[:5]
            if len(neg_hits) >= 3:
                warnings.append(f"发现多个负面关键词: {neg_hits[:3]}")

        analyst_signal = 0.0
        for word in _BULLISH_WORDS:
            if word in report_text:
                analyst_signal += 0.2
        for word in _BEARISH_WORDS:
            if word in report_text:
                analyst_signal -= 0.2
        analyst_signal = self._clamp(analyst_signal, -0.6, 0.6)
        evidence["analyst_signal"] = round(analyst_signal, 3)

        total_signal = self._clamp(rule_signal * 0.6 + analyst_signal * 0.4)

        llm_signal = self._get_llm_signal(news_texts, report_text)
        if llm_signal is not None:
            total_signal = self._clamp(total_signal * 0.5 + llm_signal * 0.5)
            evidence["llm_signal"] = round(llm_signal, 3)

        confidence = 0.4
        if len(news_texts) >= 3:
            confidence += 0.2
        if len(report_items) >= 1:
            confidence += 0.15
        if llm_signal is not None:
            confidence += 0.15
        confidence = min(0.9, confidence)

        if total_signal > 0.3:
            summary = f"市场情绪偏正面（{len(pos_hits)}个积极信号）"
        elif total_signal < -0.3:
            summary = f"市场情绪偏负面（{len(neg_hits)}个消极信号）"
        else:
            summary = "市场情绪中性"

        return AgentSignal(
            agent_name="SentimentAgent",
            ticker=self.ticker,
            signal=total_signal,
            confidence=confidence,
            summary=summary[:50],
            evidence=evidence,
            warnings=warnings,
        )

    def _analyze_tfidf(self) -> AgentSignal:
        evidence: dict = {"model_version": "tfidf_v2"}
        warnings: list[str] = []

        news_items = self.context.get("news_context", [])
        news_texts = []
        for item in news_items[:10]:
            if isinstance(item, dict):
                text = item.get("text", "") or item.get("title", "")
            else:
                text = str(item)
            if text.strip():
                news_texts.append(text[:300])

        if not news_texts:
            evidence["note"] = "无新闻文本，降级到规则"
            return self._analyze_rules()

        try:
            vectorizer = self._ml_model.get("vectorizer")
            classifier = self._ml_model.get("classifier")
            if vectorizer is None or classifier is None:
                return self._analyze_rules()

            X = vectorizer.transform(news_texts)
            probs = classifier.predict_proba(X)
            avg_probs = probs.mean(axis=0)

            sentiment_score = float(avg_probs[2] - avg_probs[0])
            signal = self._clamp(sentiment_score * 1.5)

            evidence.update({
                "prob_negative": round(float(avg_probs[0]), 3),
                "prob_neutral": round(float(avg_probs[1]), 3),
                "prob_positive": round(float(avg_probs[2]), 3),
                "news_count": len(news_texts),
            })

            if signal > 0.3:
                summary = f"TF-IDF情感正面（P={avg_probs[2]:.1%}）"
            elif signal < -0.3:
                summary = f"TF-IDF情感负面（P={avg_probs[0]:.1%}）"
                if avg_probs[0] > 0.5:
                    warnings.append("多条新闻显示负面情绪")
            else:
                summary = f"TF-IDF情感中性（P={avg_probs[1]:.1%}）"

            confidence = min(0.85, 0.55 + len(news_texts) * 0.03)
            return AgentSignal(
                agent_name="SentimentAgent",
                ticker=self.ticker,
                signal=signal,
                confidence=confidence,
                summary=summary[:50],
                evidence=evidence,
                warnings=warnings,
            )
        except Exception as e:
            logger.warning(f"[SentimentAgent/tfidf_v2] 预测失败: {e}，降级")
            return self._analyze_rules()

    def _get_llm_signal(self, news_texts: list[str], report_text: str) -> float | None:
        """rules_v1 路径的 LLM 辅助信号（保持向后兼容）。"""
        score, _ = self._get_llm_synthesis(news_texts[:3])
        return score
