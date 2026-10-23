"""quantmind.line_a.evidence.ann_classifier — 学习得到的公告标题分类器（X4 v2，T-049）.

用户 2026-10-23："不想用标题规则……不要搞这种卡死的硬规则"。T-049 在金标准 v2 上判定采用哪个模型之后，
旁路事件监测和公告工具通过这里取类型：
- 14 类事件（X4 的范围）由模型判断：MacBERT 多标签（本机，阈值来自校准集）或 DeepSeek 直接分类；
- 模型不可用时（模型目录不存在、显卡和网络都不可用）降级到 `ann_types.classify_title`，并在结果里标明来源；
- 14 类以外的类型（担保、质押、可转债等）X4 没有建模，仍由标题规则给出，标明来源为 rule。
阶段（计划 / 进展 / 实施 / 完成 / 终止……）仍用 `ann_types` 的阶段规则，X4 没有评测阶段。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from quantmind.line_a.evidence.ann_types import RULES, classify_title

X4_TYPES = ("performance_forecast", "performance_express", "repurchase", "holder_reduction", "holder_increase",
            "unlock", "dividend", "st_change", "investigation", "penalty", "regulatory_letter", "litigation",
            "restructuring", "control_change")
IMPORTANCE = {typ: im for _, typ, _, im in RULES}          # 每类的默认重要性与标题规则一致（只是政策，不是分类）
IMPORTANCE.setdefault("performance_express", "watch")


@dataclass
class Label:
    types: tuple[str, ...]
    importance: str
    source: str                                              # model:<名字> / rule


class TitleClassifier:
    """model_dir：x4v2.train_macbert 的输出目录；thresholds：x4v2_score_*.json 里的 thresholds。"""

    def __init__(self, model_dir: Path | None, thresholds: dict[str, float] | None, *, device: str | None = None):
        self.model_dir, self.thr = (Path(model_dir) if model_dir else None), thresholds or {}
        self._tok = self._model = None
        self.device = device

    @classmethod
    def from_adoption(cls, adoption_json: Path) -> TitleClassifier:
        a = json.loads(Path(adoption_json).read_text(encoding="utf-8"))
        return cls(a.get("model_dir"), a.get("thresholds"))

    def _load(self) -> bool:
        if self._model is not None:
            return True
        if not self.model_dir or not (self.model_dir / "config.json").exists():
            return False
        import torch
        import transformers
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = transformers.AutoTokenizer.from_pretrained(self.model_dir)
        self._model = transformers.AutoModelForSequenceClassification.from_pretrained(self.model_dir).to(self.device).eval()
        return True

    def classify(self, titles: list[str], batch: int = 128) -> list[Label]:
        rule = [classify_title(t) for t in titles]
        if not self._load():
            return [Label((r.event_type,) if r.event_type != "other" else (), r.importance, "rule") for r in rule]
        import torch
        cfg = json.loads((self.model_dir / "x4v2_config.json").read_text(encoding="utf-8"))
        keys = cfg["types"]
        probs = []
        with torch.no_grad():
            for i in range(0, len(titles), batch):
                enc = self._tok(titles[i:i + batch], truncation=True, max_length=cfg["cfg"]["max_len"], padding=True,
                                return_tensors="pt").to(self.device)
                probs += torch.sigmoid(self._model(**enc).logits.float()).cpu().tolist()
        out = []
        rank = {"major": 2, "watch": 1, "routine": 0}
        for r, p in zip(rule, probs, strict=True):
            ts = [k for k, v in zip(keys, p, strict=True) if v >= self.thr.get(k, 0.5)]
            if r.event_type not in X4_TYPES and r.event_type != "other":
                ts.append(r.event_type)                          # X4 没建模的类型：沿用规则，来源记 rule
            imp = max((IMPORTANCE.get(t, "routine") for t in ts), key=lambda x: rank[x], default="routine")
            if r.stage in ("terminated", "withdrawn") and imp == "routine" and ts:
                imp = "watch"
            out.append(Label(tuple(ts), imp, "model:macbert" + ("+rule" if r.event_type not in (*X4_TYPES, "other") else "")))
        return out
