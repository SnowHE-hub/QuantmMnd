"""quantmind.line_a.evidence.ann_classifier — 学习得到的公告标题分类器（X4 v2，T-049 采用 MacBERT 多标签）.

用户 2026-10-23："不想用标题规则……不要搞这种卡死的硬规则"。T-049 在金标准 v2 上的判定：
MacBERT 多标签 F1 0.967（规则 0.877），严重类型召回 0.946（规则 0.851），采用。旁路事件监测、F6 触发和公告工具都从这里取类型：
- 14 类事件（X4 的范围）由模型判断，可以多标签；排除项（回复函、子公司业绩、回购注销……）由模型按指南 v2 学到，
  不再用标题规则的"摘要 / 说明会"过滤；
- 14 类以外的类型（担保、质押、可转债、退市风险……）X4 没有建模，仍由标题规则给出，来源标 rule；
- 阶段（计划 / 进展 / 实施 / 完成 / 终止……）仍用 `ann_types` 的阶段规则，X4 没有评测阶段；
- 模型不可用时（模型目录不存在、torch 加载失败）降级到标题规则，来源标 rule，并在 `degraded` 里说明。
重要性是政策（哪些类型值得优先研究），不是分类，沿用 `ann_types.RULES` 的默认值。
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from quantmind.line_a.evidence.ann_types import RULES, classify_title

X4_TYPES = ("performance_forecast", "performance_express", "repurchase", "holder_reduction", "holder_increase",
            "unlock", "dividend", "st_change", "investigation", "penalty", "regulatory_letter", "litigation",
            "restructuring", "control_change")
IMPORTANCE = {typ: im for _, typ, _, im in RULES}
RANK = {"major": 2, "watch": 1, "routine": 0}
ADOPTION = Path(__file__).resolve().parents[3] / "configs/line_a_v2/ann_classifier.json"


@dataclass(frozen=True)
class AnnClass:
    """与 `ann_types.TitleClass` 字段兼容（event_type / stage / importance / type_rule / stage_rule / is_summary_or_meta），
    另加 types（全部类型）和 source。event_type 取 types 里重要性最高的一个，没有类型时为 other。"""
    event_type: str
    stage: str
    importance: str
    type_rule: str
    stage_rule: str
    is_summary_or_meta: bool
    types: tuple[str, ...]
    source: str                                               # model:<版本> / model:<版本>+rule / rule


class TitleClassifier:
    """model_dir：x4v2.train_macbert 的输出目录；thresholds：x4v2_score_*.json 里的每类阈值。"""

    def __init__(self, model_dir: Path | str | None, thresholds: dict[str, float] | None, *,
                 version: str = "macbert", device: str | None = None, cache_size: int = 200_000):
        self.model_dir = Path(model_dir) if model_dir else None
        self.thr, self.version, self.device = thresholds or {}, version, device or "cpu"
        self._tok = self._model = self._keys = self._max_len = None
        self.degraded: str | None = None
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[str, ...]] = {}
        self._cache_size = cache_size

    @classmethod
    def from_adoption(cls, adoption_json: Path | str = ADOPTION) -> TitleClassifier:
        a = json.loads(Path(adoption_json).read_text(encoding="utf-8"))
        return cls(a.get("model_dir"), a.get("thresholds"), version=a.get("version", "macbert"))

    def _load(self) -> bool:
        if self._model is not None:
            return True
        if self.degraded:
            return False
        if not self.model_dir or not (self.model_dir / "config.json").exists():
            self.degraded = f"model dir missing: {self.model_dir}"
            return False
        try:
            import torch
            import transformers
            cfg = json.loads((self.model_dir / "x4v2_config.json").read_text(encoding="utf-8"))
            self._keys, self._max_len = cfg["types"], cfg["cfg"]["max_len"]
            self._tok = transformers.AutoTokenizer.from_pretrained(self.model_dir)
            self._model = (transformers.AutoModelForSequenceClassification.from_pretrained(self.model_dir)
                           .to(self.device).eval())
            torch.set_grad_enabled(False)
        except Exception as e:  # noqa: BLE001  模型加载失败：降级到规则并记下原因
            self.degraded = f"{type(e).__name__}: {str(e)[:200]}"
            self._model = None
            return False
        return True

    def model_types(self, titles: list[str], batch: int = 256) -> list[tuple[str, ...]] | None:
        """每条标题的 14 类判断（多标签）；模型不可用时返回 None。"""
        with self._lock:
            if not self._load():
                return None
            todo = [t for t in dict.fromkeys(titles) if t not in self._cache]
            if len(self._cache) + len(todo) > self._cache_size:
                self._cache.clear()
                todo = list(dict.fromkeys(titles))
            if todo:
                import torch
                with torch.no_grad():
                    for i in range(0, len(todo), batch):
                        chunk = todo[i:i + batch]
                        enc = self._tok(chunk, truncation=True, max_length=self._max_len, padding=True,
                                        return_tensors="pt").to(self.device)
                        probs = torch.sigmoid(self._model(**enc).logits.float()).cpu().tolist()
                        for t, p in zip(chunk, probs, strict=True):
                            self._cache[t] = tuple(k for k, v in zip(self._keys, p, strict=True)
                                                   if v >= self.thr.get(k, 0.5))
            return [self._cache[t] for t in titles]

    def classify(self, titles: list[str]) -> list[AnnClass]:
        titles = [str(t or "").strip() for t in titles]
        rule = [classify_title(t) for t in titles]
        mt = self.model_types(titles)
        out = []
        for i, r in enumerate(rule):
            if mt is None:                                    # 降级：完全按标题规则
                ts = (r.event_type,) if r.event_type != "other" else ()
                out.append(AnnClass(r.event_type, r.stage, r.importance, r.type_rule, r.stage_rule,
                                    r.is_summary_or_meta, ts, "rule"))
                continue
            ts = list(mt[i])
            extra = r.event_type not in X4_TYPES and r.event_type != "other"
            if extra:                                         # X4 没建模的类型：沿用规则
                ts.append(r.event_type)
            if not ts:
                out.append(AnnClass("other", r.stage, "routine", f"M:{self.version}", r.stage_rule, False, (),
                                    f"model:{self.version}"))
                continue
            primary = max(ts, key=lambda t: RANK[IMPORTANCE.get(t, "routine")])
            imp = IMPORTANCE.get(primary, "routine")
            if r.stage in ("terminated", "withdrawn") and imp == "routine":
                imp = "watch"                                 # 终止 / 撤回本身值得看（与规则版一致）
            from_rule = primary not in X4_TYPES
            out.append(AnnClass(primary, r.stage, imp, r.type_rule if from_rule else f"M:{self.version}", r.stage_rule,
                                r.is_summary_or_meta if from_rule else False, tuple(ts),
                                f"model:{self.version}" + ("+rule" if extra else "")))
        return out

    def classify_one(self, title: str) -> AnnClass:
        return self.classify([title])[0]


_DEFAULT: TitleClassifier | None = None
_DEFAULT_LOCK = threading.Lock()


def default_classifier() -> TitleClassifier:
    """进程内单例。环境变量 QM_ANN_CLASSIFIER=rule 时强制用标题规则（对照或排障用）。"""
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            if os.environ.get("QM_ANN_CLASSIFIER") == "rule" or not ADOPTION.exists():
                _DEFAULT = TitleClassifier(None, None)
                _DEFAULT.degraded = "rule forced" if ADOPTION.exists() else f"no adoption file {ADOPTION}"
            else:
                _DEFAULT = TitleClassifier.from_adoption(ADOPTION)
        return _DEFAULT


def classify_titles(titles: list[str]) -> list[AnnClass]:
    return default_classifier().classify(list(titles))


def classifier_version() -> str:
    c = default_classifier()
    return "rule" if c.degraded or not c._load() else f"model:{c.version}"
