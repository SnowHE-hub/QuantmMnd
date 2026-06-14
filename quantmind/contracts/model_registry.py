"""ModelRegistry —— 模型注册表（model_id → 制式/标签/口径/版本/gate/产物）.

JSON 落盘（data/registry/model_registry.json），支持版本化 + 可复现加载。
gate_status 取值：
  research_candidate              —— 研究候选（默认）
  research_candidate_pending_nav  —— 研究层赢家，但净超额 < formal gate，待 executable NAV 回测
  production                      —— 已签收进生产（须 NAV 过 gate）
  retired                         —— 退役

与 run_manifest 联动：每次训练写 manifest 后自动 register_from_manifest 一条候选。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
REGISTRY_PATH = _REPO / "data" / "registry" / "model_registry.json"

VALID_GATE = {"research_candidate", "research_candidate_pending_nav", "production", "retired"}


@dataclass
class ModelRecord:
    model_id: str
    label: str                       # forward_return_12d ...
    horizon: str                     # HorizonRegistry name: short|robust|long
    feature_set: str                 # tab35 | tab35_plus16 | full_35_16_158 | alpha360_seq60
    data_version: str                # v5 | v6
    wf_config: dict = field(default_factory=dict)
    neutralization: dict = field(default_factory=dict)
    gate_status: str = "research_candidate"
    metrics: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)  # preds 路径 / manifest sha256 / verdict doc
    git_sha: str | None = None
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    updated: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def __post_init__(self):
        if self.gate_status not in VALID_GATE:
            raise ValueError(f"gate_status {self.gate_status!r} not in {VALID_GATE}")


def _load() -> dict[str, dict]:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {}


def _save(d: dict[str, dict]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def register(record: ModelRecord, *, overwrite: bool = True) -> ModelRecord:
    """注册/更新一条模型记录。overwrite=False 时已存在则保留旧记录。"""
    d = _load()
    if record.model_id in d and not overwrite:
        return ModelRecord(**d[record.model_id])
    if record.model_id in d:
        record.created = d[record.model_id].get("created", record.created)
    record.updated = datetime.now().isoformat(timespec="seconds")
    d[record.model_id] = asdict(record)
    _save(d)
    return record


def get(model_id: str) -> ModelRecord:
    d = _load()
    if model_id not in d:
        raise KeyError(f"model_id {model_id!r} not registered; have={list(d)}")
    return ModelRecord(**d[model_id])


def list_models(*, gate_status: str | None = None, horizon: str | None = None) -> list[ModelRecord]:
    recs = [ModelRecord(**v) for v in _load().values()]
    if gate_status:
        recs = [r for r in recs if r.gate_status == gate_status]
    if horizon:
        recs = [r for r in recs if r.horizon == horizon]
    return recs


def set_gate_status(model_id: str, gate_status: str) -> ModelRecord:
    if gate_status not in VALID_GATE:
        raise ValueError(f"invalid gate_status {gate_status!r}")
    r = get(model_id)
    r.gate_status = gate_status
    return register(r)


def register_from_manifest(manifest: dict | str | Path) -> ModelRecord | None:
    """run_manifest 联动：从 run_manifest JSON（或 dict）注册一条候选。

    仅当 manifest 含 extra.model_id（或 extra.model）时注册；否则返回 None（非模型训练）。
    默认 gate_status=research_candidate（不抬到 pending_nav/production——那是人工决策）。
    """
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    extra = manifest.get("extra", {}) or {}
    model_id = extra.get("model_id") or (
        f"{extra['model']}_{manifest.get('feature_set')}_{manifest.get('horizon')}_{manifest.get('data_version')}"
        if extra.get("model") else None)
    if not model_id:
        return None
    horizon_name = "short" if str(manifest.get("horizon", "")).startswith("12") else (
        "long" if str(manifest.get("horizon", "")).startswith("63") else
        "robust" if str(manifest.get("horizon", "")).startswith("21") else "short")
    rec = ModelRecord(
        model_id=model_id, label=manifest.get("label") or "", horizon=horizon_name,
        feature_set=manifest.get("feature_set") or "", data_version=manifest.get("data_version") or "",
        wf_config={k: extra[k] for k in ("refit", "alpha", "n_feats", "n_folds", "n_flips") if k in extra},
        gate_status="research_candidate", git_sha=manifest.get("git_sha"),
        artifacts={"manifest_sha256": manifest.get("artifacts_sha256", {}),
                   "preds": extra.get("preds")},
        metrics={"train_seconds": extra.get("train_seconds")})
    return register(rec, overwrite=True)


__all__ = ["ModelRecord", "register", "get", "list_models", "set_gate_status",
           "register_from_manifest", "REGISTRY_PATH", "VALID_GATE"]
