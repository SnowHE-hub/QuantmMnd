"""quantmind.line_a.contracts — 数据契约（03_final_plan_v2 第 4.1、6 节）.

- RecordTimes：每条数据的双时间字段与可用性判断（前瞻运行 / 历史重放分开）；
- Claim：Agent 输出的最小单位，每条声明有自己的支持状态和校验状态；
- SnapshotManifest：一次运行允许使用的数据版本（不可变；哈希、行数、水位、缺失分类）。
"""
from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from quantmind.line_a.atomic import canonical_json, sha256_bytes, sha256_file


class AvailabilityBasis(StrEnum):
    OBSERVED = "observed"          # 系统实测的首次取得时间
    SOURCE = "source"              # 来源提供的发布时间（精确到分钟）
    CONSERVATIVE = "conservative"  # 保守推定（只有日期等）


class RecordTimes(BaseModel):
    event_time: datetime | None = None
    published_at: datetime | None = None
    first_seen_at: datetime | None = None
    effective_from: datetime | None = None
    effective_to: datetime | None = None
    revision_id: str | None = None
    source_hash: str | None = None
    availability_basis: AvailabilityBasis

    def usable_forward(self, data_cutoff_ts: datetime, decision_ts: datetime) -> bool:
        """前瞻运行：公开时间不晚于数据截止，且系统在决策前已经实际取得。"""
        return (self.published_at is not None and self.published_at <= data_cutoff_ts
                and self.first_seen_at is not None and self.first_seen_at <= decision_ts)

    def usable_history(self, data_cutoff_ts: datetime, conservative_delay: timedelta = timedelta(0)) -> bool:
        """历史重放：按历史公开时间加预先登记的保守延迟判断（今天回补的历史数据不因 first_seen_at 失效）。"""
        return self.published_at is not None and self.published_at + conservative_delay <= data_cutoff_ts


class ClaimType(StrEnum):
    REPORTED_FACT = "reported_fact"
    COMPUTED = "computed"
    INFERENCE = "inference"
    COUNTER_EVIDENCE = "counter_evidence"
    OPEN_QUESTION = "open_question"


class SupportStatus(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNRESOLVED = "unresolved"
    SEARCHED_NO_SUPPORTING_EVIDENCE = "searched_no_supporting_evidence"


class EvidenceStatus(StrEnum):
    """一项证据获取任务的结果；不能把失败或未查写成"没有风险"（原则 P6）。"""
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    NOT_REQUESTED = "not_requested"
    SEARCHED_NO_SUPPORTING_EVIDENCE = "searched_no_supporting_evidence"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNRESOLVED = "unresolved"


class VerificationStatus(StrEnum):
    VERIFIED = "verified"
    FAILED = "failed"
    NOT_CHECKED = "not_checked"


class Producer(BaseModel):
    role: str
    model: str | None = None              # 例如 "qwen3.5-9b-q4km@sha256:..."
    prompt_schema_hash: str | None = None


class Claim(BaseModel):
    claim_id: str
    claim_type: ClaimType
    subject_id: str                       # 例如 "issuer:600000.SH"
    predicate: str
    value: str | float | int | bool | None = None
    unit: str | None = None
    currency: str | None = None
    period: str | None = None
    formula: str | None = None
    input_claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    effective_at: datetime | None = None
    support_status: SupportStatus
    verification_status: VerificationStatus = VerificationStatus.NOT_CHECKED
    supersedes_claim_id: str | None = None
    limitations: list[str] = Field(default_factory=list)
    producer: Producer

    @model_validator(mode="after")
    def _binding_rules(self) -> Claim:
        if self.claim_type == ClaimType.REPORTED_FACT and not self.evidence_ids:
            raise ValueError("reported_fact needs evidence_ids")
        if self.claim_type == ClaimType.COMPUTED and not (self.formula and self.input_claim_ids):
            raise ValueError("computed claim needs formula and input_claim_ids")
        if self.support_status == SupportStatus.SUPPORTED and not (self.evidence_ids or self.input_claim_ids):
            raise ValueError("supported claim needs evidence or inputs")
        return self


class MissingClass(StrEnum):
    SUSPENDED = "suspended"
    NOT_PUBLISHED = "not_published"
    NO_PERMISSION = "no_permission"
    FETCH_FAILED = "fetch_failed"


class SnapshotInput(BaseModel):
    name: str
    path: str
    sha256: str
    bytes: int
    rows: int | None = None
    watermark: str | None = None          # 该数据源的最新日期或时间
    missing: dict[MissingClass, int] = Field(default_factory=dict)


class SnapshotManifest(BaseModel):
    as_of: str
    data_cutoff_ts: datetime
    created_utc: datetime
    inputs: list[SnapshotInput]
    notes: list[str] = Field(default_factory=list)

    def content_hash(self) -> str:
        """快照的身份：只由输入的名称、路径和哈希决定（与创建时间无关），用作任务键的一部分。"""
        return sha256_bytes(canonical_json(sorted((i.name, i.path, i.sha256) for i in self.inputs)))


def snapshot_input(name: str, path: str | Path, *, rows: int | None = None, watermark: str | None = None,
                   missing: dict[MissingClass, int] | None = None) -> SnapshotInput:
    p = Path(path)
    return SnapshotInput(name=name, path=str(p), sha256=sha256_file(p), bytes=p.stat().st_size, rows=rows,
                         watermark=watermark, missing=missing or {})
