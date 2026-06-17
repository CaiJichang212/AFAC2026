from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


EvidenceType = Literal["support", "refute", "unclear"]
ConfidenceLevel = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class UsageRecord:
    qid: str
    stage: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: int
    success: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UsageSummary:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ParsedQuestion:
    qid: str
    domain: str
    split: str
    question: str
    options: dict[str, str]
    answer_format: str
    type: str
    doc_ids: list[str]
    mentioned_products: list[str] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateNode:
    doc_id: str
    node_id: str
    title: str
    page_range: str
    matched_signals: list[str]
    reason: str
    needs_page_fetch: bool = True


@dataclass(frozen=True)
class EvidenceRecord:
    qid: str
    doc_id: str
    node_id: str
    pages: str
    option: str
    evidence_type: EvidenceType
    quote: str
    normalized_fact: str
    numbers: list[dict[str, Any]] = field(default_factory=list)
    confidence: ConfidenceLevel = "medium"


@dataclass(frozen=True)
class CalculationRecord:
    name: str
    inputs: dict[str, Any]
    formula: str
    result: float | int | str
    unit: str
    source_evidence_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AnswerRecord:
    qid: str
    answer: str
    option_judgements: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
