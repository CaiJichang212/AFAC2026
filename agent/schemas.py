"""Data contracts (schemas) for the insurance QA pipeline.

These dataclasses define the shapes of data flowing through every stage:
parse -> candidate retrieval -> evidence extraction -> answer assembly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Question
# ---------------------------------------------------------------------------

@dataclass
class ParsedQuestion:
    """A single parsed question from the questions JSON file."""

    qid: str
    domain: str
    split: str
    question: str
    options: dict[str, str]  # e.g. {"A": "...", "B": "...", "C": "...", "D": "..."}
    answer_format: str  # "mcq" | "multi" | "tf"
    type: str  # e.g. "推理判断", "计算题"
    doc_ids: list[str] = field(default_factory=list)

    # Optional parsed signals (fleshed out in later tasks)
    mentioned_products: list[str] = field(default_factory=list)
    doc_product_map: dict[str, list[str]] = field(default_factory=dict)
    number_conditions: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Retrieval candidates
# ---------------------------------------------------------------------------

@dataclass
class CandidateNode:
    """A document node identified as relevant during retrieval."""

    doc_id: str
    node_id: str
    title: str
    page_range: tuple[int, int] | None = None  # (start, end) 1-indexed
    matched_signals: list[str] = field(default_factory=list)
    reason: str = ""
    needs_page_fetch: bool = True


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

@dataclass
class EvidenceRecord:
    """A single piece of evidence extracted from a document node."""

    qid: str
    doc_id: str
    node_id: str
    pages: list[int] = field(default_factory=list)  # page numbers referenced
    option: str = ""  # which option this evidence relates to
    evidence_type: str = "unclear"  # "support" | "refute" | "unclear"
    quote: str = ""
    normalized_fact: str = ""
    numbers: list[dict[str, Any]] = field(default_factory=list)
    confidence: str = "medium"  # "high" | "medium" | "low"


# ---------------------------------------------------------------------------
# Answer
# ---------------------------------------------------------------------------

@dataclass
class AnswerRecord:
    """The final answer for a single question, with full provenance."""

    qid: str
    answer: str = ""
    candidate_docs: list[str] = field(default_factory=list)  # doc_ids considered
    selected_nodes: list[str] = field(default_factory=list)  # node_ids used
    evidence: list[EvidenceRecord] = field(default_factory=list)
    calculations: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)  # UsageRecord fields
    fallbacks: list[str] = field(default_factory=list)  # fallback paths taken
    warnings: list[str] = field(default_factory=list)
    option_judgements: dict[str, str] = field(default_factory=dict)  # option -> "correct"/"incorrect"/"unclear"


# ---------------------------------------------------------------------------
# Usage / token tracking
# ---------------------------------------------------------------------------

@dataclass
class UsageRecord:
    """Token usage for a single LLM call."""

    qid: str = ""
    stage: str = ""  # e.g. "preprocess", "pageindex", "retrieval", "evidence", "judge"
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""
