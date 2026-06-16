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
    doc_product_map: dict[str, str] = field(default_factory=dict)
    liability_signals: list[str] = field(default_factory=list)
    number_conditions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def stem_number_conditions(self) -> list[dict[str, Any]]:
        """Number conditions extracted from the question stem only.

        Useful for calculation engines that need stem inputs but NOT
        candidate-answer numbers from options.
        """
        return [c for c in self.number_conditions if c.get("source") == "stem"]


# ---------------------------------------------------------------------------
# Retrieval candidates
# ---------------------------------------------------------------------------

@dataclass
class CandidateNode:
    """A document node identified as relevant during retrieval."""

    doc_id: str
    node_id: str
    title: str
    page_range: str = ""  # "start-end" string format, e.g. "6-8" or single page "6"
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
    pages: str = ""  # "start-end" string format, e.g. "6-8" or single page "6"
    option: str = ""  # which option this evidence relates to
    evidence_type: str = "unclear"  # "support" | "refute" | "unclear"
    quote: str = ""
    normalized_fact: str = ""
    numbers: list[dict[str, Any]] = field(default_factory=list)
    confidence: str = "medium"  # "high" | "medium" | "low"


# ---------------------------------------------------------------------------
# Calculation
# ---------------------------------------------------------------------------

@dataclass
class CalculationRecord:
    """A single deterministic calculation result.

    Used by the CalculationEngine to record structured arithmetic on
    question-stem numbers and evidence-supplied formulas/parameters.
    """

    qid: str
    calc_type: str  # e.g. "death_benefit_comparison", "medical_payout", "ranking"
    inputs: dict[str, Any]  # the numeric inputs used
    formula: str = ""  # human-readable formula string
    result: float = 0.0
    unit: str = ""  # "元" / "%" / ""
    source_evidence_ids: list[str] = field(default_factory=list)


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
    option_judgements: dict[str, dict[str, Any]] = field(default_factory=dict)  # option -> {support_count, refute_count, ...}


# ---------------------------------------------------------------------------
# Usage / token tracking
# ---------------------------------------------------------------------------

@dataclass
class UsageRecord:
    """Token usage for a single LLM call."""

    qid: str
    stage: str  # e.g. "preprocess", "pageindex", "retrieval", "evidence", "judge"
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""
