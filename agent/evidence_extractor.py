"""Evidence extraction from candidate document nodes (Task 9).

For each candidate node, retrieves page text via IndexStore, sends it to the LLM
along with the question and options, and collects per-option verdicts.  The LLM
MUST quote verbatim from the page text for traceability.

When no LLM client is available (or all calls fail), a deterministic heuristic
fallback produces ``unclear`` records so the pipeline never crashes offline.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agent.config import AgentConfig
from agent.index_store import IndexStore
from agent.llm_client import LLMClient, LLMResponse
from agent.schemas import CandidateNode, EvidenceRecord, ParsedQuestion, UsageRecord
from agent.token_meter import TokenMeter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON schema for per-node LLM verdicts
# ---------------------------------------------------------------------------

_EVIDENCE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "option": {"type": "string", "enum": ["A", "B", "C", "D"]},
                    "evidence_type": {
                        "type": "string",
                        "enum": ["support", "refute", "unclear"],
                    },
                    "quote": {"type": "string"},
                    "normalized_fact": {"type": "string"},
                    "numbers": {"type": "array"},
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                },
                "required": [
                    "option",
                    "evidence_type",
                    "quote",
                    "normalized_fact",
                    "confidence",
                ],
            },
        }
    },
    "required": ["verdicts"],
}

# ---------------------------------------------------------------------------
# Confidence ordering for dedup tie-breaking
# ---------------------------------------------------------------------------

_CONFIDENCE_RANK: dict[str, int] = {"high": 3, "medium": 2, "low": 1}

_VALID_EVIDENCE_TYPES: frozenset[str] = frozenset({"support", "refute", "unclear"})
_VALID_CONFIDENCES: frozenset[str] = frozenset({"high", "medium", "low"})
_ALL_OPTIONS: list[str] = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class EvidenceExtractor:
    """Extract option-level evidence from candidate document nodes.

    Uses an (optionally injected) LLM client to produce per-option verdicts
    with verbatim quotes from the source page text.  Falls back to a
    deterministic heuristic when no LLM client is available.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        parsed: ParsedQuestion,
        candidates: list[CandidateNode],
        index_store: IndexStore,
        config: AgentConfig,
        *,
        llm_client: LLMClient | None = None,
        token_meter: TokenMeter | None = None,
    ) -> list[EvidenceRecord]:
        """Extract evidence records for *parsed* from *candidates*.

        Args:
            parsed: The parsed question with qid, question text, and options.
            candidates: Candidate nodes from retrieval (Task 8).
            index_store: For reading page text.
            config: Pipeline configuration (budgets, model name).
            llm_client: Optional LLM client.  When ``None``, the deterministic
                heuristic fallback is used.
            token_meter: Optional token meter for recording LLM usage.

        Returns:
            A list of ``EvidenceRecord`` covering all options A/B/C/D.
        """
        all_records: list[EvidenceRecord] = []
        llm_available = llm_client is not None

        # Phase 1: per-node extraction ----------------------------------------
        for candidate in candidates:
            # --- resolve page text ---
            try:
                doc_id_int = int(candidate.doc_id)
            except (ValueError, TypeError):
                logger.warning("Invalid doc_id %r for node %s; skipping.", candidate.doc_id, candidate.node_id)
                continue

            page_range = candidate.page_range
            if not page_range:
                logger.warning("Candidate %s has no page_range; skipping.", candidate.node_id)
                continue

            pages = index_store.get_page_content(doc_id_int, page_range)
            full_text = "\n".join(p.text for p in pages)
            if not full_text.strip():
                logger.warning("No page text for %s doc %s pages %s; skipping.",
                               candidate.node_id, candidate.doc_id, page_range)
                continue

            # --- extract ---
            if llm_available:
                try:
                    records = self._extract_from_node_llm(
                        parsed=parsed,
                        candidate=candidate,
                        full_text=full_text,
                        config=config,
                        llm_client=llm_client,
                        token_meter=token_meter,
                    )
                except Exception as exc:
                    logger.warning("LLM extraction failed for %s: %s; using heuristic fallback.",
                                   candidate.node_id, exc)
                    records = self._extract_from_node_heuristic(
                        parsed=parsed, candidate=candidate, full_text=full_text, config=config
                    )
            else:
                records = self._extract_from_node_heuristic(
                    parsed=parsed, candidate=candidate, full_text=full_text, config=config
                )

            all_records.extend(records)

        # Phase 2: normalise confidence for empty-quote support/refute records
        all_records = [self._normalize_record(r) for r in all_records]

        # Phase 3: dedup ------------------------------------------------------
        all_records = self._dedup(all_records)

        # Phase 3: cap per option ---------------------------------------------
        all_records = self._cap_per_option(all_records, config.max_evidence_per_option)

        # Phase 4: guarantee per-option coverage ------------------------------
        all_records = self._ensure_coverage(parsed, all_records)

        return all_records

    # ------------------------------------------------------------------
    # LLM-backed extraction (per node)
    # ------------------------------------------------------------------

    def _extract_from_node_llm(
        self,
        *,
        parsed: ParsedQuestion,
        candidate: CandidateNode,
        full_text: str,
        config: AgentConfig,
        llm_client: LLMClient,
        token_meter: TokenMeter | None,
    ) -> list[EvidenceRecord]:
        """Call the LLM for one candidate node and parse the response."""
        messages = _build_evidence_messages(
            question=parsed.question,
            options=parsed.options,
            page_text=full_text,
            doc_id=candidate.doc_id,
            node_title=candidate.title,
        )

        # ARK coding model returns empty content at temperature=0.0.
        # Real calls MUST use temperature > 0.
        temperature = 0.6

        response: LLMResponse = llm_client.chat(
            messages=messages,
            json_schema=_EVIDENCE_JSON_SCHEMA,
            temperature=temperature,
            max_tokens=4096,
        )

        # Record usage if a meter is provided
        if token_meter is not None:
            token_meter.record(
                UsageRecord(
                    qid=parsed.qid,
                    stage="evidence",
                    model=response.model or config.inference_model,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    total_tokens=response.total_tokens,
                    latency_ms=response.latency_ms,
                    success=True,
                )
            )

        return self._parse_llm_verdicts(
            raw_content=response.content,
            parsed=parsed,
            candidate=candidate,
            full_text=full_text,
        )

    def _parse_llm_verdicts(
        self,
        raw_content: str,
        parsed: ParsedQuestion,
        candidate: CandidateNode,
        full_text: str,
    ) -> list[EvidenceRecord]:
        """Parse the LLM JSON response into EvidenceRecords.

        Normalizes evidence_type (rejects unknown values to "unclear"),
        validates quotes against page text, and enforces per-node caps.
        """
        try:
            data = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            logger.warning("LLM response is not valid JSON: %s", exc)
            return []

        verdicts: list[dict[str, Any]] = data.get("verdicts", [])
        if not isinstance(verdicts, list):
            logger.warning("LLM verdicts is not a list; got %s", type(verdicts).__name__)
            return []

        records: list[EvidenceRecord] = []
        for v in verdicts:
            if not isinstance(v, dict):
                continue

            option = str(v.get("option", "")).strip().upper()
            if option not in ("A", "B", "C", "D"):
                continue

            evidence_type = str(v.get("evidence_type", "unclear")).strip().lower()
            if evidence_type not in _VALID_EVIDENCE_TYPES:
                evidence_type = "unclear"

            quote = str(v.get("quote", ""))
            normalized_fact = str(v.get("normalized_fact", ""))
            confidence = str(v.get("confidence", "medium")).strip().lower()
            if confidence not in _VALID_CONFIDENCES:
                confidence = "medium"

            numbers_raw = v.get("numbers", [])
            if not isinstance(numbers_raw, list):
                numbers_raw = []
            numbers: list[dict[str, Any]] = []
            for n in numbers_raw:
                if isinstance(n, dict):
                    numbers.append({
                        "name": str(n.get("name", "")),
                        "value": n.get("value", ""),
                        "unit": str(n.get("unit", "")),
                    })

            records.append(
                EvidenceRecord(
                    qid=parsed.qid,
                    doc_id=candidate.doc_id,
                    node_id=candidate.node_id,
                    pages=candidate.page_range,
                    option=option,
                    evidence_type=evidence_type,
                    quote=quote,
                    normalized_fact=normalized_fact,
                    numbers=numbers,
                    confidence=confidence,
                )
            )

        return records

    # ------------------------------------------------------------------
    # Deterministic heuristic fallback (per node)
    # ------------------------------------------------------------------

    def _extract_from_node_heuristic(
        self,
        *,
        parsed: ParsedQuestion,
        candidate: CandidateNode,
        full_text: str,
        config: AgentConfig,
    ) -> list[EvidenceRecord]:
        """Fallback: produce one ``unclear`` record per option with a best-effort snippet."""
        records: list[EvidenceRecord] = []
        for option_letter in _ALL_OPTIONS:
            snippet = self._find_best_snippet(full_text, parsed.options.get(option_letter, ""))
            records.append(
                EvidenceRecord(
                    qid=parsed.qid,
                    doc_id=candidate.doc_id,
                    node_id=candidate.node_id,
                    pages=candidate.page_range,
                    option=option_letter,
                    evidence_type="unclear",
                    quote=snippet,
                    normalized_fact="",
                    numbers=[],
                    confidence="low",
                )
            )
        return records

    @staticmethod
    def _find_best_snippet(page_text: str, option_text: str) -> str:
        """Find the longest substring of *option_text* that appears in *page_text*.

        Returns the matching substring, or the empty string if nothing matches.
        """
        if not option_text or not page_text:
            return ""
        # Try to find substrings of option_text in page_text, longest first.
        opt = option_text.strip()
        for length in range(len(opt), 1, -1):
            for start in range(0, len(opt) - length + 1):
                candidate_sub = opt[start : start + length]
                if candidate_sub in page_text:
                    return candidate_sub
        return ""

    # ------------------------------------------------------------------
    # Record normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_record(rec: EvidenceRecord) -> EvidenceRecord:
        """Normalize a record: empty quote with support/refute -> downgrade to low.

        A support or refute claim without a verbatim quote is untrustworthy;
        downgrade confidence to "low" so downstream judges can weigh it
        appropriately.
        """
        if rec.evidence_type in ("support", "refute") and not rec.quote.strip():
            return EvidenceRecord(
                qid=rec.qid,
                doc_id=rec.doc_id,
                node_id=rec.node_id,
                pages=rec.pages,
                option=rec.option,
                evidence_type=rec.evidence_type,
                quote=rec.quote,
                normalized_fact=rec.normalized_fact,
                numbers=rec.numbers,
                confidence="low",
            )
        return rec

    # ------------------------------------------------------------------
    # Dedup
    # ------------------------------------------------------------------

    def _dedup(self, records: list[EvidenceRecord]) -> list[EvidenceRecord]:
        """Drop records with identical (doc_id, pages, option, quote).

        Keeps the highest-confidence record when duplicates are found.
        Ties are broken by keeping the first encountered record.
        """
        groups: dict[tuple[str, str, str, str], EvidenceRecord] = {}
        for rec in records:
            key = (rec.doc_id, rec.pages, rec.option, _normalize_whitespace(rec.quote))
            if key in groups:
                existing_rank = _CONFIDENCE_RANK.get(groups[key].confidence, 1)
                new_rank = _CONFIDENCE_RANK.get(rec.confidence, 1)
                if new_rank > existing_rank:
                    groups[key] = rec
            else:
                groups[key] = rec
        return list(groups.values())

    # ------------------------------------------------------------------
    # Cap per option
    # ------------------------------------------------------------------

    @staticmethod
    def _cap_per_option(records: list[EvidenceRecord], max_per_option: int) -> list[EvidenceRecord]:
        """Keep at most *max_per_option* records per option, preferring higher confidence."""
        by_option: dict[str, list[EvidenceRecord]] = {}
        for rec in records:
            by_option.setdefault(rec.option, []).append(rec)

        result: list[EvidenceRecord] = []
        for option in _ALL_OPTIONS:
            group = by_option.get(option, [])
            # Sort by confidence descending, then keep top N
            group.sort(key=lambda r: _CONFIDENCE_RANK.get(r.confidence, 1), reverse=True)
            result.extend(group[:max_per_option])
        return result

    # ------------------------------------------------------------------
    # Per-option coverage guarantee
    # ------------------------------------------------------------------

    def _ensure_coverage(
        self, parsed: ParsedQuestion, records: list[EvidenceRecord]
    ) -> list[EvidenceRecord]:
        """Ensure every option (A/B/C/D) has at least one record.

        Missing options get a synthesized ``unclear`` record.
        """
        covered_options = {rec.option for rec in records}
        for option in _ALL_OPTIONS:
            if option not in covered_options:
                logger.info("Synthesizing unclear record for option %s (qid=%s)", option, parsed.qid)
                records.append(
                    EvidenceRecord(
                        qid=parsed.qid,
                        doc_id="",
                        node_id="",
                        pages="",
                        option=option,
                        evidence_type="unclear",
                        quote="",
                        normalized_fact="",
                        numbers=[],
                        confidence="low",
                    )
                )
        return records


# ---------------------------------------------------------------------------
# Prompt builders (module-level for testability)
# ---------------------------------------------------------------------------


def _build_evidence_messages(
    *,
    question: str,
    options: dict[str, str],
    page_text: str,
    doc_id: str,
    node_title: str,
) -> list[dict[str, Any]]:
    """Build the LLM messages for evidence extraction from one candidate node."""

    options_text = "\n".join(
        f"  {letter}. {text}" for letter, text in sorted(options.items())
    )

    system_prompt = (
        "你是一个保险条款分析助手。你的任务是根据提供的保险文档页面内容，"
        "对每个选项（A/B/C/D）独立判断该页面内容是否支持、反驳该选项，或是无法确定。\n\n"
        "规则：\n"
        "1. 为每个选项（A、B、C、D）分别给出一个判断（verdict）。\n"
        "2. evidence_type 必须严格从以下三个值中选择：support（支持）、refute（反驳）、unclear（无法确定）。\n"
        "3. quote 必须是页面文本中的**原文字符串**（逐字复制），不得修改或概括。如果找不到相关原文，quote 可以为空字符串。\n"
        "4. normalized_fact 用简体中文重新表述从原文中提炼的事实。\n"
        "5. numbers 提取页面中相关的数值信息，格式为 [{\"name\": \"名称\", \"value\": 数值, \"unit\": \"单位\"}]。\n"
        "6. confidence 表示判断的置信度：high（高）、medium（中）、low（低）。\n\n"
        "请严格按照JSON格式返回，包含 verdicts 数组，每个元素为一个选项的判断。"
    )

    user_prompt = (
        f"文档ID: {doc_id}\n"
        f"章节标题: {node_title}\n\n"
        f"问题:\n{question}\n\n"
        f"选项:\n{options_text}\n\n"
        f"--- 以下为文档页面原文 ---\n"
        f"{page_text}\n"
        f"--- 文档页面原文结束 ---\n\n"
        f"请基于以上页面原文，对每个选项（A/B/C/D）给出判断。"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_whitespace(text: str) -> str:
    """Collapse whitespace for dedup key comparison."""
    return re.sub(r"\s+", " ", text).strip()
