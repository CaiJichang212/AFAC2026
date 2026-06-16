"""End-to-end insurance QA pipeline orchestration (Task 12).

Orchestrates all stages: question parsing -> doc retrieval -> tree retrieval ->
evidence extraction -> calculation -> answer judging, with best-effort fallbacks.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agent.answer_judge import AnswerJudge
from agent.calculation import CalculationEngine
from agent.config import AgentConfig
from agent.doc_retriever import DocRetriever
from agent.domain_profiles import DomainProfile, get_profile
from agent.evidence_extractor import EvidenceExtractor
from agent.index_store import IndexStore
from agent.llm_client import LLMClient
from agent.question_parser import QuestionParser
from agent.schemas import AnswerRecord, CandidateNode, EvidenceRecord
from agent.token_meter import TokenMeter
from agent.tree_retriever import TreeRetriever, _count_pages_in_range

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Page-range widening helper
# ---------------------------------------------------------------------------


def _widen_page_range(page_range: str, by: int = 1) -> str:
    """Expand a page-range string by *by* pages on each side.

    >>> _widen_page_range("6-8", 1)
    "5-9"
    >>> _widen_page_range("6", 1)
    "5-7"
    """
    if not page_range:
        return ""
    page_range = page_range.strip()
    if "-" in page_range:
        parts = page_range.split("-", 1)
        start = max(1, int(parts[0].strip()) - by)
        end = int(parts[1].strip()) + by
        return f"{start}-{end}"
    else:
        p = int(page_range)
        start = max(1, p - by)
        end = p + by
        if start == end:
            return str(start)
        return f"{start}-{end}"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_question(
    parsed: Any,  # ParsedQuestion
    *,
    config: AgentConfig,
    profile: DomainProfile,
    catalog: Any,  # DocCatalog
    index_store: IndexStore,
    llm_client: LLMClient | None,
    token_meter: TokenMeter,
) -> AnswerRecord:
    """Run the full pipeline for a single parsed question.

    Stages:
    1. DocRetriever.retrieve (A-split passthrough)
    2. TreeRetriever.retrieve per doc (LLM node selection)
    3. EvidenceExtractor.extract (LLM verdicts)
    4. CalculationEngine.compute
    5. AnswerJudge.judge

    Best-effort fallbacks (each capped by config.max_retry_per_question):
    - node-insufficient: widen page ranges and re-extract evidence
    - evidence-insufficient: retry extraction with widened pages
    - low_confidence / answer_unclear: re-judge with filtered evidence

    Returns an AnswerRecord with all provenance data.
    """
    doc_retriever = DocRetriever()
    tree_retriever = TreeRetriever()
    evidence_extractor = EvidenceExtractor()
    calc_engine = CalculationEngine(doc_product_map=profile.doc_product_map)
    answer_judge = AnswerJudge()

    max_retries = config.max_retry_per_question
    retry_count = 0
    fallbacks: list[str] = []
    all_warnings: list[str] = []

    # ------------------------------------------------------------------
    # Stage 1: Document retrieval (A-split passthrough)
    # ------------------------------------------------------------------
    try:
        doc_ids = doc_retriever.retrieve(parsed, catalog)
    except Exception as exc:
        logger.warning("Doc retrieval failed for %s: %s", parsed.qid, exc)
        return _empty_answer(parsed, token_meter, ["doc_retrieval_failed"],
                             [f"Doc retrieval failed: {exc}"])

    if not doc_ids:
        logger.warning("No candidate docs for %s; producing empty answer.", parsed.qid)
        return _empty_answer(parsed, token_meter, ["no_docs_retrieved"],
                             ["No candidate documents retrieved"])

    # ------------------------------------------------------------------
    # Stage 2: Tree retrieval per document
    # ------------------------------------------------------------------
    all_candidates: list[CandidateNode] = []
    for doc_id_str in doc_ids:
        try:
            doc_id_int = int(doc_id_str)
        except (ValueError, TypeError):
            logger.warning("Invalid doc_id %r for %s; skipping.", doc_id_str, parsed.qid)
            continue

        try:
            compact_tree = index_store.get_document_structure(doc_id_int)
        except Exception as exc:
            logger.warning("Failed to get structure for doc %s (qid=%s): %s",
                           doc_id_int, parsed.qid, exc)
            all_warnings.append(f"doc_{doc_id_int}_structure_failed")
            continue

        try:
            candidates = tree_retriever.retrieve(
                parsed,
                doc_id_str,
                compact_tree,
                config,
                profile,
                llm_client=llm_client,
                token_meter=token_meter,
            )
        except Exception as exc:
            logger.warning("Tree retrieval failed for doc %s (qid=%s): %s",
                           doc_id_int, parsed.qid, exc)
            all_warnings.append(f"tree_retrieval_failed_doc_{doc_id_int}")
            continue

        all_candidates.extend(candidates)

    # ------------------------------------------------------------------
    # Stage 3: Evidence extraction
    # ------------------------------------------------------------------
    evidence = _extract_evidence(
        parsed, all_candidates, index_store, config,
        evidence_extractor, llm_client, token_meter,
    )

    # ------------------------------------------------------------------
    # Fallback: node-insufficient
    # ------------------------------------------------------------------
    total_pages = sum(_count_pages_in_range(c.page_range) for c in all_candidates)
    if (len(all_candidates) < 2 or total_pages < 3):
        if retry_count < max_retries and all_candidates:
            logger.info(
                "Node-insufficient for %s (%d candidates, %d pages); widening.",
                parsed.qid, len(all_candidates), total_pages,
            )
            fallbacks.append("node_insufficient_widen")
            widened = _widen_candidates(all_candidates, by=2)
            evidence_retry = _extract_evidence(
                parsed, widened, index_store, config,
                evidence_extractor, llm_client, token_meter,
            )
            if evidence_retry:
                evidence = evidence_extractor._dedup(evidence + evidence_retry)
            retry_count += 1

    # ------------------------------------------------------------------
    # Stage 4: Calculation
    # ------------------------------------------------------------------
    calculations = calc_engine.compute(parsed, evidence)

    # ------------------------------------------------------------------
    # Stage 5: Answer judging
    # ------------------------------------------------------------------
    answer_record = answer_judge.judge(parsed, evidence, calculations)

    # ------------------------------------------------------------------
    # Fallback: evidence-insufficient (selected answer lacks support)
    # ------------------------------------------------------------------
    if not _has_support_for_answer(answer_record, parsed):
        if retry_count < max_retries and all_candidates:
            logger.info(
                "Evidence-insufficient for %s (answer=%s); retrying extraction.",
                parsed.qid, answer_record.answer,
            )
            fallbacks.append("evidence_insufficient_retry")
            widened = _widen_candidates(all_candidates, by=2)
            evidence_retry = _extract_evidence(
                parsed, widened, index_store, config,
                evidence_extractor, llm_client, token_meter,
            )
            if evidence_retry:
                evidence = evidence_extractor._dedup(evidence + evidence_retry)
            # Re-calculate and re-judge
            calculations = calc_engine.compute(parsed, evidence)
            answer_record = answer_judge.judge(parsed, evidence, calculations)
            retry_count += 1

    # ------------------------------------------------------------------
    # Fallback: illegal-answer self-check (low_confidence / answer_unclear)
    # ------------------------------------------------------------------
    has_low_confidence = "low_confidence" in answer_record.fallbacks
    has_answer_unclear = any("answer_unclear" in w for w in answer_record.warnings)
    if (has_low_confidence or has_answer_unclear) and retry_count < max_retries:
        logger.info("Low-confidence answer for %s; retrying judge with filtered evidence.",
                     parsed.qid)
        fallbacks.append("low_confidence_rejudge")
        # Re-judge with only high/medium confidence evidence
        filtered_evidence = [e for e in evidence if e.confidence in ("high", "medium")]
        if filtered_evidence and len(filtered_evidence) >= len(evidence) // 2:
            answer_record = answer_judge.judge(parsed, filtered_evidence, calculations)
        retry_count += 1

    # ------------------------------------------------------------------
    # Assemble final AnswerRecord
    # ------------------------------------------------------------------
    answer_record.candidate_docs = list(doc_ids)
    answer_record.fallbacks = fallbacks + answer_record.fallbacks
    answer_record.warnings = all_warnings + answer_record.warnings
    answer_record.usage = _qid_usage(token_meter, parsed.qid)

    return answer_record


def run_all(
    *,
    config: AgentConfig,
    profile: DomainProfile | None = None,
    catalog: Any = None,
    index_store: IndexStore | None = None,
    llm_client: LLMClient | None = None,
    token_meter: TokenMeter | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run the full pipeline on all questions and write output artifacts.

    Args:
        config: Agent configuration.
        profile: Domain profile (loaded if None).
        catalog: DocCatalog (loaded if None).
        index_store: IndexStore (created if None).
        llm_client: LLM client (real or mock).
        token_meter: Token meter (created if None).
        limit: If set, only process the first *limit* questions.

    Returns a dict with keys:
        paths: dict of written file paths
        summary: dict with questions_run, fallbacks_triggered, total_tokens
    """
    if profile is None:
        profile = get_profile(config.domain)

    if catalog is None:
        from agent.catalog import DocCatalog
        catalog = DocCatalog.load(config.catalog_path)

    if index_store is None:
        index_store = IndexStore(config)

    if token_meter is None:
        token_meter = TokenMeter(logs_dir=config.logs_dir)

    # Parse questions
    parser = QuestionParser()
    parsed_questions = parser.parse_questions(config.questions_path, profile)

    if limit is not None:
        parsed_questions = parsed_questions[:limit]

    # Create output dirs
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.logs_dir.mkdir(parents=True, exist_ok=True)

    evidence_path = config.output_dir / "evidence.jsonl"

    total_fallbacks = 0
    answers: list[dict[str, Any]] = []  # for answer.csv

    with open(evidence_path, "w", encoding="utf-8") as efh:
        for pq in parsed_questions:
            logger.info("Processing %s ...", pq.qid)
            record = run_question(
                pq,
                config=config,
                profile=profile,
                catalog=catalog,
                index_store=index_store,
                llm_client=llm_client,
                token_meter=token_meter,
            )
            # Write to evidence.jsonl (one JSON line per question)
            efh.write(json.dumps(_answer_record_to_dict(record), ensure_ascii=False) + "\n")
            efh.flush()

            # Collect for answer.csv
            qid_usage = record.usage
            answers.append({
                "qid": record.qid,
                "answer": record.answer,
                "prompt_tokens": qid_usage.get("prompt_tokens", 0),
                "completion_tokens": qid_usage.get("completion_tokens", 0),
                "total_tokens": qid_usage.get("total_tokens", 0),
            })
            total_fallbacks += len(record.fallbacks)

    # Write answer.csv
    answer_csv_path = _write_answer_csv(config.output_dir / "answer.csv", answers)

    # Write usage log
    usage_path = token_meter.write_log()

    # Summary
    total_tokens = sum(a["total_tokens"] for a in answers)

    return {
        "paths": {
            "evidence_jsonl": str(evidence_path),
            "answer_csv": str(answer_csv_path),
            "usage_jsonl": str(usage_path),
        },
        "summary": {
            "questions_run": len(parsed_questions),
            "fallbacks_triggered": total_fallbacks,
            "total_tokens": total_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_evidence(
    parsed: Any,
    candidates: list[CandidateNode],
    index_store: IndexStore,
    config: AgentConfig,
    extractor: EvidenceExtractor,
    llm_client: LLMClient | None,
    token_meter: TokenMeter,
) -> list[EvidenceRecord]:
    """Extract evidence, returning coverage-only records on failure."""
    try:
        return extractor.extract(
            parsed, candidates, index_store, config,
            llm_client=llm_client, token_meter=token_meter,
        )
    except Exception as exc:
        logger.warning("Evidence extraction failed for %s: %s", parsed.qid, exc)
        return extractor.extract(
            parsed, [], index_store, config,
            llm_client=llm_client, token_meter=token_meter,
        )


def _widen_candidates(
    candidates: list[CandidateNode], by: int = 1
) -> list[CandidateNode]:
    """Return a copy of candidates with widened page ranges."""
    widened: list[CandidateNode] = []
    for c in candidates:
        new_range = _widen_page_range(c.page_range, by)
        if new_range and new_range != c.page_range:
            widened.append(CandidateNode(
                doc_id=c.doc_id,
                node_id=c.node_id,
                title=c.title,
                page_range=new_range,
                matched_signals=list(c.matched_signals),
                reason=f"{c.reason} (widened by +-{by})",
                needs_page_fetch=True,
            ))
    return widened


def _has_support_for_answer(record: AnswerRecord, parsed: Any) -> bool:
    """Check if every selected answer option has at least one support evidence record."""
    answer = record.answer
    if not answer:
        return False

    fmt = getattr(parsed, "answer_format", "mcq")
    if fmt == "tf":
        selected_opts = [answer] if answer in ("A", "B") else []
    elif fmt == "mcq":
        selected_opts = [answer] if len(answer) == 1 else []
    elif fmt == "multi":
        selected_opts = list(answer)
    else:
        selected_opts = list(answer)

    for opt in selected_opts:
        has_support = any(
            e.evidence_type == "support" and e.option == opt
            for e in record.evidence
        )
        if not has_support:
            return False
    return True


def _qid_usage(token_meter: TokenMeter, qid: str) -> dict[str, Any]:
    """Return per-qid token summary dict."""
    records = token_meter.records_for_qid(qid)
    return {
        "prompt_tokens": sum(r.prompt_tokens for r in records),
        "completion_tokens": sum(r.completion_tokens for r in records),
        "total_tokens": sum(r.total_tokens for r in records),
    }


def _answer_record_to_dict(record: AnswerRecord) -> dict[str, Any]:
    """Serialize an AnswerRecord to a JSON-safe dict."""
    return {
        "qid": record.qid,
        "answer": record.answer,
        "candidate_docs": record.candidate_docs,
        "selected_nodes": record.selected_nodes,
        "evidence": [asdict(e) for e in record.evidence],
        "calculations": record.calculations,
        "usage": record.usage,
        "fallbacks": record.fallbacks,
        "warnings": record.warnings,
        "option_judgements": record.option_judgements,
    }


def _write_answer_csv(path: Path, answers: list[dict[str, Any]]) -> Path:
    """Write answer.csv with summary row + question rows.

    Header: qid,answer,prompt_tokens,completion_tokens,total_tokens
    Row 1: summary row (sums of all question rows, answer blank)
    Rows 2..N: per-question rows in question-file order
    """
    total_prompt = sum(a["prompt_tokens"] for a in answers)
    total_completion = sum(a["completion_tokens"] for a in answers)
    total_tokens = sum(a["total_tokens"] for a in answers)

    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"])
        # Summary row: answer blank, token columns are sums
        writer.writerow(["summary", "", total_prompt, total_completion, total_tokens])
        # Question rows (in file order)
        for a in answers:
            writer.writerow([
                a["qid"],
                a["answer"],
                a["prompt_tokens"],
                a["completion_tokens"],
                a["total_tokens"],
            ])

    return path


def _empty_answer(
    parsed: Any,
    token_meter: TokenMeter,
    fallbacks: list[str],
    warnings: list[str],
) -> AnswerRecord:
    """Produce a minimal fallback AnswerRecord when the pipeline cannot proceed."""
    return AnswerRecord(
        qid=parsed.qid,
        answer="A",
        candidate_docs=[],
        fallbacks=fallbacks,
        warnings=warnings,
        usage=_qid_usage(token_meter, parsed.qid),
    )
