"""Tests for agent/evidence_extractor.py (Task 9) — mocked LLM, NO network.

Also serves as the shared test file for output validation (Task 11).
"""

from __future__ import annotations

import csv
import json
from io import StringIO
from unittest.mock import patch

import pytest

from agent.config import AgentConfig
from agent.evidence_extractor import (
    _ALL_OPTIONS,
    _EVIDENCE_JSON_SCHEMA,
    _build_evidence_messages,
    _normalize_whitespace,
    EvidenceExtractor,
)
from agent.index_store import IndexStore, PageText
from agent.llm_client import LLMClient, MockApiCaller
from agent.schemas import CandidateNode, EvidenceRecord, ParsedQuestion
from scripts.validate_outputs import (
    validate_answer_csv_rows,
    validate_evidence_jsonl_lines,
    validate_usage_jsonl_lines,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_parsed_question(**overrides: object) -> ParsedQuestion:
    """Build a minimal ParsedQuestion with sensible defaults."""
    defaults: dict = {
        "qid": "ins_a_001",
        "domain": "insurance",
        "split": "A",
        "question": "张先生购买了平安智盈金生产品，身故保险金如何计算？",
        "options": {
            "A": "已交保费",
            "B": "基本保额的160%",
            "C": "现金价值",
            "D": "三者取大",
        },
        "answer_format": "mcq",
        "type": "推理判断",
        "doc_ids": ["1"],
    }
    defaults.update(overrides)
    return ParsedQuestion(**defaults)


def _make_candidate(**overrides: object) -> CandidateNode:
    """Build a minimal CandidateNode."""
    defaults: dict = {
        "doc_id": "1",
        "node_id": "n1",
        "title": "身故保险金",
        "page_range": "6-8",
        "matched_signals": ["身故保险金"],
        "reason": "liability match",
        "needs_page_fetch": True,
    }
    defaults.update(overrides)
    return CandidateNode(**defaults)


def _make_page_text(page: int, text: str) -> PageText:
    """Build a PageText for synthetic tests."""
    return PageText(doc_id="1", page=page, text=text, char_count=len(text))


PAGE_6_TEXT = (
    "第三章 身故保险金\n"
    "被保险人在保险期间内身故，我们将按以下方式计算身故保险金：\n"
    "已交保费、基本保险金额的160%、现金价值三者取大。\n"
    "具体计算公式为：身故保险金 = max(已交保费, 基本保额×160%, 现金价值)。\n"
)

PAGE_7_TEXT = (
    "第四章 免赔额规定\n"
    "每年免赔额为10000元，超过免赔额的部分按80%比例进行赔付。\n"
)

PAGE_8_TEXT = (
    "第五章 退保规定\n"
    "投保人在猴年期内退保，退还现金价值。\n"
)

COMBINED_PAGE_TEXT = PAGE_6_TEXT + "\n" + PAGE_7_TEXT + "\n" + PAGE_8_TEXT


def _make_canned_evidence_response(
    verdicts: list[dict],
    *,
    model: str = "mock",
) -> dict:
    """Build a canned OpenAI-compatible response for evidence extraction."""
    content = json.dumps({"verdicts": verdicts}, ensure_ascii=False)
    return {
        "choices": [{"message": {"content": content}}],
        "model": model,
        "usage": {"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300},
    }


def _make_mock_llm_client(verdicts: list[dict]) -> LLMClient:
    """Build an LLMClient with a MockApiCaller returning *verdicts*."""
    mock = MockApiCaller(responses=[_make_canned_evidence_response(verdicts)])
    return LLMClient(model="mock", api_caller=mock)


def _make_support_verdicts() -> list[dict]:
    """Return a standard set of verdicts that quote PAGE_6_TEXT."""
    return [
        {
            "option": "A",
            "evidence_type": "support",
            "quote": "已交保费、基本保险金额的160%、现金价值三者取大。",
            "normalized_fact": "身故保险金由已交保费、基本保额的160%、现金价值三者取大确定。",
            "numbers": [],
            "confidence": "high",
        },
        {
            "option": "B",
            "evidence_type": "support",
            "quote": "基本保险金额的160%",
            "normalized_fact": "基本保险金额的160%作为身故保险金计算因子之一。",
            "numbers": [{"name": "比例", "value": 160, "unit": "%"}],
            "confidence": "high",
        },
        {
            "option": "C",
            "evidence_type": "support",
            "quote": "现金价值三者取大",
            "normalized_fact": "现金价值是身故保险金计算因子之一。",
            "numbers": [],
            "confidence": "medium",
        },
        {
            "option": "D",
            "evidence_type": "support",
            "quote": "三者取大",
            "normalized_fact": "身故保险金采用三者取大方式确定。",
            "numbers": [],
            "confidence": "high",
        },
    ]


# ===========================================================================
# _normalize_whitespace
# ===========================================================================


def test_normalize_whitespace_collapses_spaces() -> None:
    assert _normalize_whitespace("  hello   world  ") == "hello world"


def test_normalize_whitespace_newlines() -> None:
    assert _normalize_whitespace("line1\nline2\n\nline3") == "line1 line2 line3"


def test_normalize_whitespace_empty() -> None:
    assert _normalize_whitespace("") == ""


# ===========================================================================
# TestEvidenceExtraction — core unit tests (mocked LLM, NO network)
# ===========================================================================


class TestEvidenceExtraction:
    """Tests for EvidenceExtractor with mocked LLM (no real network calls)."""

    # ------------------------------------------------------------------
    # Basic extraction
    # ------------------------------------------------------------------

    def test_extract_basic_mocked_llm(self) -> None:
        """Basic flow: MockApiCaller returns per-option verdicts -> EvidenceRecords."""
        extractor = EvidenceExtractor()
        parsed = _make_parsed_question()
        candidate = _make_candidate()
        llm_client = _make_mock_llm_client(_make_support_verdicts())

        with patch.object(
            IndexStore, "get_page_content", return_value=[
                _make_page_text(6, PAGE_6_TEXT),
                _make_page_text(7, PAGE_7_TEXT),
                _make_page_text(8, PAGE_8_TEXT),
            ]
        ):
            records = extractor.extract(
                parsed=parsed,
                candidates=[candidate],
                index_store=IndexStore(AgentConfig()),
                config=AgentConfig(),
                llm_client=llm_client,
            )

        assert len(records) >= 4  # at least one per option
        options_seen = {r.option for r in records}
        assert options_seen >= {"A", "B", "C", "D"}

        for rec in records:
            assert rec.qid == "ins_a_001"
            assert rec.doc_id == "1"
            assert rec.node_id == "n1"
            assert rec.pages == "6-8"
            assert rec.evidence_type in ("support", "refute", "unclear")
            assert rec.confidence in ("high", "medium", "low")

    # ------------------------------------------------------------------
    # Evidence type normalization
    # ------------------------------------------------------------------

    def test_evidence_type_normalized_to_valid_set(self) -> None:
        """Invalid evidence_type values are normalized to 'unclear'."""
        extractor = EvidenceExtractor()
        parsed = _make_parsed_question()
        candidate = _make_candidate()

        verdicts = [
            {
                "option": "A",
                "evidence_type": "contradict",  # invalid
                "quote": "已交保费",
                "normalized_fact": "test",
                "confidence": "high",
            },
            {
                "option": "B",
                "evidence_type": "SUPPORT",  # case-normalized
                "quote": "基本保额的160%",
                "normalized_fact": "test",
                "confidence": "medium",
            },
        ]
        llm_client = _make_mock_llm_client(verdicts)

        with patch.object(
            IndexStore, "get_page_content", return_value=[
                _make_page_text(6, PAGE_6_TEXT),
            ]
        ):
            records = extractor.extract(
                parsed=parsed,
                candidates=[candidate],
                index_store=IndexStore(AgentConfig()),
                config=AgentConfig(),
                llm_client=llm_client,
            )

        rec_a = next(r for r in records if r.option == "A")
        rec_b = next(r for r in records if r.option == "B")
        assert rec_a.evidence_type == "unclear"  # invalid -> unclear
        assert rec_b.evidence_type == "support"  # case-insensitive OK

    # ------------------------------------------------------------------
    # Per-option coverage
    # ------------------------------------------------------------------

    def test_each_option_covered(self) -> None:
        """Every option A/B/C/D has at least one record returned."""
        extractor = EvidenceExtractor()
        parsed = _make_parsed_question()
        candidate = _make_candidate()

        # LLM only returns verdicts for A and B
        verdicts = [
            {
                "option": "A",
                "evidence_type": "support",
                "quote": "已交保费",
                "normalized_fact": "A is supported",
                "confidence": "high",
            },
            {
                "option": "B",
                "evidence_type": "refute",
                "quote": "基本保额的160%",
                "normalized_fact": "B is refuted",
                "confidence": "medium",
            },
        ]
        llm_client = _make_mock_llm_client(verdicts)

        with patch.object(
            IndexStore, "get_page_content", return_value=[
                _make_page_text(6, PAGE_6_TEXT),
            ]
        ):
            records = extractor.extract(
                parsed=parsed,
                candidates=[candidate],
                index_store=IndexStore(AgentConfig()),
                config=AgentConfig(),
                llm_client=llm_client,
            )

        options_seen = {r.option for r in records}
        assert options_seen == {"A", "B", "C", "D"}

        # C and D should be synthesized unclear records
        for opt in ("C", "D"):
            rec = next(r for r in records if r.option == opt)
            assert rec.evidence_type == "unclear"
            assert rec.confidence == "low"

    # ------------------------------------------------------------------
    # Traceability: quotes MUST be substrings of page text
    # ------------------------------------------------------------------

    def test_traceability_quotes_in_page_text(self) -> None:
        """Every record's quote (if non-empty) is a substring of the page text."""
        extractor = EvidenceExtractor()
        parsed = _make_parsed_question()
        candidate = _make_candidate()
        llm_client = _make_mock_llm_client(_make_support_verdicts())

        with patch.object(
            IndexStore, "get_page_content", return_value=[
                _make_page_text(6, PAGE_6_TEXT),
                _make_page_text(7, PAGE_7_TEXT),
                _make_page_text(8, PAGE_8_TEXT),
            ]
        ):
            records = extractor.extract(
                parsed=parsed,
                candidates=[candidate],
                index_store=IndexStore(AgentConfig()),
                config=AgentConfig(),
                llm_client=llm_client,
            )

        for rec in records:
            if rec.quote:
                assert rec.quote in COMBINED_PAGE_TEXT, (
                    f"Quote for option {rec.option} not found in page text: {rec.quote!r}"
                )

    def test_bad_quote_not_in_page_text_not_allowed_high_confidence(self) -> None:
        """A support record with a quote NOT in the page text is downgraded to low."""
        extractor = EvidenceExtractor()
        parsed = _make_parsed_question()
        candidate = _make_candidate()

        # This verdict has a quote NOT in PAGE_6_TEXT
        bad_verdicts = [
            {
                "option": "A",
                "evidence_type": "support",
                "quote": "这句话不在原文中",
                "normalized_fact": "bad quote",
                "confidence": "high",
            },
        ]
        llm_client = _make_mock_llm_client(bad_verdicts)

        with patch.object(
            IndexStore, "get_page_content", return_value=[
                _make_page_text(6, PAGE_6_TEXT),
            ]
        ):
            records = extractor.extract(
                parsed=parsed,
                candidates=[candidate],
                index_store=IndexStore(AgentConfig()),
                config=AgentConfig(),
                llm_client=llm_client,
            )

        rec_a = next(r for r in records if r.option == "A")
        # The quote is faithfully passed through but confidence is downgraded
        # because it is not traceable to the page text.
        assert rec_a.quote == "这句话不在原文中"
        assert rec_a.confidence == "low"

    # ------------------------------------------------------------------
    # Quote traceability enforcement (whitespace-normalized substring check)
    # ------------------------------------------------------------------

    def test_traceable_quote_keeps_claimed_confidence(self) -> None:
        """A support/refute record whose quote IS a substring of page text
        keeps its claimed confidence (e.g. high)."""
        extractor = EvidenceExtractor()
        parsed = _make_parsed_question()
        candidate = _make_candidate()

        # Quote IS a verbatim substring of PAGE_6_TEXT
        verdicts = [
            {
                "option": "A",
                "evidence_type": "support",
                "quote": "已交保费、基本保险金额的160%、现金价值三者取大。",
                "normalized_fact": "fact",
                "confidence": "high",
            },
        ]
        llm_client = _make_mock_llm_client(verdicts)

        with patch.object(
            IndexStore, "get_page_content", return_value=[
                _make_page_text(6, PAGE_6_TEXT),
            ]
        ):
            records = extractor.extract(
                parsed=parsed,
                candidates=[candidate],
                index_store=IndexStore(AgentConfig()),
                config=AgentConfig(),
                llm_client=llm_client,
            )

        rec_a = next(r for r in records if r.option == "A" and r.doc_id == "1")
        assert rec_a.confidence == "high", (
            f"Traceable quote should keep high confidence, got {rec_a.confidence}"
        )

    def test_support_fabricated_quote_downgraded_to_low(self) -> None:
        """A support record whose non-empty quote is NOT a substring of
        the page text gets its confidence downgraded to 'low'."""
        extractor = EvidenceExtractor()
        parsed = _make_parsed_question()
        candidate = _make_candidate()

        # Quote is hallucinated, not in PAGE_6_TEXT
        verdicts = [
            {
                "option": "A",
                "evidence_type": "support",
                "quote": "完全不在原文中的句子",
                "normalized_fact": "fabricated",
                "confidence": "high",
            },
            {
                "option": "B",
                "evidence_type": "refute",
                "quote": "另一个捏造的引用",
                "normalized_fact": "fabricated refute",
                "confidence": "medium",
            },
        ]
        llm_client = _make_mock_llm_client(verdicts)

        with patch.object(
            IndexStore, "get_page_content", return_value=[
                _make_page_text(6, PAGE_6_TEXT),
            ]
        ):
            records = extractor.extract(
                parsed=parsed,
                candidates=[candidate],
                index_store=IndexStore(AgentConfig()),
                config=AgentConfig(),
                llm_client=llm_client,
            )

        rec_a = next(r for r in records if r.option == "A" and r.doc_id == "1")
        rec_b = next(r for r in records if r.option == "B" and r.doc_id == "1")

        # Both should be downgraded to low (not dropped, quote preserved)
        assert rec_a.confidence == "low", (
            f"Fabricated quote 'support' should be downgraded, got {rec_a.confidence}"
        )
        assert rec_a.quote == "完全不在原文中的句子"
        assert rec_b.confidence == "low", (
            f"Fabricated quote 'refute' should be downgraded, got {rec_b.confidence}"
        )
        assert rec_b.quote == "另一个捏造的引用"

    def test_quote_traceability_whitespace_tolerance(self) -> None:
        """Quotes with extra/normalized whitespace vs page text still count
        as traceable (not downgraded)."""
        extractor = EvidenceExtractor()
        parsed = _make_parsed_question()
        candidate = _make_candidate()

        # Quote has leading/trailing spaces that get stripped by normalisation
        quote_with_extra_ws = "  已交保费、基本保险金额的160%、现金价值三者取大。  "
        page_text = PAGE_6_TEXT

        verdicts = [
            {
                "option": "A",
                "evidence_type": "support",
                "quote": quote_with_extra_ws,
                "normalized_fact": "fact",
                "confidence": "high",
            },
        ]
        llm_client = _make_mock_llm_client(verdicts)

        with patch.object(
            IndexStore, "get_page_content", return_value=[
                _make_page_text(6, page_text),
            ]
        ):
            records = extractor.extract(
                parsed=parsed,
                candidates=[candidate],
                index_store=IndexStore(AgentConfig()),
                config=AgentConfig(),
                llm_client=llm_client,
            )

        rec_a = next(r for r in records if r.option == "A" and r.doc_id == "1")
        assert rec_a.confidence == "high", (
            f"Whitespace-tolerant quote should keep high confidence, got {rec_a.confidence}"
        )

    def test_unclear_with_fabricated_quote_not_downgraded(self) -> None:
        """Unclear records with non-substring quotes are NOT downgraded by the
        traceability check (their confidence is independent)."""
        extractor = EvidenceExtractor()
        parsed = _make_parsed_question()
        candidate = _make_candidate()

        # Unclear record with a quote NOT in page text
        verdicts = [
            {
                "option": "A",
                "evidence_type": "unclear",
                "quote": "这句话不在原文中",
                "normalized_fact": "uncertain",
                "confidence": "medium",
            },
        ]
        llm_client = _make_mock_llm_client(verdicts)

        with patch.object(
            IndexStore, "get_page_content", return_value=[
                _make_page_text(6, PAGE_6_TEXT),
            ]
        ):
            records = extractor.extract(
                parsed=parsed,
                candidates=[candidate],
                index_store=IndexStore(AgentConfig()),
                config=AgentConfig(),
                llm_client=llm_client,
            )

        rec_a = next(r for r in records if r.option == "A" and r.doc_id == "1")
        # Unclear records are NOT affected by traceability downgrade
        assert rec_a.evidence_type == "unclear"
        assert rec_a.confidence == "medium", (
            f"Unclear record should keep original confidence, got {rec_a.confidence}"
        )

    # ------------------------------------------------------------------
    # Dedup
    # ------------------------------------------------------------------

    def test_dedup_identical_records(self) -> None:
        """Two nodes returning identical (doc_id, pages, quote) -> only one kept."""
        extractor = EvidenceExtractor()
        parsed = _make_parsed_question()

        cand1 = _make_candidate(node_id="n1", page_range="6-8")
        cand2 = _make_candidate(node_id="n2", page_range="6-8")

        # Both return the same quote for option A
        same_quote = "已交保费、基本保险金额的160%、现金价值三者取大。"
        verdicts1 = [
            {
                "option": "A",
                "evidence_type": "support",
                "quote": same_quote,
                "normalized_fact": "fact 1",
                "confidence": "medium",
            },
            {
                "option": "B",
                "evidence_type": "unclear",
                "quote": "",
                "normalized_fact": "",
                "confidence": "low",
            },
            {
                "option": "C",
                "evidence_type": "unclear",
                "quote": "",
                "normalized_fact": "",
                "confidence": "low",
            },
            {
                "option": "D",
                "evidence_type": "unclear",
                "quote": "",
                "normalized_fact": "",
                "confidence": "low",
            },
        ]
        # Second node returns same quote with HIGH confidence
        verdicts2 = [
            {
                "option": "A",
                "evidence_type": "support",
                "quote": same_quote,
                "normalized_fact": "fact 2 (better)",
                "confidence": "high",
            },
            {
                "option": "B",
                "evidence_type": "unclear",
                "quote": "",
                "normalized_fact": "",
                "confidence": "low",
            },
            {
                "option": "C",
                "evidence_type": "unclear",
                "quote": "",
                "normalized_fact": "",
                "confidence": "low",
            },
            {
                "option": "D",
                "evidence_type": "unclear",
                "quote": "",
                "normalized_fact": "",
                "confidence": "low",
            },
        ]

        # Use a mock with multiple responses
        mock = MockApiCaller(responses=[
            _make_canned_evidence_response(verdicts1),
            _make_canned_evidence_response(verdicts2),
        ])
        llm_client = LLMClient(model="mock", api_caller=mock)

        with patch.object(
            IndexStore, "get_page_content", return_value=[
                _make_page_text(6, PAGE_6_TEXT),
            ]
        ):
            records = extractor.extract(
                parsed=parsed,
                candidates=[cand1, cand2],
                index_store=IndexStore(AgentConfig()),
                config=AgentConfig(),
                llm_client=llm_client,
            )

        # Option A records with that quote - should be exactly 1 (deduped)
        a_records = [r for r in records if r.option == "A" and r.quote == same_quote]
        assert len(a_records) == 1, (
            f"Expected 1 deduped record for option A, got {len(a_records)}"
        )
        # The kept record should be the high-confidence one
        assert a_records[0].confidence == "high"

    # ------------------------------------------------------------------
    # Coverage / synthesis
    # ------------------------------------------------------------------

    def test_coverage_synthesizes_unclear_for_missing_option(self) -> None:
        """An option with no verdict from any node gets a synthesized unclear record."""
        extractor = EvidenceExtractor()
        parsed = _make_parsed_question()
        candidate = _make_candidate()

        # LLM only returns verdict for A
        verdicts = [
            {
                "option": "A",
                "evidence_type": "support",
                "quote": "已交保费",
                "normalized_fact": "A is supported",
                "confidence": "high",
            },
        ]
        llm_client = _make_mock_llm_client(verdicts)

        with patch.object(
            IndexStore, "get_page_content", return_value=[
                _make_page_text(6, PAGE_6_TEXT),
            ]
        ):
            records = extractor.extract(
                parsed=parsed,
                candidates=[candidate],
                index_store=IndexStore(AgentConfig()),
                config=AgentConfig(),
                llm_client=llm_client,
            )

        # Check each option has at least one record
        for opt in _ALL_OPTIONS:
            opt_records = [r for r in records if r.option == opt]
            assert len(opt_records) >= 1, f"Option {opt} has no records"

        # Synthesized records for B, C, D should be unclear/low
        for opt in ("B", "C", "D"):
            syn = next(r for r in records if r.option == opt and r.doc_id == "")
            assert syn.evidence_type == "unclear"
            assert syn.confidence == "low"
            assert syn.quote == ""
            assert syn.normalized_fact == ""

    # ------------------------------------------------------------------
    # Fallback (no LLM client)
    # ------------------------------------------------------------------

    def test_fallback_no_llm_client(self) -> None:
        """Without an LLM client, deterministic unclear records are produced, no crash."""
        extractor = EvidenceExtractor()
        parsed = _make_parsed_question()
        candidate = _make_candidate()

        with patch.object(
            IndexStore, "get_page_content", return_value=[
                _make_page_text(6, PAGE_6_TEXT),
            ]
        ):
            records = extractor.extract(
                parsed=parsed,
                candidates=[candidate],
                index_store=IndexStore(AgentConfig()),
                config=AgentConfig(),
                llm_client=None,
            )

        assert len(records) >= 4
        for rec in records:
            assert rec.qid == "ins_a_001"
            assert rec.evidence_type == "unclear"
            assert rec.confidence == "low"
            assert rec.option in ("A", "B", "C", "D")

    def test_fallback_no_llm_client_no_crash(self) -> None:
        """Fallback with empty candidates -> no crash, still coverage records."""
        extractor = EvidenceExtractor()
        parsed = _make_parsed_question()

        records = extractor.extract(
            parsed=parsed,
            candidates=[],
            index_store=IndexStore(AgentConfig()),
            config=AgentConfig(),
            llm_client=None,
        )

        # Should still get 4 synthesized records (one per option)
        assert len(records) == 4
        for rec in records:
            assert rec.evidence_type == "unclear"
            assert rec.confidence == "low"

    # ------------------------------------------------------------------
    # max_evidence_per_option cap
    # ------------------------------------------------------------------

    def test_max_evidence_per_option_cap_respected(self) -> None:
        """max_evidence_per_option=2 should limit per-option records to 2."""
        extractor = EvidenceExtractor()
        parsed = _make_parsed_question()

        # Two candidates, each returning records for A
        cand1 = _make_candidate(node_id="n1", page_range="1-2")
        cand2 = _make_candidate(node_id="n2", page_range="3-4")

        verdicts_a1 = [
            {
                "option": "A",
                "evidence_type": "support",
                "quote": "已交保费 quote 1",
                "normalized_fact": "fact 1",
                "confidence": "high",
            },
            {
                "option": "B",
                "evidence_type": "unclear",
                "quote": "",
                "normalized_fact": "",
                "confidence": "low",
            },
            {
                "option": "C",
                "evidence_type": "unclear",
                "quote": "",
                "normalized_fact": "",
                "confidence": "low",
            },
            {
                "option": "D",
                "evidence_type": "unclear",
                "quote": "",
                "normalized_fact": "",
                "confidence": "low",
            },
        ]
        verdicts_a2 = [
            {
                "option": "A",
                "evidence_type": "support",
                "quote": "已交保费 quote 2",
                "normalized_fact": "fact 2",
                "confidence": "medium",
            },
            {
                "option": "A",
                "evidence_type": "support",
                "quote": "已交保费 quote 3",
                "normalized_fact": "fact 3",
                "confidence": "low",
            },
            {
                "option": "B",
                "evidence_type": "unclear",
                "quote": "",
                "normalized_fact": "",
                "confidence": "low",
            },
            {
                "option": "C",
                "evidence_type": "unclear",
                "quote": "",
                "normalized_fact": "",
                "confidence": "low",
            },
            {
                "option": "D",
                "evidence_type": "unclear",
                "quote": "",
                "normalized_fact": "",
                "confidence": "low",
            },
        ]

        mock = MockApiCaller(responses=[
            _make_canned_evidence_response(verdicts_a1),
            _make_canned_evidence_response(verdicts_a2),
        ])
        llm_client = LLMClient(model="mock", api_caller=mock)

        page_text_1 = "已交保费 quote 1 in page text here."
        page_text_2 = "已交保费 quote 2 in page. 已交保费 quote 3 also in page."

        with patch.object(IndexStore, "get_page_content") as mock_get:
            mock_get.side_effect = [
                [_make_page_text(1, page_text_1)],
                [_make_page_text(3, page_text_2)],
            ]

            records = extractor.extract(
                parsed=parsed,
                candidates=[cand1, cand2],
                index_store=IndexStore(AgentConfig()),
                config=AgentConfig(max_evidence_per_option=2),
                llm_client=llm_client,
            )

        # Option A should have at most 2 records
        a_records = [r for r in records if r.option == "A"]
        assert len(a_records) <= 2, f"Expected <=2 records for option A, got {len(a_records)}"

    # ------------------------------------------------------------------
    # TokenMeter integration (optional param)
    # ------------------------------------------------------------------

    def test_token_meter_records_usage(self) -> None:
        """When token_meter is provided, usage records are appended."""
        from agent.token_meter import TokenMeter

        extractor = EvidenceExtractor()
        parsed = _make_parsed_question()
        candidate = _make_candidate()
        llm_client = _make_mock_llm_client(_make_support_verdicts())

        meter = TokenMeter()

        with patch.object(
            IndexStore, "get_page_content", return_value=[
                _make_page_text(6, PAGE_6_TEXT),
            ]
        ):
            records = extractor.extract(
                parsed=parsed,
                candidates=[candidate],
                index_store=IndexStore(AgentConfig()),
                config=AgentConfig(),
                llm_client=llm_client,
                token_meter=meter,
            )

        assert len(records) >= 4
        assert meter.record_count >= 1  # one LLM call recorded
        evidence_records = meter.records_for_stage("evidence")
        assert len(evidence_records) >= 1
        for rec in evidence_records:
            assert rec.qid == "ins_a_001"
            assert rec.stage == "evidence"
            assert rec.success is True

    def test_token_meter_none_is_fine(self) -> None:
        """When token_meter is None, no error occurs."""
        extractor = EvidenceExtractor()
        parsed = _make_parsed_question()
        candidate = _make_candidate()
        llm_client = _make_mock_llm_client(_make_support_verdicts())

        with patch.object(
            IndexStore, "get_page_content", return_value=[
                _make_page_text(6, PAGE_6_TEXT),
            ]
        ):
            records = extractor.extract(
                parsed=parsed,
                candidates=[candidate],
                index_store=IndexStore(AgentConfig()),
                config=AgentConfig(),
                llm_client=llm_client,
                token_meter=None,
            )

        assert len(records) >= 4

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_candidate_invalid_doc_id_skipped(self) -> None:
        """Candidates with non-numeric doc_id are skipped with a warning."""
        extractor = EvidenceExtractor()
        parsed = _make_parsed_question()
        bad_candidate = _make_candidate(doc_id="not_a_number")

        llm_client = _make_mock_llm_client(_make_support_verdicts())

        records = extractor.extract(
            parsed=parsed,
            candidates=[bad_candidate],
            index_store=IndexStore(AgentConfig()),
            config=AgentConfig(),
            llm_client=llm_client,
        )

        # Should still get coverage records (synthesized)
        for opt in _ALL_OPTIONS:
            assert any(r.option == opt for r in records)

    def test_candidate_empty_page_range_skipped(self) -> None:
        """Candidates with empty page_range are skipped."""
        extractor = EvidenceExtractor()
        parsed = _make_parsed_question()
        bad_candidate = _make_candidate(doc_id="1", page_range="")

        llm_client = _make_mock_llm_client(_make_support_verdicts())

        with patch.object(
            IndexStore, "get_page_content", return_value=[]
        ):
            records = extractor.extract(
                parsed=parsed,
                candidates=[bad_candidate],
                index_store=IndexStore(AgentConfig()),
                config=AgentConfig(),
                llm_client=llm_client,
            )

        # Should still get coverage records
        for opt in _ALL_OPTIONS:
            assert any(r.option == opt for r in records)

    def test_candidate_empty_page_text_skipped(self) -> None:
        """Candidates whose page text is all empty are skipped."""
        extractor = EvidenceExtractor()
        parsed = _make_parsed_question()
        candidate = _make_candidate()

        llm_client = _make_mock_llm_client(_make_support_verdicts())

        with patch.object(
            IndexStore, "get_page_content", return_value=[
                _make_page_text(6, ""),
                _make_page_text(7, ""),
                _make_page_text(8, ""),
            ]
        ):
            records = extractor.extract(
                parsed=parsed,
                candidates=[candidate],
                index_store=IndexStore(AgentConfig()),
                config=AgentConfig(),
                llm_client=llm_client,
            )

        # Should still get coverage records
        for opt in _ALL_OPTIONS:
            assert any(r.option == opt for r in records)

    def test_llm_returns_invalid_json(self) -> None:
        """When LLM returns invalid JSON, no records from that node (graceful)."""
        extractor = EvidenceExtractor()
        parsed = _make_parsed_question()
        candidate = _make_candidate()

        mock = MockApiCaller(responses=[{
            "choices": [{"message": {"content": "not valid json!!!"}}],
            "model": "mock",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }])
        llm_client = LLMClient(model="mock", api_caller=mock)

        with patch.object(
            IndexStore, "get_page_content", return_value=[
                _make_page_text(6, PAGE_6_TEXT),
            ]
        ):
            records = extractor.extract(
                parsed=parsed,
                candidates=[candidate],
                index_store=IndexStore(AgentConfig()),
                config=AgentConfig(),
                llm_client=llm_client,
            )

        # Should get coverage records (synthesized)
        for opt in _ALL_OPTIONS:
            assert any(r.option == opt for r in records)

    def test_llm_returns_verdicts_not_a_list(self) -> None:
        """When verdicts is not a list, no records from that node."""
        extractor = EvidenceExtractor()
        parsed = _make_parsed_question()
        candidate = _make_candidate()

        content = json.dumps({"verdicts": "not a list"})
        mock = MockApiCaller(responses=[{
            "choices": [{"message": {"content": content}}],
            "model": "mock",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }])
        llm_client = LLMClient(model="mock", api_caller=mock)

        with patch.object(
            IndexStore, "get_page_content", return_value=[
                _make_page_text(6, PAGE_6_TEXT),
            ]
        ):
            records = extractor.extract(
                parsed=parsed,
                candidates=[candidate],
                index_store=IndexStore(AgentConfig()),
                config=AgentConfig(),
                llm_client=llm_client,
            )

        # Coverage records synthesized
        for opt in _ALL_OPTIONS:
            assert any(r.option == opt for r in records)

    def test_missing_required_fields_get_defaults(self) -> None:
        """Missing fields in LLM verdict get safe defaults."""
        extractor = EvidenceExtractor()
        parsed = _make_parsed_question()
        candidate = _make_candidate()

        # Verdict missing evidence_type, quote, normalized_fact, confidence
        verdicts = [{"option": "A"}]  # bare minimum
        llm_client = _make_mock_llm_client(verdicts)

        with patch.object(
            IndexStore, "get_page_content", return_value=[
                _make_page_text(6, PAGE_6_TEXT),
            ]
        ):
            records = extractor.extract(
                parsed=parsed,
                candidates=[candidate],
                index_store=IndexStore(AgentConfig()),
                config=AgentConfig(),
                llm_client=llm_client,
            )

        rec_a = next(r for r in records if r.option == "A" and r.doc_id == "1")
        assert rec_a.evidence_type == "unclear"
        assert rec_a.quote == ""
        assert rec_a.normalized_fact == ""
        assert rec_a.confidence == "medium"

    # ------------------------------------------------------------------
    # Verdict-level numbers field
    # ------------------------------------------------------------------

    def test_empty_quote_support_confidence_downgraded_to_low(self) -> None:
        """A support/refute record with empty quote gets confidence downgraded to low."""
        extractor = EvidenceExtractor()
        parsed = _make_parsed_question()
        candidate = _make_candidate()

        verdicts = [
            {
                "option": "A",
                "evidence_type": "support",
                "quote": "",  # empty quote but support claim
                "normalized_fact": "some fact without quote",
                "confidence": "high",  # should be downgraded
            },
            {
                "option": "B",
                "evidence_type": "unclear",
                "quote": "",
                "normalized_fact": "",
                "confidence": "medium",
            },
            {
                "option": "C",
                "evidence_type": "unclear",
                "quote": "",
                "normalized_fact": "",
                "confidence": "low",
            },
            {
                "option": "D",
                "evidence_type": "unclear",
                "quote": "",
                "normalized_fact": "",
                "confidence": "low",
            },
        ]
        llm_client = _make_mock_llm_client(verdicts)

        with patch.object(
            IndexStore, "get_page_content", return_value=[
                _make_page_text(6, PAGE_6_TEXT),
            ]
        ):
            records = extractor.extract(
                parsed=parsed,
                candidates=[candidate],
                index_store=IndexStore(AgentConfig()),
                config=AgentConfig(),
                llm_client=llm_client,
            )

        # Option A: empty quote + support -> confidence downgraded to low
        rec_a = next(r for r in records if r.option == "A" and r.doc_id == "1")
        assert rec_a.confidence == "low", (
            f"Expected confidence=low for support+empty-quote, got {rec_a.confidence}"
        )
        assert rec_a.evidence_type == "support"

    # ------------------------------------------------------------------
    # Verdict-level numbers field
    # ------------------------------------------------------------------

    def test_numbers_field_propagated_correctly(self) -> None:
        """Numbers from LLM verdict are mapped into EvidenceRecord.numbers."""
        extractor = EvidenceExtractor()
        parsed = _make_parsed_question()
        candidate = _make_candidate()

        verdicts = [
            {
                "option": "A",
                "evidence_type": "support",
                "quote": "免赔额为10000元",
                "normalized_fact": "免赔额10000元",
                "confidence": "high",
                "numbers": [
                    {"name": "免赔额", "value": 10000, "unit": "元"},
                    {"name": "比例", "value": 80, "unit": "%"},
                ],
            },
            {
                "option": "B",
                "evidence_type": "unclear",
                "quote": "",
                "normalized_fact": "",
                "confidence": "low",
            },
            {
                "option": "C",
                "evidence_type": "unclear",
                "quote": "",
                "normalized_fact": "",
                "confidence": "low",
            },
            {
                "option": "D",
                "evidence_type": "unclear",
                "quote": "",
                "normalized_fact": "",
                "confidence": "low",
            },
        ]
        llm_client = _make_mock_llm_client(verdicts)

        with patch.object(
            IndexStore, "get_page_content", return_value=[
                _make_page_text(6, PAGE_6_TEXT + "\n免赔额为10000元按80%赔付。"),
            ]
        ):
            records = extractor.extract(
                parsed=parsed,
                candidates=[candidate],
                index_store=IndexStore(AgentConfig()),
                config=AgentConfig(),
                llm_client=llm_client,
            )

        rec_a = next(r for r in records if r.option == "A" and r.doc_id == "1")
        assert len(rec_a.numbers) == 2
        assert rec_a.numbers[0]["name"] == "免赔额"
        assert rec_a.numbers[0]["value"] == 10000
        assert rec_a.numbers[0]["unit"] == "元"

    # ------------------------------------------------------------------
    # Dedup with whitespace normalization
    # ------------------------------------------------------------------

    def test_dedup_whitespace_normalization(self) -> None:
        """Quotes with different whitespace are treated as identical for dedup."""
        extractor = EvidenceExtractor()
        parsed = _make_parsed_question()

        cand1 = _make_candidate(node_id="n1", page_range="1-2")
        cand2 = _make_candidate(node_id="n2", page_range="1-2")

        verdicts1 = [
            {
                "option": "A",
                "evidence_type": "support",
                "quote": "  hello   world  ",
                "normalized_fact": "fact 1",
                "confidence": "high",
            },
            {
                "option": "B",
                "evidence_type": "unclear",
                "quote": "",
                "normalized_fact": "",
                "confidence": "low",
            },
            {
                "option": "C",
                "evidence_type": "unclear",
                "quote": "",
                "normalized_fact": "",
                "confidence": "low",
            },
            {
                "option": "D",
                "evidence_type": "unclear",
                "quote": "",
                "normalized_fact": "",
                "confidence": "low",
            },
        ]
        verdicts2 = [
            {
                "option": "A",
                "evidence_type": "support",
                "quote": "hello world",
                "normalized_fact": "fact 2",
                "confidence": "low",
            },
            {
                "option": "B",
                "evidence_type": "unclear",
                "quote": "",
                "normalized_fact": "",
                "confidence": "low",
            },
            {
                "option": "C",
                "evidence_type": "unclear",
                "quote": "",
                "normalized_fact": "",
                "confidence": "low",
            },
            {
                "option": "D",
                "evidence_type": "unclear",
                "quote": "",
                "normalized_fact": "",
                "confidence": "low",
            },
        ]

        mock = MockApiCaller(responses=[
            _make_canned_evidence_response(verdicts1),
            _make_canned_evidence_response(verdicts2),
        ])
        llm_client = LLMClient(model="mock", api_caller=mock)

        with patch.object(
            IndexStore, "get_page_content", return_value=[
                _make_page_text(1, "hello world in text"),
            ]
        ):
            records = extractor.extract(
                parsed=parsed,
                candidates=[cand1, cand2],
                index_store=IndexStore(AgentConfig()),
                config=AgentConfig(),
                llm_client=llm_client,
            )

        # Option A records with "hello world" (normalized) - should be exactly 1
        a_records = [r for r in records if r.option == "A" and "hello" in _normalize_whitespace(r.quote)]
        assert len(a_records) == 1
        # Should keep the high-confidence one
        assert a_records[0].confidence == "high"

    # ------------------------------------------------------------------
    # LLM call failure fallback
    # ------------------------------------------------------------------

    def test_llm_call_exception_triggers_heuristic_fallback(self) -> None:
        """When an LLM call raises an exception, the heuristic fallback is used."""
        extractor = EvidenceExtractor()
        parsed = _make_parsed_question()
        candidate = _make_candidate()

        # MockApiCaller that raises on first call
        mock = MockApiCaller(
            raise_on_call={1: RuntimeError("simulated LLM failure")}
        )
        llm_client = LLMClient(model="mock", api_caller=mock, max_retries=0)

        with patch.object(
            IndexStore, "get_page_content", return_value=[
                _make_page_text(6, PAGE_6_TEXT),
            ]
        ):
            records = extractor.extract(
                parsed=parsed,
                candidates=[candidate],
                index_store=IndexStore(AgentConfig()),
                config=AgentConfig(),
                llm_client=llm_client,
            )

        # Should get coverage records via fallback
        assert len(records) >= 4
        for rec in records:
            assert rec.option in _ALL_OPTIONS
            assert rec.qid == "ins_a_001"

    # ------------------------------------------------------------------
    # Synthetic integration (real IndexStore pattern, mocked LLM)
    # ------------------------------------------------------------------

    def test_synthetic_integration_multiple_candidates(self) -> None:
        """Multiple candidates across different docs -> merged and deduped records."""
        extractor = EvidenceExtractor()
        parsed = _make_parsed_question()

        cand1 = _make_candidate(doc_id="1", node_id="n1", page_range="6-8")
        cand2 = _make_candidate(doc_id="2", node_id="n2", page_range="3-4")

        page_text_d1 = "Doc 1: 身故保险金 = max(已交保费, 基本保额×160%, 现金价值)"
        page_text_d2 = "Doc 2: 身故保险金仅为已交保费"

        verdicts_d1 = [
            {
                "option": "A",
                "evidence_type": "support",
                "quote": "已交保费",
                "normalized_fact": "已交保费是计算因子",
                "confidence": "high",
            },
            {
                "option": "B",
                "evidence_type": "support",
                "quote": "基本保额×160%",
                "normalized_fact": "基本保额160%是因子",
                "confidence": "high",
            },
            {
                "option": "C",
                "evidence_type": "support",
                "quote": "现金价值",
                "normalized_fact": "现金价值是因子",
                "confidence": "medium",
            },
            {
                "option": "D",
                "evidence_type": "support",
                "quote": "max(已交保费, 基本保额×160%, 现金价值)",
                "normalized_fact": "三者取大",
                "confidence": "high",
            },
        ]
        verdicts_d2 = [
            {
                "option": "A",
                "evidence_type": "support",
                "quote": "已交保费",
                "normalized_fact": "身故保险金为已交保费",
                "confidence": "high",
            },
            {
                "option": "B",
                "evidence_type": "refute",
                "quote": "仅为已交保费",
                "normalized_fact": "不包含基本保额",
                "confidence": "high",
            },
            {
                "option": "C",
                "evidence_type": "unclear",
                "quote": "",
                "normalized_fact": "",
                "confidence": "low",
            },
            {
                "option": "D",
                "evidence_type": "refute",
                "quote": "仅为已交保费",
                "normalized_fact": "不是三者取大",
                "confidence": "medium",
            },
        ]

        mock = MockApiCaller(responses=[
            _make_canned_evidence_response(verdicts_d1),
            _make_canned_evidence_response(verdicts_d2),
        ])
        llm_client = LLMClient(model="mock", api_caller=mock)

        with patch.object(IndexStore, "get_page_content") as mock_get:
            mock_get.side_effect = [
                [_make_page_text(6, page_text_d1)],
                [_make_page_text(3, page_text_d2)],
            ]

            records = extractor.extract(
                parsed=parsed,
                candidates=[cand1, cand2],
                index_store=IndexStore(AgentConfig()),
                config=AgentConfig(),
                llm_client=llm_client,
            )

        # Verify coverage
        for opt in _ALL_OPTIONS:
            assert any(r.option == opt for r in records), f"Missing option {opt}"

        # Option A: both docs have a record, but dedup on "已交保费" quote
        # They have same (doc_id=1 vs 2, pages, quote="已交保费") -> different doc_id so different keys
        a_records = [r for r in records if r.option == "A"]
        assert len(a_records) >= 1

        # All records should have valid fields
        for rec in records:
            assert rec.evidence_type in ("support", "refute", "unclear")
            assert rec.confidence in ("high", "medium", "low")


    # ------------------------------------------------------------------
    # Phase C: compact JSON prompt
    # ------------------------------------------------------------------

    def test_evidence_prompt_instructs_compact_json(self) -> None:
        """The system prompt demands short quotes (<=80 chars), short facts
        (<=40 chars), and only-JSON output with no markdown fences."""
        messages = _build_evidence_messages(
            question="测试问题",
            options={"A": "选项A", "B": "选项B", "C": "选项C", "D": "选项D"},
            page_text="测试页面内容",
            doc_id="1",
            node_title="测试节点",
        )
        system = messages[0]["content"]
        # Quote length instruction
        assert "≤80" in system or "80字" in system, (
            f"System prompt should cap quote length, got: {system}"
        )
        # Normalized fact length instruction
        assert "≤40" in system or "40字" in system, (
            f"System prompt should cap normalized_fact length, got: {system}"
        )
        # Only-JSON / no-markdown-fences instruction
        assert "只输出 JSON" in system or "不要加 markdown" in system or "不要加任何解释" in system, (
            f"System prompt should demand JSON-only output, got: {system}"
        )
        # Numbers constraint
        assert "0-2" in system, (
            f"System prompt should constrain numbers to 0-2 entries, got: {system}"
        )

    # ------------------------------------------------------------------
    # Phase C: max_tokens from config
    # ------------------------------------------------------------------

    def test_extract_from_node_llm_uses_config_evidence_max_tokens(self) -> None:
        """_extract_from_node_llm passes config.evidence_max_tokens to llm_client.chat."""
        extractor = EvidenceExtractor()
        parsed = _make_parsed_question()
        candidate = _make_candidate()

        mock = MockApiCaller(responses=[_make_canned_evidence_response(_make_support_verdicts())])
        llm_client = LLMClient(model="mock", api_caller=mock)

        config = AgentConfig(evidence_max_tokens=9999)

        with patch.object(
            IndexStore, "get_page_content", return_value=[
                _make_page_text(6, PAGE_6_TEXT),
            ]
        ):
            extractor._extract_from_node_llm(
                parsed=parsed,
                candidate=candidate,
                full_text=PAGE_6_TEXT,
                config=config,
                llm_client=llm_client,
                token_meter=None,
            )

        assert len(mock.calls) >= 1, "Expected at least one LLM call"
        actual_max_tokens = mock.calls[0]["kwargs"].get("max_tokens")
        assert actual_max_tokens == 9999, (
            f"Expected max_tokens=9999 from config, got {actual_max_tokens}"
        )


# ===========================================================================
# Light integration test (real IndexStore with temp files + mocked LLM)
# ===========================================================================


class TestEvidenceExtractionIntegration:
    """Light integration: real IndexStore on temp files + mocked LLM."""

    def test_integration_with_temp_index_store(self, tmp_path) -> None:
        """IndexStore reads from real temp files, LLM is mocked."""
        import json
        from pathlib import Path

        # Build a minimal AgentConfig pointing at tmp_path
        processed_root = tmp_path / "processed_data"
        config = AgentConfig(processed_root=processed_root)

        # Write pageindex files
        pi_dir = config.pageindex_dir
        pi_dir.mkdir(parents=True, exist_ok=True)
        tree = {"doc_name": "1", "line_count": 10, "structure": []}
        with open(pi_dir / "1.json", "w", encoding="utf-8") as f:
            json.dump(tree, f)
        spans = [{
            "node_id": "n1", "title": "Test",
            "source_page_range": "1-3",
            "start_line": 1, "end_line": 10,
            "start_page": 1, "end_page": 3,
            "bad": False, "index_source": "markdown",
        }]
        with open(pi_dir / "1.node_spans.json", "w", encoding="utf-8") as f:
            json.dump(spans, f)

        # Write page content
        pages_dir = config.pages_dir
        pages_dir.mkdir(parents=True, exist_ok=True)
        with open(pages_dir / "1.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"doc_id": "1", "page": 1, "text": PAGE_6_TEXT, "char_count": len(PAGE_6_TEXT)}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"doc_id": "1", "page": 2, "text": PAGE_7_TEXT, "char_count": len(PAGE_7_TEXT)}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"doc_id": "1", "page": 3, "text": PAGE_8_TEXT, "char_count": len(PAGE_8_TEXT)}, ensure_ascii=False) + "\n")

        store = IndexStore(config)

        extractor = EvidenceExtractor()
        parsed = _make_parsed_question()
        candidate = _make_candidate(page_range="1-3")
        llm_client = _make_mock_llm_client(_make_support_verdicts())

        records = extractor.extract(
            parsed=parsed,
            candidates=[candidate],
            index_store=store,
            config=config,
            llm_client=llm_client,
        )

        assert len(records) >= 4
        for rec in records:
            if rec.quote and rec.doc_id == "1":
                full_text = PAGE_6_TEXT + "\n" + PAGE_7_TEXT + "\n" + PAGE_8_TEXT
                assert rec.quote in full_text, (
                    f"Quote not in page text for option {rec.option}: {rec.quote!r}"
                )


# ===========================================================================
# TestOutputValidation — output artifact validation (Task 11)
# ===========================================================================


def _make_questions() -> list[dict]:
    """Return a minimal set of synthetic questions matching the real structure."""
    return [
        {"qid": "ins_a_001", "answer_format": "mcq"},
        {"qid": "ins_a_002", "answer_format": "mcq"},
        {"qid": "ins_a_003", "answer_format": "mcq"},
        {"qid": "ins_a_004", "answer_format": "mcq"},
        {"qid": "ins_a_005", "answer_format": "mcq"},
        {"qid": "ins_a_006", "answer_format": "mcq"},
        {"qid": "ins_a_007", "answer_format": "mcq"},
        {"qid": "ins_a_008", "answer_format": "mcq"},
        {"qid": "ins_a_009", "answer_format": "mcq"},
        {"qid": "ins_a_010", "answer_format": "mcq"},
        {"qid": "ins_a_011", "answer_format": "multi"},
        {"qid": "ins_a_012", "answer_format": "multi"},
        {"qid": "ins_a_013", "answer_format": "multi"},
        {"qid": "ins_a_014", "answer_format": "multi"},
        {"qid": "ins_a_015", "answer_format": "multi"},
        {"qid": "ins_a_016", "answer_format": "multi"},
        {"qid": "ins_a_017", "answer_format": "tf"},
        {"qid": "ins_a_018", "answer_format": "tf"},
        {"qid": "ins_a_019", "answer_format": "tf"},
        {"qid": "ins_a_020", "answer_format": "tf"},
    ]


def _make_valid_answer_csv_rows(
    questions: list[dict],
    *,
    summary_tokens: int = 5000,
    per_q_tokens: int = 250,
) -> list[dict[str, str]]:
    """Build a fully valid set of answer.csv rows."""
    rows: list[dict[str, str]] = []
    # summary row
    rows.append({
        "qid": "summary",
        "answer": "",
        "prompt_tokens": str(summary_tokens // 2),
        "completion_tokens": str(summary_tokens // 4),
        "total_tokens": str(summary_tokens),
    })
    # question rows
    for q in questions:
        fmt = q["answer_format"]
        if fmt == "mcq":
            answer = "A"
        elif fmt == "multi":
            answer = "AC"
        elif fmt == "tf":
            answer = "A"
        else:
            answer = "A"
        rows.append({
            "qid": q["qid"],
            "answer": answer,
            "prompt_tokens": str(per_q_tokens // 2),
            "completion_tokens": str(per_q_tokens // 4),
            "total_tokens": str(per_q_tokens),
        })
    return rows


def _make_valid_evidence_jsonl_lines(
    questions: list[dict],
) -> list[dict]:
    """Build fully valid evidence.jsonl records (traceable)."""
    lines: list[dict] = []
    for q in questions:
        fmt = q["answer_format"]
        if fmt == "mcq":
            answer = "A"
            selected_opts = ["A"]
        elif fmt == "multi":
            answer = "AC"
            selected_opts = ["A", "C"]
        elif fmt == "tf":
            answer = "A"
            selected_opts = ["A"]
        else:
            answer = "A"
            selected_opts = ["A"]

        evidence = []
        for opt in selected_opts:
            evidence.append({
                "qid": q["qid"],
                "doc_id": "1",
                "node_id": f"n_{opt}",
                "pages": "1-3",
                "option": opt,
                "evidence_type": "support",
                "quote": f"quote for {opt}",
                "normalized_fact": f"fact for {opt}",
                "confidence": "high",
            })
        # Add some unrelated evidence
        for opt in ("B", "D"):
            evidence.append({
                "qid": q["qid"],
                "doc_id": "1",
                "node_id": f"n_{opt}",
                "pages": "1-3",
                "option": opt,
                "evidence_type": "unclear",
                "quote": "",
                "normalized_fact": "",
                "confidence": "low",
            })

        lines.append({
            "qid": q["qid"],
            "answer": answer,
            "evidence": evidence,
            "option_judgements": {},
        })
    return lines


class TestOutputValidation:
    """Tests for scripts/validate_outputs.py core validation functions."""

    # ------------------------------------------------------------------
    # Valid inputs pass
    # ------------------------------------------------------------------

    def test_valid_answer_csv_passes(self) -> None:
        questions = _make_questions()
        rows = _make_valid_answer_csv_rows(questions)
        failures = validate_answer_csv_rows(rows, questions)
        assert failures == [], f"Expected no failures, got: {failures}"

    def test_valid_evidence_jsonl_passes(self) -> None:
        questions = _make_questions()
        lines = _make_valid_evidence_jsonl_lines(questions)
        failures = validate_evidence_jsonl_lines(lines, questions)
        assert failures == [], f"Expected no failures, got: {failures}"

    # ------------------------------------------------------------------
    # Summary row checks
    # ------------------------------------------------------------------

    def test_missing_summary_row(self) -> None:
        questions = _make_questions()
        # No "summary" qid in first position
        rows = [
            {"qid": "ins_a_001", "answer": "A",
             "prompt_tokens": "100", "completion_tokens": "50", "total_tokens": "150"},
        ] * 20
        failures = validate_answer_csv_rows(rows, questions)
        assert any("summary" in f.lower() for f in failures)

    def test_summary_wrong_qid_label(self) -> None:
        questions = _make_questions()
        rows = _make_valid_answer_csv_rows(questions)
        rows[0]["qid"] = "totals"  # wrong label
        failures = validate_answer_csv_rows(rows, questions)
        assert any("summary" in f for f in failures)

    # ------------------------------------------------------------------
    # QID coverage
    # ------------------------------------------------------------------

    def test_wrong_qid_count(self) -> None:
        questions = _make_questions()
        rows = _make_valid_answer_csv_rows(questions)
        # Remove one question row
        rows = [rows[0]] + rows[2:]  # drop ins_a_001
        failures = validate_answer_csv_rows(rows, questions)
        assert any("19" in f or "20" in f or "Expected" in f for f in failures)

    def test_missing_qid(self) -> None:
        questions = _make_questions()
        rows = _make_valid_answer_csv_rows(questions)
        # Rename a qid to simulate missing
        for r in rows:
            if r["qid"] == "ins_a_001":
                r["qid"] = "ins_a_099"
                break
        failures = validate_answer_csv_rows(rows, questions)
        assert any("missing" in f.lower() for f in failures) or any("Missing" in f for f in failures)

    # ------------------------------------------------------------------
    # Answer format validation
    # ------------------------------------------------------------------

    def test_bad_mcq_format_lowercase(self) -> None:
        questions = _make_questions()
        rows = _make_valid_answer_csv_rows(questions)
        rows[1]["answer"] = "a"  # lowercase is invalid for mcq
        failures = validate_answer_csv_rows(rows, questions)
        assert any("mcq" in f.lower() for f in failures)

    def test_bad_mcq_format_multiple_letters(self) -> None:
        questions = _make_questions()
        rows = _make_valid_answer_csv_rows(questions)
        rows[1]["answer"] = "AB"  # mcq must be single letter
        failures = validate_answer_csv_rows(rows, questions)
        assert any("mcq" in f.lower() for f in failures)

    def test_bad_multi_format_duplicates(self) -> None:
        questions = _make_questions()
        rows = _make_valid_answer_csv_rows(questions)
        # Find a multi row
        for i, r in enumerate(rows):
            q = next((q for q in questions if q["qid"] == r["qid"]), None)
            if q and q["answer_format"] == "multi":
                rows[i]["answer"] = "AAC"  # duplicates
                break
        failures = validate_answer_csv_rows(rows, questions)
        assert any("duplicate" in f.lower() or "no duplicates" in f.lower() for f in failures)

    def test_bad_multi_format_not_sorted(self) -> None:
        questions = _make_questions()
        rows = _make_valid_answer_csv_rows(questions)
        for i, r in enumerate(rows):
            q = next((q for q in questions if q["qid"] == r["qid"]), None)
            if q and q["answer_format"] == "multi":
                rows[i]["answer"] = "CA"  # not sorted
                break
        failures = validate_answer_csv_rows(rows, questions)
        assert any("sorted" in f.lower() for f in failures)

    def test_bad_multi_format_with_spaces(self) -> None:
        questions = _make_questions()
        rows = _make_valid_answer_csv_rows(questions)
        for i, r in enumerate(rows):
            q = next((q for q in questions if q["qid"] == r["qid"]), None)
            if q and q["answer_format"] == "multi":
                rows[i]["answer"] = "A C"  # spaces not allowed
                break
        failures = validate_answer_csv_rows(rows, questions)
        assert any(failures)

    def test_bad_tf_format(self) -> None:
        questions = _make_questions()
        rows = _make_valid_answer_csv_rows(questions)
        for i, r in enumerate(rows):
            q = next((q for q in questions if q["qid"] == r["qid"]), None)
            if q and q["answer_format"] == "tf":
                rows[i]["answer"] = "C"  # must be A or B
                break
        failures = validate_answer_csv_rows(rows, questions)
        assert any("A' or 'B'" in f for f in failures)

    def test_empty_answer(self) -> None:
        questions = _make_questions()
        rows = _make_valid_answer_csv_rows(questions)
        rows[1]["answer"] = ""
        failures = validate_answer_csv_rows(rows, questions)
        assert any("empty" in f.lower() for f in failures)

    # ------------------------------------------------------------------
    # Token sum check
    # ------------------------------------------------------------------

    def test_summary_token_mismatch(self) -> None:
        questions = _make_questions()
        rows = _make_valid_answer_csv_rows(questions, summary_tokens=9999, per_q_tokens=100)
        failures = validate_answer_csv_rows(rows, questions)
        assert any("total_tokens" in f for f in failures)

    def test_token_sum_matches(self) -> None:
        """When summary matches sum of per-question tokens, no failure."""
        questions = _make_questions()
        per_q = 250
        rows = _make_valid_answer_csv_rows(
            questions, summary_tokens=per_q * len(questions), per_q_tokens=per_q
        )
        failures = validate_answer_csv_rows(rows, questions)
        assert failures == [], f"Expected no failures, got: {failures}"

    # ------------------------------------------------------------------
    # Evidence traceability
    # ------------------------------------------------------------------

    def test_selected_option_lacks_support_evidence(self) -> None:
        questions = _make_questions()
        lines = _make_valid_evidence_jsonl_lines(questions)
        # Change the first mcq question's evidence: remove support for its answer
        lines[0]["answer"] = "B"  # but evidence has support for A, not B
        # Remove support evidence for B
        lines[0]["evidence"] = [e for e in lines[0]["evidence"] if e["option"] != "B"]
        failures = validate_evidence_jsonl_lines(lines, questions)
        assert any("no support evidence" in f.lower() for f in failures)

    def test_each_multi_letter_needs_support(self) -> None:
        questions = _make_questions()
        lines = _make_valid_evidence_jsonl_lines(questions)
        # Multi question: answer "AC" but remove support for C
        for line in lines:
            q = next((q for q in questions if q["qid"] == line["qid"]), None)
            if q and q["answer_format"] == "multi":
                # Remove support for C
                line["evidence"] = [e for e in line["evidence"]
                                    if not (e["option"] == "C" and e["evidence_type"] == "support")]
                break
        failures = validate_evidence_jsonl_lines(lines, questions)
        assert any("no support evidence" in f.lower() for f in failures)

    def test_missing_qid_in_evidence(self) -> None:
        questions = _make_questions()
        lines = _make_valid_evidence_jsonl_lines(questions)
        # Remove one line
        lines = lines[:-1]
        failures = validate_evidence_jsonl_lines(lines, questions)
        assert any("missing record" in f.lower() for f in failures)

    # ------------------------------------------------------------------
    # usage.jsonl validation
    # ------------------------------------------------------------------

    def test_usage_jsonl_missing_qid(self) -> None:
        questions = _make_questions()
        # Only include half the qids
        lines = [{"qid": q["qid"]} for q in questions[:-1]]
        failures = validate_usage_jsonl_lines(lines, questions)
        assert any("missing qids" in f.lower() for f in failures)

    def test_usage_jsonl_all_present(self) -> None:
        questions = _make_questions()
        lines = [{"qid": q["qid"]} for q in questions]
        failures = validate_usage_jsonl_lines(lines, questions)
        assert failures == []

    # ------------------------------------------------------------------
    # Edge: empty inputs
    # ------------------------------------------------------------------

    def test_empty_csv_rows(self) -> None:
        questions = _make_questions()
        failures = validate_answer_csv_rows([], questions)
        assert any(failures)
        assert any("empty" in f.lower() for f in failures)
