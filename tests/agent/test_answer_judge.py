"""Tests for agent/answer_judge.py — deterministic, rule-based answer judging.

All tests use synthetic evidence + calculations; no LLM calls, no network.
"""

from __future__ import annotations

import pytest

from agent.answer_judge import (
    AnswerJudge,
    _build_option_judgements,
    _normalize_mcq,
    _normalize_multi,
    _normalize_tf,
)
from agent.schemas import CalculationRecord, EvidenceRecord, ParsedQuestion


# ===========================================================================
# Helpers
# ===========================================================================


def _make_parsed(fmt: str = "mcq", **overrides: object) -> ParsedQuestion:
    """Build a minimal ParsedQuestion with sensible defaults."""
    defaults: dict = {
        "qid": "ins_a_001",
        "domain": "insurance",
        "split": "A",
        "question": "测试问题？",
        "options": {"A": "选项A", "B": "选项B", "C": "选项C", "D": "选项D"},
        "answer_format": fmt,
        "type": "推理判断",
        "doc_ids": ["1"],
    }
    defaults.update(overrides)
    return ParsedQuestion(**defaults)


def _support_rec(option: str, confidence: str = "high", **overrides: object) -> EvidenceRecord:
    """Shorthand for a support EvidenceRecord."""
    defaults: dict = {
        "qid": "ins_a_001",
        "doc_id": "1",
        "node_id": f"n_{option}",
        "pages": "1-3",
        "option": option,
        "evidence_type": "support",
        "quote": f"quote for {option}",
        "normalized_fact": f"fact for {option}",
        "confidence": confidence,
    }
    defaults.update(overrides)
    return EvidenceRecord(**defaults)


def _refute_rec(option: str, confidence: str = "medium", **overrides: object) -> EvidenceRecord:
    """Shorthand for a refute EvidenceRecord."""
    defaults: dict = {
        "qid": "ins_a_001",
        "doc_id": "1",
        "node_id": f"n_{option}",
        "pages": "1-3",
        "option": option,
        "evidence_type": "refute",
        "quote": f"refute quote for {option}",
        "normalized_fact": f"refute fact for {option}",
        "confidence": confidence,
    }
    defaults.update(overrides)
    return EvidenceRecord(**defaults)


def _unclear_rec(option: str) -> EvidenceRecord:
    """Shorthand for an unclear EvidenceRecord."""
    return EvidenceRecord(
        qid="ins_a_001",
        doc_id="1",
        node_id=f"n_{option}",
        pages="1-3",
        option=option,
        evidence_type="unclear",
        quote="",
        normalized_fact="",
        confidence="low",
    )


# ===========================================================================
# Normalisation helpers
# ===========================================================================


class TestNormalisation:
    def test_normalize_mcq_single_letter(self) -> None:
        assert _normalize_mcq("B", ["A", "B", "C", "D"]) == "B"

    def test_normalize_mcq_lowercase(self) -> None:
        assert _normalize_mcq("b", ["A", "B", "C", "D"]) == "B"

    def test_normalize_mcq_with_punctuation(self) -> None:
        assert _normalize_mcq("B,", ["A", "B", "C", "D"]) == "B"

    def test_normalize_mcq_with_spaces(self) -> None:
        assert _normalize_mcq("  C  ", ["A", "B", "C", "D"]) == "C"

    def test_normalize_mcq_invalid_falls_back(self) -> None:
        result = _normalize_mcq("XYZ", ["A", "B", "C", "D"])
        assert len(result) == 1
        assert result in "ABCD"

    def test_normalize_mcq_empty_falls_back(self) -> None:
        result = _normalize_mcq("", ["A", "B", "C", "D"])
        assert len(result) == 1
        assert result in "ABCD"

    def test_normalize_multi_sorts_and_dedups(self) -> None:
        assert _normalize_multi("C A C", ["A", "B", "C", "D"]) == "AC"

    def test_normalize_multi_lowercase(self) -> None:
        assert _normalize_multi("ca", ["A", "B", "C", "D"]) == "AC"

    def test_normalize_multi_with_commas(self) -> None:
        assert _normalize_multi("A,C", ["A", "B", "C", "D"]) == "AC"

    def test_normalize_multi_empty(self) -> None:
        assert _normalize_multi("", ["A", "B", "C", "D"]) == ""

    def test_normalize_tf_a(self) -> None:
        assert _normalize_tf("A") == "A"

    def test_normalize_tf_b(self) -> None:
        assert _normalize_tf("B") == "B"

    def test_normalize_tf_lowercase(self) -> None:
        assert _normalize_tf("a") == "A"

    def test_normalize_tf_empty_fallback(self) -> None:
        assert _normalize_tf("") == "A"  # best-guess


# ===========================================================================
# Option judgements
# ===========================================================================


class TestOptionJudgements:
    def test_counts_and_scores(self) -> None:
        evidence = [
            _support_rec("A", "high"),
            _support_rec("A", "medium"),
            _refute_rec("A", "high"),
            _unclear_rec("A"),
            _support_rec("B", "high"),
            _refute_rec("B", "medium"),
            _refute_rec("B", "low"),
        ]
        judgements = _build_option_judgements(evidence, ["A", "B", "C", "D"])

        # Option A: 2 support (high=3, medium=2) = 5; 1 refute (high=3) = 3; net=2
        assert judgements["A"]["support_count"] == 2
        assert judgements["A"]["refute_count"] == 1
        assert judgements["A"]["unclear_count"] == 1
        assert judgements["A"]["support_score"] == 5
        assert judgements["A"]["refute_score"] == 3
        assert judgements["A"]["net_score"] == 2

        # Option B: 1 support (high=3) = 3; 2 refute (medium=2, low=1) = 3; net=0
        assert judgements["B"]["support_count"] == 1
        assert judgements["B"]["refute_count"] == 2
        assert judgements["B"]["support_score"] == 3
        assert judgements["B"]["refute_score"] == 3
        assert judgements["B"]["net_score"] == 0

        # Options C, D: no evidence records
        for opt in ("C", "D"):
            assert judgements[opt]["support_count"] == 0
            assert judgements[opt]["refute_count"] == 0
            assert judgements[opt]["net_score"] == 0

    def test_all_options_present(self) -> None:
        evidence: list[EvidenceRecord] = []
        judgements = _build_option_judgements(evidence, ["A", "B", "C", "D"])
        assert set(judgements.keys()) == {"A", "B", "C", "D"}


# ===========================================================================
# MCQ format
# ===========================================================================


class TestMcqJudging:
    def test_strong_support_for_b_selects_b(self) -> None:
        judge = AnswerJudge()
        parsed = _make_parsed("mcq")
        evidence = [
            _support_rec("A", "low"),
            _support_rec("B", "high"),
            _support_rec("B", "high"),
            _support_rec("B", "medium"),
            _unclear_rec("C"),
            _unclear_rec("D"),
        ]
        result = judge.judge(parsed, evidence)
        assert result.answer == "B"

    def test_tie_break_by_support_score_then_first_letter(self) -> None:
        """A and B both have net_score=3 (one high support each).
        A also has an unclear record — doesn't affect score.
        Tie-break: same support_score (3 vs 3), A comes first alphabetically.
        """
        judge = AnswerJudge()
        parsed = _make_parsed("mcq")
        evidence = [
            _support_rec("A", "high"),
            _support_rec("B", "high"),
        ]
        result = judge.judge(parsed, evidence)
        # Both have net_score=3, support_score=3; A < B alphabetically
        assert result.answer == "A"

    def test_tie_break_by_higher_support_score(self) -> None:
        """A has medium support (2), B has high support (3).
        Both net=0 from support only. B wins by higher support_score.
        """
        judge = AnswerJudge()
        parsed = _make_parsed("mcq")
        evidence = [
            _support_rec("A", "medium"),
            _support_rec("B", "high"),
        ]
        result = judge.judge(parsed, evidence)
        assert result.answer == "B"

    def test_all_unclear_best_guess_with_warning(self) -> None:
        judge = AnswerJudge()
        parsed = _make_parsed("mcq")
        evidence = [
            _unclear_rec("A"),
            _unclear_rec("B"),
            _unclear_rec("C"),
            _unclear_rec("D"),
        ]
        result = judge.judge(parsed, evidence)
        # Should still return a single letter
        assert result.answer in ("A", "B", "C", "D")
        assert len(result.answer) == 1
        assert any("answer_unclear" in w for w in result.warnings)

    def test_option_judgements_populated(self) -> None:
        judge = AnswerJudge()
        parsed = _make_parsed("mcq")
        evidence = [
            _support_rec("A", "high"),
            _refute_rec("B", "medium"),
            _unclear_rec("C"),
            _unclear_rec("D"),
        ]
        result = judge.judge(parsed, evidence)
        assert "A" in result.option_judgements
        assert result.option_judgements["A"]["support_count"] == 1
        assert result.option_judgements["B"]["refute_count"] == 1
        assert result.option_judgements["C"]["unclear_count"] >= 1

    def test_selected_nodes_from_support_evidence(self) -> None:
        judge = AnswerJudge()
        parsed = _make_parsed("mcq")
        evidence = [
            _support_rec("B", "high", node_id="n1"),
            _support_rec("B", "high", node_id="n2"),
            _support_rec("A", "low", node_id="nA"),
        ]
        result = judge.judge(parsed, evidence)
        assert result.answer == "B"
        assert "n1" in result.selected_nodes
        assert "n2" in result.selected_nodes
        assert "nA" not in result.selected_nodes


# ===========================================================================
# Multi format
# ===========================================================================


class TestMultiJudging:
    def test_ac_support_selects_ac(self) -> None:
        judge = AnswerJudge()
        parsed = _make_parsed("multi")
        evidence = [
            _support_rec("A", "high"),
            _support_rec("C", "high"),
            _refute_rec("B", "high"),
            _unclear_rec("D"),
        ]
        result = judge.judge(parsed, evidence)
        assert result.answer == "AC"

    def test_sorted_no_separators(self) -> None:
        """Even if we'd naturally get "CA", the answer is sorted to "AC"."""
        judge = AnswerJudge()
        parsed = _make_parsed("multi")
        evidence = [
            _support_rec("C", "high"),
            _support_rec("A", "high"),
            _refute_rec("B", "high"),
            _refute_rec("D", "high"),
        ]
        result = judge.judge(parsed, evidence)
        assert result.answer == "AC"

    def test_dedup_sorted(self) -> None:
        """Redundant letters are deduplicated."""
        judge = AnswerJudge()
        parsed = _make_parsed("multi")
        evidence = [
            _support_rec("A", "high"),
            _support_rec("C", "high"),
            _support_rec("A", "medium"),  # duplicate A
            _refute_rec("B", "high"),
            _refute_rec("D", "high"),
        ]
        result = judge.judge(parsed, evidence)
        assert result.answer == "AC"

    def test_none_qualify_fallback_single_best(self) -> None:
        """When no option has net_score > 0, fall back to the single best."""
        judge = AnswerJudge()
        parsed = _make_parsed("multi")
        evidence = [
            _refute_rec("A", "high"),
            _refute_rec("B", "medium"),
            _support_rec("C", "low"),  # net=1
            _refute_rec("D", "high"),
        ]
        result = judge.judge(parsed, evidence)
        # C is the only one with positive net_score
        assert result.answer == "C"

    def test_all_refuted_fallback_with_warning(self) -> None:
        judge = AnswerJudge()
        parsed = _make_parsed("multi")
        evidence = [
            _refute_rec("A", "high"),
            _refute_rec("B", "high"),
            _refute_rec("C", "high"),
            _refute_rec("D", "high"),
        ]
        result = judge.judge(parsed, evidence)
        assert len(result.answer) >= 1
        assert all(c in "ABCD" for c in result.answer)
        assert any("answer_unclear" in w for w in result.warnings)

    def test_multi_only_positive_net_selected(self) -> None:
        """Options with net_score == 0 are NOT selected in multi."""
        judge = AnswerJudge()
        parsed = _make_parsed("multi")
        evidence = [
            _support_rec("A", "high"),  # net=3
            _support_rec("B", "low"),  # net=1
            _support_rec("C", "medium"),  # net=2
            _refute_rec("C", "high"),  # C now: support=medium(2) - refute=high(3) = -1
            _unclear_rec("D"),
        ]
        result = judge.judge(parsed, evidence)
        # A net=3, B net=1, C net=-1, D net=0
        # Only A and B have net > 0
        assert result.answer == "AB"


# ===========================================================================
# TF format
# ===========================================================================


class TestTfJudging:
    def test_a_supported_selects_a(self) -> None:
        judge = AnswerJudge()
        parsed = _make_parsed(
            "tf",
            options={"A": "正确", "B": "错误"},
        )
        evidence = [
            _support_rec("A", "high"),
            _refute_rec("B", "medium"),
        ]
        result = judge.judge(parsed, evidence)
        assert result.answer == "A"

    def test_b_supported_selects_b(self) -> None:
        judge = AnswerJudge()
        parsed = _make_parsed(
            "tf",
            options={"A": "正确", "B": "错误"},
        )
        evidence = [
            _refute_rec("A", "high"),
            _support_rec("B", "high"),
        ]
        result = judge.judge(parsed, evidence)
        assert result.answer == "B"

    def test_tf_neither_supported_best_guess(self) -> None:
        judge = AnswerJudge()
        parsed = _make_parsed(
            "tf",
            options={"A": "正确", "B": "错误"},
        )
        evidence = [
            _unclear_rec("A"),
            _unclear_rec("B"),
        ]
        result = judge.judge(parsed, evidence)
        assert result.answer in ("A", "B")
        assert any("answer_unclear" in w for w in result.warnings)

    def test_tf_always_returns_a_or_b(self) -> None:
        judge = AnswerJudge()
        parsed = _make_parsed(
            "tf",
            options={"A": "正确", "B": "错误"},
        )
        evidence: list[EvidenceRecord] = []
        result = judge.judge(parsed, evidence)
        assert result.answer in ("A", "B")


# ===========================================================================
# Calculation-informed judging
# ===========================================================================


class TestCalculationInformedJudging:
    def test_ranking_calc_matches_option_b_overrides(self) -> None:
        """When the ranking calculation's ranked_order matches option B's text,
        the answer is B even if evidence weakly supports A."""
        judge = AnswerJudge()
        parsed = _make_parsed(
            "mcq",
            options={
                "A": "产品X(100万) > 产品Y(80万) > 产品Z(60万)",
                "B": "产品Y(90万) > 产品X(85万) > 产品Z(60万)",
                "C": "产品Z(60万) > 产品Y(50万) > 产品X(40万)",
                "D": "产品X(100万) = 产品Y(100万) > 产品Z(60万)",
            },
        )
        # Evidence weakly supports A
        evidence = [
            _support_rec("A", "low"),
            _unclear_rec("B"),
            _unclear_rec("C"),
            _unclear_rec("D"),
        ]
        # But calculation says ranked order is: 产品Y, 产品X, 产品Z (matches B's text)
        calc = CalculationRecord(
            qid="ins_a_001",
            calc_type="ranking",
            inputs={
                "ranked_order": ["产品Y", "产品X", "产品Z"],
                "ranked_values": {"产品Y": "90万", "产品X": "85万", "产品Z": "60万"},
                "product_values": {"产品X": "85", "产品Y": "90", "产品Z": "60"},
            },
            formula="产品Y(90万) > 产品X(85万) > 产品Z(60万)",
            result=0.0,
            source_evidence_ids=["1/n1"],
        )
        result = judge.judge(parsed, evidence, [calc])
        # Calculation should override to B
        assert result.answer == "B"
        assert any("calculation_override" in fb for fb in result.fallbacks)

    def test_ranking_calc_ambiguous_no_override(self) -> None:
        """When the ranking calculation result is ambiguous (matches multiple
        options equally), the evidence-based decision stands."""
        judge = AnswerJudge()
        parsed = _make_parsed(
            "mcq",
            options={
                "A": "产品X(100万) > 产品Y(80万)",
                "B": "产品X(100万) > 产品Y(80万)",
                "C": "产品Z(60万)",
                "D": "产品W(40万)",
            },
        )
        evidence = [
            _support_rec("A", "high"),
            _unclear_rec("B"),
            _unclear_rec("C"),
            _unclear_rec("D"),
        ]
        calc = CalculationRecord(
            qid="ins_a_001",
            calc_type="ranking",
            inputs={
                "ranked_order": ["产品X", "产品Y"],
                "ranked_values": {"产品X": "100万", "产品Y": "80万"},
            },
            formula="产品X(100万) > 产品Y(80万)",
            result=0.0,
            source_evidence_ids=["1/n1"],
        )
        result = judge.judge(parsed, evidence, [calc])
        # Both A and B match equally; evidence selects A
        assert result.answer == "A"


# ===========================================================================
# AnswerRecord population
# ===========================================================================


class TestAnswerRecordPopulation:
    def test_fields_populated_correctly(self) -> None:
        judge = AnswerJudge()
        parsed = _make_parsed("mcq")
        evidence = [
            _support_rec("B", "high", node_id="n1"),
            _support_rec("B", "medium", node_id="n2"),
        ]
        result = judge.judge(parsed, evidence)
        assert result.qid == "ins_a_001"
        assert result.answer == "B"
        assert result.candidate_docs == ["1"]
        assert "n1" in result.selected_nodes
        assert len(result.evidence) == 2
        assert isinstance(result.usage, dict)
        assert isinstance(result.fallbacks, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.option_judgements, dict)

    def test_calculations_serialized(self) -> None:
        judge = AnswerJudge()
        parsed = _make_parsed("mcq")
        evidence = [_support_rec("A", "high")]
        calc = CalculationRecord(
            qid="ins_a_001",
            calc_type="medical_payout",
            inputs={"a": 1},
            formula="1+1=2",
            result=2.0,
            unit="元",
            source_evidence_ids=["1/n1"],
        )
        result = judge.judge(parsed, evidence, [calc])
        assert len(result.calculations) == 1
        assert result.calculations[0]["calc_type"] == "medical_payout"
        assert result.calculations[0]["result"] == 2.0

    def test_empty_calculations_ok(self) -> None:
        judge = AnswerJudge()
        parsed = _make_parsed("mcq")
        evidence = [_support_rec("A", "high")]
        result = judge.judge(parsed, evidence, None)
        assert result.answer == "A"
        assert result.calculations == []


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_empty_evidence_still_produces_answer(self) -> None:
        judge = AnswerJudge()
        parsed = _make_parsed("mcq")
        result = judge.judge(parsed, [])
        assert result.answer in ("A", "B", "C", "D")
        assert len(result.answer) == 1

    def test_tf_only_options(self) -> None:
        judge = AnswerJudge()
        parsed = _make_parsed("tf", options={"A": "True", "B": "False"})
        evidence = [_support_rec("A", "high")]
        result = judge.judge(parsed, evidence)
        assert result.answer == "A"

    def test_confidence_weights_applied_correctly(self) -> None:
        """high=3, medium=2, low=1."""
        judge = AnswerJudge()
        parsed = _make_parsed("mcq")
        evidence = [
            _support_rec("A", "high"),  # score 3
            _support_rec("B", "medium"),  # score 2
            _support_rec("B", "medium"),  # score 2 -> total 4
            _unclear_rec("C"),
            _unclear_rec("D"),
        ]
        result = judge.judge(parsed, evidence)
        # B: net=4 > A: net=3
        assert result.answer == "B"

    def test_refute_reduces_net_score(self) -> None:
        """A has one high support (3) and one high refute (3) = net 0.
        B has one low support (1) = net 1. B wins."""
        judge = AnswerJudge()
        parsed = _make_parsed("mcq")
        evidence = [
            _support_rec("A", "high"),
            _refute_rec("A", "high"),
            _support_rec("B", "low"),
        ]
        result = judge.judge(parsed, evidence)
        assert result.answer == "B"


# ===========================================================================
# Phase D: Computation-derived verdicts with synthesized support evidence
# ===========================================================================


class TestComputationDerivedVerdicts:
    """Tests that the judge derives answers from calculation results and
    synthesizes support EvidenceRecords for the selected option."""

    def test_ranking_derives_answer_b_with_support_evidence(self):
        """ins_a_001 scenario: ranking calc identifies option B, and the
        judge synthesizes support EvidenceRecords with non-empty quotes."""
        judge = AnswerJudge()
        parsed = _make_parsed(
            "mcq",
            qid="ins_a_001",
            question="根据以下产品身故保险金按降序排列，正确的排序是？",
            options={
                "A": "平安智盈金生(90万) > 国寿增益宝(144万) > 平安富鸿金生(85万) > 国寿鑫享添盈(80万)",
                "B": "国寿增益宝(144万) > 平安智盈金生(90万) > 平安富鸿金生(85万) > 国寿鑫享添盈(80万)",
                "C": "平安富鸿金生(85万) > 国寿鑫享添盈(80万) > 国寿增益宝(144万) > 平安智盈金生(90万)",
                "D": "国寿鑫享添盈(80万) > 平安富鸿金生(85万) > 平安智盈金生(90万) > 国寿增益宝(144万)",
            },
        )

        # Calculation with supporting facts embedded
        calc = CalculationRecord(
            qid="ins_a_001",
            calc_type="ranking",
            inputs={
                "ranked_order": [
                    "国寿增益宝", "平安智盈金生", "平安富鸿金生", "国寿鑫享添盈",
                ],
                "ranked_values": {
                    "国寿增益宝": "1440000",
                    "平安智盈金生": "900000",
                    "平安富鸿金生": "850000",
                    "国寿鑫享添盈": "800000",
                },
                "product_values": {
                    "国寿增益宝": "1440000",
                    "平安智盈金生": "900000",
                    "平安富鸿金生": "850000",
                    "国寿鑫享添盈": "800000",
                },
                "supporting_facts": [
                    {
                        "product": "国寿增益宝",
                        "field": "身故保险金",
                        "formula_or_value": "基本保额*1.6",
                        "quote": "按基本保额的160%给付身故保险金",
                        "source_doc_id": "2",
                        "source_node_id": "n2",
                        "source_pages": "4-6",
                    },
                    {
                        "product": "平安智盈金生",
                        "field": "身故保险金",
                        "formula_or_value": "保单账户价值",
                        "quote": "保单账户价值为90万元",
                        "source_doc_id": "1",
                        "source_node_id": "n1",
                        "source_pages": "1-3",
                    },
                    {
                        "product": "平安富鸿金生",
                        "field": "身故保险金",
                        "formula_or_value": "已交保费-已领养老年金",
                        "quote": "按已交保费减去已领年金给付",
                        "source_doc_id": "16",
                        "source_node_id": "n16",
                        "source_pages": "3-4",
                    },
                    {
                        "product": "国寿鑫享添盈",
                        "field": "身故保险金",
                        "formula_or_value": "已交保费-已领养老年金",
                        "quote": "已交保费扣除已领养老年金",
                        "source_doc_id": "15",
                        "source_node_id": "n15",
                        "source_pages": "1-2",
                    },
                ],
            },
            formula="...",
            result=0.0,
            source_evidence_ids=["1/n1", "2/n2", "15/n15", "16/n16"],
        )

        # Start with empty evidence — judge must synthesize support from calc
        result = judge.judge(parsed, [], [calc])

        # Answer must be B
        assert result.answer == "B", f"Expected B, got {result.answer}"

        # Must have support evidence for B
        support_for_b = [
            e for e in result.evidence
            if e.option == "B" and e.evidence_type == "support"
        ]
        assert len(support_for_b) >= 1, (
            f"Expected >=1 support evidence for B, got {len(support_for_b)}"
        )

        # At least one support evidence must have a non-empty quote
        has_quote = any(e.quote for e in support_for_b)
        assert has_quote, "Support evidence for B must have non-empty quote"

        # Verify the calculation_override fallback is present
        assert any("calculation_override" in fb for fb in result.fallbacks), (
            "Should have calculation_override fallback"
        )

    def test_ranking_derives_answer_with_synthesized_evidence_fields(self):
        """Synthesized EvidenceRecords have correct doc_id/node_id/pages from facts."""
        judge = AnswerJudge()
        parsed = _make_parsed(
            "mcq",
            qid="test",
            question="排序问题？",
            options={
                "A": "产品A > 产品B",
                "B": "产品B > 产品A",
                "C": "产品A = 产品B",
                "D": "无法比较",
            },
        )

        calc = CalculationRecord(
            qid="test",
            calc_type="ranking",
            inputs={
                "ranked_order": ["产品B", "产品A"],
                "ranked_values": {"产品B": "100", "产品A": "50"},
                "product_values": {"产品B": "100", "产品A": "50"},
                "supporting_facts": [
                    {
                        "product": "产品B",
                        "field": "身故保险金",
                        "formula_or_value": "保单账户价值",
                        "quote": "产品B的账户价值为100万",
                        "source_doc_id": "10",
                        "source_node_id": "n10",
                        "source_pages": "5-7",
                    },
                ],
            },
            formula="...",
            result=0.0,
            source_evidence_ids=["10/n10"],
        )

        result = judge.judge(parsed, [], [calc])
        assert result.answer == "B"

        # Check synthesized evidence fields
        support_evidence = [
            e for e in result.evidence if e.evidence_type == "support"
        ]
        assert len(support_evidence) >= 1
        se = support_evidence[0]
        assert se.option == "B"
        assert se.doc_id == "10"
        assert se.node_id == "n10"
        assert se.pages == "5-7"
        assert se.quote == "产品B的账户价值为100万"
        assert "产品B" in se.normalized_fact
        assert se.confidence == "high"

    def test_medical_calc_without_ranking_uses_evidence_path(self):
        """Medical calculations without ranked_order should still work via
        evidence-based path (no override)."""
        judge = AnswerJudge()
        parsed = _make_parsed("mcq")
        evidence = [
            _support_rec("C", "high"),
            _unclear_rec("A"),
            _unclear_rec("B"),
            _unclear_rec("D"),
        ]
        calc = CalculationRecord(
            qid="test",
            calc_type="medical_payout",
            inputs={
                "product": "测试产品",
                "deductible": 5000,
                "supporting_facts": [
                    {
                        "product": "测试产品",
                        "field": "免赔额",
                        "formula_or_value": "5000",
                        "quote": "年度免赔额5000元",
                        "source_doc_id": "5",
                        "source_node_id": "n5",
                        "source_pages": "3-4",
                    },
                ],
            },
            formula="...",
            result=45000,
            unit="元",
            source_evidence_ids=["5/n5"],
        )
        result = judge.judge(parsed, evidence, [calc])
        # C has high support, medical calc doesn't override (no ranking)
        assert result.answer == "C"

        # But synthesized support should still be present for C from evidence
        support_for_c = [
            e for e in result.evidence
            if e.option == "C" and e.evidence_type == "support"
        ]
        assert len(support_for_c) >= 1

    def test_no_calculations_no_synthesis(self):
        """Without calculations, no synthetic evidence is created."""
        judge = AnswerJudge()
        parsed = _make_parsed("mcq")
        evidence = [_support_rec("A", "high")]
        result = judge.judge(parsed, evidence, [])
        assert result.answer == "A"
        # All evidence should be from the input, not synthesized
        assert len(result.evidence) == 1
        assert result.evidence[0].option == "A"

    def test_synthesized_evidence_passes_has_support_check(self):
        """The synthesized support evidence satisfies _has_support_for_answer.
        This is critical for the validator."""
        from agent.pipeline import _has_support_for_answer

        judge = AnswerJudge()
        parsed = _make_parsed(
            "mcq",
            qid="test",
            question="排序问题？",
            options={
                "A": "产品A > 产品B > 产品C > 产品D",
                "B": "产品D > 产品C > 产品B > 产品A",
                "C": "无法排序",
                "D": "产品B > 产品D > 产品A > 产品C",
            },
        )

        calc = CalculationRecord(
            qid="test",
            calc_type="ranking",
            inputs={
                "ranked_order": ["产品D", "产品C", "产品B", "产品A"],
                "ranked_values": {"产品D": "400", "产品C": "300", "产品B": "200", "产品A": "100"},
                "product_values": {"产品D": "400", "产品C": "300", "产品B": "200", "产品A": "100"},
                "supporting_facts": [
                    {
                        "product": "产品D",
                        "field": "身故保险金",
                        "formula_or_value": "400",
                        "quote": "产品D给付400万元",
                        "source_doc_id": "4",
                        "source_node_id": "n4",
                        "source_pages": "1-2",
                    },
                ],
            },
            formula="...",
            result=0.0,
            source_evidence_ids=["4/n4"],
        )

        result = judge.judge(parsed, [], [calc])

        # Check that has_support_for_answer passes
        assert _has_support_for_answer(result, parsed), (
            "Synthesized evidence should satisfy support check"
        )
