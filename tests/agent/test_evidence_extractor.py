"""Tests for agent/evidence_extractor.py — fact-matrix extraction (Phase D).

All tests use mocked LLM; NO network calls.
"""

from __future__ import annotations

import json

import pytest

from agent.config import AgentConfig
from agent.evidence_extractor import (
    EvidenceExtractor,
    _build_fact_matrix_messages,
)
from agent.index_store import IndexStore
from agent.llm_client import LLMClient, MockApiCaller
from agent.schemas import CandidateNode, FactRecord, ParsedQuestion
from agent.token_meter import TokenMeter


# ===========================================================================
# Helpers
# ===========================================================================


def _make_parsed(**overrides: object) -> ParsedQuestion:
    """Build a minimal ParsedQuestion for fact extraction tests."""
    defaults: dict = {
        "qid": "ins_a_001",
        "domain": "insurance",
        "split": "A",
        "question": "关于身故保险金的排序问题？",
        "options": {"A": "选项A", "B": "选项B", "C": "选项C", "D": "选项D"},
        "answer_format": "mcq",
        "type": "推理判断",
        "doc_ids": ["1"],
    }
    defaults.update(overrides)
    return ParsedQuestion(**defaults)


def _make_candidate(
    doc_id: str = "1", node_id: str = "n1",
    title: str = "身故保险金条款", page_range: str = "1-3",
) -> CandidateNode:
    """Build a minimal CandidateNode."""
    return CandidateNode(
        doc_id=doc_id,
        node_id=node_id,
        title=title,
        page_range=page_range,
        reason="test",
    )


def _make_mock_llm_response(facts: list[dict]) -> dict:
    """Build a mock LLM response dict with the given facts."""
    return {
        "choices": [{"message": {"content": json.dumps({"facts": facts}, ensure_ascii=False)}}],
        "model": "mock-model",
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


# ===========================================================================
# Prompt builder tests
# ===========================================================================


class TestFactMatrixMessages:
    """Tests for _build_fact_matrix_messages."""

    def test_returns_list_of_two_messages(self):
        msgs = _build_fact_matrix_messages(
            question="测试问题？",
            page_text="测试文本内容。",
            doc_id="1",
            node_title="测试章节",
        )
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_system_prompt_contains_field_vocabulary(self):
        msgs = _build_fact_matrix_messages(
            question="测试问题？",
            page_text="测试文本。",
            doc_id="1",
            node_title="测试",
        )
        system = msgs[0]["content"]
        assert "身故保险金" in system
        assert "免赔额" in system
        assert "formula_or_value" in system
        assert "基本保额*1.6" in system

    def test_user_prompt_contains_page_text(self):
        msgs = _build_fact_matrix_messages(
            question="测试问题？",
            page_text="这是测试原文。",
            doc_id="1",
            node_title="测试章节",
        )
        user = msgs[1]["content"]
        assert "这是测试原文" in user
        assert "测试问题" in user
        assert "文档ID: 1" in user
        assert "测试章节" in user


# ===========================================================================
# extract_fact_matrix with mocked LLM
# ===========================================================================


class TestExtractFactMatrixMocked:
    """Tests for extract_fact_matrix with mocked LLM responses."""

    def test_extracts_facts_from_mock_llm(self, tmp_path):
        """Mock LLM returns facts → extract_fact_matrix produces FactRecords."""
        config = AgentConfig(domain="insurance", split="A")

        # Build a minimal index store with fake page content
        index_dir = tmp_path / "index"
        index_dir.mkdir()
        index_store = _FakeIndexStore(page_text="基本保额的160%给付身故保险金。")

        parsed = _make_parsed()
        candidates = [_make_candidate()]

        # Mock LLM that returns valid fact JSON
        mock_caller = MockApiCaller(responses=[
            _make_mock_llm_response([
                {
                    "product": "国寿增益宝",
                    "field": "身故保险金",
                    "formula_or_value": "基本保额*1.6",
                    "unit": "",
                    "quote": "基本保额的160%给付身故保险金",
                },
            ]),
        ])
        llm_client = LLMClient(model="mock", api_caller=mock_caller)
        token_meter = TokenMeter(logs_dir=tmp_path)

        extractor = EvidenceExtractor()
        facts = extractor.extract_fact_matrix(
            parsed, candidates, index_store, config,
            llm_client=llm_client, token_meter=token_meter,
        )

        assert len(facts) == 1
        f = facts[0]
        assert isinstance(f, FactRecord)
        assert f.product == "国寿增益宝"
        assert f.field == "身故保险金"
        assert f.formula_or_value == "基本保额*1.6"
        assert f.quote == "基本保额的160%给付身故保险金"
        assert f.source_doc_id == "1"
        assert f.source_node_id == "n1"
        assert f.source_pages == "1-3"

    def test_multiple_facts_per_node(self, tmp_path):
        """One node → multiple facts for different products/fields."""
        config = AgentConfig()
        index_store = _FakeIndexStore(page_text="产品A免赔额0元。产品B免赔额5000元。")

        parsed = _make_parsed(question="免赔额计算？", type="计算题")
        candidates = [_make_candidate()]

        mock_caller = MockApiCaller(responses=[
            _make_mock_llm_response([
                {
                    "product": "产品A",
                    "field": "免赔额",
                    "formula_or_value": "0",
                    "unit": "元",
                    "quote": "产品A免赔额0元",
                },
                {
                    "product": "产品B",
                    "field": "免赔额",
                    "formula_or_value": "5000",
                    "unit": "元",
                    "quote": "产品B免赔额5000元",
                },
            ]),
        ])
        llm_client = LLMClient(model="mock", api_caller=mock_caller)
        token_meter = TokenMeter(logs_dir=tmp_path)

        extractor = EvidenceExtractor()
        facts = extractor.extract_fact_matrix(
            parsed, candidates, index_store, config,
            llm_client=llm_client, token_meter=token_meter,
        )

        assert len(facts) == 2
        products = {f.product for f in facts}
        assert products == {"产品A", "产品B"}

    def test_dedup_same_product_field(self, tmp_path):
        """When two nodes return the same (product, field), keep the one with quote."""
        config = AgentConfig()
        index_store = _FakeIndexStore(page_text="保单账户价值为90万元。")

        parsed = _make_parsed()
        candidates = [
            _make_candidate(node_id="n1"),
            _make_candidate(node_id="n2"),
        ]

        # First node returns fact without quote; second with quote
        mock_caller = MockApiCaller(responses=[
            _make_mock_llm_response([
                {
                    "product": "平安智盈金生",
                    "field": "身故保险金",
                    "formula_or_value": "保单账户价值",
                    "unit": "",
                    "quote": "",
                },
            ]),
            _make_mock_llm_response([
                {
                    "product": "平安智盈金生",
                    "field": "身故保险金",
                    "formula_or_value": "保单账户价值",
                    "unit": "",
                    "quote": "保单账户价值为90万元",
                },
            ]),
        ])
        llm_client = LLMClient(model="mock", api_caller=mock_caller)
        token_meter = TokenMeter(logs_dir=tmp_path)

        extractor = EvidenceExtractor()
        facts = extractor.extract_fact_matrix(
            parsed, candidates, index_store, config,
            llm_client=llm_client, token_meter=token_meter,
        )

        # Dedup should keep the one with the quote
        assert len(facts) == 1
        assert facts[0].quote == "保单账户价值为90万元"

    def test_quote_traceability_check(self, tmp_path):
        """A quote not found in page text is cleared but the fact is kept."""
        config = AgentConfig()
        index_store = _FakeIndexStore(page_text="这是完全不相关的文本。")

        parsed = _make_parsed()
        candidates = [_make_candidate()]

        mock_caller = MockApiCaller(responses=[
            _make_mock_llm_response([
                {
                    "product": "产品A",
                    "field": "免赔额",
                    "formula_or_value": "5000",
                    "unit": "元",
                    "quote": "这段文字不在页面原文中",
                },
            ]),
        ])
        llm_client = LLMClient(model="mock", api_caller=mock_caller)
        token_meter = TokenMeter(logs_dir=tmp_path)

        extractor = EvidenceExtractor()
        facts = extractor.extract_fact_matrix(
            parsed, candidates, index_store, config,
            llm_client=llm_client, token_meter=token_meter,
        )

        # Fact is still returned (formula_or_value is usable), but quote is cleared
        assert len(facts) == 1
        assert facts[0].formula_or_value == "5000"
        assert facts[0].quote == ""  # cleared because not traceable

    def test_invalid_field_filtered(self, tmp_path):
        """Facts with invalid field names are silently dropped."""
        config = AgentConfig()
        index_store = _FakeIndexStore(page_text="测试内容。")

        parsed = _make_parsed()
        candidates = [_make_candidate()]

        mock_caller = MockApiCaller(responses=[
            _make_mock_llm_response([
                {
                    "product": "产品A",
                    "field": "不存在的字段",
                    "formula_or_value": "123",
                    "unit": "元",
                    "quote": "测试内容",
                },
            ]),
        ])
        llm_client = LLMClient(model="mock", api_caller=mock_caller)
        token_meter = TokenMeter(logs_dir=tmp_path)

        extractor = EvidenceExtractor()
        facts = extractor.extract_fact_matrix(
            parsed, candidates, index_store, config,
            llm_client=llm_client, token_meter=token_meter,
        )

        assert len(facts) == 0

    def test_no_llm_returns_empty(self, tmp_path):
        """Without LLM client, returns empty list (caller falls back)."""
        config = AgentConfig()
        index_store = _FakeIndexStore(page_text="测试。")

        parsed = _make_parsed()
        candidates = [_make_candidate()]

        extractor = EvidenceExtractor()
        facts = extractor.extract_fact_matrix(
            parsed, candidates, index_store, config,
            llm_client=None,
        )

        assert facts == []

    def test_json_parse_error_returns_empty(self, tmp_path):
        """Malformed JSON from LLM → empty list for that node."""
        config = AgentConfig()
        index_store = _FakeIndexStore(page_text="测试。")

        parsed = _make_parsed()
        candidates = [_make_candidate()]

        mock_caller = MockApiCaller(responses=[
            {
                "choices": [{"message": {"content": "not valid json"}}],
                "model": "mock",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        ])
        llm_client = LLMClient(model="mock", api_caller=mock_caller)
        token_meter = TokenMeter(logs_dir=tmp_path)

        extractor = EvidenceExtractor()
        facts = extractor.extract_fact_matrix(
            parsed, candidates, index_store, config,
            llm_client=llm_client, token_meter=token_meter,
        )

        assert facts == []

    def test_respects_budget(self, tmp_path):
        """When LLM budget is exhausted, no LLM calls are made."""
        from agent.pipeline import LLMBudget

        config = AgentConfig()
        index_store = _FakeIndexStore(page_text="测试。")

        parsed = _make_parsed()
        candidates = [_make_candidate()]

        mock_caller = MockApiCaller(responses=[
            _make_mock_llm_response([]),
        ])
        llm_client = LLMClient(model="mock", api_caller=mock_caller)
        token_meter = TokenMeter(logs_dir=tmp_path)

        # Budget already exhausted
        budget = LLMBudget(max_calls=0)

        extractor = EvidenceExtractor()
        facts = extractor.extract_fact_matrix(
            parsed, candidates, index_store, config,
            llm_client=llm_client, token_meter=token_meter,
            budget=budget,
        )

        # No facts (budget exhausted → no LLM call)
        assert facts == []
        # LLM should not have been called
        assert len(mock_caller.calls) == 0

    def test_records_usage_on_success(self, tmp_path):
        """TokenMeter records usage for successful fact extraction."""
        config = AgentConfig()
        index_store = _FakeIndexStore(page_text="基本保额160%给付。")

        parsed = _make_parsed()
        candidates = [_make_candidate()]

        mock_caller = MockApiCaller(responses=[
            _make_mock_llm_response([
                {
                    "product": "产品A",
                    "field": "身故保险金",
                    "formula_or_value": "基本保额*1.6",
                    "unit": "",
                    "quote": "基本保额160%给付",
                },
            ]),
        ])
        llm_client = LLMClient(model="mock", api_caller=mock_caller)
        token_meter = TokenMeter(logs_dir=tmp_path)

        extractor = EvidenceExtractor()
        extractor.extract_fact_matrix(
            parsed, candidates, index_store, config,
            llm_client=llm_client, token_meter=token_meter,
        )

        # Check token meter has records
        records = token_meter.records_for_qid("ins_a_001")
        fact_records = [r for r in records if r.stage == "fact_matrix"]
        assert len(fact_records) >= 1
        assert fact_records[0].success is True


# ===========================================================================
# Fake index store for tests
# ===========================================================================


class _FakePageText:
    """Minimal page-text object for FakeIndexStore."""
    def __init__(self, text: str):
        self.text = text


class _FakeIndexStore:
    """Minimal IndexStore stub that returns canned page text."""

    def __init__(self, page_text: str = ""):
        self._text = page_text
        self._calls: list[tuple[int, str]] = []

    def get_page_content(self, doc_id: int, page_range: str):
        self._calls.append((doc_id, page_range))
        return [_FakePageText(self._text)]

    def get_document_structure(self, doc_id: int):
        return {}
