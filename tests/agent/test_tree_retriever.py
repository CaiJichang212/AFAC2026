"""Tests for agent/tree_retriever.py (Task 8).

Covers:
- Rule-prescreen fallback (no LLM client)
- Mocked LLM selection
- Page-budget enforcement (5 nodes exceeding 8 pages dropped)
- Large-tree path (>12 nodes, only prescreened to K sent to LLM)
- No line_num/text leakage
- needs_page_fetch always True, node cap at 5
- Integration test with real doc-1 tree + mocked LLM
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.config import AgentConfig
from agent.domain_profiles import get_profile
from agent.index_store import IndexStore, _flatten_tree
from agent.llm_client import LLMClient, MockApiCaller
from agent.schemas import CandidateNode, ParsedQuestion
from agent.tree_retriever import (
    TreeRetriever,
    _SMALL_TREE_NODE_LIMIT,
    _count_pages_in_range,
    _extract_amount_ratio_tokens,
)

# ======================================================================
# Helpers
# ======================================================================


def _make_parsed_question(**overrides: object) -> ParsedQuestion:
    """Build a minimal ParsedQuestion with sensible defaults."""
    defaults: dict = {
        "qid": "ins_a_001",
        "domain": "insurance",
        "split": "A",
        "question": "在平安智盈金生产品中，身故保险金的计算方式是什么？",
        "options": {
            "A": "已交保费",
            "B": "基本保额的160%",
            "C": "现金价值",
            "D": "三者取大",
        },
        "answer_format": "mcq",
        "type": "推理判断",
        "doc_ids": ["1"],
        "mentioned_products": ["平安智盈金生专属商业养老保险"],
        "doc_product_map": {"1": "平安智盈金生专属商业养老保险"},
        "liability_signals": ["身故保险金"],
        "number_conditions": [],
    }
    defaults.update(overrides)
    return ParsedQuestion(**defaults)


def _make_synthetic_compact_tree(
    specs: list[tuple[str, str, str, list | None]],
) -> list[dict]:
    """Build a compact tree from (node_id, title, page_range, children) specs.

    *children* is either None (leaf) or a list of child specs in the same format.
    """
    result: list[dict] = []
    for nid, title, page_range, children in specs:
        node: dict = {
            "node_id": nid,
            "title": title,
            "page_range": page_range,
            "summary": None,
            "index_source": "markdown",
        }
        if children is not None:
            node["nodes"] = _make_synthetic_compact_tree(children)
        result.append(node)
    return result


def _make_canned_llm_response(node_ids: list[str], reasons: list[str] | None = None) -> dict:
    """Build a canned MockApiCaller response for node selection."""
    if reasons is None:
        reasons = [f"Relevant because matches question" for _ in node_ids]
    nodes = [
        {"node_id": nid, "reason": r}
        for nid, r in zip(node_ids, reasons)
    ]
    content = json.dumps({"nodes": nodes}, ensure_ascii=False)
    return {
        "choices": [{"message": {"content": content}}],
        "model": "mock",
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


def _make_mock_llm_client(node_ids: list[str], reasons: list[str] | None = None) -> LLMClient:
    """Build an LLMClient wired to a MockApiCaller with a canned response."""
    mock = MockApiCaller(responses=[_make_canned_llm_response(node_ids, reasons)])
    return LLMClient(model="mock", api_caller=mock)


# ======================================================================
# _extract_amount_ratio_tokens
# ======================================================================


def test_extract_amount_ratio_tokens() -> None:
    text = "保费为100万元，赔付比例为80%，年龄40岁"
    tokens = _extract_amount_ratio_tokens(text)
    assert "100万元" in tokens
    assert "80%" in tokens
    assert "40岁" in tokens


def test_extract_amount_ratio_tokens_empty() -> None:
    assert _extract_amount_ratio_tokens("没有数字") == []


# ======================================================================
# _count_pages_in_range
# ======================================================================


def test_count_pages_single() -> None:
    assert _count_pages_in_range("6") == 1


def test_count_pages_span() -> None:
    assert _count_pages_in_range("6-8") == 3


def test_count_pages_empty() -> None:
    assert _count_pages_in_range("") == 0


# ======================================================================
# TreeRetriever -- rule-prescreen fallback (no LLM client)
# ======================================================================


def test_retrieve_rule_fallback_no_llm_client() -> None:
    """Without an LLM client, the retriever falls back to rule prescreen top-5."""
    profile = get_profile("insurance")
    config = AgentConfig()
    parsed = _make_parsed_question(
        mentioned_products=["平安智盈金生专属商业养老保险"],
        liability_signals=["身故保险金"],
    )

    tree = _make_synthetic_compact_tree([
        ("n1", "身故保险金的计算方式", "1-2", None),       # liability match
        ("n2", "保险责任概述", "3", None),                   # keyword match
        ("n3", "责任免除条款", "4", None),                   # keyword match
        ("n4", "释义", "5", None),                           # keyword match
        ("n5", "保险期间与合同终止", "6-7", None),            # keyword match
        ("n6", "平安智盈金生产品介绍", "8", None),            # product match
        ("n7", "与保险无关的内容", "9", None),               # no match
    ])

    retriever = TreeRetriever()
    result = retriever.retrieve(parsed, "1", tree, config, profile)

    assert len(result) <= 5
    assert len(result) >= 1

    # Top result should be the liability match (score 3)
    assert result[0].node_id == "n1"
    assert "身故保险金" in str(result[0].matched_signals)

    # Every candidate has needs_page_fetch=True
    for c in result:
        assert c.needs_page_fetch is True
        assert c.doc_id == "1"
        assert c.page_range != ""
        assert c.reason == "rule prescreen"

    # n7 (no match) should not appear
    result_ids = {c.node_id for c in result}
    assert "n7" not in result_ids


def test_retrieve_rule_fallback_with_failing_llm() -> None:
    """When the LLM raises, fall back to rule prescreen."""
    profile = get_profile("insurance")
    config = AgentConfig()
    parsed = _make_parsed_question()

    tree = _make_synthetic_compact_tree([
        ("n1", "身故保险金的计算方式", "1-2", None),
        ("n2", "保险责任概述", "3", None),
        ("n3", "责任免除条款", "4", None),
    ])

    # Build a MockApiCaller whose response is invalid JSON -> validation fails
    mock = MockApiCaller(responses=[
        {"choices": [{"message": {"content": "not json"}}], "model": "mock", "usage": {}},
    ])
    bad_client = LLMClient(model="mock", api_caller=mock, max_retries=0)

    retriever = TreeRetriever()
    result = retriever.retrieve(parsed, "1", tree, config, profile, llm_client=bad_client)

    # Should fall back to rule prescreen
    assert len(result) > 0
    assert all(c.reason == "rule prescreen" for c in result)


# ======================================================================
# TreeRetriever -- mocked LLM selection
# ======================================================================


def test_retrieve_mocked_llm_large_tree() -> None:
    """With a mock LLM and a large tree, LLM is called and its returned node_ids are used."""
    profile = get_profile("insurance")
    config = AgentConfig()
    parsed = _make_parsed_question()

    # Build a large tree (>12 nodes) so the LLM path is still exercised
    specs: list[tuple[str, str, str, list | None]] = [
        ("n1", "保险责任", "1", None),
        ("n2", "责任免除", "2", None),
        ("n3", "释义", "3", None),
    ]
    for i in range(10):
        specs.append((f"extra{i}", f"附加条款{i}", f"{4+i}", None))
    tree = _make_synthetic_compact_tree(specs)
    assert len(_flatten_tree(tree)) == 13  # large tree

    llm_client = _make_mock_llm_client(
        node_ids=["n2", "n3"],
        reasons=["matches insurance liability", "covers exclusion clauses"],
    )

    retriever = TreeRetriever()
    result = retriever.retrieve(parsed, "1", tree, config, profile, llm_client=llm_client)

    assert len(result) == 2
    assert result[0].node_id == "n2"
    assert result[0].reason == "matches insurance liability"
    assert result[0].doc_id == "1"
    assert result[0].page_range == "2"
    assert result[0].needs_page_fetch is True
    # matched_signals from prescreen still populated
    assert "keyword:保险责任" not in str(result[0].matched_signals)  # n2 has 责任免除

    assert result[1].node_id == "n3"
    assert result[1].reason == "covers exclusion clauses"


def test_retrieve_node_cap_at_5() -> None:
    """Even if LLM returns more, cap at max_nodes_per_doc (5)."""
    profile = get_profile("insurance")
    config = AgentConfig()  # max_nodes_per_doc=5
    parsed = _make_parsed_question()

    # Build a tree with 10 nodes (all different)
    specs = [(f"n{i}", f"保险责任 第{i}节", f"{i}", None) for i in range(10)]
    tree = _make_synthetic_compact_tree(specs)

    # LLM returns 7 node_ids
    llm_client = _make_mock_llm_client(node_ids=[f"n{i}" for i in range(7)])

    retriever = TreeRetriever()
    result = retriever.retrieve(parsed, "1", tree, config, profile, llm_client=llm_client)

    assert len(result) <= 5, f"Expected <= 5, got {len(result)}"


def test_retrieve_needs_page_fetch_always_true() -> None:
    """Every CandidateNode must have needs_page_fetch=True."""
    profile = get_profile("insurance")
    config = AgentConfig()
    parsed = _make_parsed_question()

    tree = _make_synthetic_compact_tree([
        ("n1", "保险责任", "1-2", None),
        ("n2", "责任免除", "3", None),
    ])

    llm_client = _make_mock_llm_client(node_ids=["n1", "n2"])

    retriever = TreeRetriever()
    result = retriever.retrieve(parsed, "1", tree, config, profile, llm_client=llm_client)

    for c in result:
        assert c.needs_page_fetch is True, f"Node {c.node_id} has needs_page_fetch={c.needs_page_fetch}"


# ======================================================================
# TreeRetriever -- page budget enforcement
# ======================================================================


def test_retrieve_page_budget_enforcement() -> None:
    """When 5 nodes would exceed 8 pages, drop lowest-ranked to stay within budget."""
    profile = get_profile("insurance")
    config = AgentConfig()  # max_pages_per_doc=8
    parsed = _make_parsed_question()

    # 5 nodes, total pages = 2+4+4+3+4 = 17 > 8
    tree = _make_synthetic_compact_tree([
        ("n1", "身故保险金的计算方式", "1-2", None),   # 2 pages
        ("n2", "保险责任概述", "3-6", None),             # 4 pages
        ("n3", "责任免除条款", "7-10", None),            # 4 pages
        ("n4", "释义条款", "11-13", None),               # 3 pages
        ("n5", "保险期间与合同终止", "14-17", None),     # 4 pages
    ])

    llm_client = _make_mock_llm_client(node_ids=["n1", "n2", "n3", "n4", "n5"])

    retriever = TreeRetriever()
    result = retriever.retrieve(parsed, "1", tree, config, profile, llm_client=llm_client)

    # Should have dropped some nodes to keep total pages <= 8
    total_pages = sum(_count_pages_in_range(c.page_range) for c in result)
    assert total_pages <= 8, f"Total pages {total_pages} exceeds budget 8"
    assert len(result) >= 1
    # First node(s) should be kept, later ones dropped
    kept_ids = {c.node_id for c in result}
    assert "n1" in kept_ids  # first-ranked, 2 pages, should always fit
    # At least one of the larger nodes should be dropped
    assert len(result) < 5, f"Expected some nodes dropped, got {len(result)}"


def test_retrieve_page_budget_single_node_fits() -> None:
    """A single node within budget is kept."""
    profile = get_profile("insurance")
    config = AgentConfig()
    parsed = _make_parsed_question()

    tree = _make_synthetic_compact_tree([
        ("n1", "保险责任条款", "1-5", None),  # 5 pages, within budget of 8
    ])

    retriever = TreeRetriever()
    result = retriever.retrieve(parsed, "1", tree, config, profile)

    assert len(result) == 1
    total_pages = sum(_count_pages_in_range(c.page_range) for c in result)
    assert total_pages <= 8


# ======================================================================
# TreeRetriever -- large-tree path (>12 nodes)
# ======================================================================


def test_retrieve_large_tree_sends_only_prescreened_to_llm() -> None:
    """When tree has >12 nodes, only prescreened top-K are in the LLM prompt."""
    profile = get_profile("insurance")
    config = AgentConfig()
    parsed = _make_parsed_question(
        question="身故保险金的计算方式是什么？",
        liability_signals=["身故保险金"],
    )

    # Build a tree with 15 nodes: 1 good match + 14 irrelevant
    specs: list[tuple[str, str, str, list | None]] = [
        ("match", "身故保险金的计算方式", "1-2", None),
    ]
    for i in range(14):
        specs.append((f"irr{i}", f"无关内容第{i}节", "3", None))
    tree = _make_synthetic_compact_tree(specs)

    assert len(_flatten_tree(tree)) == 15  # large tree

    # Mock that inspects the prompt
    mock = MockApiCaller(responses=[_make_canned_llm_response(["match"], ["good match"])])
    llm_client = LLMClient(model="mock", api_caller=mock)

    retriever = TreeRetriever()
    result = retriever.retrieve(parsed, "1", tree, config, profile, llm_client=llm_client)

    assert len(result) == 1
    assert result[0].node_id == "match"

    # Check that the mock received the call and its messages
    assert len(mock.calls) == 1
    messages = mock.calls[0]["messages"]
    user_content = messages[1]["content"]

    # The prompt should NOT contain all 15 nodes (only prescreened top-K)
    # The irrelevant nodes (irr0...irr13) should not appear
    for i in range(5, 14):
        assert f"irr{i}" not in user_content, (
            f"Large-tree prompt should not contain irrelevant node irr{i}"
        )


def test_retrieve_small_tree_skips_llm() -> None:
    """Phase B: when tree has <=12 nodes, skip LLM entirely and use rule prescreen."""
    profile = get_profile("insurance")
    config = AgentConfig()
    parsed = _make_parsed_question()

    tree = _make_synthetic_compact_tree([
        ("n1", "保险责任", "1", None),
        ("n2", "责任免除", "2", None),
        ("n3", "释义", "3", None),
    ])

    mock = MockApiCaller(responses=[_make_canned_llm_response(["n1", "n2"])])
    llm_client = LLMClient(model="mock", api_caller=mock)

    retriever = TreeRetriever()
    result = retriever.retrieve(parsed, "1", tree, config, profile, llm_client=llm_client)

    # Small tree: LLM should NOT have been called
    assert len(mock.calls) == 0, f"Expected 0 LLM calls for small tree, got {len(mock.calls)}"
    # Results come from rule prescreen
    assert len(result) >= 1
    assert all(c.reason == "rule prescreen" for c in result)


# ======================================================================
# TreeRetriever -- no line_num/text leakage
# ======================================================================


def test_retrieve_no_line_num_or_text_in_output() -> None:
    """CandidateNode output must not contain line_num or text keys."""
    profile = get_profile("insurance")
    config = AgentConfig()
    parsed = _make_parsed_question()

    tree = _make_synthetic_compact_tree([
        ("n1", "保险责任", "1-2", None),
    ])

    # Specifically verify: no line_num/text/start_index/end_index in tree
    def _recursive_keys(obj, prefix=""):
        keys = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                path = f"{prefix}.{k}" if prefix else k
                keys.append(path)
                if isinstance(v, (dict, list)):
                    keys.extend(_recursive_keys(v, path))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                path = f"{prefix}[{i}]"
                if isinstance(item, (dict, list)):
                    keys.extend(_recursive_keys(item, path))
        return keys

    all_keys = set(k.split(".")[-1] for k in _recursive_keys(tree))
    forbidden = {"line_num", "text", "start_index", "end_index"}
    found = all_keys & forbidden
    assert not found, f"Tree should not contain forbidden keys, found: {found}"

    # Small tree (1 node) -> rule prescreen, no LLM call
    retriever = TreeRetriever()
    result = retriever.retrieve(parsed, "1", tree, config, profile)

    assert len(result) == 1
    c = result[0]

    # CandidateNode never has line_num/text
    assert not hasattr(c, "line_num")
    assert not hasattr(c, "text")


# ======================================================================
# TreeRetriever -- product name matching in prescreen
# ======================================================================


def test_prescreen_product_name_match() -> None:
    """Node titles containing the mentioned product get scored higher."""
    profile = get_profile("insurance")
    config = AgentConfig()
    parsed = _make_parsed_question(
        mentioned_products=["平安智盈金生专属商业养老保险"],
        liability_signals=[],
    )

    tree = _make_synthetic_compact_tree([
        ("n1", "平安智盈金生产品条款说明", "1", None),    # product match
        ("n2", "保险责任概述", "2", None),                  # keyword only
        ("n3", "无关内容", "3", None),                      # no match
    ])

    retriever = TreeRetriever()
    result = retriever.retrieve(parsed, "1", tree, config, profile)

    # n1 should rank first (product match = score 3 vs keyword = score 1)
    assert result[0].node_id == "n1"
    assert any("product:" in s for s in result[0].matched_signals)

    # n3 should not appear
    result_ids = {c.node_id for c in result}
    assert "n3" not in result_ids


# ======================================================================
# TreeRetriever -- amount/ratio token matching
# ======================================================================


def test_prescreen_amount_ratio_match() -> None:
    """Amount/ratio tokens from the question are matched in node titles."""
    profile = get_profile("insurance")
    config = AgentConfig()
    parsed = _make_parsed_question(
        question="赔付比例为80%的情况",
        options={"A": "100万元", "B": "50万", "C": "200元"},
        mentioned_products=[],
        liability_signals=[],
    )

    tree = _make_synthetic_compact_tree([
        ("n1", "80%赔付比例说明", "1", None),
        ("n2", "100万元保额条款", "2", None),
        ("n3", "无关条款", "3", None),
    ])

    retriever = TreeRetriever()
    result = retriever.retrieve(parsed, "1", tree, config, profile)

    # n1 and n2 should both match
    result_ids = {c.node_id for c in result}
    assert "n1" in result_ids
    assert "n2" in result_ids
    assert "n3" not in result_ids


# ======================================================================
# TreeRetriever -- empty tree edge case
# ======================================================================


def test_retrieve_empty_tree() -> None:
    """An empty tree returns an empty list."""
    profile = get_profile("insurance")
    config = AgentConfig()
    parsed = _make_parsed_question()

    retriever = TreeRetriever()
    result = retriever.retrieve(parsed, "1", [], config, profile)
    assert result == []


def test_retrieve_no_matching_nodes() -> None:
    """When no nodes match any signals, return empty list."""
    profile = get_profile("insurance")
    config = AgentConfig()
    parsed = _make_parsed_question(
        question="完全无关的问题文本",
        mentioned_products=[],
        liability_signals=[],
    )

    tree = _make_synthetic_compact_tree([
        ("n1", "XYZ 123 ABC", "1", None),
        ("n2", "DEF 456 GHI", "2", None),
    ])

    retriever = TreeRetriever()
    result = retriever.retrieve(parsed, "1", tree, config, profile)
    # No signals match -> scored is empty -> no LLM call -> empty result
    assert result == []


# ======================================================================
# TreeRetriever -- nested tree handling
# ======================================================================


def test_retrieve_nested_tree() -> None:
    """The retriever handles nested compact trees (nodes with children) with LLM for large trees."""
    profile = get_profile("insurance")
    config = AgentConfig()
    parsed = _make_parsed_question()

    tree = _make_synthetic_compact_tree([
        ("n1", "保险责任", "1-3", [
            ("n1a", "身故保险金", "1", None),
            ("n1b", "养老保险金", "2-3", None),
        ]),
        ("n2", "责任免除", "4", None),
    ] + [(f"extra{i}", f"附加{i}", f"{5+i}", None) for i in range(10)])
    # 2 + 10 = 12 extra nodes + 2 (n1, n2) = 14 flat nodes -> large tree
    assert len(_flatten_tree(tree)) >= 13

    llm_client = _make_mock_llm_client(node_ids=["n1a", "n1b", "n2"])

    retriever = TreeRetriever()
    result = retriever.retrieve(parsed, "1", tree, config, profile, llm_client=llm_client)

    assert len(result) == 3
    ids = [c.node_id for c in result]
    assert ids == ["n1a", "n1b", "n2"]

    # Check page_ranges from child nodes
    page_ranges = {c.node_id: c.page_range for c in result}
    assert page_ranges["n1a"] == "1"
    assert page_ranges["n1b"] == "2-3"


# ======================================================================
# TreeRetriever -- multiple signal types simultaneously
# ======================================================================


def test_retrieve_multiple_signal_types() -> None:
    """A node matching both a liability and a keyword should have both in matched_signals."""
    profile = get_profile("insurance")
    config = AgentConfig()
    parsed = _make_parsed_question(
        liability_signals=["身故保险金"],
    )

    tree = _make_synthetic_compact_tree([
        ("n1", "身故保险金的计算方式与释义", "1-2", None),
    ])

    retriever = TreeRetriever()
    result = retriever.retrieve(parsed, "1", tree, config, profile)

    assert len(result) == 1
    signals = result[0].matched_signals
    # Should have at least liability:身故保险金 and keyword:释义 or keyword:身故保险金
    has_liability = any("liability:身故保险金" in s for s in signals)
    assert has_liability, f"Expected liability signal, got {signals}"


# ======================================================================
# TreeRetriever -- config respects max_nodes_per_doc and max_pages_per_doc
# ======================================================================


def test_retrieve_respects_config_budgets() -> None:
    """Different config values are respected."""
    profile = get_profile("insurance")
    config = AgentConfig(max_nodes_per_doc=3, max_pages_per_doc=6)
    parsed = _make_parsed_question()

    tree = _make_synthetic_compact_tree([
        ("n1", "保险责任条款", "1-2", None),
        ("n2", "责任免除条款", "3-4", None),
        ("n3", "释义条款", "5-6", None),
        ("n4", "附录条款", "7-8", None),
        ("n5", "附则条款", "9-10", None),
    ])

    llm_client = _make_mock_llm_client(node_ids=["n1", "n2", "n3", "n4", "n5"])

    retriever = TreeRetriever()
    result = retriever.retrieve(parsed, "1", tree, config, profile, llm_client=llm_client)

    # Capped at 3 nodes
    assert len(result) <= 3
    # Total pages <= 6
    total_pages = sum(_count_pages_in_range(c.page_range) for c in result)
    assert total_pages <= 6, f"Total pages {total_pages} > 6"


# ======================================================================
# TreeRetriever -- edge: LLM returns non-existent node_ids
# ======================================================================


def test_retrieve_llm_returns_nonexistent_nodes() -> None:
    """Large tree: LLM returning invalid node_ids are silently filtered out; valid kept."""
    profile = get_profile("insurance")
    config = AgentConfig()
    parsed = _make_parsed_question()

    tree = _make_synthetic_compact_tree([
        ("n1", "保险责任", "1-2", None),
        ("n2", "责任免除", "3", None),
    ] + [(f"extra{i}", f"附加{i}", f"{4+i}", None) for i in range(11)])
    assert len(_flatten_tree(tree)) >= 13  # large tree

    # LLM returns one valid, one invalid node_id
    llm_client = _make_mock_llm_client(node_ids=["n1", "ghost_node"])

    retriever = TreeRetriever()
    result = retriever.retrieve(parsed, "1", tree, config, profile, llm_client=llm_client)

    assert len(result) == 1
    assert result[0].node_id == "n1"


# ======================================================================
# TreeRetriever integration test with real doc-1 tree + mocked LLM
# ======================================================================

@pytest.mark.integration
def test_retrieve_integration_doc1_mocked_llm() -> None:
    """Light integration test: real doc-1 compact tree + mocked LLM."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    pi_dir = repo_root / "data" / "processed_data" / "pageindex" / "insurance"

    if not (pi_dir / "1.json").exists() or not (pi_dir / "1.node_spans.json").exists():
        pytest.skip("Real doc-1 artifacts not found; run build_pageindex first")

    profile = get_profile("insurance")
    config = AgentConfig()

    store = IndexStore(config)
    compact = store.get_document_structure(1)

    # Ensure compact tree is well-formed
    flat = _flatten_tree(compact)
    assert len(flat) > 0

    # Build a ParsedQuestion matching doc 1's product
    parsed = _make_parsed_question(
        qid="ins_a_001",
        question="平安智盈金生产品中，身故保险金的计算方式是什么？",
        options={"A": "已交保费", "B": "现金价值", "C": "基本保额", "D": "三者取大"},
        mentioned_products=["平安智盈金生专属商业养老保险"],
        liability_signals=["身故保险金", "养老保险金"],
    )

    # Mock LLM to return some reasonable node_ids from the real tree
    real_node_ids = [n["node_id"] for n in flat[:5]]
    llm_client = _make_mock_llm_client(node_ids=real_node_ids[:3])

    retriever = TreeRetriever()
    result = retriever.retrieve(parsed, "1", compact, config, profile, llm_client=llm_client)

    assert len(result) >= 1
    assert len(result) <= 5

    for c in result:
        assert c.doc_id == "1"
        assert c.node_id
        assert c.title
        assert c.page_range
        assert c.needs_page_fetch is True
        # matched_signals should be populated from prescreen
        assert isinstance(c.matched_signals, list)
        assert isinstance(c.reason, str)
        assert len(c.reason) > 0

    # Total pages within budget
    total_pages = sum(_count_pages_in_range(c.page_range) for c in result)
    assert total_pages <= config.max_pages_per_doc


# ======================================================================
# TreeRetriever -- llm_client=None with large tree also works (fallback)
# ======================================================================


def test_retrieve_large_tree_fallback_no_llm() -> None:
    """Large tree (>12 nodes) without LLM: falls back to rule prescreen top-5."""
    profile = get_profile("insurance")
    config = AgentConfig()
    parsed = _make_parsed_question(
        liability_signals=["身故保险金"],
    )

    specs: list[tuple[str, str, str, list | None]] = [
        ("n_good", "身故保险金的计算方式", "1-2", None),
    ]
    for i in range(20):
        specs.append((f"n_{i}", f"无关条款第{i}节", "3", None))
    tree = _make_synthetic_compact_tree(specs)

    assert len(_flatten_tree(tree)) == 21  # large tree

    retriever = TreeRetriever()
    result = retriever.retrieve(parsed, "1", tree, config, profile)

    assert len(result) >= 1
    assert result[0].node_id == "n_good"
    assert all(c.reason == "rule prescreen" for c in result)
    assert len(result) <= 5


# ======================================================================
# Phase B: LLM-call budget exhaustion
# ======================================================================


def test_retrieve_budget_exhausted_falls_back_to_prescreen() -> None:
    """When the per-question LLM budget is exhausted, skip LLM and use prescreen."""
    from agent.pipeline import LLMBudget

    profile = get_profile("insurance")
    config = AgentConfig()
    parsed = _make_parsed_question()

    # Large tree so LLM would normally be called
    specs: list[tuple[str, str, str, list | None]] = [
        ("n1", "身故保险金", "1-2", None),
    ]
    for i in range(15):
        specs.append((f"extra{i}", f"附加条款{i}", "3", None))
    tree = _make_synthetic_compact_tree(specs)
    assert len(_flatten_tree(tree)) >= 13  # large tree

    mock = MockApiCaller(responses=[_make_canned_llm_response(["n1"])])
    llm_client = LLMClient(model="mock", api_caller=mock)

    # Exhausted budget: 0 remaining calls
    budget = LLMBudget(max_calls=0)

    retriever = TreeRetriever()
    result = retriever.retrieve(parsed, "1", tree, config, profile,
                                llm_client=llm_client, budget=budget)

    # LLM should NOT have been called (budget exhausted)
    assert len(mock.calls) == 0, f"Expected 0 LLM calls, got {len(mock.calls)}"
    # Results come from rule prescreen
    assert len(result) >= 1
    assert all(c.reason == "rule prescreen" for c in result)


# ======================================================================
# Phase C: tree_max_tokens from config
# ======================================================================


def test_retrieve_uses_config_tree_max_tokens() -> None:
    """Tree retrieval passes config.tree_max_tokens to llm_client.chat."""
    profile = get_profile("insurance")
    config = AgentConfig(tree_max_tokens=7777)
    parsed = _make_parsed_question()

    # Large tree so LLM is called
    specs: list[tuple[str, str, str, list | None]] = [
        ("n1", "身故保险金", "1-2", None),
    ]
    for i in range(15):
        specs.append((f"extra{i}", f"附加条款{i}", "3", None))
    tree = _make_synthetic_compact_tree(specs)
    assert len(_flatten_tree(tree)) >= 13  # large tree

    mock = MockApiCaller(responses=[_make_canned_llm_response(["n1"])])
    llm_client = LLMClient(model="mock", api_caller=mock)

    retriever = TreeRetriever()
    retriever.retrieve(parsed, "1", tree, config, profile, llm_client=llm_client)

    assert len(mock.calls) >= 1, "Expected at least one LLM call"
    actual_max_tokens = mock.calls[0]["kwargs"].get("max_tokens")
    assert actual_max_tokens == 7777, (
        f"Expected max_tokens=7777 from config, got {actual_max_tokens}"
    )
