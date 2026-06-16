"""Tests for agent/index_store.py (Task 5).

Covers:
- ``compute_node_spans`` with synthetic trees + page maps (flat and nested)
- ``get_document_structure`` compact output contract (no line_num/text/…)
- bad / missing page mapping handling
- ``get_page_content`` page-range parsing
- Light integration test against real doc-1 artifacts
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.config import AgentConfig
from agent.index_store import (
    IndexStore,
    PageText,
    _flatten_tree,
    _parse_page_range,
    compute_index_quality,
    compute_node_spans,
)


# ======================================================================
# _flatten_tree
# ======================================================================


def test_flatten_tree_flat() -> None:
    nodes = [
        {"node_id": "a", "title": "A", "line_num": 10},
        {"node_id": "b", "title": "B", "line_num": 20},
        {"node_id": "c", "title": "C", "line_num": 30},
    ]
    flat = _flatten_tree(nodes)
    assert len(flat) == 3
    assert all(n["_level"] == 0 for n in flat)
    assert [n["node_id"] for n in flat] == ["a", "b", "c"]


def test_flatten_tree_nested() -> None:
    nodes = [
        {
            "node_id": "a",
            "title": "A",
            "line_num": 10,
            "structure": [
                {"node_id": "a1", "title": "A1", "line_num": 12},
                {"node_id": "a2", "title": "A2", "line_num": 15},
            ],
        },
        {"node_id": "b", "title": "B", "line_num": 30},
    ]
    flat = _flatten_tree(nodes)
    assert len(flat) == 4
    ids_levels = [(n["node_id"], n["_level"]) for n in flat]
    assert ids_levels == [("a", 0), ("a1", 1), ("a2", 1), ("b", 0)]


# ======================================================================
# compute_node_spans -- flat tree (rule: end = next line_num - 1)
# ======================================================================


def _make_flat_tree(nodes_spec: list[tuple[str, str, int]],
                    line_count: int | None = None) -> dict:
    """Build a tree dict with a flat ``structure`` list."""
    structure = [
        {"node_id": nid, "title": title, "line_num": ln}
        for nid, title, ln in nodes_spec
    ]
    lc = line_count if line_count is not None else max(ln for _, _, ln in nodes_spec) + 50
    return {"doc_name": "test", "line_count": lc, "structure": structure}


def _make_page_map(lines: list[int], pages: list[int]) -> dict[str, int]:
    """Build line_to_page with given line->page mapping."""
    return {str(ln): pg for ln, pg in zip(lines, pages)}


def test_compute_node_spans_flat_three_nodes() -> None:
    """End of node N = line_num of node N+1 - 1; last node = line_count."""
    tree = _make_flat_tree([
        ("a", "Node A", 10),
        ("b", "Node B", 20),
        ("c", "Node C", 30),
    ], line_count=100)

    # line_to_page: lines 10-99 -> page 1, line 100 -> page 2
    ltp = _make_page_map([10, 19, 20, 29, 30, 100], [1, 1, 1, 1, 1, 2])

    spans = compute_node_spans(tree, ltp)

    assert len(spans) == 3

    # Node A: start=10, end=19 (next node B.line_num-1)
    assert spans[0]["node_id"] == "a"
    assert spans[0]["start_line"] == 10
    assert spans[0]["end_line"] == 19
    assert spans[0]["start_page"] == 1
    assert spans[0]["end_page"] == 1
    assert spans[0]["source_page_range"] == "1"
    assert spans[0]["bad"] is False

    # Node B: start=20, end=29 (next node C.line_num-1)
    assert spans[1]["node_id"] == "b"
    assert spans[1]["start_line"] == 20
    assert spans[1]["end_line"] == 29
    assert spans[1]["source_page_range"] == "1"

    # Node C: start=30, end=100 (last node -> line_count)
    assert spans[2]["node_id"] == "c"
    assert spans[2]["start_line"] == 30
    assert spans[2]["end_line"] == 100  # line_count
    assert spans[2]["start_page"] == 1
    assert spans[2]["end_page"] == 2
    assert spans[2]["source_page_range"] == "1-2"


# ======================================================================
# compute_node_spans -- nested tree (parent span NOT ended by child)
# ======================================================================


def test_compute_node_spans_nested_parent_not_ended_by_child() -> None:
    """Parent's span should end at the next sibling-or-ancestor, NOT at a child."""
    tree = {
        "doc_name": "test",
        "line_count": 100,
        "structure": [
            {
                "node_id": "parent",
                "title": "Parent",
                "line_num": 10,
                "structure": [
                    {"node_id": "child1", "title": "Child 1", "line_num": 12},
                    {"node_id": "child2", "title": "Child 2", "line_num": 18},
                ],
            },
            {"node_id": "sibling", "title": "Sibling", "line_num": 30},
        ],
    }
    ltp = _make_page_map(
        [10, 11, 12, 17, 18, 29, 30, 100],
        [1, 1, 1, 1, 1, 1, 2, 2],
    )

    spans = compute_node_spans(tree, ltp)
    assert len(spans) == 4

    # Parent: start=10, end=29 (sibling.line_num - 1 = 29, NOT child1)
    parent_span = next(s for s in spans if s["node_id"] == "parent")
    assert parent_span["start_line"] == 10
    assert parent_span["end_line"] == 29
    assert parent_span["source_page_range"] == "1"

    # child1: start=12, end=17 (child2.line_num - 1)
    c1 = next(s for s in spans if s["node_id"] == "child1")
    assert c1["start_line"] == 12
    assert c1["end_line"] == 17

    # child2: start=18, end=29 (sibling.line_num - 1 = 29, since sibling is level 0 <= level 1)
    c2 = next(s for s in spans if s["node_id"] == "child2")
    assert c2["start_line"] == 18
    assert c2["end_line"] == 29

    # sibling: start=30, end=100 (last node)
    sib = next(s for s in spans if s["node_id"] == "sibling")
    assert sib["start_line"] == 30
    assert sib["end_line"] == 100


# ======================================================================
# compute_node_spans -- bad page mapping
# ======================================================================


def test_compute_node_spans_bad_page_mapping() -> None:
    """A node whose line_start or line_end is not in line_to_page gets ``bad=True``
    and ``source_page_range=""``."""
    tree = _make_flat_tree([
        ("a", "Good", 10),
        ("b", "Bad", 20),  # line 20 not mapped
    ], line_count=50)

    # Only map line 10 and line 19 (start/end of "a") and line 50 (end of "b")
    ltp = _make_page_map([10, 19, 50], [1, 1, 2])
    # line 20 is NOT mapped -> node "b" start_line=20 has no mapping

    spans = compute_node_spans(tree, ltp)

    assert spans[0]["bad"] is False
    assert spans[0]["source_page_range"] == "1"

    assert spans[1]["bad"] is True
    assert spans[1]["source_page_range"] == ""
    assert spans[1]["node_id"] == "b"


def test_compute_node_spans_malformed_start_gt_end() -> None:
    """When start_line > end_line (malformed tree), mark as bad."""
    tree = {
        "doc_name": "test",
        "line_count": 5,
        "structure": [
            {"node_id": "x", "title": "X", "line_num": 10},
        ],
    }
    ltp = _make_page_map([10], [1])
    spans = compute_node_spans(tree, ltp)
    assert len(spans) == 1
    assert spans[0]["bad"] is True
    assert spans[0]["start_line"] == 10
    assert spans[0]["end_line"] == 5  # line_count < start_line


# ======================================================================
# compute_node_spans -- empty tree
# ======================================================================


def test_compute_node_spans_empty_tree() -> None:
    spans = compute_node_spans({"structure": []}, {})
    assert spans == []


# ======================================================================
# _parse_page_range
# ======================================================================


def test_parse_page_range_single() -> None:
    assert _parse_page_range("6") == [6]


def test_parse_page_range_span() -> None:
    assert _parse_page_range("6-8") == [6, 7, 8]


def test_parse_page_range_span_with_spaces() -> None:
    assert _parse_page_range(" 6 - 8 ") == [6, 7, 8]


# ======================================================================
# compute_index_quality
# ======================================================================


def test_compute_index_quality_ok() -> None:
    spans = [
        {"node_id": "a", "title": "保险责任", "bad": False},
        {"node_id": "b", "title": "责任免除", "bad": False},
        {"node_id": "c", "title": "释义", "bad": False},
        {"node_id": "d", "title": "其他", "bad": False},
    ]
    keywords = ["保险", "责任", "释义"]
    q = compute_index_quality(1, spans, keywords, "markdown")
    assert q["node_count"] == 4
    assert q["empty_title_count"] == 0
    assert q["bad_page_range_count"] == 0
    assert q["keyword_title_hits"] == 3  # 保险责任, 责任免除, 释义
    assert q["page_mapping_coverage"] == 1.0
    assert q["index_source"] == "markdown"
    assert q["status"] == "ok"


def test_compute_index_quality_degraded_bad_coverage() -> None:
    spans = [
        {"node_id": "a", "title": "A", "bad": False},
        {"node_id": "b", "title": "B", "bad": True},
        {"node_id": "c", "title": "C", "bad": True},
    ]
    q = compute_index_quality(2, spans, [], "markdown")
    assert q["bad_page_range_count"] == 2
    assert q["page_mapping_coverage"] == round(1.0 / 3.0, 4)
    assert q["status"] == "degraded"


def test_compute_index_quality_degraded_few_nodes() -> None:
    spans = [{"node_id": "a", "title": "唯一", "bad": False}]
    q = compute_index_quality(3, spans, [], "markdown")
    assert q["node_count"] == 1
    assert q["status"] == "degraded"


def test_compute_index_quality_error_empty() -> None:
    q = compute_index_quality(4, [], [], "markdown")
    assert q["status"] == "error"
    assert q["node_count"] == 0


# ======================================================================
# IndexStore -- compact structure contract (NO line_num/text/…)
# ======================================================================


def _recursive_keys(obj: dict | list, prefix: str = "") -> list[str]:
    """Collect all dotted key paths from a nested dict/list."""
    keys: list[str] = []
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


def _make_temp_config() -> AgentConfig:
    """Build a config pointing to a temp directory."""
    return AgentConfig(processed_root=Path(tempfile.mkdtemp()))


def _write_artifacts(config: AgentConfig, doc_id: int,
                     tree: dict, spans: list[dict],
                     pages_records: list[dict] | None = None) -> None:
    """Write tree, node_spans, and (optional) pages files for a doc_id."""
    pi_dir = config.pageindex_dir
    pi_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = config.pages_dir
    pages_dir.mkdir(parents=True, exist_ok=True)

    with open(pi_dir / f"{doc_id}.json", "w", encoding="utf-8") as f:
        json.dump(tree, f, ensure_ascii=False)

    with open(pi_dir / f"{doc_id}.node_spans.json", "w", encoding="utf-8") as f:
        json.dump(spans, f, ensure_ascii=False)

    if pages_records:
        with open(pages_dir / f"{doc_id}.jsonl", "w", encoding="utf-8") as f:
            for rec in pages_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def test_get_document_structure_no_line_num() -> None:
    """``get_document_structure`` output must contain NO ``line_num``, ``text``,
    ``start_index``, ``end_index`` keys at any depth."""
    config = _make_temp_config()
    tree = {
        "doc_name": "1",
        "line_count": 50,
        "structure": [
            {
                "node_id": "a",
                "title": "Parent A",
                "line_num": 10,
                "structure": [
                    {"node_id": "a1", "title": "Child A1", "line_num": 12},
                ],
            },
            {"node_id": "b", "title": "Node B", "line_num": 20},
        ],
    }
    spans = [
        {"node_id": "a", "title": "Parent A", "source_page_range": "1-2",
         "start_line": 10, "end_line": 19, "start_page": 1, "end_page": 2, "bad": False,
         "index_source": "markdown"},
        {"node_id": "a1", "title": "Child A1", "source_page_range": "1",
         "start_line": 12, "end_line": 19, "start_page": 1, "end_page": 1, "bad": False,
         "index_source": "markdown"},
        {"node_id": "b", "title": "Node B", "source_page_range": "2",
         "start_line": 20, "end_line": 50, "start_page": 2, "end_page": 2, "bad": False,
         "index_source": "markdown"},
    ]
    _write_artifacts(config, 1, tree, spans)

    store = IndexStore(config)
    structure = store.get_document_structure(1)

    all_keys = _recursive_keys(structure)
    forbidden = {"line_num", "text", "start_index", "end_index"}
    found_forbidden = [k for k in all_keys if k.split(".")[-1] in forbidden]
    assert not found_forbidden, f"Forbidden keys found: {found_forbidden}"

    # Expected keys present (check terminal key names via suffix match)
    terminal_keys = {k.split(".")[-1] for k in all_keys}
    assert "node_id" in terminal_keys
    assert "title" in terminal_keys
    assert "page_range" in terminal_keys
    assert "index_source" in terminal_keys

    # Check nested structure preserved
    top_ids = [n["node_id"] for n in structure]
    assert top_ids == ["a", "b"]

    parent_a = structure[0]
    assert "nodes" in parent_a
    assert len(parent_a["nodes"]) == 1
    assert parent_a["nodes"][0]["node_id"] == "a1"
    assert parent_a["nodes"][0]["page_range"] == "1"


def test_get_document_structure_bad_page_range_sentinel() -> None:
    """A node with bad mapping gets page_range="" (sentinel), not a fabricated page."""
    config = _make_temp_config()
    tree = {
        "doc_name": "1",
        "line_count": 30,
        "structure": [
            {"node_id": "x", "title": "Bad Node", "line_num": 10},
        ],
    }
    spans = [
        {"node_id": "x", "title": "Bad Node", "source_page_range": "",
         "start_line": 10, "end_line": 30, "start_page": -1, "end_page": -1, "bad": True,
         "index_source": "markdown"},
    ]
    _write_artifacts(config, 1, tree, spans)

    store = IndexStore(config)
    structure = store.get_document_structure(1)
    assert structure[0]["page_range"] == ""


def test_get_page_content() -> None:
    """``get_page_content(doc_id, "6-8")`` returns pages 6,7,8 in order."""
    config = _make_temp_config()
    tree = {"doc_name": "1", "line_count": 10, "structure": []}
    spans = [
        {"node_id": "a", "title": "A", "source_page_range": "1",
         "start_line": 1, "end_line": 10, "start_page": 1, "end_page": 1, "bad": False,
         "index_source": "markdown"},
    ]
    pages_records = [
        {"doc_id": "1", "page": 6, "text": "page six text", "char_count": 5},
        {"doc_id": "1", "page": 7, "text": "page seven text", "char_count": 5},
        {"doc_id": "1", "page": 8, "text": "page eight text", "char_count": 5},
    ]
    _write_artifacts(config, 1, tree, spans, pages_records)

    store = IndexStore(config)
    results = store.get_page_content(1, "6-8")
    assert len(results) == 3
    assert results[0].page == 6
    assert results[0].text == "page six text"
    assert results[1].page == 7
    assert results[2].page == 8
    assert all(isinstance(r, PageText) for r in results)


def test_get_page_content_single_page() -> None:
    """``get_page_content(doc_id, "6")`` returns just page 6."""
    config = _make_temp_config()
    tree = {"doc_name": "1", "line_count": 10, "structure": []}
    spans = [
        {"node_id": "a", "title": "A", "source_page_range": "1",
         "start_line": 1, "end_line": 10, "start_page": 1, "end_page": 1, "bad": False,
         "index_source": "markdown"},
    ]
    pages_records = [
        {"doc_id": "1", "page": 6, "text": "page six", "char_count": 5},
    ]
    _write_artifacts(config, 1, tree, spans, pages_records)

    store = IndexStore(config)
    results = store.get_page_content(1, "6")
    assert len(results) == 1
    assert results[0].page == 6


def test_get_document_metadata() -> None:
    """``get_document_metadata`` returns expected keys and product_name."""
    config = _make_temp_config()
    tree = {"doc_name": "1", "line_count": 50, "structure": []}
    spans = [
        {"node_id": "a", "title": "A", "source_page_range": "1",
         "start_line": 1, "end_line": 50, "start_page": 1, "end_page": 1, "bad": False,
         "index_source": "markdown"},
    ]
    pages_records = [
        {"doc_id": "1", "page": 1, "text": "text", "char_count": 10},
        {"doc_id": "1", "page": 2, "text": "text", "char_count": 10},
    ]
    _write_artifacts(config, 1, tree, spans, pages_records)

    store = IndexStore(config)
    meta = store.get_document_metadata(1)
    assert meta["doc_id"] == 1
    assert "平安智盈金生" in meta["product_name"]
    assert meta["page_count"] == 2
    assert meta["index_status"] == "ok"
    assert meta["index_source"] == "markdown"


# ======================================================================
# Light integration test: real doc-1 tree + page_map
# ======================================================================

@pytest.mark.integration
def test_integration_doc1_node_spans() -> None:
    """Verify node_spans from REAL doc-1 artifacts are non-empty and all ranges valid.

    Uses committed-artifacts from prior tasks (present on disk in data/).
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    pi_dir = repo_root / "data" / "processed_data" / "pageindex" / "insurance"
    md_dir = repo_root / "data" / "processed_data" / "markdown" / "insurance"

    tree_path = pi_dir / "1.json"
    pm_path = md_dir / "1.page_map.json"

    if not tree_path.exists() or not pm_path.exists():
        pytest.skip("Real doc-1 artifacts not found; run build_pageindex first")

    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    pm = json.loads(pm_path.read_text(encoding="utf-8"))
    line_to_page = pm["line_to_page"]

    spans = compute_node_spans(tree, line_to_page)
    assert len(spans) > 0, "Expected non-empty node spans for doc 1"

    bad_count = sum(1 for s in spans if s["bad"])
    good_count = len(spans) - bad_count
    assert good_count > 0, "Expected at least some valid page ranges"

    # Every non-bad span must have a non-empty source_page_range
    for s in spans:
        if not s["bad"]:
            assert s["source_page_range"], f"Node {s['node_id']} has empty source_page_range"
            assert s["start_page"] > 0
            assert s["end_page"] > 0

    # Compute quality
    keywords = ["保险", "责任", "条款", "合同", "投保", "理赔"]
    q = compute_index_quality(1, spans, keywords, "markdown")
    assert q["doc_id"] == 1
    assert q["node_count"] == len(spans)
    assert "page_mapping_coverage" in q
    assert "status" in q
