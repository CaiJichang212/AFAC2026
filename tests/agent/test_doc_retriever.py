"""Tests for agent.doc_retriever — A-split passthrough + B-split stub."""

from __future__ import annotations

import logging
import warnings

import pytest

from agent.doc_retriever import DocRetriever
from agent.schemas import ParsedQuestion


# ---------------------------------------------------------------------------
# Tiny fake catalog for testing
# ---------------------------------------------------------------------------

class _FakeCatalog:
    """Minimal fake DocCatalog for testing the retriever."""

    def __init__(self, known_ids: set[str] | None = None):
        self._known = known_ids or set()

    def contains(self, doc_id: str) -> bool:
        return doc_id in self._known

    def get(self, doc_id: str) -> dict:
        if doc_id in self._known:
            return {"doc_id": doc_id, "product_name": f"product_{doc_id}"}
        raise KeyError(doc_id)


# ---------------------------------------------------------------------------
# Helper to build a minimal ParsedQuestion
# ---------------------------------------------------------------------------

def _make_pq(
    qid: str = "ins_a_001",
    split: str = "A",
    doc_ids: list[str] | None = None,
) -> ParsedQuestion:
    if doc_ids is None:
        doc_ids = ["1", "2", "3"]
    return ParsedQuestion(
        qid=qid,
        domain="insurance",
        split=split,
        question="test question",
        options={"A": "yes", "B": "no"},
        answer_format="tf",
        type="事实查询",
        doc_ids=doc_ids,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDocRetrieverSplitA:
    """Tests for split='A' passthrough behaviour."""

    def test_returns_doc_ids_exactly(self):
        """Split A: retrieve returns exactly the question's doc_ids in order."""
        retriever = DocRetriever()
        catalog = _FakeCatalog(known_ids={"1", "2", "3"})
        pq = _make_pq(doc_ids=["3", "1", "2"])
        result = retriever.retrieve(pq, catalog)
        assert result == ["3", "1", "2"]
        # Verify it's a copy (not the same list object)
        assert result is not pq.doc_ids

    def test_returns_empty_list_for_no_doc_ids(self):
        """Question with empty doc_ids returns empty list."""
        retriever = DocRetriever()
        catalog = _FakeCatalog()
        pq = _make_pq(doc_ids=[])
        result = retriever.retrieve(pq, catalog)
        assert result == []

    def test_no_extra_docs_added(self):
        """Split A never adds documents beyond the question's doc_ids."""
        retriever = DocRetriever()
        # Catalog has extra docs not in the question
        catalog = _FakeCatalog(known_ids={"1", "2", "3", "4", "5"})
        pq = _make_pq(doc_ids=["1", "2"])
        result = retriever.retrieve(pq, catalog)
        assert result == ["1", "2"]

    def test_unknown_doc_id_warns_but_still_returned(self, caplog):
        """Unknown doc_id emits a warning but is still returned (A-split rule)."""
        retriever = DocRetriever()
        catalog = _FakeCatalog(known_ids={"1"})  # "9" is unknown
        pq = _make_pq(doc_ids=["1", "9"])

        with caplog.at_level(logging.WARNING):
            result = retriever.retrieve(pq, catalog)

        # Result still includes the unknown doc_id
        assert result == ["1", "9"]
        # Warning was logged
        assert "doc_id '9'" in caplog.text or "'9'" in caplog.text

    def test_all_doc_ids_returned_regardless_of_catalog(self):
        """Even when no doc_ids are in the catalog, all are returned."""
        retriever = DocRetriever()
        catalog = _FakeCatalog(known_ids=set())  # empty catalog
        pq = _make_pq(doc_ids=["7", "8", "9"])
        result = retriever.retrieve(pq, catalog)
        assert result == ["7", "8", "9"]


class TestDocRetrieverSplitB:
    """Tests for split='B' (not yet implemented)."""

    def test_split_b_raises_not_implemented(self):
        """Split B raises NotImplementedError."""
        retriever = DocRetriever()
        catalog = _FakeCatalog()
        pq = _make_pq(split="B", doc_ids=["1", "2"])
        with pytest.raises(NotImplementedError):
            retriever.retrieve(pq, catalog)

    def test_split_b_error_message_is_descriptive(self):
        """The NotImplementedError message mentions split='B'."""
        retriever = DocRetriever()
        catalog = _FakeCatalog()
        pq = _make_pq(split="B")
        with pytest.raises(NotImplementedError, match="split"):
            retriever.retrieve(pq, catalog)


class TestDocRetrieverRetrieveAll:
    """Tests for the retrieve_all convenience method."""

    def test_retrieve_all_returns_dict(self):
        """retrieve_all maps qid -> doc_ids for all questions."""
        retriever = DocRetriever()
        catalog = _FakeCatalog(known_ids={"1", "2", "3", "4", "5"})
        pqs = [
            _make_pq(qid="ins_a_001", doc_ids=["1", "2"]),
            _make_pq(qid="ins_a_002", doc_ids=["3", "4"]),
            _make_pq(qid="ins_a_003", doc_ids=["5"]),
        ]
        result = retriever.retrieve_all(pqs, catalog)
        assert result == {
            "ins_a_001": ["1", "2"],
            "ins_a_002": ["3", "4"],
            "ins_a_003": ["5"],
        }

    def test_retrieve_all_empty_list(self):
        """retrieve_all with empty list returns empty dict."""
        retriever = DocRetriever()
        catalog = _FakeCatalog()
        result = retriever.retrieve_all([], catalog)
        assert result == {}

    def test_retrieve_all_raises_on_b_split(self):
        """retrieve_all raises NotImplementedError if any question is B-split."""
        retriever = DocRetriever()
        catalog = _FakeCatalog()
        pqs = [
            _make_pq(qid="ins_a_001", doc_ids=["1"]),
            _make_pq(qid="ins_b_001", split="B", doc_ids=["1"]),
        ]
        with pytest.raises(NotImplementedError):
            retriever.retrieve_all(pqs, catalog)


class TestDocRetrieverEdgeCases:
    """Edge case tests."""

    def test_single_doc_question(self):
        """Question with a single doc_id works."""
        retriever = DocRetriever()
        catalog = _FakeCatalog(known_ids={"1"})
        pq = _make_pq(doc_ids=["1"])
        result = retriever.retrieve(pq, catalog)
        assert result == ["1"]

    def test_doc_ids_not_mutated(self):
        """The original parsed.doc_ids list is not mutated."""
        pq = _make_pq(doc_ids=["1", "2", "3"])
        original = pq.doc_ids[:]
        retriever = DocRetriever()
        catalog = _FakeCatalog(known_ids={"1", "2", "3"})
        _ = retriever.retrieve(pq, catalog)
        assert pq.doc_ids == original
