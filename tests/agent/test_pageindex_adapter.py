"""Tests for agent/pageindex_adapter.py.

Gate for Task 4: verifies exact competition-required kwargs, no PageIndexClient
usage, and that every call passes explicit settings (never relies on config.yaml
defaults).

All tests are **synchronous** -- async adapter functions are driven via
``asyncio.run()`` so that ``pytest-asyncio`` is NOT required.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

from agent.config import AgentConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_pageindex_on_path() -> None:
    """Idempotent helper: add ``open_projects/PageIndex`` to sys.path."""
    pageindex_root = str(Path("open_projects/PageIndex").resolve())
    if pageindex_root not in sys.path:
        sys.path.insert(0, pageindex_root)


def _make_config(**overrides) -> AgentConfig:
    """Build a config with defaults suitable for testing."""
    return AgentConfig(**overrides)


# ---------------------------------------------------------------------------
# build_md_index -- exact kwarg assertions (section 4.2)
# ---------------------------------------------------------------------------


def test_build_md_index_exact_kwargs() -> None:
    """``build_md_index`` must call ``md_to_tree`` with every competition-required
    setting passed explicitly."""
    _ensure_pageindex_on_path()

    recorded: dict = {}

    async def fake_md_to_tree(**kwargs) -> dict:
        recorded.update(kwargs)
        return {"doc_name": "test", "line_count": 1, "structure": {}}

    with patch("pageindex.page_index_md.md_to_tree", side_effect=fake_md_to_tree):
        from agent.pageindex_adapter import build_md_index

        config = _make_config()
        md_path = Path("/fake/path/1.md")
        asyncio.run(build_md_index(md_path, config))

    assert recorded["md_path"] == str(md_path)
    assert recorded["if_add_node_summary"] == "no"
    assert recorded["if_add_doc_description"] == "no"
    assert recorded["if_add_node_text"] == "no"
    assert recorded["if_add_node_id"] == "yes"
    assert recorded["model"] == config.inference_model

    # No extra/unexpected kwargs passed
    expected_keys = {
        "md_path",
        "if_add_node_summary",
        "if_add_doc_description",
        "if_add_node_text",
        "if_add_node_id",
        "model",
    }
    assert set(recorded.keys()) == expected_keys


# ---------------------------------------------------------------------------
# build_pdf_fallback -- exact kwarg assertions (section 4.2)
# ---------------------------------------------------------------------------


def test_build_pdf_fallback_exact_kwargs() -> None:
    """``build_pdf_fallback`` must call ``page_index`` with every competition-required
    setting passed explicitly, including ``toc_check_page_num``, ``max_page_num_each_node``,
    and ``max_token_num_each_node``."""
    _ensure_pageindex_on_path()

    recorded: dict = {}

    def fake_page_index(**kwargs) -> dict:
        recorded.update(kwargs)
        return {"doc_name": "test", "structure": {}}

    with patch("pageindex.page_index.page_index", side_effect=fake_page_index):
        from agent.pageindex_adapter import build_pdf_fallback

        config = _make_config()
        pdf_path = Path("/fake/path/1.pdf")
        build_pdf_fallback(pdf_path, config)

    assert recorded["doc"] == str(pdf_path)
    assert recorded["model"] == config.inference_model
    assert recorded["toc_check_page_num"] == config.toc_check_page_num  # 20
    assert recorded["max_page_num_each_node"] == config.max_page_num_each_node  # 8
    assert recorded["max_token_num_each_node"] == config.max_token_num_each_node  # 20000
    assert recorded["if_add_node_summary"] == "no"
    assert recorded["if_add_doc_description"] == "no"
    assert recorded["if_add_node_text"] == "no"
    assert recorded["if_add_node_id"] == "yes"

    # No extra/unexpected kwargs passed
    expected_keys = {
        "doc",
        "model",
        "toc_check_page_num",
        "max_page_num_each_node",
        "max_token_num_each_node",
        "if_add_node_summary",
        "if_add_doc_description",
        "if_add_node_text",
        "if_add_node_id",
    }
    assert set(recorded.keys()) == expected_keys


# ---------------------------------------------------------------------------
# PageIndexClient.index is NEVER called
# ---------------------------------------------------------------------------


def test_pageindex_client_index_never_called() -> None:
    """The adapter must never invoke ``PageIndexClient.index``.

    We monkeypatch ``PageIndexClient.index`` to raise ``RuntimeError`` if called,
    then exercise both adapter entry points (with ``md_to_tree`` / ``page_index``
    themselves monkeypatched so the calls succeed).
    """
    _ensure_pageindex_on_path()

    def _raise_if_called(*args, **kwargs):
        raise RuntimeError("PageIndexClient.index must not be called by the adapter!")

    with patch("pageindex.client.PageIndexClient.index", side_effect=_raise_if_called):
        from agent.pageindex_adapter import build_md_index, build_pdf_fallback

        config = _make_config()

        # --- exercise markdown path ---
        async def fake_md_to_tree(**kwargs) -> dict:
            return {"doc_name": "md", "line_count": 1, "structure": {}}

        with patch("pageindex.page_index_md.md_to_tree", side_effect=fake_md_to_tree):
            asyncio.run(build_md_index(Path("/fake/1.md"), config))

        # --- exercise PDF fallback path ---
        def fake_page_index(**kwargs) -> dict:
            return {"doc_name": "pdf", "structure": {}}

        with patch("pageindex.page_index.page_index", side_effect=fake_page_index):
            build_pdf_fallback(Path("/fake/1.pdf"), config)

    # If we reach here, PageIndexClient.index was *never* called.


# ---------------------------------------------------------------------------
# Explicit settings -- adapter never passes None for model / summaries
# ---------------------------------------------------------------------------


def test_build_md_index_always_passes_model_explicitly() -> None:
    """``build_md_index`` always passes ``model`` (never ``None``) so it does NOT
    fall back to PageIndex's ``config.yaml`` defaults."""
    _ensure_pageindex_on_path()

    called: dict = {}

    async def fake_md_to_tree(**kwargs) -> dict:
        called.update(kwargs)
        return {"doc_name": "test", "line_count": 1, "structure": {}}

    with patch("pageindex.page_index_md.md_to_tree", side_effect=fake_md_to_tree):
        from agent.pageindex_adapter import build_md_index

        config = _make_config()
        asyncio.run(build_md_index(Path("/fake/test.md"), config))

    assert called.get("model") is not None, "model must be passed explicitly, never None"
    assert called["model"] == config.inference_model
    assert called["if_add_node_summary"] == "no"
    assert called["if_add_doc_description"] == "no"
    assert called["if_add_node_text"] == "no"
    assert called["if_add_node_id"] == "yes"


def test_build_pdf_fallback_always_passes_model_explicitly() -> None:
    """``build_pdf_fallback`` always passes ``model`` (never ``None``) so it does NOT
    fall back to PageIndex's ``config.yaml`` defaults."""
    _ensure_pageindex_on_path()

    called: dict = {}

    def fake_page_index(**kwargs) -> dict:
        called.update(kwargs)
        return {"doc_name": "test", "structure": {}}

    with patch("pageindex.page_index.page_index", side_effect=fake_page_index):
        from agent.pageindex_adapter import build_pdf_fallback

        config = _make_config()
        build_pdf_fallback(Path("/fake/test.pdf"), config)

    assert called.get("model") is not None, "model must be passed explicitly, never None"
    assert called["model"] == config.inference_model
    assert called["if_add_node_summary"] == "no"
    assert called["if_add_doc_description"] == "no"
    assert called["if_add_node_text"] == "no"
    assert called["if_add_node_id"] == "yes"
