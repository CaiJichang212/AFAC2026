"""PageIndex adapter: competition-required explicit overrides for md_to_tree and page_index.

Only uses the low-level ``md_to_tree`` (async) and ``page_index`` (sync) functions.
Never imports or uses ``PageIndexClient`` (the high-level entry point) -- the adapter
must stay at the low-level API so every setting is explicitly controlled.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agent.config import AgentConfig


def _ensure_pageindex_on_path(config: AgentConfig) -> None:
    """Add the PageIndex root directory to ``sys.path`` so that ``pageindex.*``
    imports resolve.  Idempotent -- only inserts once per session."""
    pageindex_root = str(config.pageindex_root.resolve())
    if pageindex_root not in sys.path:
        sys.path.insert(0, pageindex_root)


async def build_md_index(md_path: Path, config: AgentConfig) -> dict:
    """Build a PageIndex tree from a markdown file.

    Calls ``md_to_tree`` with every competition-required setting passed
    **explicitly** -- we never rely on PageIndex's default ``config.yaml``.
    """
    _ensure_pageindex_on_path(config)
    from pageindex.page_index_md import md_to_tree  # type: ignore[import-not-found]

    result = await md_to_tree(
        md_path=str(md_path),
        if_add_node_summary="no",
        if_add_doc_description="no",
        if_add_node_text="no",
        if_add_node_id="yes",
        model=config.inference_model,
    )
    return result


def build_pdf_fallback(pdf_path: Path, config: AgentConfig) -> dict:
    """Build a PageIndex tree from a PDF (TOC-based fallback path).

    Calls ``page_index`` with every competition-required setting passed
    **explicitly** -- we never rely on PageIndex's default ``config.yaml``.
    """
    _ensure_pageindex_on_path(config)
    from pageindex.page_index import page_index  # type: ignore[import-not-found]

    result = page_index(
        doc=str(pdf_path),
        model=config.inference_model,
        toc_check_page_num=config.toc_check_page_num,
        max_page_num_each_node=config.max_page_num_each_node,
        max_token_num_each_node=config.max_token_num_each_node,
        if_add_node_summary="no",
        if_add_doc_description="no",
        if_add_node_text="no",
        if_add_node_id="yes",
    )
    return result
