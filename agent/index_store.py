"""IndexStore: load and query PageIndex trees, node spans, and page content.

Provides the storage/query layer used by Task 8 (tree retrieval) and Task 9
(evidence extraction).  All IO is lazy: trees, spans, page-maps, and pages are
cached on first access.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.config import AgentConfig
from agent.domain_profiles import get_profile


# ---------------------------------------------------------------------------
# Pure helper: flatten a PageIndex tree depth-first with depth/level
# ---------------------------------------------------------------------------


def _flatten_tree(nodes: list[dict], level: int = 0) -> list[dict]:
    """Flatten *nodes* depth-first, injecting ``_level`` into each node dict."""
    result: list[dict] = []
    for node in nodes:
        flat_node: dict[str, Any] = dict(node)
        flat_node["_level"] = level
        result.append(flat_node)
        children = node.get("structure") or node.get("nodes") or []
        if children:
            result.extend(_flatten_tree(children, level + 1))
    return result


# ---------------------------------------------------------------------------
# Pure helper: compute node spans
# ---------------------------------------------------------------------------


def compute_node_spans(
    tree: dict,
    line_to_page: dict[str, int],
) -> list[dict]:
    """Compute source-page spans for every node in *tree*.

    Returns a list of dicts, one per node in depth-first order, with keys:
    ``node_id``, ``title``, ``start_line``, ``end_line``, ``start_page``,
    ``end_page``, ``source_page_range``, ``bad``.
    """
    line_count: int = tree.get("line_count", 0)
    nodes: list[dict] = tree.get("structure", [])
    flat = _flatten_tree(nodes)

    if not flat:
        return []

    # Fallback line_count: max key from line_to_page
    if line_count == 0 and line_to_page:
        line_count = max(int(k) for k in line_to_page)

    spans: list[dict] = []
    n = len(flat)

    for i, node in enumerate(flat):
        start_line: int = node["line_num"]
        level: int = node.get("_level", 0)

        # Find next node at same-or-higher level
        end_line: int = line_count
        for j in range(i + 1, n):
            if flat[j].get("_level", 0) <= level:
                end_line = flat[j]["line_num"] - 1
                break

        # Guard malformed spans
        bad: bool = False
        if start_line > end_line or start_line < 1:
            bad = True
            start_page: int = -1
            end_page: int = -1
            source_page_range: str = ""
        else:
            sp = line_to_page.get(str(start_line))
            ep = line_to_page.get(str(end_line))
            if sp is None or ep is None:
                bad = True
                start_page = sp if sp is not None else -1
                end_page = ep if ep is not None else -1
                source_page_range = ""
            else:
                start_page = sp
                end_page = ep
                source_page_range = (
                    str(start_page) if start_page == end_page else f"{start_page}-{end_page}"
                )

        spans.append({
            "node_id": node["node_id"],
            "title": node.get("title", ""),
            "start_line": start_line,
            "end_line": end_line,
            "start_page": start_page,
            "end_page": end_page,
            "source_page_range": source_page_range,
            "bad": bad,
        })

    return spans


# ---------------------------------------------------------------------------
# Quality helpers
# ---------------------------------------------------------------------------


def compute_index_quality(
    doc_id: int,
    spans: list[dict],
    keywords: list[str],
    index_source: str,
    thresholds: dict | None = None,
) -> dict:
    """Compute quality metrics from a node-spans list.

    Returns a dict suitable for writing as one line of an index-quality JSONL.
    """
    thresholds = thresholds or {}
    node_count = len(spans)
    empty_title_count = sum(1 for s in spans if not s.get("title", "").strip())
    bad_page_range_count = sum(1 for s in spans if s.get("bad", False))

    # Keyword title hits
    keyword_title_hits = 0
    for s in spans:
        title = s.get("title", "")
        if any(kw in title for kw in keywords):
            keyword_title_hits += 1

    page_mapping_coverage = (node_count - bad_page_range_count) / max(node_count, 1)

    # Decide status
    min_nodes = thresholds.get("min_title_count", 3)
    max_empty_ratio = thresholds.get("max_empty_title_ratio", 0.3)
    min_coverage = thresholds.get("min_page_mapping_coverage", 0.95)

    if node_count == 0:
        status = "error"
    elif (
        node_count < min_nodes
        or (empty_title_count / max(node_count, 1)) > max_empty_ratio
        or page_mapping_coverage < min_coverage
    ):
        status = "degraded"
    else:
        status = "ok"

    return {
        "doc_id": doc_id,
        "node_count": node_count,
        "empty_title_count": empty_title_count,
        "bad_page_range_count": bad_page_range_count,
        "keyword_title_hits": keyword_title_hits,
        "page_mapping_coverage": round(page_mapping_coverage, 4),
        "index_source": index_source,
        "status": status,
    }


# ---------------------------------------------------------------------------
# PageText dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PageText:
    """A single page of extracted document text."""

    doc_id: str
    page: int
    text: str
    char_count: int = 0
    source_path: str = ""


# ---------------------------------------------------------------------------
# IndexStore
# ---------------------------------------------------------------------------


class IndexStore:
    """Loads and caches PageIndex trees, node spans, page maps, and page text.

    All paths are derived from an ``AgentConfig`` instance; no hard-coded paths.
    """

    def __init__(self, config: AgentConfig):
        self._config = config
        self._domain = config.domain

        # Lazy caches: doc_id (int) -> data
        self._trees: dict[int, dict] = {}
        self._node_spans: dict[int, list[dict]] = {}
        self._page_maps: dict[int, dict] = {}
        self._pages: dict[int, dict[int, dict]] = {}

        # Profile (keywords, quality thresholds, product map)
        self._profile = get_profile(self._domain)

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _load_tree(self, doc_id: int) -> dict:
        if doc_id not in self._trees:
            path = self._config.pageindex_dir / f"{doc_id}.json"
            with open(path, encoding="utf-8") as f:
                self._trees[doc_id] = json.load(f)
        return self._trees[doc_id]

    def _load_node_spans(self, doc_id: int) -> list[dict]:
        if doc_id not in self._node_spans:
            path = self._config.pageindex_dir / f"{doc_id}.node_spans.json"
            with open(path, encoding="utf-8") as f:
                self._node_spans[doc_id] = json.load(f)
        return self._node_spans[doc_id]

    def _load_page_map(self, doc_id: int) -> dict:
        if doc_id not in self._page_maps:
            path = self._config.markdown_dir / f"{doc_id}.page_map.json"
            with open(path, encoding="utf-8") as f:
                self._page_maps[doc_id] = json.load(f)
        return self._page_maps[doc_id]

    def _load_pages(self, doc_id: int) -> dict[int, dict]:
        if doc_id not in self._pages:
            pages: dict[int, dict] = {}
            path = self._config.pages_dir / f"{doc_id}.jsonl"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        pages[rec["page"]] = rec
            self._pages[doc_id] = pages
        return self._pages[doc_id]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_document_metadata(self, doc_id: int) -> dict:
        """Return metadata for a document.

        Keys: ``doc_id``, ``product_name``, ``index_status``, ``page_count``,
        ``index_source``.
        """
        tree = self._load_tree(doc_id)
        product_name = self._profile.doc_product_map.get(doc_id, "")

        # Determine index_status from node_spans if available
        index_status = "unknown"
        index_source = "markdown"
        page_count = 0

        try:
            spans = self._load_node_spans(doc_id)
            # Infer status from spans: if any are bad, degraded
            bad_count = sum(1 for s in spans if s.get("bad", False))
            index_status = "degraded" if bad_count > 0 else "ok"
            if spans and spans[0].get("index_source"):
                index_source = spans[0]["index_source"]
        except FileNotFoundError:
            pass

        # page_count from pages file
        pages = self._load_pages(doc_id)
        page_count = len(pages)

        # Fallback: use page_map max page or tree line_count
        if page_count == 0:
            try:
                pm = self._load_page_map(doc_id)
                ltp = pm.get("line_to_page", {})
                page_count = max(ltp.values()) if ltp else 0
            except FileNotFoundError:
                page_count = tree.get("line_count", 0)

        return {
            "doc_id": doc_id,
            "product_name": product_name,
            "index_status": index_status,
            "page_count": page_count,
            "index_source": index_source,
        }

    def get_document_structure(self, doc_id: int) -> list[dict]:
        """Return the compact document tree for retrieval.

        The returned dicts contain ONLY: ``node_id``, ``title``, ``summary``
        (optional/nullable), ``page_range``, ``nodes`` (children, recursively),
        ``index_source``.  Keys such as ``line_num``, ``start_index``,
        ``end_index``, ``text`` MUST NOT appear -- this is a hard contract
        enforced by tests.
        """
        tree = self._load_tree(doc_id)
        spans = self._load_node_spans(doc_id)

        # Build a lookup: node_id -> span
        span_map: dict[str, dict] = {s["node_id"]: s for s in spans}

        # Determine index_source once
        index_source = "markdown"
        if spans:
            index_source = spans[0].get("index_source", "markdown")

        def _compact(node: dict) -> dict:
            sid = node.get("node_id", "")
            span = span_map.get(sid, {})
            result: dict[str, Any] = {
                "node_id": sid,
                "title": node.get("title", ""),
                "summary": node.get("summary"),
                "page_range": span.get("source_page_range", ""),
                "index_source": index_source,
            }
            children = node.get("structure") or node.get("nodes") or []
            if children:
                result["nodes"] = [_compact(c) for c in children]
            return result

        return [_compact(n) for n in tree.get("structure", [])]

    def get_page_content(self, doc_id: int, pages: str) -> list[PageText]:
        """Return page records for a page-range string like ``"6-8"`` or ``"6"``.

        Pages are returned in ascending order.
        """
        all_pages = self._load_pages(doc_id)
        page_nums = _parse_page_range(pages)

        results: list[PageText] = []
        for pn in page_nums:
            rec = all_pages.get(pn)
            if rec is None:
                results.append(PageText(
                    doc_id=str(doc_id),
                    page=pn,
                    text="",
                ))
            else:
                results.append(PageText(
                    doc_id=rec.get("doc_id", str(doc_id)),
                    page=rec["page"],
                    text=rec.get("text", ""),
                    char_count=rec.get("char_count", 0),
                    source_path=rec.get("source_path", ""),
                ))
        return results


# ---------------------------------------------------------------------------
# Parse a page-range string
# ---------------------------------------------------------------------------


def _parse_page_range(pages: str) -> list[int]:
    """Parse ``"6-8"`` -> ``[6, 7, 8]`` or ``"6"`` -> ``[6]``."""
    pages = pages.strip()
    if "-" in pages:
        parts = pages.split("-", 1)
        start = int(parts[0].strip())
        end = int(parts[1].strip())
        return list(range(start, end + 1))
    return [int(pages)]
