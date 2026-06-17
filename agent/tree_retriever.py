from __future__ import annotations

from dataclasses import asdict
from typing import Any

from agent.domain_profiles import DomainProfile
from agent.index_store import IndexStore
from agent.schemas import CandidateNode, ParsedQuestion


class TreeRetriever:
    def __init__(
        self,
        store: IndexStore,
        profile: DomainProfile,
        max_nodes_per_doc: int = 5,
        max_pages_per_doc: int = 8,
    ) -> None:
        self.store = store
        self.profile = profile
        self.max_nodes_per_doc = max_nodes_per_doc
        self.max_pages_per_doc = max_pages_per_doc

    def retrieve(self, parsed: ParsedQuestion, doc_id: str) -> list[CandidateNode]:
        compact_tree = self.store.get_document_structure(doc_id)
        flattened = list(_flatten_tree(compact_tree))
        scored: list[tuple[int, dict[str, Any], list[str]]] = []
        signals = _query_signals(parsed, self.profile)
        for node in flattened:
            title = str(node.get("title", ""))
            matched = [signal for signal in signals if signal and signal in title]
            score = len(matched) * 10
            if node.get("page_range"):
                score += 1
            if score > 0:
                scored.append((score, node, matched))

        if not scored:
            scored = [(1, node, []) for node in flattened if node.get("page_range")]

        candidates: list[CandidateNode] = []
        for _score, node, matched in sorted(scored, key=lambda item: item[0], reverse=True):
            page_range = _trim_page_range(str(node.get("page_range") or ""), self.max_pages_per_doc)
            if not page_range:
                continue
            candidates.append(
                CandidateNode(
                    doc_id=doc_id,
                    node_id=str(node.get("node_id", "")),
                    title=str(node.get("title", "")),
                    page_range=page_range,
                    matched_signals=matched,
                    reason="标题、页码范围和题目关键词用于定位候选页段",
                    needs_page_fetch=True,
                )
            )
            if len(candidates) >= self.max_nodes_per_doc:
                break
        return candidates


def _flatten_tree(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for node in nodes:
        flattened.append(node)
        flattened.extend(_flatten_tree(node.get("nodes", [])))
    return flattened


def _query_signals(parsed: ParsedQuestion, profile: DomainProfile) -> list[str]:
    signals: list[str] = []
    signals.extend(parsed.signals.get("keywords", []))
    signals.extend(profile.liability_terms)
    for product in parsed.mentioned_products:
        signals.append(product)
        signals.extend(profile.product_aliases.get(product, ()))
    return list(dict.fromkeys(signals))


def _trim_page_range(page_range: str, max_pages: int) -> str:
    if not page_range:
        return ""
    if "-" not in page_range:
        return page_range
    start_text, end_text = page_range.split("-", 1)
    start = int(start_text)
    end = int(end_text)
    if end - start + 1 > max_pages:
        end = start + max_pages - 1
    return f"{start}-{end}"
