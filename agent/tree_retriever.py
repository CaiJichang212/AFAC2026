from __future__ import annotations

from typing import Any

from agent.domain_profiles import DomainProfile
from agent.index_store import IndexStore
from agent.schemas import CandidateNode, ParsedQuestion

INSURANCE_EXPANSION_TERMS = (
    "保单账户价值",
    "个人账户价值",
    "身故保险金额",
    "身故给付比例",
    "基本保险金额",
    "累计已交保险费",
    "累计已给付",
    "退保费用",
    "解除合同",
    "全部保险费",
    "保险金计算方法",
    "赔付比例",
    "费用补偿",
    "免赔额余额",
    "指定药店",
    "处方审核",
    "效力中止期间",
    "不承担保险责任",
    "保单贷款",
    "个人养老金制度",
    "施救费用",
    "最高不超过保险金额",
)


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
        page_signals = _page_query_signals(parsed)
        for node in flattened:
            title = str(node.get("title", ""))
            matched = [signal for signal in signals if signal and signal in title]
            score = len(matched) * 10
            if node.get("page_range"):
                score += 1
            if score > 0:
                scored.append((score, node, matched))

        for page_node, matched in _page_keyword_nodes(
            self.store, doc_id, page_signals or signals, self.max_nodes_per_doc
        ):
            score = len(matched) * 15 + 5
            scored.append((score, page_node, matched))

        if not scored:
            scored = [(1, node, []) for node in flattened if node.get("page_range")]

        candidates: list[CandidateNode] = []
        seen_pages: set[str] = set()
        for _score, node, matched in sorted(scored, key=lambda item: item[0], reverse=True):
            page_range = _trim_page_range(str(node.get("page_range") or ""), self.max_pages_per_doc)
            if not page_range:
                continue
            dedupe_key = f"{node.get('node_id')}:{page_range}"
            if dedupe_key in seen_pages:
                continue
            seen_pages.add(dedupe_key)
            reason = str(node.get("reason") or "标题、页码范围和题目关键词用于定位候选页段")
            candidates.append(
                CandidateNode(
                    doc_id=doc_id,
                    node_id=str(node.get("node_id", "")),
                    title=str(node.get("title", "")),
                    page_range=page_range,
                    matched_signals=matched,
                    reason=reason,
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
    signals.extend(_page_query_signals(parsed))
    signals.extend(profile.liability_terms)
    for product in parsed.mentioned_products:
        signals.append(product)
        signals.extend(profile.product_aliases.get(product, ()))
    return list(dict.fromkeys(signals))


def _page_query_signals(parsed: ParsedQuestion) -> list[str]:
    signals: list[str] = []
    combined_text = parsed.question + "\n" + "\n".join(parsed.options.values())
    signals.extend(parsed.signals.get("keywords", []))
    signals.extend(term for term in INSURANCE_EXPANSION_TERMS if term in combined_text)
    signals.extend(str(amount.get("raw", "")) for amount in parsed.signals.get("amounts", []))
    return list(dict.fromkeys(signals))


def _page_keyword_nodes(
    store: IndexStore,
    doc_id: str,
    signals: list[str],
    max_pages: int,
) -> list[tuple[dict[str, Any], list[str]]]:
    pages = store.search_page_content(doc_id, signals, max_pages=max_pages)
    nodes: list[tuple[dict[str, Any], list[str]]] = []
    for page in pages:
        page_number = int(page["page"])
        matched = list(page.get("matched_terms", []))
        nodes.append(
            (
                {
                    "title": "页文本关键词: " + "、".join(matched[:4]),
                    "node_id": f"page-{page_number}",
                    "page_range": f"{page_number}-{page_number}",
                    "reason": "页文本关键词补召回，用于弥补 PageIndex 标题或页码范围未命中真实条款",
                },
                matched,
            )
        )
    return nodes


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
