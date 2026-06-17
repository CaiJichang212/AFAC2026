from __future__ import annotations

from agent.index_store import IndexStore
from agent.schemas import CandidateNode, EvidenceRecord, ParsedQuestion


class EvidenceExtractor:
    def __init__(self, store: IndexStore) -> None:
        self.store = store

    def extract(
        self, parsed: ParsedQuestion, candidates: list[CandidateNode]
    ) -> list[EvidenceRecord]:
        records: list[EvidenceRecord] = []
        seen: set[tuple[str, str, str, str]] = set()
        for candidate in candidates:
            pages = self.store.get_page_content(candidate.doc_id, candidate.page_range)
            page_text = "\n".join(page["text"] for page in pages)
            for option, option_text in parsed.options.items():
                quote = _find_quote(page_text, option_text, parsed.question)
                evidence_type = "support" if quote else "unclear"
                if not quote:
                    quote = page_text[:120].strip()
                key = (candidate.doc_id, candidate.page_range, quote, option)
                if key in seen:
                    continue
                seen.add(key)
                records.append(
                    EvidenceRecord(
                        qid=parsed.qid,
                        doc_id=candidate.doc_id,
                        node_id=candidate.node_id,
                        pages=candidate.page_range,
                        option=option,
                        evidence_type=evidence_type,
                        quote=quote,
                        normalized_fact=quote[:200],
                        numbers=[],
                        confidence="high" if evidence_type == "support" else "low",
                    )
                )
        return records


def _find_quote(page_text: str, option_text: str, question: str) -> str:
    terms = sorted(
        {
            term
            for term in ("身故保险金", "保险责任", "责任免除", "免赔额", "现金价值", "退保", "保单贷款")
            if term in option_text or term in question
        },
        key=len,
        reverse=True,
    )
    for term in terms:
        index = page_text.find(term)
        if index >= 0:
            start = max(0, index - 40)
            end = min(len(page_text), index + 120)
            return " ".join(page_text[start:end].split())
    return ""
