from __future__ import annotations

from agent.schemas import ParsedQuestion


class DocRetriever:
    def retrieve(self, parsed: ParsedQuestion, catalog: dict | None = None) -> list[str]:
        if parsed.split.upper() == "A":
            return list(parsed.doc_ids)
        if catalog is None:
            return []
        return [doc_id for doc_id in parsed.doc_ids if doc_id in catalog]
