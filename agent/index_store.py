from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent.config import AgentConfig

PAGE_SUBSTANTIVE_MARKERS = (
    "按",
    "给付",
    "等于",
    "扣除",
    "比例",
    "不承担",
    "不接受",
    "退还",
    "不得",
    "最高不超过",
    "计算",
    "赔付",
)
PAGE_PREFACE_MARKERS = ("阅读指引", "条款目录")


@dataclass(frozen=True)
class DocumentMetadata:
    doc_id: str
    product_name: str
    index_status: str
    page_count: int
    index_source: str


def validate_node_spans(
    spans: list[dict[str, Any]], page_count: int | None = None
) -> dict[str, Any]:
    bad_page_range_count = 0
    empty_title_count = 0
    keyword_title_hits = 0

    for span in spans:
        title = str(span.get("title", "")).strip()
        if not title:
            empty_title_count += 1
        if any(keyword in title for keyword in ("保险责任", "责任免除", "身故保险金", "现金价值")):
            keyword_title_hits += 1
        start_page = int(span.get("start_page", 0))
        end_page = int(span.get("end_page", 0))
        if start_page <= 0 or end_page <= 0 or end_page < start_page:
            bad_page_range_count += 1
        if page_count is not None and end_page > page_count:
            bad_page_range_count += 1

    status = "markdown" if bad_page_range_count == 0 and empty_title_count == 0 else "page_keyword"
    return {
        "node_count": len(spans),
        "empty_title_count": empty_title_count,
        "bad_page_range_count": bad_page_range_count,
        "keyword_title_hits": keyword_title_hits,
        "page_mapping_coverage": 1.0 if spans else 0.0,
        "index_source": status,
        "status": status,
    }


class IndexStore:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def _pageindex_path(self, doc_id: str) -> Path:
        path = self.config.pageindex_dir / f"{doc_id}.json"
        if path.exists():
            return path
        return self.config.pageindex_dir / f"{doc_id}.pdf_fallback.json"

    def _spans_path(self, doc_id: str) -> Path:
        return self.config.pageindex_dir / f"{doc_id}.node_spans.json"

    def _load_json(self, path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    def get_document_metadata(self, doc_id: str) -> dict[str, Any]:
        pdf_path = self.config.raw_dir / f"{doc_id}.pdf"
        page_count = 0
        if pdf_path.exists():
            import fitz

            with fitz.open(pdf_path) as doc:
                page_count = doc.page_count
        return asdict(
            DocumentMetadata(
                doc_id=doc_id,
                product_name=doc_id,
                index_status="available" if self._pageindex_path(doc_id).exists() else "missing",
                page_count=page_count,
                index_source="markdown" if (self.config.pageindex_dir / f"{doc_id}.json").exists() else "pdf",
            )
        )

    def get_document_structure(self, doc_id: str) -> list[dict[str, Any]]:
        pageindex_path = self._pageindex_path(doc_id)
        if not pageindex_path.exists():
            return []
        structure = self._load_json(pageindex_path).get("structure", [])
        spans_path = self._spans_path(doc_id)
        spans = self._load_json(spans_path) if spans_path.exists() else []
        span_map = {span["node_id"]: span for span in spans}

        def _clean(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
            cleaned: list[dict[str, Any]] = []
            for node in nodes:
                current = {
                    "title": node.get("title", ""),
                    "node_id": node.get("node_id", ""),
                    "summary": node.get("summary"),
                    "page_range": span_map.get(node.get("node_id", ""), {}).get("source_page_range"),
                    "nodes": _clean(node.get("nodes", [])),
                    "index_source": "pdf" if ".pdf_fallback." in pageindex_path.name else "markdown",
                }
                cleaned.append(current)
            return cleaned

        return _clean(structure)

    def get_page_content(self, doc_id: str, pages: str) -> list[dict[str, Any]]:
        page_file = self.config.pages_dir / f"{doc_id}.jsonl"
        if not page_file.exists():
            return []
        page_nums = _parse_pages(pages)
        results: list[dict[str, Any]] = []
        for line in page_file.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["page"] in page_nums:
                results.append(record)
        return results

    def search_page_content(
        self, doc_id: str, terms: list[str], max_pages: int = 5
    ) -> list[dict[str, Any]]:
        page_file = self.config.pages_dir / f"{doc_id}.jsonl"
        if not page_file.exists():
            return []
        unique_terms = [
            term.strip()
            for term in dict.fromkeys(terms)
            if term and len(term.strip()) >= 2
        ]
        if not unique_terms:
            return []

        scored: list[tuple[int, dict[str, Any], list[str]]] = []
        for line in page_file.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            text = str(record.get("text", ""))
            matched = [term for term in unique_terms if term in text]
            if not matched:
                continue
            occurrence_score = sum(text.count(term) for term in matched)
            score = len(matched) * 15 + min(occurrence_score, 10)
            if any(marker in text for marker in PAGE_SUBSTANTIVE_MARKERS):
                score += 12
            if any(marker in text for marker in PAGE_PREFACE_MARKERS):
                score -= 25
            scored.append((score, record, matched))

        results: list[dict[str, Any]] = []
        for score, record, matched in sorted(scored, key=lambda item: item[0], reverse=True):
            enriched = dict(record)
            enriched["matched_terms"] = matched
            enriched["score"] = score
            results.append(enriched)
            if len(results) >= max_pages:
                break
        return results


def _parse_pages(pages: str) -> list[int]:
    result: list[int] = []
    for part in pages.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return sorted(set(result))
