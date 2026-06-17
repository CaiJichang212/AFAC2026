from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import fitz

from agent.config import AgentConfig
from agent.schemas import PageText

TITLE_PATTERNS = (
    r"^\d+(?:\.\d+)*\s*[\u4e00-\u9fff]",
    r"^[\u4e00-\u9fff]{2,12}$",
    r"^第[一二三四五六七八九十百千0-9]+[章节编条]",
    r"^第[一二三四五六七八九十百千0-9]+条",
    r"^保险责任",
    r"^责任免除",
    r"^释义",
    r"^现金价值",
    r"^退保",
    r"^身故保险金",
)


def _clean_line(text: str) -> str:
    return " ".join(text.replace("\u3000", " ").split())


def _looks_like_title(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return any(re.search(pattern, stripped) for pattern in TITLE_PATTERNS)


def _heading_level(line: str) -> int:
    stripped = line.strip()
    if re.match(r"^\d+\.\d+\.\d+", stripped):
        return 4
    if re.match(r"^\d+\.\d+", stripped):
        return 3
    if re.match(r"^\d+", stripped):
        return 2
    return 2


def _extract_page_text(doc: fitz.Document, page_num: int) -> str:
    page = doc[page_num - 1]
    text = page.get_text("text")
    return text.strip()


def _build_markdown_lines(page_texts: list[PageText]) -> tuple[list[str], dict[str, int]]:
    markdown_lines: list[str] = []
    line_to_page: dict[str, int] = {}
    for page_text in page_texts:
        lines = [_clean_line(line) for line in page_text.text.splitlines()]
        lines = [line for line in lines if line]
        if not lines:
            continue
        title_lines = [line for line in lines if _looks_like_title(line)]
        for line in lines:
            if _looks_like_title(line):
                markdown_lines.append(f"{'#' * _heading_level(line)} {line}")
            else:
                markdown_lines.append(line)
            line_to_page[str(len(markdown_lines))] = page_text.page
        if title_lines:
            markdown_lines.append("")
            line_to_page[str(len(markdown_lines))] = page_text.page
    return markdown_lines, line_to_page


def preprocess_domain(config: AgentConfig) -> dict[str, int]:
    config.pages_dir.mkdir(parents=True, exist_ok=True)
    config.markdown_dir.mkdir(parents=True, exist_ok=True)
    config.quality_dir.mkdir(parents=True, exist_ok=True)

    doc_count = 0
    total_pages = 0
    quality_records: list[dict[str, object]] = []

    pdf_paths = sorted(config.raw_dir.glob("*.pdf"), key=lambda path: int(path.stem))
    for pdf_path in pdf_paths:
        doc_count += 1
        doc_id = pdf_path.stem
        page_texts: list[PageText] = []
        with fitz.open(pdf_path) as doc:
            page_count = doc.page_count
            total_pages += page_count
            for page_num in range(1, page_count + 1):
                text = _extract_page_text(doc, page_num)
                page_texts.append(
                    PageText(
                        doc_id=doc_id,
                        page=page_num,
                        text=text,
                        char_count=len(text),
                        source_path=str(pdf_path),
                    )
                )

        page_file = config.pages_dir / f"{doc_id}.jsonl"
        with page_file.open("w", encoding="utf-8") as handle:
            for record in page_texts:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

        markdown_lines, line_to_page = _build_markdown_lines(page_texts)
        markdown_path = config.markdown_dir / f"{doc_id}.md"
        markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
        page_map_path = config.markdown_dir / f"{doc_id}.page_map.json"
        page_map_path.write_text(
            json.dumps(
                {
                    "doc_id": doc_id,
                    "markdown_path": str(markdown_path),
                    "line_to_page": line_to_page,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        title_count = sum(1 for line in markdown_lines if _looks_like_title(line))
        quality_records.append(
            {
                "doc_id": doc_id,
                "page_count": page_count,
                "text_pages": page_count,
                "char_count": sum(record.char_count for record in page_texts),
                "title_count": title_count,
                "outline_count": 0,
                "status": "ok",
                "error": "",
            }
        )

    quality_path = config.quality_dir / f"{config.domain}_parse_quality.jsonl"
    with quality_path.open("w", encoding="utf-8") as handle:
        for record in quality_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {"doc_count": doc_count, "page_count": total_pages}
