"""PDF preprocessing: extract pages, build markdown, track quality.

Pure functions that take paths and return/write results.  The CLI entry point
lives in ``scripts/build_preprocess.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import fitz

# ---------------------------------------------------------------------------
# Heading detection signals
# ---------------------------------------------------------------------------

# Priority-ordered structural patterns: (regex, markdown_heading_level)
_STRUCTURAL_PATTERNS: list[tuple[str, int]] = [
    (r"第[一二三四五六七八九十百千\d]+部分", 1),  # H1  #
    (r"第[一二三四五六七八九十百千\d]+章", 2),  # H2  ##
    (r"第[一二三四五六七八九十百千\d]+条", 3),  # H3  ###
]

# Insurance clause heading signals (standalone or line-starting terms).
# Each is mapped to H3 (###) per the spec.
_CLAUSE_SIGNALS: set[str] = {
    # Explicitly listed in the spec
    "保险责任",
    "责任免除",
    "释义",
    "现金价值",
    "退保",
    "解除合同",
    "身故保险金",
    "养老保险金",
    "满期生存保险金",
    "保险金额",
    "免赔额",
    "给付比例",
    "犹豫期",
    "宽限期",
    "受益人",
    "等待期",
    "红利",
    "保单贷款",
    "减额交清",
    "保险期间",
    # Additional common insurance clause headings
    "阅读指引",
    "条款目录",
    "总则",
    "附录",
    "附表",
    "保险合同",
    "保险费",
    "投保人",
    "被保险人",
    "保险人",
    "保险金",
    "保险单",
    "续保",
    "保证续保",
    "保单年度",
    "合同成立",
    "合同生效",
    "合同解除",
    "合同终止",
    "如实告知",
    "保险事故通知",
    "保险金申请",
    "保险金给付",
    "索赔",
    "理赔",
    "核保",
    "承保",
    "基本保额",
    "基本保险金额",
    "保险费率",
    "保单账户价值",
    "个人账户价值",
    "退保费用",
    "初始费用",
    "部分领取",
    "分红",
    "利息",
    "重大疾病保险金",
    "医疗保险金",
    "意外伤残保险金",
    "意外身故保险金",
    "赔偿限额",
    "补偿原则",
    "合同效力中止",
    "合同效力恢复",
    "复效",
    "养老年金",
    "满期保险金",
    "保险费自动垫交",
    "特定药品费用",
    "住院医疗保险金",
    "医疗费用补偿",
    "第三者责任",
    "保险凭证",
    "保单价值",
    "投资账户",
    "个人账户",
}

MAX_HEADING_LENGTH: int = 40


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------


def is_heading_line(line: str) -> tuple[bool, int]:
    """Check whether *line* is a likely heading.

    Returns ``(is_heading, markdown_level)`` where *markdown_level* is the
    number of ``#`` prefixes to use (1-3).  Returns ``(False, 0)`` for body
    text.

    Detection is conservative: false positives (body text promoted to heading)
    are worse than false negatives, so when in doubt the line stays as body.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > MAX_HEADING_LENGTH:
        return False, 0

    # 1. Structural patterns (第X部分 / 第X章 / 第X条)
    for pattern, level in _STRUCTURAL_PATTERNS:
        if re.search(pattern, stripped):
            return True, level

    # 2. Insurance clause signals -- line must *start* with the signal
    for signal in _CLAUSE_SIGNALS:
        if stripped.startswith(signal):
            # Guard: the signal must make up most of the line to avoid
            # matching mid-sentence occurrences that happen to start a line.
            if len(signal) >= len(stripped) * 0.5:
                return True, 3  # H3

    return False, 0


# ---------------------------------------------------------------------------
# PDF scanning
# ---------------------------------------------------------------------------


def scan_pdfs(raw_dir: Path) -> list[Path]:
    """Return PDF paths in *raw_dir* sorted by numeric filename stem.

    Sorts numerically (1, 2, ..., 16) rather than lexicographically
    (1, 10, 11, ...).
    """
    pdfs = list(raw_dir.glob("*.pdf"))
    pdfs.sort(key=lambda p: int(p.stem))
    return pdfs


# ---------------------------------------------------------------------------
# Page extraction
# ---------------------------------------------------------------------------


def extract_page_text(doc: fitz.Document) -> list[dict]:
    """Extract text from every page of *doc*.

    Returns a list of dicts with keys: page (1-based), text, char_count.
    """
    pages: list[dict] = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")
        pages.append(
            {
                "page": i,
                "text": text,
                "char_count": len(text),
            }
        )
    return pages


def write_page_cache(
    pages: list[dict],
    doc_id: str,
    source_path: str,
    pages_dir: Path,
) -> Path:
    """Write per-page JSONL cache to ``{pages_dir}/{doc_id}.jsonl``.

    One JSON object per line.  Returns the output path.
    """
    pages_dir.mkdir(parents=True, exist_ok=True)
    out_path = pages_dir / f"{doc_id}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in pages:
            record = {
                "doc_id": doc_id,
                "page": entry["page"],
                "text": entry["text"],
                "char_count": entry["char_count"],
                "source_path": source_path,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_path


# ---------------------------------------------------------------------------
# Markdown building with page mapping
# ---------------------------------------------------------------------------


def build_markdown_with_map(pages: list[dict], doc_id: str) -> tuple[str, dict]:
    """Build markdown text and a ``line_to_page`` mapping.

    Returns ``(markdown_text, page_map)`` where *page_map* is a dict like
    ``{"1": 1, "2": 1, "83": 2}`` mapping 1-based markdown line numbers
    (as strings) to PDF physical page numbers.
    """
    md_lines: list[str] = []
    line_pages: list[int] = []

    for entry in pages:
        page_num: int = entry["page"]
        text: str = entry["text"]
        raw_lines = text.splitlines()

        for line in raw_lines:
            stripped = line.strip()
            is_heading, level = is_heading_line(line)

            if is_heading:
                prefix = "#" * level
                md_lines.append(f"{prefix} {stripped}")
            elif stripped:
                md_lines.append(stripped)
            else:
                md_lines.append("")

            line_pages.append(page_num)

    md_text = "\n".join(md_lines) + "\n"
    line_to_page = {str(i): p for i, p in enumerate(line_pages, start=1)}

    return md_text, {"doc_id": doc_id, "line_to_page": line_to_page}


def write_markdown_with_map(
    md_text: str,
    page_map: dict,
    doc_id: str,
    markdown_dir: Path,
) -> tuple[Path, Path]:
    """Write ``.md`` and ``.page_map.json`` files.

    Returns ``(md_path, map_path)``.
    """
    markdown_dir.mkdir(parents=True, exist_ok=True)

    md_path = markdown_dir / f"{doc_id}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    map_path = markdown_dir / f"{doc_id}.page_map.json"
    page_map_full = {
        "doc_id": doc_id,
        "markdown_path": str(md_path),
        "line_to_page": page_map["line_to_page"],
    }
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(page_map_full, f, ensure_ascii=False, indent=2)

    return md_path, map_path


# ---------------------------------------------------------------------------
# Quality tracking
# ---------------------------------------------------------------------------


def count_headings_in_markdown(md_text: str) -> int:
    """Count markdown heading lines (lines starting with ``#``)."""
    count = 0
    for line in md_text.splitlines():
        if line.lstrip().startswith("#"):
            count += 1
    return count


def build_quality_record(
    doc_id: str,
    pages: list[dict],
    title_count: int,
    outline_count: int,
    status: str = "ok",
    error: str = "",
) -> dict:
    """Build a quality record for a single document."""
    text_pages = sum(1 for p in pages if p["text"].strip())
    char_count = sum(p["char_count"] for p in pages)
    return {
        "doc_id": doc_id,
        "page_count": len(pages),
        "text_pages": text_pages,
        "char_count": char_count,
        "title_count": title_count,
        "outline_count": outline_count,
        "status": status,
        "error": error,
    }


def write_quality_log(records: list[dict], quality_path: Path) -> Path:
    """Write quality records as JSONL to *quality_path*."""
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    with open(quality_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return quality_path


# ---------------------------------------------------------------------------
# Per-PDF and batch processing
# ---------------------------------------------------------------------------


def process_pdf(
    pdf_path: Path,
    doc_id: str,
    pages_dir: Path,
    markdown_dir: Path,
) -> dict:
    """Process a single PDF end-to-end.

    Extracts pages, writes page cache, builds markdown with page mapping,
    and returns a quality record (not yet written to disk).

    Returns a quality record dict.
    """
    source_path = str(pdf_path)
    error_msg = ""

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        return build_quality_record(
            doc_id=doc_id,
            pages=[],
            title_count=0,
            outline_count=0,
            status="error",
            error=f"Failed to open PDF: {exc}",
        )

    try:
        # Extract pages
        pages = extract_page_text(doc)

        # Check for empty text
        if not pages or all(not p["text"].strip() for p in pages):
            doc.close()
            return build_quality_record(
                doc_id=doc_id,
                pages=pages,
                title_count=0,
                outline_count=0,
                status="no_text",
                error="No extractable text in any page",
            )

        # Get PDF outline (table of contents)
        toc = doc.get_toc()
        outline_count = len(toc) if toc else 0

        # Write page cache
        write_page_cache(pages, doc_id, source_path, pages_dir)

        # Build markdown with page mapping
        md_text, page_map = build_markdown_with_map(pages, doc_id)

        # Count headings
        title_count = count_headings_in_markdown(md_text)

        # Write markdown and page map
        write_markdown_with_map(md_text, page_map, doc_id, markdown_dir)

        doc.close()

        return build_quality_record(
            doc_id=doc_id,
            pages=pages,
            title_count=title_count,
            outline_count=outline_count,
            status="ok",
            error="",
        )

    except Exception as exc:
        try:
            doc.close()
        except Exception:
            pass
        return build_quality_record(
            doc_id=doc_id,
            pages=pages if "pages" in dir() else [],
            title_count=0,
            outline_count=0,
            status="error",
            error=str(exc),
        )


def process_all(
    raw_dir: Path,
    pages_dir: Path,
    markdown_dir: Path,
    quality_path: Path,
    doc_ids: list[str] | None = None,
) -> list[dict]:
    """Process all (or selected) PDFs in *raw_dir*.

    Parameters
    ----------
    raw_dir:
        Directory containing ``{doc_id}.pdf`` files.
    pages_dir:
        Output directory for per-page JSONL cache files.
    markdown_dir:
        Output directory for ``.md`` and ``.page_map.json`` files.
    quality_path:
        Path for the parse-quality JSONL log.
    doc_ids:
        If given, only process PDFs whose stem is in this list.

    Returns
    -------
    list[dict]:
        Quality records, one per document, sorted by numeric doc_id.
    """
    pdfs = scan_pdfs(raw_dir)

    if doc_ids is not None:
        doc_id_set = set(doc_ids)
        pdfs = [p for p in pdfs if p.stem in doc_id_set]

    records: list[dict] = []
    for pdf_path in pdfs:
        doc_id = pdf_path.stem
        record = process_pdf(pdf_path, doc_id, pages_dir, markdown_dir)
        records.append(record)

    # Sort by numeric doc_id for deterministic output
    records.sort(key=lambda r: int(r["doc_id"]))

    # Write quality log
    write_quality_log(records, quality_path)

    return records
