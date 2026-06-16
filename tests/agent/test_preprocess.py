"""Tests for agent/preprocess.py — PDF page extraction, markdown, quality.

These tests call the preprocess functions directly on the real PDFs and
assert on in-memory / freshly-written results.  They do NOT depend on
pre-existing committed artifacts (data/ is gitignored).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.preprocess import (
    build_markdown_with_map,
    build_quality_record,
    count_headings_in_markdown,
    extract_page_text,
    is_heading_line,
    process_all,
    process_pdf,
    scan_pdfs,
    write_page_cache,
    write_quality_log,
)

# ---------------------------------------------------------------------------
# Paths to real data
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/public_dataset_upload/raw/insurance")


# ---------------------------------------------------------------------------
# scan_pdfs
# ---------------------------------------------------------------------------


class TestScanPdfs:
    def test_finds_all_16_pdfs(self) -> None:
        pdfs = scan_pdfs(RAW_DIR)
        assert len(pdfs) == 16

    def test_sorted_numerically_not_lexicographically(self) -> None:
        pdfs = scan_pdfs(RAW_DIR)
        stems = [p.stem for p in pdfs]
        assert stems == [str(i) for i in range(1, 17)]


# ---------------------------------------------------------------------------
# is_heading_line
# ---------------------------------------------------------------------------


class TestHeadingDetection:
    def test_part_heading(self) -> None:
        is_h, level = is_heading_line("第一部分 总则")
        assert is_h is True
        assert level == 1

    def test_chapter_heading(self) -> None:
        is_h, level = is_heading_line("第二章 保险责任")
        assert is_h is True
        assert level == 2

    def test_article_heading(self) -> None:
        is_h, level = is_heading_line("第五条 保险期间")
        assert is_h is True
        assert level == 3

    def test_clause_signal_heading(self) -> None:
        is_h, level = is_heading_line("保险责任")
        assert is_h is True
        assert level == 3

    def test_long_line_is_not_heading(self) -> None:
        long_text = (
            "本保险合同（以下简称本合同）是投保人与保险人约定保险权利义务关系的协议，"
            "由保险条款、投保单、保险单、保险凭证以及批单等组成。"
        )
        is_h, level = is_heading_line(long_text)
        assert is_h is False

    def test_empty_line_is_not_heading(self) -> None:
        is_h, level = is_heading_line("")
        assert is_h is False

    def test_whitespace_only_is_not_heading(self) -> None:
        is_h, level = is_heading_line("   ")
        assert is_h is False

    def test_ordinary_text_is_not_heading(self) -> None:
        is_h, level = is_heading_line("凡涉及本合同的约定，均应当采用书面形式。")
        assert is_h is False

    def test_arabic_numeral_article(self) -> None:
        is_h, level = is_heading_line("第1条 定义")
        assert is_h is True
        assert level == 3

    def test_signal_must_start_line(self) -> None:
        """A line containing a signal mid-sentence is NOT a heading."""
        is_h, level = is_heading_line("关于保险责任的详细说明")
        assert is_h is False


# ---------------------------------------------------------------------------
# Full pipeline on real PDFs
# ---------------------------------------------------------------------------


class TestPreprocessPipeline:
    """End-to-end tests that call preprocess functions directly on real PDFs."""

    def test_all_16_pdfs_process_total_299_pages(self, tmp_path: Path) -> None:
        """Process all PDFs into tmp_path and verify page totals."""
        pages_dir = tmp_path / "pages" / "insurance"
        markdown_dir = tmp_path / "markdown" / "insurance"
        quality_path = tmp_path / "quality" / "insurance_parse_quality.jsonl"

        records = process_all(
            raw_dir=RAW_DIR,
            pages_dir=pages_dir,
            markdown_dir=markdown_dir,
            quality_path=quality_path,
        )

        assert len(records) == 16
        total_pages = sum(r["page_count"] for r in records)
        assert total_pages == 299, f"Expected 299 pages, got {total_pages}"

    def test_every_page_has_nonempty_text(self, tmp_path: Path) -> None:
        """Every extracted page must have non-empty text."""
        import fitz

        pdfs = scan_pdfs(RAW_DIR)
        empty_pages: list[tuple[str, int]] = []
        for pdf_path in pdfs:
            doc = fitz.open(pdf_path)
            try:
                pages = extract_page_text(doc)
                for p in pages:
                    if not p["text"].strip():
                        empty_pages.append((pdf_path.stem, p["page"]))
            finally:
                doc.close()

        assert len(empty_pages) == 0, (
            f"Found {len(empty_pages)} empty page(s): {empty_pages}"
        )

    def test_markdown_nonempty_and_page_map_covers_all_lines(
        self, tmp_path: Path
    ) -> None:
        """For doc 1: markdown is non-empty and line_to_page covers all lines."""
        import fitz

        pdf_path = RAW_DIR / "1.pdf"
        doc = fitz.open(pdf_path)
        try:
            pages = extract_page_text(doc)
        finally:
            doc.close()

        md_text, page_map = build_markdown_with_map(pages, "1")

        # Markdown is non-empty
        assert len(md_text.strip()) > 0

        line_to_page = page_map["line_to_page"]
        md_lines = md_text.splitlines()

        # Every markdown line must have a page mapping
        for i in range(1, len(md_lines) + 1):
            key = str(i)
            assert key in line_to_page, f"Line {i} missing from line_to_page"
            page = line_to_page[key]
            assert 1 <= page <= 21, (
                f"Line {i} mapped to invalid page {page}"
            )

        # All line_to_page values must be valid page numbers
        page_count = len(pages)
        for line_str, pgnum in line_to_page.items():
            assert 1 <= pgnum <= page_count, (
                f"line_to_page[{line_str}] = {pgnum} not in [1, {page_count}]"
            )

    def test_quality_records_have_required_fields(self, tmp_path: Path) -> None:
        """Quality records must contain all required fields."""
        pages_dir = tmp_path / "pages" / "insurance"
        markdown_dir = tmp_path / "markdown" / "insurance"
        quality_path = tmp_path / "quality" / "insurance_parse_quality.jsonl"

        records = process_all(
            raw_dir=RAW_DIR,
            pages_dir=pages_dir,
            markdown_dir=markdown_dir,
            quality_path=quality_path,
        )

        required_fields = {
            "doc_id",
            "page_count",
            "text_pages",
            "char_count",
            "title_count",
            "outline_count",
            "status",
            "error",
        }

        for record in records:
            for field in required_fields:
                assert field in record, (
                    f"doc {record.get('doc_id', '?')} missing field '{field}'"
                )

    def test_all_docs_with_text_have_status_ok(self, tmp_path: Path) -> None:
        """Every doc with extracted text must have status=='ok'."""
        pages_dir = tmp_path / "pages" / "insurance"
        markdown_dir = tmp_path / "markdown" / "insurance"
        quality_path = tmp_path / "quality" / "insurance_parse_quality.jsonl"

        records = process_all(
            raw_dir=RAW_DIR,
            pages_dir=pages_dir,
            markdown_dir=markdown_dir,
            quality_path=quality_path,
        )

        for record in records:
            if record["text_pages"] > 0:
                assert record["status"] == "ok", (
                    f"doc {record['doc_id']} has {record['text_pages']} text pages "
                    f"but status={record['status']}"
                )

    def test_page_cache_line_count_equals_page_count(self, tmp_path: Path) -> None:
        """For doc 1, the page-cache JSONL line count == page_count."""
        pages_dir = tmp_path / "pages" / "insurance"
        markdown_dir = tmp_path / "markdown" / "insurance"
        quality_path = tmp_path / "quality" / "insurance_parse_quality.jsonl"

        records = process_all(
            raw_dir=RAW_DIR,
            pages_dir=pages_dir,
            markdown_dir=markdown_dir,
            quality_path=quality_path,
        )

        # doc 1 has 21 pages
        doc1_cache = pages_dir / "1.jsonl"
        assert doc1_cache.exists()

        with open(doc1_cache, "r", encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
        assert len(lines) == 21, f"Expected 21 JSONL lines, got {len(lines)}"

    def test_page_cache_record_structure(self, tmp_path: Path) -> None:
        """Verify a page-cache JSONL record has the expected fields."""
        pages_dir = tmp_path / "pages" / "insurance"
        markdown_dir = tmp_path / "markdown" / "insurance"
        quality_path = tmp_path / "quality" / "insurance_parse_quality.jsonl"

        process_all(
            raw_dir=RAW_DIR,
            pages_dir=pages_dir,
            markdown_dir=markdown_dir,
            quality_path=quality_path,
        )

        doc1_cache = pages_dir / "1.jsonl"
        with open(doc1_cache, "r", encoding="utf-8") as f:
            first_line = f.readline()
        record = json.loads(first_line)

        assert record["doc_id"] == "1"
        assert record["page"] == 1
        assert "text" in record
        assert "char_count" in record
        assert record["char_count"] == len(record["text"])
        assert "source_path" in record
        assert record["source_path"].endswith("1.pdf")

    def test_page_map_json_structure(self, tmp_path: Path) -> None:
        """Verify the .page_map.json has the expected structure."""
        pages_dir = tmp_path / "pages" / "insurance"
        markdown_dir = tmp_path / "markdown" / "insurance"
        quality_path = tmp_path / "quality" / "insurance_parse_quality.jsonl"

        process_all(
            raw_dir=RAW_DIR,
            pages_dir=pages_dir,
            markdown_dir=markdown_dir,
            quality_path=quality_path,
        )

        map_path = markdown_dir / "1.page_map.json"
        assert map_path.exists()

        with open(map_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["doc_id"] == "1"
        assert "markdown_path" in data
        assert "line_to_page" in data
        assert isinstance(data["line_to_page"], dict)
        # Keys should be 1-based line numbers as strings
        for key in data["line_to_page"]:
            assert isinstance(key, str)
            assert int(key) >= 1

    def test_process_pdf_returns_quality_record(self, tmp_path: Path) -> None:
        """process_pdf returns a valid quality record."""
        pages_dir = tmp_path / "pages"
        markdown_dir = tmp_path / "markdown"

        record = process_pdf(
            pdf_path=RAW_DIR / "1.pdf",
            doc_id="1",
            pages_dir=pages_dir,
            markdown_dir=markdown_dir,
        )

        assert record["doc_id"] == "1"
        assert record["page_count"] == 21
        assert record["status"] == "ok"
        assert record["text_pages"] == 21
        assert record["char_count"] > 0
        assert record["title_count"] >= 0
        assert record["outline_count"] >= 0
        assert record["error"] == ""

    def test_single_doc_id_filter(self, tmp_path: Path) -> None:
        """--doc-id filter processes only the requested doc."""
        pages_dir = tmp_path / "pages" / "insurance"
        markdown_dir = tmp_path / "markdown" / "insurance"
        quality_path = tmp_path / "quality" / "insurance_parse_quality.jsonl"

        records = process_all(
            raw_dir=RAW_DIR,
            pages_dir=pages_dir,
            markdown_dir=markdown_dir,
            quality_path=quality_path,
            doc_ids=["5"],
        )

        assert len(records) == 1
        assert records[0]["doc_id"] == "5"

    def test_heading_count_is_reasonable(self, tmp_path: Path) -> None:
        """Each doc should have at least some detected headings."""
        pages_dir = tmp_path / "pages" / "insurance"
        markdown_dir = tmp_path / "markdown" / "insurance"
        quality_path = tmp_path / "quality" / "insurance_parse_quality.jsonl"

        records = process_all(
            raw_dir=RAW_DIR,
            pages_dir=pages_dir,
            markdown_dir=markdown_dir,
            quality_path=quality_path,
        )

        headings_per_doc = {r["doc_id"]: r["title_count"] for r in records}
        total_headings = sum(headings_per_doc.values())
        # Across 16 insurance PDFs with ~300 pages, we expect *some* headings.
        assert total_headings > 0, "Expected at least some headings across all docs"

    def test_quality_log_is_valid_jsonl(self, tmp_path: Path) -> None:
        """The quality log written to disk is valid JSONL."""
        pages_dir = tmp_path / "pages" / "insurance"
        markdown_dir = tmp_path / "markdown" / "insurance"
        quality_path = tmp_path / "quality" / "insurance_parse_quality.jsonl"

        process_all(
            raw_dir=RAW_DIR,
            pages_dir=pages_dir,
            markdown_dir=markdown_dir,
            quality_path=quality_path,
        )

        assert quality_path.exists()
        with open(quality_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    assert "doc_id" in record


# ---------------------------------------------------------------------------
# Unit tests for individual functions
# ---------------------------------------------------------------------------


class TestCountHeadingsInMarkdown:
    def test_empty_markdown(self) -> None:
        assert count_headings_in_markdown("") == 0

    def test_only_body_text(self) -> None:
        md = "这是一段普通文本。\n又是一段文本。\n"
        assert count_headings_in_markdown(md) == 0

    def test_mixed_headings_and_body(self) -> None:
        md = "# 第一部分\n正文内容。\n## 第一章\n更多内容。\n### 第一条\n条款内容。\n"
        assert count_headings_in_markdown(md) == 3


class TestBuildQualityRecord:
    def test_ok_record(self) -> None:
        pages = [
            {"page": 1, "text": "hello world", "char_count": 11},
            {"page": 2, "text": "page two content", "char_count": 15},
        ]
        record = build_quality_record(
            doc_id="99",
            pages=pages,
            title_count=5,
            outline_count=2,
            status="ok",
        )
        assert record["doc_id"] == "99"
        assert record["page_count"] == 2
        assert record["text_pages"] == 2
        assert record["char_count"] == 26
        assert record["title_count"] == 5
        assert record["outline_count"] == 2
        assert record["status"] == "ok"
        assert record["error"] == ""

    def test_error_record(self) -> None:
        record = build_quality_record(
            doc_id="99",
            pages=[],
            title_count=0,
            outline_count=0,
            status="error",
            error="PDF corrupted",
        )
        assert record["status"] == "error"
        assert record["error"] == "PDF corrupted"

    def test_text_pages_count_excludes_empty(self) -> None:
        pages = [
            {"page": 1, "text": "   ", "char_count": 3},
            {"page": 2, "text": "real text", "char_count": 9},
        ]
        record = build_quality_record(
            doc_id="x", pages=pages, title_count=0, outline_count=0
        )
        assert record["text_pages"] == 1  # only page 2


class TestWritePageCache:
    def test_writes_jsonl_with_expected_fields(self, tmp_path: Path) -> None:
        pages = [
            {"page": 1, "text": "hello", "char_count": 5},
            {"page": 2, "text": "world", "char_count": 5},
        ]
        out = write_page_cache(
            pages=pages,
            doc_id="test",
            source_path="data/raw/test.pdf",
            pages_dir=tmp_path,
        )
        assert out.exists()
        with open(out, "r", encoding="utf-8") as f:
            lines = [json.loads(ln) for ln in f if ln.strip()]
        assert len(lines) == 2
        assert lines[0]["doc_id"] == "test"
        assert lines[0]["page"] == 1
        assert lines[0]["text"] == "hello"
        assert lines[1]["page"] == 2


class TestWriteQualityLog:
    def test_writes_jsonl(self, tmp_path: Path) -> None:
        records = [
            {"doc_id": "1", "status": "ok", "error": ""},
            {"doc_id": "2", "status": "ok", "error": ""},
        ]
        qpath = tmp_path / "quality.jsonl"
        write_quality_log(records, qpath)
        assert qpath.exists()
        with open(qpath, "r", encoding="utf-8") as f:
            data = [json.loads(ln) for ln in f if ln.strip()]
        assert len(data) == 2
        assert data[0]["doc_id"] == "1"
