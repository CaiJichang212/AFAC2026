"""扫描 raw 数据集：统计 PDF 与纯文本文件信息。

PDF: 提取目录 (outline 优先 -> 否则尝试用首部页面解析"目录"页 -> 失败标记)、
     页数、每页字符数、总字符数、平均每页字符数。
TXT/MD: 行数、总字符数。

输出 JSON 到 stdout 同时落盘到 ./data_analysis/dataset_stats.json，
摘要表打印到 stdout。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

ROOT = Path("/Users/lzc/TNTprojectZ/AprojectZ/AFAC2026/data/public_dataset_upload/raw")
OUT_DIR = Path("/Users/lzc/TNTprojectZ/AprojectZ/AFAC2026/data_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PDF_EXTS = {".pdf"}
TEXT_EXTS = {".txt", ".md"}

# 目录页正则：标题...页码 形式（兼容中英文 / 全半角点）
TOC_LINE_RE = re.compile(
    r"^\s*(?P<title>.+?)[\s\.·\u2026．。\-–—_]{2,}(?P<page>\d{1,4})\s*$"
)
# 备用：标题  页码 (大量空格)
TOC_LINE_RE_LOOSE = re.compile(
    r"^\s*(?P<title>\S.+?\S)\s{2,}(?P<page>\d{1,4})\s*$"
)


def extract_outline_pymupdf(doc: fitz.Document) -> list[dict]:
    """读取 PDF 内嵌 outline。返回 [{level, title, page}, ...]"""
    toc = doc.get_toc(simple=True)  # [[level, title, page], ...]
    items: list[dict] = []
    for entry in toc:
        if len(entry) >= 3:
            level, title, page = entry[0], entry[1], entry[2]
            items.append({
                "level": int(level),
                "title": str(title).strip(),
                "page": int(page),
            })
    return items


def find_toc_page_text(pdf_path: Path, max_pages: int = 10) -> tuple[list[dict], str]:
    """在前 max_pages 页中尝试找包含"目录"的页，按行解析 (title, page)。

    返回 (items, method)。method 可为 "text-pymupdf" / "text-pdfplumber" / ""(失败)
    """
    # 优先用 PyMuPDF 提取文字
    items = _try_text_with_pymupdf(pdf_path, max_pages)
    if items:
        return items, "text-pymupdf"
    items = _try_text_with_pdfplumber(pdf_path, max_pages)
    if items:
        return items, "text-pdfplumber"
    return [], ""


def _parse_toc_lines(lines: list[str]) -> list[dict]:
    parsed: list[dict] = []
    for raw in lines:
        line = raw.replace("\u3000", " ").strip()
        if not line:
            continue
        m = TOC_LINE_RE.match(line) or TOC_LINE_RE_LOOSE.match(line)
        if not m:
            continue
        title = m.group("title").strip(" .·\u2026．。-–—_\t")
        try:
            page = int(m.group("page"))
        except ValueError:
            continue
        if not title:
            continue
        parsed.append({"level": 1, "title": title, "page": page})
    return parsed


def _try_text_with_pymupdf(pdf_path: Path, max_pages: int) -> list[dict]:
    try:
        with fitz.open(pdf_path) as doc:
            n = min(max_pages, doc.page_count)
            for i in range(n):
                text = doc.load_page(i).get_text("text") or ""
                if "目录" not in text and "目  录" not in text and "Contents" not in text:
                    continue
                lines = text.splitlines()
                # 收集本页 + 后续 1~3 页（目录可能跨页）
                for j in range(i + 1, min(i + 4, doc.page_count)):
                    lines.extend((doc.load_page(j).get_text("text") or "").splitlines())
                items = _parse_toc_lines(lines)
                if len(items) >= 3:
                    return items
    except Exception:
        return []
    return []


def _try_text_with_pdfplumber(pdf_path: Path, max_pages: int) -> list[dict]:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            n = min(max_pages, len(pdf.pages))
            for i in range(n):
                text = pdf.pages[i].extract_text() or ""
                if "目录" not in text and "目  录" not in text and "Contents" not in text:
                    continue
                lines = text.splitlines()
                for j in range(i + 1, min(i + 4, len(pdf.pages))):
                    lines.extend((pdf.pages[j].extract_text() or "").splitlines())
                items = _parse_toc_lines(lines)
                if len(items) >= 3:
                    return items
    except Exception:
        return []
    return []


def stat_pdf(pdf_path: Path) -> dict:
    info: dict = {
        "path": str(pdf_path.relative_to(ROOT)),
        "type": "pdf",
        "page_count": 0,
        "total_chars": 0,
        "avg_chars_per_page": 0.0,
        "per_page_chars": [],
        "outline_method": "",  # outline / text-pymupdf / text-pdfplumber / failed
        "outline_count": 0,
        "outline": [],
        "error": None,
    }
    try:
        with fitz.open(pdf_path) as doc:
            info["page_count"] = doc.page_count
            per_page = []
            total = 0
            for i in range(doc.page_count):
                txt = doc.load_page(i).get_text("text") or ""
                c = len(txt)
                per_page.append(c)
                total += c
            info["per_page_chars"] = per_page
            info["total_chars"] = total
            info["avg_chars_per_page"] = round(total / doc.page_count, 2) if doc.page_count else 0.0

            outline = extract_outline_pymupdf(doc)
            if outline:
                info["outline_method"] = "outline"
                info["outline"] = outline
                info["outline_count"] = len(outline)
                return info

        # outline 缺失 -> 解析"目录"页
        items, method = find_toc_page_text(pdf_path)
        if items:
            info["outline_method"] = method
            info["outline"] = items
            info["outline_count"] = len(items)
        else:
            info["outline_method"] = "failed"
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def stat_text(p: Path) -> dict:
    try:
        data = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {
            "path": str(p.relative_to(ROOT)),
            "type": p.suffix.lstrip("."),
            "error": f"{type(e).__name__}: {e}",
        }
    lines = data.splitlines()
    return {
        "path": str(p.relative_to(ROOT)),
        "type": p.suffix.lstrip("."),
        "line_count": len(lines),
        "total_chars": len(data),
    }


def main() -> int:
    pdfs: list[Path] = []
    texts: list[Path] = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in PDF_EXTS:
            pdfs.append(p)
        elif ext in TEXT_EXTS:
            texts.append(p)
    pdfs.sort()
    texts.sort()

    pdf_results: list[dict] = []
    for i, p in enumerate(pdfs, 1):
        print(f"[PDF {i}/{len(pdfs)}] {p.relative_to(ROOT)}", file=sys.stderr, flush=True)
        pdf_results.append(stat_pdf(p))

    text_results: list[dict] = []
    for p in texts:
        text_results.append(stat_text(p))

    # 按 5 个子目录分组
    sub_dirs = ["financial_contracts", "financial_reports", "insurance", "regulatory", "research"]
    groups: dict[str, dict] = {sub: {"pdfs": [], "texts": []} for sub in sub_dirs}
    other = {"pdfs": [], "texts": []}

    def _sub_of(rel: str) -> str:
        return rel.split("/", 1)[0]

    for r in pdf_results:
        sub = _sub_of(r["path"])
        (groups[sub] if sub in groups else other)["pdfs"].append(r)
    for r in text_results:
        sub = _sub_of(r["path"])
        (groups[sub] if sub in groups else other)["texts"].append(r)

    # 为每个分组生成统计摘要
    grouped_summary: dict[str, dict] = {}
    for sub in sub_dirs:
        items_pdf = groups[sub]["pdfs"]
        items_txt = groups[sub]["texts"]
        pages = sum(i.get("page_count", 0) or 0 for i in items_pdf)
        chars = sum(i.get("total_chars", 0) or 0 for i in items_pdf)
        method_counter: dict[str, int] = {}
        for i in items_pdf:
            m = i.get("outline_method", "")
            method_counter[m] = method_counter.get(m, 0) + 1
        grouped_summary[sub] = {
            "summary": {
                "pdf_count": len(items_pdf),
                "text_count": len(items_txt),
                "pdf_total_pages": pages,
                "pdf_total_chars": chars,
                "pdf_avg_pages_per_file": round(pages / len(items_pdf), 2) if items_pdf else 0.0,
                "pdf_avg_chars_per_page": round(chars / pages, 2) if pages else 0.0,
                "outline_method_distribution": method_counter,
                "text_total_lines": sum(i.get("line_count", 0) or 0 for i in items_txt),
                "text_total_chars": sum(i.get("total_chars", 0) or 0 for i in items_txt),
            },
            "pdfs": items_pdf,
            "texts": items_txt,
        }

    summary = {
        "root": str(ROOT),
        "pdf_count": len(pdf_results),
        "text_count": len(text_results),
        "groups": grouped_summary,
    }
    if other["pdfs"] or other["texts"]:
        summary["groups"]["_other"] = {
            "summary": {
                "pdf_count": len(other["pdfs"]),
                "text_count": len(other["texts"]),
            },
            "pdfs": other["pdfs"],
            "texts": other["texts"],
        }

    out_json = OUT_DIR / "dataset_stats.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] 写入 {out_json}")
    print(f"PDF 文件: {len(pdf_results)}  文本文件: {len(text_results)}")

    # 打印简要统计
    method_counter: dict[str, int] = {}
    total_pages = 0
    total_chars = 0
    for r in pdf_results:
        method_counter[r.get("outline_method", "")] = method_counter.get(r.get("outline_method", ""), 0) + 1
        total_pages += r.get("page_count", 0) or 0
        total_chars += r.get("total_chars", 0) or 0
    print(f"PDF 总页数: {total_pages}, PDF 总字符数: {total_chars}")
    print(f"PDF 目录提取方法分布: {method_counter}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
