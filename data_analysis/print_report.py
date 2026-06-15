"""读取 dataset_stats.json，按子目录分组输出 CSV 报告。

CSV 包含 3 个 section（用空行分隔）：
1. 子目录汇总（GROUP SUMMARY）
2. PDF 文件明细（PDF FILES）
3. 文本文件明细（TEXT FILES）
"""
import csv
import io
import json
from pathlib import Path

DATA = Path("/Users/lzc/TNTprojectZ/AprojectZ/AFAC2026/data_analysis/dataset_stats.json")
OUT = Path("/Users/lzc/TNTprojectZ/AprojectZ/AFAC2026/data_analysis/report.csv")

s = json.loads(DATA.read_text(encoding="utf-8"))
groups: dict = s.get("groups", {})

buf = io.StringIO()
writer = csv.writer(buf)

# Section 1: 子目录汇总
writer.writerow(["# section", "group_summary"])
writer.writerow([
    "group", "pdf_count", "pdf_total_pages", "pdf_total_chars",
    "pdf_avg_pages_per_file", "pdf_avg_chars_per_page",
    "outline_count", "text_toc_count", "failed_count",
    "text_count", "text_total_lines", "text_total_chars",
])
total_pdf = total_pages = total_chars = total_outline = total_text_toc = total_failed = 0
total_text = total_lines = total_text_chars = 0
for name, g in groups.items():
    summ = g.get("summary", {})
    method = summ.get("outline_method_distribution", {}) or {}
    outline_n = method.get("outline", 0)
    text_toc_n = sum(v for k, v in method.items() if k.startswith("text-"))
    failed_n = method.get("failed", 0)
    writer.writerow([
        name,
        summ.get("pdf_count", 0),
        summ.get("pdf_total_pages", 0),
        summ.get("pdf_total_chars", 0),
        summ.get("pdf_avg_pages_per_file", 0),
        summ.get("pdf_avg_chars_per_page", 0),
        outline_n,
        text_toc_n,
        failed_n,
        summ.get("text_count", 0),
        summ.get("text_total_lines", 0),
        summ.get("text_total_chars", 0),
    ])
    total_pdf += summ.get("pdf_count", 0)
    total_pages += summ.get("pdf_total_pages", 0)
    total_chars += summ.get("pdf_total_chars", 0)
    total_outline += outline_n
    total_text_toc += text_toc_n
    total_failed += failed_n
    total_text += summ.get("text_count", 0)
    total_lines += summ.get("text_total_lines", 0)
    total_text_chars += summ.get("text_total_chars", 0)

avg_total = round(total_chars / total_pages, 2) if total_pages else 0
avg_pages_total = round(total_pages / total_pdf, 2) if total_pdf else 0
writer.writerow([
    "TOTAL", total_pdf, total_pages, total_chars, avg_pages_total, avg_total,
    total_outline, total_text_toc, total_failed,
    total_text, total_lines, total_text_chars,
])
writer.writerow([])

# Section 2: PDF 文件明细
writer.writerow(["# section", "pdf_files"])
writer.writerow([
    "group", "path", "page_count", "total_chars", "avg_chars_per_page",
    "outline_method", "outline_count", "error",
])
for name, g in groups.items():
    for r in g.get("pdfs", []):
        writer.writerow([
            name,
            r.get("path", ""),
            r.get("page_count", 0),
            r.get("total_chars", 0),
            r.get("avg_chars_per_page", 0),
            r.get("outline_method", ""),
            r.get("outline_count", 0),
            r.get("error") or "",
        ])
writer.writerow([])

# Section 3: 文本文件明细
writer.writerow(["# section", "text_files"])
writer.writerow(["group", "path", "type", "line_count", "total_chars", "error"])
for name, g in groups.items():
    for r in g.get("texts", []):
        writer.writerow([
            name,
            r.get("path", ""),
            r.get("type", ""),
            r.get("line_count", 0),
            r.get("total_chars", 0),
            r.get("error") or "",
        ])

OUT.write_text(buf.getvalue(), encoding="utf-8")
print(f"[OK] CSV 报告已写入 {OUT}")
print(f"分组数: {len(groups)}  PDF 总数: {total_pdf}  文本总数: {total_text}")
