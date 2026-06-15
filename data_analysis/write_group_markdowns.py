"""将 dataset_stats.json / report.csv 中的 5 个分组分别保存为 Markdown 文档。"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path("/Users/lzc/TNTprojectZ/AprojectZ/AFAC2026/data_analysis")
DATA = BASE / "dataset_stats.json"
OUT_DIR = BASE / "markdown_groups"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUPS = [
    "financial_contracts",
    "insurance",
    "research",
    "financial_reports",
    "regulatory",
]


def esc(s: object) -> str:
    text = "" if s is None else str(s)
    return text.replace("|", "\\|").replace("\n", "<br>")


def write_table(lines: list[str], headers: list[str], rows: list[list[object]]) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(esc(v) for v in row) + " |")


def main() -> int:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    groups = data.get("groups", {})
    written: list[Path] = []

    for group in GROUPS:
        g = groups.get(group, {})
        summary = g.get("summary", {})
        pdfs = g.get("pdfs", [])
        texts = g.get("texts", [])
        method = summary.get("outline_method_distribution", {}) or {}

        lines: list[str] = []
        lines.append(f"# {group} 统计报告")
        lines.append("")
        lines.append("## 分组汇总")
        lines.append("")
        write_table(
            lines,
            ["字段", "值"],
            [
                ["pdf_count", summary.get("pdf_count", 0)],
                ["pdf_total_pages", summary.get("pdf_total_pages", 0)],
                ["pdf_total_chars", summary.get("pdf_total_chars", 0)],
                ["pdf_avg_pages_per_file", summary.get("pdf_avg_pages_per_file", 0)],
                ["pdf_avg_chars_per_page", summary.get("pdf_avg_chars_per_page", 0)],
                ["outline_count", method.get("outline", 0)],
                ["text_toc_count", sum(v for k, v in method.items() if str(k).startswith("text-"))],
                ["failed_count", method.get("failed", 0)],
                ["text_count", summary.get("text_count", 0)],
                ["text_total_lines", summary.get("text_total_lines", 0)],
                ["text_total_chars", summary.get("text_total_chars", 0)],
            ],
        )
        lines.append("")

        lines.append("## PDF 文件明细")
        lines.append("")
        if pdfs:
            write_table(
                lines,
                ["path", "page_count", "total_chars", "avg_chars_per_page", "outline_method", "outline_count", "error"],
                [
                    [
                        r.get("path", ""),
                        r.get("page_count", 0),
                        r.get("total_chars", 0),
                        r.get("avg_chars_per_page", 0),
                        r.get("outline_method", ""),
                        r.get("outline_count", 0),
                        r.get("error") or "",
                    ]
                    for r in pdfs
                ],
            )
        else:
            lines.append("无 PDF 文件。")
        lines.append("")

        lines.append("## 文本文件明细")
        lines.append("")
        if texts:
            write_table(
                lines,
                ["path", "type", "line_count", "total_chars", "error"],
                [
                    [
                        r.get("path", ""),
                        r.get("type", ""),
                        r.get("line_count", 0),
                        r.get("total_chars", 0),
                        r.get("error") or "",
                    ]
                    for r in texts
                ],
            )
        else:
            lines.append("无文本文件。")
        lines.append("")

        lines.append("## PDF 目录提取结果")
        lines.append("")
        for r in pdfs:
            lines.append(f"### {r.get('path', '')}")
            lines.append("")
            lines.append(f"- outline_method: `{r.get('outline_method', '')}`")
            lines.append(f"- outline_count: `{r.get('outline_count', 0)}`")
            outline = r.get("outline", []) or []
            if outline:
                write_table(
                    lines,
                    ["level", "title", "page"],
                    [[item.get("level", ""), item.get("title", ""), item.get("page", "")] for item in outline],
                )
            else:
                lines.append("目录读取失败或无目录。")
            lines.append("")

        out = OUT_DIR / f"{group}.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        written.append(out)

    print("[OK] 已生成 Markdown 文档：")
    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
