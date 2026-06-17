from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PAGEINDEX_ROOT = Path(__file__).resolve().parents[1] / "open_projects" / "PageIndex"
if str(PAGEINDEX_ROOT) not in sys.path:
    sys.path.insert(0, str(PAGEINDEX_ROOT))

from pageindex.page_index import page_index  # type: ignore  # noqa: E402
from pageindex.page_index_md import md_to_tree  # type: ignore  # noqa: E402
from pageindex.client import PageIndexClient  # type: ignore  # noqa: E402

from agent.config import AgentConfig


@dataclass(frozen=True)
class NodeSpan:
    doc_id: str
    node_id: str
    title: str
    start_line: int
    end_line: int
    start_page: int
    end_page: int
    source_page_range: str


def _run_async(coro):
    return asyncio.run(coro)


def _load_page_map(config: AgentConfig, doc_id: str) -> dict[str, Any]:
    page_map_path = config.markdown_dir / f"{doc_id}.page_map.json"
    return json.loads(page_map_path.read_text(encoding="utf-8"))


def _build_spans_from_structure(
    doc_id: str, structure: list[dict[str, Any]], line_to_page: dict[str, int]
) -> list[NodeSpan]:
    flat: list[tuple[dict[str, Any], int]] = []

    def walk(nodes: list[dict[str, Any]], level: int = 1) -> None:
        for node in nodes:
            flat.append((node, level))
            if node.get("nodes"):
                walk(node["nodes"], level + 1)

    walk(structure)

    spans: list[NodeSpan] = []
    for index, (node, level) in enumerate(flat):
        start_line = int(node.get("line_num", 1))
        end_line = None
        for next_node, next_level in flat[index + 1 :]:
            if next_level <= level:
                end_line = int(next_node.get("line_num", start_line))
                break
        if end_line is None:
            end_line = max(int(key) for key in line_to_page) if line_to_page else start_line
        end_line = max(end_line - 1, start_line)
        start_page = int(line_to_page.get(str(start_line), 1))
        end_page = int(line_to_page.get(str(end_line), start_page))
        spans.append(
            NodeSpan(
                doc_id=doc_id,
                node_id=str(node.get("node_id", "")),
                title=str(node.get("title", "")),
                start_line=start_line,
                end_line=end_line,
                start_page=start_page,
                end_page=end_page,
                source_page_range=f"{start_page}-{end_page}",
            )
        )
    return spans


def build_markdown_pageindex(config: AgentConfig, doc_id: str, md_path: Path) -> Path:
    result = _run_async(
        md_to_tree(
            md_path=str(md_path),
            if_add_node_summary="no",
            if_add_doc_description="no",
            if_add_node_text="no",
            if_add_node_id="yes",
            model=config.inference_model,
        )
    )
    out_path = config.pageindex_dir / f"{doc_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    page_map = _load_page_map(config, doc_id)
    spans = _build_spans_from_structure(
        doc_id, result.get("structure", []), {k: int(v) for k, v in page_map["line_to_page"].items()}
    )
    (config.pageindex_dir / f"{doc_id}.node_spans.json").write_text(
        json.dumps([asdict(span) for span in spans], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path


def build_pdf_pageindex(config: AgentConfig, doc_id: str, pdf_path: Path) -> Path:
    result = page_index(
        doc=str(pdf_path),
        model=config.inference_model,
        toc_check_page_num=config.toc_check_page_num,
        max_page_num_each_node=config.max_page_num_each_node,
        max_token_num_each_node=config.max_token_num_each_node,
        if_add_node_summary="no",
        if_add_doc_description="no",
        if_add_node_text="no",
        if_add_node_id="yes",
    )
    out_path = config.pageindex_dir / f"{doc_id}.pdf_fallback.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
