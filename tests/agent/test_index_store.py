import json
from pathlib import Path

from agent.config import AgentConfig
from agent.index_store import IndexStore, validate_node_spans


def test_document_structure_hides_line_num_and_text(tmp_path: Path) -> None:
    config = AgentConfig(processed_root=tmp_path / "processed", output_root=tmp_path / "outputs")
    config.pageindex_dir.mkdir(parents=True, exist_ok=True)
    (config.pageindex_dir / "1.json").write_text(
        json.dumps(
            {
                "doc_name": "doc1",
                "structure": [
                    {
                        "title": "保险责任",
                        "node_id": "0001",
                        "line_num": 2,
                        "text": "hidden",
                        "nodes": [],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (config.pageindex_dir / "1.node_spans.json").write_text(
        json.dumps(
            [
                {
                    "doc_id": "1",
                    "node_id": "0001",
                    "title": "保险责任",
                    "start_line": 2,
                    "end_line": 5,
                    "start_page": 1,
                    "end_page": 2,
                    "source_page_range": "1-2",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    structure = IndexStore(config).get_document_structure("1")

    assert structure == [
        {
            "title": "保险责任",
            "node_id": "0001",
            "summary": None,
            "page_range": "1-2",
            "nodes": [],
            "index_source": "markdown",
        }
    ]
    assert "line_num" not in structure[0]
    assert "text" not in structure[0]


def test_bad_page_range_is_marked_unusable() -> None:
    records = [
        {
            "doc_id": "1",
            "node_id": "bad",
            "title": "坏节点",
            "start_line": 5,
            "end_line": 6,
            "start_page": 3,
            "end_page": 2,
            "source_page_range": "3-2",
        }
    ]

    report = validate_node_spans(records, page_count=5)

    assert report["bad_page_range_count"] == 1
    assert report["status"] == "page_keyword"
