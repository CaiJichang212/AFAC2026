import json
from pathlib import Path

from agent.config import AgentConfig
from agent.preprocess import preprocess_domain


def test_preprocess_generates_full_insurance_artifacts(tmp_path: Path) -> None:
    config = AgentConfig(
        processed_root=tmp_path / "processed",
        output_root=tmp_path / "outputs",
    )

    result = preprocess_domain(config)

    assert result["doc_count"] == 16
    assert result["page_count"] == 299

    page_files = sorted(config.pages_dir.glob("*.jsonl"), key=lambda path: int(path.stem))
    markdown_files = sorted(config.markdown_dir.glob("*.md"), key=lambda path: int(path.stem))
    page_map_files = sorted(
        config.markdown_dir.glob("*.page_map.json"), key=lambda path: int(path.stem.split(".")[0])
    )

    assert len(page_files) == 16
    assert len(markdown_files) == 16
    assert len(page_map_files) == 16

    total_pages = 0
    for page_file in page_files:
        records = [json.loads(line) for line in page_file.read_text(encoding="utf-8").splitlines()]
        assert records
        assert all(record["text"].strip() for record in records)
        total_pages += len(records)

    assert total_pages == 299

    sample_page_map = json.loads(page_map_files[0].read_text(encoding="utf-8"))
    assert sample_page_map["doc_id"] == "1"
    assert sample_page_map["line_to_page"]

    quality_path = config.quality_dir / "insurance_parse_quality.jsonl"
    quality_records = [
        json.loads(line) for line in quality_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(quality_records) == 16
    assert all(record["status"] == "ok" for record in quality_records)
