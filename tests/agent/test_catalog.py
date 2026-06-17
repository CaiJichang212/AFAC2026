import json
from pathlib import Path

from agent.catalog import build_catalog, load_catalog
from agent.config import AgentConfig


def test_catalog_covers_insurance_documents_and_question_doc_ids(tmp_path: Path) -> None:
    config = AgentConfig(processed_root=tmp_path / "processed", output_root=tmp_path / "outputs")

    path = build_catalog(config)
    catalog = load_catalog(path)

    assert path == config.catalog_path
    assert len(catalog) == 16
    assert sorted(catalog, key=int) == [str(index) for index in range(1, 17)]

    question_doc_ids = {
        doc_id
        for item in json.loads(config.questions_path.read_text(encoding="utf-8"))
        for doc_id in item["doc_ids"]
    }
    assert question_doc_ids <= set(catalog)
    assert all(record["source_pdf"].endswith(f"{doc_id}.pdf") for doc_id, record in catalog.items())
    assert all(record["aliases"] for record in catalog.values())
