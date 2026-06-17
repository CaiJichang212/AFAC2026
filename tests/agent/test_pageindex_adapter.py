from pathlib import Path

import pytest

from agent.config import AgentConfig
from agent.pageindex_adapter import build_markdown_pageindex, build_pdf_pageindex


def test_markdown_pageindex_calls_low_level_adapter(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    async def fake_md_to_tree(**kwargs):
        calls["md_to_tree"] = kwargs
        return {"doc_name": "demo", "structure": [{"title": "T", "node_id": "0001"}]}

    def forbidden_index(*args, **kwargs):
        raise AssertionError("PageIndexClient.index() must not be used")

    monkeypatch.setattr("agent.pageindex_adapter.md_to_tree", fake_md_to_tree)
    monkeypatch.setattr("agent.pageindex_adapter.PageIndexClient", type("X", (), {"index": forbidden_index}))

    config = AgentConfig(processed_root=tmp_path / "processed", output_root=tmp_path / "outputs")
    config.markdown_dir.mkdir(parents=True, exist_ok=True)
    md_path = config.markdown_dir / "1.md"
    md_path.write_text("# Title\nbody", encoding="utf-8")
    (config.markdown_dir / "1.page_map.json").write_text(
        '{"doc_id":"1","markdown_path":"%s","line_to_page":{"1":1,"2":1}}'
        % md_path.as_posix(),
        encoding="utf-8",
    )

    out_path = build_markdown_pageindex(config, "1", md_path)

    assert out_path.name == "1.json"
    assert calls["md_to_tree"]["if_add_node_summary"] == "no"
    assert calls["md_to_tree"]["if_add_doc_description"] == "no"
    assert calls["md_to_tree"]["if_add_node_text"] == "no"
    assert calls["md_to_tree"]["if_add_node_id"] == "yes"
    assert calls["md_to_tree"]["model"] == config.inference_model


def test_pdf_pageindex_calls_low_level_adapter(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_page_index(**kwargs):
        calls["page_index"] = kwargs
        return {"doc_name": "demo", "structure": [{"title": "T", "node_id": "0001"}]}

    monkeypatch.setattr("agent.pageindex_adapter.page_index", fake_page_index)

    config = AgentConfig(processed_root=tmp_path / "processed", output_root=tmp_path / "outputs")
    pdf_path = tmp_path / "demo.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    out_path = build_pdf_pageindex(config, "1", pdf_path)

    assert out_path.name == "1.pdf_fallback.json"
    assert calls["page_index"]["toc_check_page_num"] == config.toc_check_page_num
    assert calls["page_index"]["max_page_num_each_node"] == config.max_page_num_each_node
    assert calls["page_index"]["max_token_num_each_node"] == config.max_token_num_each_node
    assert calls["page_index"]["if_add_node_summary"] == "no"
    assert calls["page_index"]["if_add_doc_description"] == "no"
    assert calls["page_index"]["if_add_node_text"] == "no"
    assert calls["page_index"]["if_add_node_id"] == "yes"
