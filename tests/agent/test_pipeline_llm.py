import csv
import json
from pathlib import Path

from agent.config import AgentConfig
from agent.pipeline import Pipeline


def _write_minimal_insurance_fixture(config: AgentConfig) -> None:
    config.questions_path.parent.mkdir(parents=True, exist_ok=True)
    config.questions_path.write_text(
        json.dumps(
            [
                {
                    "qid": "ins_a_001",
                    "domain": "insurance",
                    "split": "A",
                    "question": "该产品是否提供保险责任？",
                    "options": {"A": "无保险责任", "B": "提供保险责任"},
                    "answer_format": "mcq",
                    "type": "事实查询",
                    "doc_ids": ["1"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config.catalog_path.parent.mkdir(parents=True, exist_ok=True)
    config.catalog_path.write_text(
        json.dumps({"doc_id": "1", "product_name": "测试保险"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    config.pageindex_dir.mkdir(parents=True, exist_ok=True)
    (config.pageindex_dir / "1.json").write_text(
        json.dumps(
            {
                "structure": [
                    {
                        "title": "保险责任",
                        "node_id": "0001",
                        "nodes": [],
                    }
                ]
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
                    "start_line": 1,
                    "end_line": 1,
                    "start_page": 1,
                    "end_page": 1,
                    "source_page_range": "1-1",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config.pages_dir.mkdir(parents=True, exist_ok=True)
    (config.pages_dir / "1.jsonl").write_text(
        json.dumps(
            {
                "doc_id": "1",
                "page": 1,
                "text": "本产品提供保险责任，具体以条款约定为准。",
                "char_count": 22,
                "source_path": "1.pdf",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def test_pipeline_uses_configured_llm_model_and_records_usage(tmp_path: Path) -> None:
    config = AgentConfig(
        questions_root=tmp_path / "questions",
        processed_root=tmp_path / "processed",
        output_root=tmp_path / "outputs",
        inference_model="ark-code-latest",
        answer_mode="llm",
    )
    _write_minimal_insurance_fixture(config)
    calls = []

    def fake_transport(*, model, prompt, json_schema=None, temperature=0.0):
        calls.append(
            {
                "model": model,
                "prompt": prompt,
                "json_schema": json_schema,
                "temperature": temperature,
            }
        )
        return {
            "model": model,
            "content": {
                "answer": "B",
                "option_judgements": {"A": "refute", "B": "support"},
                "warnings": [],
            },
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 3,
                "total_tokens": 15,
            },
            "success": True,
        }

    result = Pipeline(config, llm_transport=fake_transport).run()

    assert result == {"question_count": 1, "answer_count": 1}
    assert [call["model"] for call in calls] == ["ark-code-latest"]
    assert "不得把目录、阅读指引、泛化标题当作充分证据" in calls[0]["prompt"]
    assert "证据不足时不要强行猜测" in calls[0]["prompt"]
    assert "若证据不足，选择最受证据支持的选项" not in calls[0]["prompt"]
    usage_rows = [
        json.loads(line)
        for line in (config.logs_dir / "usage.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(usage_rows) == 1
    assert usage_rows[0]["qid"] == "ins_a_001"
    assert usage_rows[0]["stage"] == "llm_answer"
    assert usage_rows[0]["model"] == "ark-code-latest"
    assert usage_rows[0]["prompt_tokens"] == 12
    assert usage_rows[0]["completion_tokens"] == 3
    assert usage_rows[0]["total_tokens"] == 15
    assert usage_rows[0]["latency_ms"] >= 0
    assert usage_rows[0]["success"] is True
    assert usage_rows[0]["error"] is None
    rows = list(csv.DictReader(config.answers_path.open(encoding="utf-8")))
    assert rows[0]["qid"] == "summary"
    assert rows[0]["total_tokens"] == "15"
    assert rows[1]["qid"] == "ins_a_001"
    assert rows[1]["answer"] == "B"
    assert rows[1]["total_tokens"] == "15"


def test_llm_mode_fails_fast_when_transport_is_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    monkeypatch.delenv("ARK_BASE_URL", raising=False)
    config = AgentConfig(
        questions_root=tmp_path / "questions",
        processed_root=tmp_path / "processed",
        output_root=tmp_path / "outputs",
        inference_model="ark-code-latest",
        answer_mode="llm",
    )
    _write_minimal_insurance_fixture(config)

    try:
        Pipeline(config).run()
    except RuntimeError as exc:
        assert "LLM transport" in str(exc)
    else:
        raise AssertionError("llm mode should fail when no LLM transport is configured")
