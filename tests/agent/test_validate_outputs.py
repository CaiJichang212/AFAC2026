import csv
import json
from pathlib import Path

import pytest

from agent.config import AgentConfig
from scripts.validate_outputs import validate_outputs


def test_validate_outputs_accepts_schema_and_traceable_evidence(tmp_path: Path) -> None:
    config = AgentConfig(processed_root=tmp_path / "processed", output_root=tmp_path / "outputs")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with config.answers_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "qid": "summary",
                "answer": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        )
        for index in range(1, 21):
            writer.writerow(
                {
                    "qid": f"ins_a_{index:03d}",
                    "answer": "A",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            )
    with config.evidence_path.open("w", encoding="utf-8") as handle:
        for index in range(1, 21):
            handle.write(
                json.dumps(
                    {
                        "qid": f"ins_a_{index:03d}",
                        "answer": "A",
                        "candidate_docs": ["1"],
                        "selected_nodes": [{"doc_id": "1", "node_id": "0001", "pages": "1-1"}],
                        "evidence": [
                            {
                                "qid": f"ins_a_{index:03d}",
                                "doc_id": "1",
                                "node_id": "0001",
                                "pages": "1-1",
                                "option": "A",
                                "evidence_type": "support",
                                "quote": "保险责任",
                                "normalized_fact": "保险责任",
                                "numbers": [],
                                "confidence": "high",
                            }
                        ],
                        "calculations": [],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        "fallbacks": [],
                        "warnings": [],
                        "option_judgements": {"A": "support"},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    validate_outputs(config)


def test_validate_outputs_rejects_missing_support_for_selected_option(tmp_path: Path) -> None:
    config = AgentConfig(processed_root=tmp_path / "processed", output_root=tmp_path / "outputs")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.answers_path.write_text(
        "qid,answer,prompt_tokens,completion_tokens,total_tokens\nsummary,,0,0,0\nins_a_001,A,0,0,0\n",
        encoding="utf-8",
    )
    config.evidence_path.write_text(
        json.dumps(
            {
                "qid": "ins_a_001",
                "answer": "A",
                "candidate_docs": ["1"],
                "selected_nodes": [],
                "evidence": [],
                "calculations": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "fallbacks": [],
                "warnings": [],
                "option_judgements": {},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="support"):
        validate_outputs(config, expected_qids=["ins_a_001"])
