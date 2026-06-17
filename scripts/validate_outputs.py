from __future__ import annotations

import csv
import json

from agent.config import AgentConfig


def validate_outputs(config: AgentConfig, expected_qids: list[str] | None = None) -> None:
    if expected_qids is None:
        questions = json.loads(config.questions_path.read_text(encoding="utf-8"))
        expected_qids = [item["qid"] for item in questions]

    answer_rows = list(csv.DictReader(config.answers_path.open(encoding="utf-8")))
    if not answer_rows or answer_rows[0]["qid"] != "summary":
        raise ValueError("answer.csv first data row must be summary")

    rows_by_qid = {row["qid"]: row for row in answer_rows[1:]}
    missing = set(expected_qids) - set(rows_by_qid)
    if missing:
        raise ValueError(f"answer.csv missing qids: {sorted(missing)}")

    summary_total = int(answer_rows[0]["total_tokens"])
    row_total = sum(int(row["total_tokens"]) for row in answer_rows[1:])
    if summary_total != row_total:
        raise ValueError("summary total_tokens must equal question row total")

    evidence_by_qid = {}
    for line in config.evidence_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        evidence_by_qid[record["qid"]] = record

    for qid in expected_qids:
        answer = rows_by_qid[qid]["answer"]
        if not answer:
            raise ValueError(f"{qid} has empty answer")
        evidence_record = evidence_by_qid.get(qid)
        if not evidence_record:
            raise ValueError(f"{qid} missing evidence")
        for option in answer:
            support = [
                item
                for item in evidence_record.get("evidence", [])
                if item.get("option") == option and item.get("evidence_type") == "support"
            ]
            if not support:
                raise ValueError(f"{qid} selected option {option} missing support evidence")


def main(argv: list[str] | None = None) -> int:
    config = AgentConfig.from_args(argv)
    validate_outputs(config)
    print(
        json.dumps(
            {
                "stage": "validate_outputs",
                "domain": config.domain,
                "split": config.split,
                "answers_path": str(config.answers_path),
                "evidence_path": str(config.evidence_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
