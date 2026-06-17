from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from agent.schemas import UsageRecord, UsageSummary


@dataclass
class TokenMeter:
    usage_by_qid: dict[str, list[UsageRecord]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def record(self, usage: UsageRecord) -> None:
        self.usage_by_qid[usage.qid].append(usage)

    def summarize_qid(self, qid: str) -> UsageSummary:
        records = self.usage_by_qid.get(qid, [])
        return UsageSummary(
            prompt_tokens=sum(record.prompt_tokens for record in records),
            completion_tokens=sum(record.completion_tokens for record in records),
            total_tokens=sum(record.total_tokens for record in records),
        )

    def summarize_all(self) -> UsageSummary:
        return UsageSummary(
            prompt_tokens=sum(
                summary.prompt_tokens
                for summary in (self.summarize_qid(qid) for qid in self.usage_by_qid)
            ),
            completion_tokens=sum(
                summary.completion_tokens
                for summary in (self.summarize_qid(qid) for qid in self.usage_by_qid)
            ),
            total_tokens=sum(
                summary.total_tokens
                for summary in (self.summarize_qid(qid) for qid in self.usage_by_qid)
            ),
        )

    def write_usage_log(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for qid in sorted(self.usage_by_qid):
                for record in self.usage_by_qid[qid]:
                    handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def write_answer_csv(self, output_path: Path, answers: dict[str, str]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "qid",
                    "answer",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                ],
            )
            writer.writeheader()
            summary = self.summarize_all()
            writer.writerow(
                {
                    "qid": "summary",
                    "answer": "",
                    "prompt_tokens": summary.prompt_tokens,
                    "completion_tokens": summary.completion_tokens,
                    "total_tokens": summary.total_tokens,
                }
            )
            for qid, answer in answers.items():
                usage = self.summarize_qid(qid)
                writer.writerow(
                    {
                        "qid": qid,
                        "answer": answer,
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                    }
                )
