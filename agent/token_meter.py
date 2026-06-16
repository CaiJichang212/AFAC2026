"""Token usage aggregation skeleton.

Tracks token usage across all pipeline stages and questions, writes JSONL logs,
and produces summary rows suitable for answer.csv.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent.schemas import UsageRecord


@dataclass
class TokenMeter:
    """Accumulates usage records and writes logs / summaries.

    Usage::

        from agent.schemas import UsageRecord

        meter = TokenMeter(logs_dir=Path("outputs/insurance_a/logs"))
        meter.record(UsageRecord(
            qid="ins_a_001", stage="evidence", model="qwen3.6-plus",
            prompt_tokens=1200, completion_tokens=300, latency_ms=4500,
        ))
        # ... more records ...
        meter.write_log()           # writes usage.jsonl
        summary = meter.summary()   # returns dict for answer.csv
    """

    logs_dir: Path = field(default_factory=lambda: Path("outputs/logs"))
    _records: list[UsageRecord] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, record: UsageRecord, /) -> None:
        """Append one usage record."""
        self._records.append(record)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def write_log(self, filename: str = "usage.jsonl") -> Path:
        """Write accumulated records as JSONL to logs_dir."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        path = self.logs_dir / filename
        with open(path, "w", encoding="utf-8") as fh:
            for rec in self._records:
                fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
        return path

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return an aggregate summary dict suitable for answer.csv metadata.

        The returned dict includes:
        - total_prompt_tokens / total_completion_tokens / total_tokens
        - total_calls / successful_calls / failed_calls
        - total_latency_ms
        - per-stage breakdowns
        """
        if not self._records:
            return {
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_latency_ms": 0.0,
            }

        total_prompt = sum(r.prompt_tokens for r in self._records)
        total_completion = sum(r.completion_tokens for r in self._records)
        total = sum(r.total_tokens for r in self._records)
        total_latency = sum(r.latency_ms for r in self._records)
        successful = sum(1 for r in self._records if r.success)
        failed = sum(1 for r in self._records if not r.success)

        # Per-stage breakdown
        stages: dict[str, dict[str, int]] = {}
        for r in self._records:
            stage = r.stage or "unknown"
            if stage not in stages:
                stages[stage] = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
            stages[stage]["calls"] += 1
            stages[stage]["prompt_tokens"] += r.prompt_tokens
            stages[stage]["completion_tokens"] += r.completion_tokens

        return {
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total,
            "total_calls": len(self._records),
            "successful_calls": successful,
            "failed_calls": failed,
            "total_latency_ms": total_latency,
            "per_stage": stages,
        }

    def summary_csv_row(self) -> dict[str, str]:
        """Return a flat dict for one row in answer.csv."""
        s = self.summary()
        row: dict[str, str] = {}
        for key in (
            "total_prompt_tokens",
            "total_completion_tokens",
            "total_tokens",
            "total_calls",
            "successful_calls",
            "failed_calls",
            "total_latency_ms",
        ):
            row[key] = str(s.get(key, ""))
        return row

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def records_for_qid(self, qid: str) -> list[UsageRecord]:
        """Return all records for a specific question."""
        return [r for r in self._records if r.qid == qid]

    def records_for_stage(self, stage: str) -> list[UsageRecord]:
        """Return all records for a specific pipeline stage."""
        return [r for r in self._records if r.stage == stage]

    @property
    def record_count(self) -> int:
        return len(self._records)
