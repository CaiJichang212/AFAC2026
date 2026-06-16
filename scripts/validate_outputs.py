#!/usr/bin/env python3
"""Validate output artifacts for the insurance QA pipeline (Task 11).

Checks:
  1. answer.csv  — summary row, qid coverage, answer format validity, token sums.
  2. evidence.jsonl — traceability (each selected answer option has support evidence).
  3. usage.jsonl  — every qid present.

All core validation logic is in pure, importable functions so tests can
exercise them without real files.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from io import StringIO
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import AgentConfig, add_cli_arguments


# =============================================================================
# Pure validation functions (importable, testable without file I/O)
# =============================================================================

def _load_questions(config: AgentConfig) -> list[dict[str, Any]]:
    """Load the questions JSON for the configured domain/split."""
    path = config.questions_path
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_answer_csv_rows(
    rows: list[dict[str, str]],
    questions: list[dict[str, Any]],
) -> list[str]:
    """Validate parsed answer.csv rows against the question set.

    Args:
        rows: List of dicts with keys qid, answer, prompt_tokens,
              completion_tokens, total_tokens (all str values from CSV).
        questions: Parsed questions list (from the JSON file).

    Returns:
        List of failure messages; empty list means all checks pass.
    """
    failures: list[str] = []

    if not rows:
        failures.append("answer.csv is empty (no rows)")
        return failures

    # --- summary row ---
    first = rows[0]
    if first.get("qid", "").strip() != "summary":
        failures.append(
            f"First data row must have qid='summary', got {first.get('qid')!r}"
        )

    # --- question qid set from the real questions file ---
    required_qids: set[str] = {q["qid"] for q in questions}
    # Identify question rows (skip summary)
    question_rows = [r for r in rows if r.get("qid", "").strip() != "summary"]
    present_qids: set[str] = {r["qid"].strip() for r in question_rows}

    missing = required_qids - present_qids
    extra = present_qids - required_qids
    if missing:
        failures.append(f"Missing qids in answer.csv: {sorted(missing)}")
    if extra:
        failures.append(f"Unexpected qids in answer.csv: {sorted(extra)}")

    if len(question_rows) != len(required_qids):
        failures.append(
            f"Expected {len(required_qids)} question rows, got {len(question_rows)}"
        )

    # --- per-row validation ---
    # Build a lookup: qid -> answer_format
    format_map: dict[str, str] = {q["qid"]: q["answer_format"] for q in questions}

    # Track token sums
    summary_total_tokens = 0
    try:
        summary_total_tokens = int(first.get("total_tokens", "0"))
    except (ValueError, TypeError):
        pass

    row_total_sum = 0

    for row in question_rows:
        qid = row.get("qid", "").strip()
        answer = row.get("answer", "").strip()
        fmt = format_map.get(qid, "")

        # --- answer format check ---
        fmt_failure = _validate_answer_format(qid, answer, fmt)
        if fmt_failure:
            failures.append(fmt_failure)

        # --- token fields ---
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            val = row.get(field, "").strip()
            try:
                ival = int(val)
                if ival < 0:
                    failures.append(f"{qid}: {field} is negative ({val})")
            except (ValueError, TypeError):
                failures.append(f"{qid}: {field} is not a non-negative integer ({val!r})")

            if field == "total_tokens":
                try:
                    row_total_sum += int(val)
                except (ValueError, TypeError):
                    pass

    # --- token sum check ---
    if summary_total_tokens > 0 and row_total_sum > 0:
        if summary_total_tokens != row_total_sum:
            failures.append(
                f"summary.total_tokens ({summary_total_tokens}) != "
                f"sum of question total_tokens ({row_total_sum})"
            )

    return failures


def _validate_answer_format(qid: str, answer: str, fmt: str) -> str | None:
    """Validate a single answer against its expected format.

    Returns a failure message string, or None if valid.
    """
    if not answer:
        return f"{qid}: answer is empty"

    if fmt == "mcq":
        if not re.fullmatch(r"^[A-Z]$", answer):
            return f"{qid}: mcq answer must be a single uppercase letter, got {answer!r}"
    elif fmt == "multi":
        # Sorted unique uppercase letters, no separators
        if not re.fullmatch(r"^[A-Z]+$", answer):
            return f"{qid}: multi answer must be uppercase letters only, got {answer!r}"
        # Check sorted
        if answer != "".join(sorted(answer)):
            return f"{qid}: multi answer must be sorted, got {answer!r}"
        # Check no duplicates
        if len(answer) != len(set(answer)):
            return f"{qid}: multi answer must have no duplicates, got {answer!r}"
    elif fmt == "tf":
        if answer not in ("A", "B"):
            return f"{qid}: tf answer must be 'A' or 'B', got {answer!r}"
    else:
        return f"{qid}: unknown answer_format {fmt!r}"

    return None


def validate_evidence_jsonl_lines(
    lines: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> list[str]:
    """Validate evidence.jsonl records for traceability.

    Each line has keys: qid, answer, ..., evidence (list), option_judgements.
    For each selected answer option, there must be at least one evidence record
    with evidence_type='support'.

    Args:
        lines: List of parsed JSON objects (one per qid).
        questions: Parsed questions list.

    Returns:
        List of failure messages; empty list means all checks pass.
    """
    failures: list[str] = []

    # Index lines by qid
    by_qid: dict[str, dict[str, Any]] = {rec.get("qid", ""): rec for rec in lines}

    for q in questions:
        qid = q["qid"]
        rec = by_qid.get(qid)
        if rec is None:
            failures.append(f"evidence.jsonl: missing record for {qid}")
            continue

        answer = rec.get("answer", "")
        evidence_list: list[dict[str, Any]] = rec.get("evidence", [])

        if not answer:
            failures.append(f"{qid}: answer field is empty in evidence.jsonl")
            continue

        fmt = q.get("answer_format", "mcq")

        # Determine which options were selected
        if fmt == "tf":
            # Only A or B is selected
            selected_opts = [answer] if answer in ("A", "B") else []
        elif fmt == "mcq":
            selected_opts = [answer] if len(answer) == 1 else []
        elif fmt == "multi":
            selected_opts = list(answer)
        else:
            selected_opts = list(answer)

        for opt in selected_opts:
            has_support = any(
                e.get("option") == opt and e.get("evidence_type") == "support"
                for e in evidence_list
            )
            if not has_support:
                failures.append(
                    f"{qid}: answer option {opt!r} has no support evidence record"
                )

    return failures


def validate_usage_jsonl_lines(
    lines: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> list[str]:
    """Validate usage.jsonl — every question qid must be present.

    Args:
        lines: List of parsed JSON objects (one per line).
        questions: Parsed questions list.

    Returns:
        List of failure messages; empty list means all checks pass.
    """
    failures: list[str] = []
    present_qids: set[str] = {rec.get("qid", "") for rec in lines}

    required_qids: set[str] = {q["qid"] for q in questions}
    missing = required_qids - present_qids
    if missing:
        failures.append(f"usage.jsonl: missing qids: {sorted(missing)}")

    return failures


# =============================================================================
# File-reading wrappers (for CLI — not needed by tests)
# =============================================================================


def _read_answer_csv(path: Path) -> list[dict[str, str]]:
    """Read answer.csv into a list of dicts."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate output artifacts")
    add_cli_arguments(parser)
    args = parser.parse_args()
    config = AgentConfig.from_args(args)

    questions = _load_questions(config)
    if not questions:
        print("[validate_outputs] ERROR: no questions loaded — cannot validate qid coverage")
        sys.exit(1)

    all_failures: list[str] = []

    # --- answer.csv ---
    answer_path = config.output_dir / "answer.csv"
    if answer_path.exists():
        rows = _read_answer_csv(answer_path)
        failures = validate_answer_csv_rows(rows, questions)
        all_failures.extend(failures)
        print(f"[validate_outputs] answer.csv : {len(rows)} rows, "
              f"{len(failures)} failure(s)")
    else:
        msg = f"answer.csv not found at {answer_path}"
        all_failures.append(msg)
        print(f"[validate_outputs] {msg}")

    # --- evidence.jsonl ---
    evidence_path = config.output_dir / "evidence.jsonl"
    if evidence_path.exists():
        lines = _read_jsonl(evidence_path)
        failures = validate_evidence_jsonl_lines(lines, questions)
        all_failures.extend(failures)
        print(f"[validate_outputs] evidence.jsonl : {len(lines)} records, "
              f"{len(failures)} failure(s)")
    else:
        msg = f"evidence.jsonl not found at {evidence_path}"
        all_failures.append(msg)
        print(f"[validate_outputs] {msg}")

    # --- usage.jsonl ---
    usage_path = config.output_dir / "usage.jsonl"
    if usage_path.exists():
        lines = _read_jsonl(usage_path)
        failures = validate_usage_jsonl_lines(lines, questions)
        all_failures.extend(failures)
        print(f"[validate_outputs] usage.jsonl : {len(lines)} records, "
              f"{len(failures)} failure(s)")
    else:
        print(f"[validate_outputs] usage.jsonl not found at {usage_path} (skipped)")

    # --- report ---
    print()
    if all_failures:
        print(f"VALIDATION FAILED ({len(all_failures)} issue(s)):")
        for f in all_failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("VALIDATION PASSED — all checks OK.")
        sys.exit(0)


if __name__ == "__main__":
    main()
