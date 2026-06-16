#!/usr/bin/env python3
"""CLI stub: run the full answer pipeline (Task 12).

Currently prints derived paths from config.  Implementation will:
1. Load questions.
2. For each question: retrieve candidate nodes, extract evidence, judge answer.
3. Write answer.csv and evidence.jsonl.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import AgentConfig, add_cli_arguments


def main() -> None:
    parser = argparse.ArgumentParser(description="Run answer pipeline")
    add_cli_arguments(parser)
    args = parser.parse_args()
    config = AgentConfig.from_args(args)

    print(f"[run_answers] domain={config.domain} split={config.split}")
    print(f"  questions_path  = {config.questions_path}")
    print(f"  output_dir      = {config.output_dir}")
    print(f"  logs_dir        = {config.logs_dir}")
    print(f"  model           = {config.inference_model}")
    print(f"  max_docs        = {config.max_docs_per_question}")
    print(f"  max_nodes       = {config.max_nodes_per_doc}")
    print(f"  max_evidence    = {config.max_evidence_per_option}")
    print(f"  max_retry       = {config.max_retry_per_question}")
    print("[run_answers] stub complete (real work in Task 12).")


if __name__ == "__main__":
    main()
