#!/usr/bin/env python3
"""CLI: run the full answer pipeline (Task 12).

Builds all dependencies from an AgentConfig, runs the end-to-end pipeline,
and writes answer.csv, evidence.jsonl, and usage.jsonl to the output directory.

Usage:
    uv run python scripts/run_answers.py --domain insurance --split A
    uv run python scripts/run_answers.py --domain insurance --split A --mock --limit 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.catalog import DocCatalog
from agent.config import AgentConfig, add_cli_arguments
from agent.domain_profiles import get_profile
from agent.index_store import IndexStore
from agent.llm_client import LLMClient
from agent.pipeline import run_all
from agent.token_meter import TokenMeter


def main() -> None:
    parser = argparse.ArgumentParser(description="Run answer pipeline")
    add_cli_arguments(parser)
    parser.add_argument(
        "--mock", action="store_true",
        help="Force mock LLM client (no network, deterministic dry-run)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit to first N questions (quick smoke test)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging",
    )
    parser.add_argument(
        "--max-llm-retries", type=int, default=1,
        help="Max LLM call retries (default: 1; set 0 for fast fail)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = AgentConfig.from_args(args)
    profile = get_profile(config.domain)

    print(f"[run_answers] domain={config.domain} split={config.split}")
    print(f"  questions_path  = {config.questions_path}")
    print(f"  output_dir      = {config.output_dir}")
    print(f"  logs_dir        = {config.logs_dir}")
    print(f"  model           = {config.inference_model}")
    print(f"  mock            = {args.mock}")
    if args.limit is not None:
        print(f"  limit           = {args.limit}")

    # Build dependencies
    catalog = DocCatalog.load(config.catalog_path)
    index_store = IndexStore(config)
    llm_client = LLMClient.from_config(config, force_mock=args.mock, max_retries=args.max_llm_retries)
    token_meter = TokenMeter(logs_dir=config.logs_dir)

    # Run pipeline
    result = run_all(
        config=config,
        profile=profile,
        catalog=catalog,
        index_store=index_store,
        llm_client=llm_client,
        token_meter=token_meter,
        limit=args.limit,
    )

    summary = result["summary"]
    paths = result["paths"]

    print()
    print(f"[run_answers] Complete!")
    print(f"  Questions run:       {summary['questions_run']}")
    print(f"  Fallbacks triggered: {summary['fallbacks_triggered']}")
    print(f"  Total tokens:        {summary['total_tokens']}")
    print(f"  Output files:")
    for name, p in sorted(paths.items()):
        print(f"    {name}: {p}")

    sys.exit(0)


if __name__ == "__main__":
    main()
