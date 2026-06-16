#!/usr/bin/env python3
"""CLI: build preprocessing artifacts (Task 3).

Extracts pages from PDFs into per-page JSONL caches, converts pages to
markdown with heading recovery, writes page-to-line mappings, and logs
parse quality.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import AgentConfig, add_cli_arguments
from agent.preprocess import process_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build preprocessing artifacts (page cache + markdown + quality log)"
    )
    add_cli_arguments(parser)
    parser.add_argument(
        "--doc-id",
        default=None,
        help="Process only the given document ID (e.g. '1') for testing",
    )
    args = parser.parse_args()
    config = AgentConfig.from_args(args)

    doc_ids = [args.doc_id] if args.doc_id else None

    print(f"[build_preprocess] domain={config.domain}  split={config.split}")
    print(f"  raw_dir       = {config.raw_dir}")
    print(f"  pages_dir     = {config.pages_dir}")
    print(f"  markdown_dir  = {config.markdown_dir}")
    print(f"  quality_path  = {config.quality_path}")
    if doc_ids:
        print(f"  limiting to doc_id(s): {doc_ids}")

    records = process_all(
        raw_dir=config.raw_dir,
        pages_dir=config.pages_dir,
        markdown_dir=config.markdown_dir,
        quality_path=config.quality_path,
        doc_ids=doc_ids,
    )

    # Summary
    total_pages = sum(r["page_count"] for r in records)
    total_chars = sum(r["char_count"] for r in records)
    errors = [r for r in records if r["status"] != "ok"]

    print(f"\n[build_preprocess] Done.")
    print(f"  Docs processed : {len(records)}")
    print(f"  Total pages    : {total_pages}")
    print(f"  Total chars    : {total_chars}")
    if errors:
        print(f"  Errors         : {len(errors)}")
        for e in errors:
            print(f"    doc {e['doc_id']}: {e['status']} - {e['error']}")
    else:
        print(f"  Errors         : 0")

    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
