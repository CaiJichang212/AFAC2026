#!/usr/bin/env python3
"""CLI: build PageIndex trees for insurance documents (Task 4).

Usage::

    # Smoke-test a single document (no real API calls for markdown path)
    uv run python scripts/build_pageindex.py --domain insurance --split A --doc-id 1

    # Build all discovered markdown files
    uv run python scripts/build_pageindex.py --domain insurance --split A
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import AgentConfig, add_cli_arguments
from agent.pageindex_adapter import build_md_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PageIndex trees")
    add_cli_arguments(parser)
    parser.add_argument(
        "--doc-id",
        type=int,
        default=None,
        help="Build index for a single doc-id (default: discover and build all)",
    )
    args = parser.parse_args()
    config = AgentConfig.from_args(args)

    markdown_dir = config.markdown_dir
    pageindex_dir = config.pageindex_dir
    pageindex_dir.mkdir(parents=True, exist_ok=True)

    # Determine which documents to process
    if args.doc_id is not None:
        doc_ids = [args.doc_id]
    else:
        doc_ids = sorted(
            int(p.stem) for p in markdown_dir.glob("*.md") if p.stem.isdigit()
        )

    print(f"[build_pageindex] domain={config.domain}  split={config.split}")
    print(f"  markdown_dir  = {markdown_dir}")
    print(f"  pageindex_dir = {pageindex_dir}")
    print(f"  doc_ids       = {doc_ids}")

    for doc_id in doc_ids:
        md_path = markdown_dir / f"{doc_id}.md"
        if not md_path.exists():
            print(f"[build_pageindex] SKIP doc {doc_id}: markdown not found at {md_path}")
            continue

        print(f"[build_pageindex] Building index for doc {doc_id} ...")
        tree = asyncio.run(build_md_index(md_path, config))

        out_path = pageindex_dir / f"{doc_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(tree, f, ensure_ascii=False, indent=2)
        print(f"[build_pageindex] Wrote {out_path}  ({tree.get('line_count', '?')} lines)")

    print("[build_pageindex] done.")


if __name__ == "__main__":
    main()
