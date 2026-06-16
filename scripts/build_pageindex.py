#!/usr/bin/env python3
"""CLI stub: build PageIndex trees (Task 4).

Currently prints derived paths from config.  Implementation will:
1. Load the document catalog.
2. Invoke PageIndex to build hierarchical trees per document.
3. Serialize trees to disk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import AgentConfig, add_cli_arguments


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PageIndex trees")
    add_cli_arguments(parser)
    args = parser.parse_args()
    config = AgentConfig.from_args(args)

    print(f"[build_pageindex] domain={config.domain} split={config.split}")
    print(f"  markdown_dir    = {config.markdown_dir}")
    print(f"  pageindex_dir   = {config.pageindex_dir}")
    print(f"  catalog_path    = {config.catalog_path}")
    print(f"  toc_check_page  = {config.toc_check_page_num}")
    print(f"  max_page_node   = {config.max_page_num_each_node}")
    print(f"  max_token_node  = {config.max_token_num_each_node}")
    print("[build_pageindex] stub complete (real work in Task 4).")


if __name__ == "__main__":
    main()
