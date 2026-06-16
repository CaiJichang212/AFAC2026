#!/usr/bin/env python3
"""CLI stub: build preprocessing artifacts (Task 3).

Currently prints derived paths from config.  Implementation will:
1. Extract pages from PDFs into per-page files.
2. Convert pages to markdown.
3. Build document catalog.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import AgentConfig, add_cli_arguments


def main() -> None:
    parser = argparse.ArgumentParser(description="Build preprocessing artifacts")
    add_cli_arguments(parser)
    args = parser.parse_args()
    config = AgentConfig.from_args(args)

    print(f"[build_preprocess] domain={config.domain} split={config.split}")
    print(f"  raw_dir         = {config.raw_dir}")
    print(f"  pages_dir       = {config.pages_dir}")
    print(f"  markdown_dir    = {config.markdown_dir}")
    print(f"  catalog_path    = {config.catalog_path}")
    print("[build_preprocess] stub complete (real work in Task 3).")


if __name__ == "__main__":
    main()
