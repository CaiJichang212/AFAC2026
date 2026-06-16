"""Build the insurance document catalog (Task 6).

Usage::

    uv run python scripts/build_catalog.py --domain insurance --split A
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent.catalog import build_catalog, write_catalog
from agent.config import AgentConfig, add_cli_arguments
from agent.domain_profiles import get_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the document catalog")
    add_cli_arguments(parser)
    args = parser.parse_args()

    config = AgentConfig.from_args(args)
    profile = get_profile(config.domain)

    rows = build_catalog(
        profile=profile,
        markdown_dir=config.markdown_dir,
        index_quality_path=config.index_quality_path,
    )

    # Ensure output directory exists
    config.catalog_path.parent.mkdir(parents=True, exist_ok=True)

    write_catalog(rows, config.catalog_path)

    # Summary
    doc_ids = list(profile.doc_product_map.keys())
    unclassified: list[str] = []
    for row in rows:
        if row["insurance_type"] == "(其他)":
            unclassified.append(f"  {row['doc_id']}: {row['product_name']}")

    print(f"Catalog written to {config.catalog_path}")
    print(f"  Rows written: {len(rows)}")
    print(f"  Doc IDs: {sorted(int(r['doc_id']) for r in rows)}")
    if unclassified:
        print(f"  Products that could not be classified:")
        for line in unclassified:
            print(line)
    else:
        print(f"  All {len(rows)} products classified successfully.")

    sys.exit(0)


if __name__ == "__main__":
    main()
