#!/usr/bin/env python3
"""CLI: build PageIndex trees for insurance documents (Tasks 4 & 5).

Usage::

    # Smoke-test a single document (no real API calls for markdown path)
    uv run python scripts/build_pageindex.py --domain insurance --split A --doc-id 1

    # Build all discovered markdown files (API-free main path)
    uv run python scripts/build_pageindex.py --domain insurance --split A

    # Build ALL with explicit PDF fallback (requires real Qwen API)
    uv run python scripts/build_pageindex.py --domain insurance --split A --enable-fallback

By default (without ``--enable-fallback``), the build is fully API-free and
deterministic: all 16 documents are built via the markdown main path.  Documents
whose index quality is BAD are assigned ``index_source="page_keyword"`` and
``status="degraded"`` WITHOUT calling the PDF fallback path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import AgentConfig, add_cli_arguments
from agent.domain_profiles import get_profile
from agent.index_store import compute_index_quality, compute_node_spans
from agent.pageindex_adapter import build_md_index, build_pdf_fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PageIndex trees and compute node spans + quality")
    add_cli_arguments(parser)
    parser.add_argument(
        "--doc-id",
        type=int,
        default=None,
        help="Build index for a single doc-id (default: discover and build all)",
    )
    parser.add_argument(
        "--enable-fallback",
        action="store_true",
        default=False,
        help="Enable PDF fallback via real LLM API calls for bad-quality docs (default: off, API-free)",
    )
    args = parser.parse_args()
    config = AgentConfig.from_args(args)

    markdown_dir = config.markdown_dir
    pageindex_dir = config.pageindex_dir
    pageindex_dir.mkdir(parents=True, exist_ok=True)
    config.quality_dir.mkdir(parents=True, exist_ok=True)

    # Domain profile (keywords for quality)
    profile = get_profile(config.domain)
    keywords: list[str] = profile.keywords
    thresholds: dict = profile.quality_thresholds

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
    print(f"  quality_dir   = {config.quality_dir}")
    print(f"  enable_fallback = {args.enable_fallback}")
    print(f"  doc_ids       = {doc_ids}")

    quality_records: list[dict] = []
    fallback_calls: int = 0
    ok_count = 0
    degraded_count = 0
    error_count = 0

    for doc_id in doc_ids:
        md_path = markdown_dir / f"{doc_id}.md"
        if not md_path.exists():
            print(f"[build_pageindex] SKIP doc {doc_id}: markdown not found at {md_path}")
            continue

        # ---- Phase 1: build tree via main (markdown) path ----
        print(f"[build_pageindex] Building index for doc {doc_id} ...")
        tree = asyncio.run(build_md_index(md_path, config))

        out_path = pageindex_dir / f"{doc_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(tree, f, ensure_ascii=False, indent=2)
        print(f"  Wrote tree: {out_path}  ({tree.get('line_count', '?')} lines, "
              f"{len(tree.get('structure', []))} nodes)")

        # ---- Phase 2: compute node_spans ----
        pm_path = markdown_dir / f"{doc_id}.page_map.json"
        if pm_path.exists():
            with open(pm_path, encoding="utf-8") as f:
                page_map = json.load(f)
            line_to_page = page_map.get("line_to_page", {})
        else:
            print(f"  WARNING: no page_map for doc {doc_id}, using empty mapping")
            line_to_page = {}

        spans = compute_node_spans(tree, line_to_page)
        spans_path = pageindex_dir / f"{doc_id}.node_spans.json"
        with open(spans_path, "w", encoding="utf-8") as f:
            json.dump(spans, f, ensure_ascii=False, indent=2)
        print(f"  Wrote node_spans: {spans_path}  ({len(spans)} nodes)")

        # ---- Phase 3: compute index quality ----
        quality = compute_index_quality(
            doc_id=doc_id,
            spans=spans,
            keywords=keywords,
            index_source="markdown",  # default; may be upgraded below
        )

        # ---- Phase 4: decide degraded / fallback ----
        if quality["status"] == "degraded" or quality["status"] == "error":
            if args.enable_fallback:
                # Attempt PDF fallback via real LLM call
                pdf_path = config.raw_dir / f"{doc_id}.pdf"
                if pdf_path.exists():
                    print(f"  [fallback] Running PDF fallback for doc {doc_id} ...")
                    try:
                        fb_tree = build_pdf_fallback(pdf_path, config)
                        fallback_calls += 1

                        fb_out_path = pageindex_dir / f"{doc_id}.json"
                        with open(fb_out_path, "w", encoding="utf-8") as f:
                            json.dump(fb_tree, f, ensure_ascii=False, indent=2)

                        fb_spans = compute_node_spans(fb_tree, line_to_page)
                        with open(spans_path, "w", encoding="utf-8") as f:
                            json.dump(fb_spans, f, ensure_ascii=False, indent=2)

                        quality = compute_index_quality(
                            doc_id=doc_id,
                            spans=fb_spans,
                            keywords=keywords,
                            index_source="pdf_fallback",
                        )
                        print(f"  [fallback] PDF fallback complete; "
                              f"status={quality['status']} nodes={len(fb_spans)}")
                    except Exception as exc:
                        print(f"  [fallback] PDF fallback FAILED for doc {doc_id}: {exc}")
                        # Keep original quality; mark as error if it was error
                        quality["index_source"] = "page_keyword"
                else:
                    print(f"  [fallback] PDF not found for doc {doc_id}, keeping markdown result")
                    quality["index_source"] = "page_keyword"
            else:
                # Fallback disabled: downgrade to page_keyword degraded status
                quality["index_source"] = "page_keyword"
                print(f"  Degraded doc {doc_id}: node_count={quality['node_count']} "
                      f"coverage={quality['page_mapping_coverage']:.2f} "
                      f"(use --enable-fallback to trigger PDF path)")

        # Track status counts
        if quality["status"] == "ok":
            ok_count += 1
        elif quality["status"] == "degraded":
            degraded_count += 1
        else:
            error_count += 1

        quality_records.append(quality)
        print(f"  Quality: status={quality['status']}  source={quality['index_source']}  "
              f"nodes={quality['node_count']}  bad_ranges={quality['bad_page_range_count']}  "
              f"coverage={quality['page_mapping_coverage']:.4f}  "
              f"keyword_hits={quality['keyword_title_hits']}")

    # ---- Phase 5: write quality log ----
    quality_path = config.index_quality_path
    with open(quality_path, "w", encoding="utf-8") as f:
        for rec in quality_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[build_pageindex] Wrote index quality log: {quality_path}  ({len(quality_records)} docs)")

    # ---- Summary ----
    print(f"\n[build_pageindex] Summary:")
    print(f"  Total docs processed: {len(quality_records)}")
    print(f"  OK:       {ok_count}")
    print(f"  Degraded: {degraded_count}")
    print(f"  Error:    {error_count}")
    print(f"  Fallback API calls: {fallback_calls}")
    print(f"[build_pageindex] done.")


if __name__ == "__main__":
    main()
