from __future__ import annotations

import json

from agent.config import AgentConfig
from agent.index_store import validate_node_spans
from agent.pageindex_adapter import build_markdown_pageindex


def main(argv: list[str] | None = None) -> int:
    config = AgentConfig.from_args(argv)
    config.pageindex_dir.mkdir(parents=True, exist_ok=True)
    config.quality_dir.mkdir(parents=True, exist_ok=True)
    results: list[str] = []
    quality_records: list[dict[str, object]] = []
    for md_path in sorted(config.markdown_dir.glob("*.md"), key=lambda path: int(path.stem)):
        doc_id = md_path.stem
        results.append(str(build_markdown_pageindex(config, doc_id, md_path)))
        spans = json.loads((config.pageindex_dir / f"{doc_id}.node_spans.json").read_text(encoding="utf-8"))
        report = validate_node_spans(spans)
        quality_records.append({"doc_id": doc_id, **report})
    quality_path = config.quality_dir / f"{config.domain}_index_quality.jsonl"
    with quality_path.open("w", encoding="utf-8") as handle:
        for record in quality_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "stage": "build_pageindex",
                "domain": config.domain,
                "split": config.split,
                "built": len(results),
                "pageindex_dir": str(config.pageindex_dir),
                "pageindex_build_options": config.pageindex_build_options,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
