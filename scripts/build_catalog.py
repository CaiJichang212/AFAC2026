from __future__ import annotations

import json

from agent.catalog import build_catalog, load_catalog
from agent.config import AgentConfig


def main(argv: list[str] | None = None) -> int:
    config = AgentConfig.from_args(argv)
    path = build_catalog(config)
    catalog = load_catalog(path)
    print(
        json.dumps(
            {
                "stage": "build_catalog",
                "domain": config.domain,
                "split": config.split,
                "catalog_path": str(path),
                "doc_count": len(catalog),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
