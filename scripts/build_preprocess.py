from __future__ import annotations

import json

from agent.config import AgentConfig
from agent.preprocess import preprocess_domain


def main(argv: list[str] | None = None) -> int:
    config = AgentConfig.from_args(argv)
    result = preprocess_domain(config)
    print(
        json.dumps(
            {
                "stage": "build_preprocess",
                "domain": config.domain,
                "split": config.split,
                "doc_count": result["doc_count"],
                "page_count": result["page_count"],
                "pages_dir": str(config.pages_dir),
                "markdown_dir": str(config.markdown_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
