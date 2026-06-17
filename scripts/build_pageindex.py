from __future__ import annotations

import json

from agent.config import AgentConfig


def main(argv: list[str] | None = None) -> int:
    config = AgentConfig.from_args(argv)
    config.pageindex_dir.mkdir(parents=True, exist_ok=True)
    config.quality_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "stage": "build_pageindex",
                "domain": config.domain,
                "split": config.split,
                "pageindex_dir": str(config.pageindex_dir),
                "pageindex_build_options": config.pageindex_build_options,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
