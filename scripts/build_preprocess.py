from __future__ import annotations

import json

from agent.config import AgentConfig


def main(argv: list[str] | None = None) -> int:
    config = AgentConfig.from_args(argv)
    config.pages_dir.mkdir(parents=True, exist_ok=True)
    config.markdown_dir.mkdir(parents=True, exist_ok=True)
    config.quality_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "stage": "build_preprocess",
                "domain": config.domain,
                "split": config.split,
                "pages_dir": str(config.pages_dir),
                "markdown_dir": str(config.markdown_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
