from __future__ import annotations

import json

from agent.config import AgentConfig


def main(argv: list[str] | None = None) -> int:
    config = AgentConfig.from_args(argv)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "stage": "run_answers",
                "domain": config.domain,
                "split": config.split,
                "output_dir": str(config.output_dir),
                "logs_dir": str(config.logs_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
