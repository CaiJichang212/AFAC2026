from __future__ import annotations

import json

from agent.config import AgentConfig


def main(argv: list[str] | None = None) -> int:
    config = AgentConfig.from_args(argv)
    print(
        json.dumps(
            {
                "stage": "validate_outputs",
                "domain": config.domain,
                "split": config.split,
                "answers_path": str(config.answers_path),
                "evidence_path": str(config.evidence_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
