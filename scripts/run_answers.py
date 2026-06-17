from __future__ import annotations

import json

from agent.config import AgentConfig
from agent.pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    config = AgentConfig.from_args(argv)
    result = run_pipeline(config)
    print(
        json.dumps(
            {
                "stage": "run_answers",
                "domain": config.domain,
                "split": config.split,
                "question_count": result["question_count"],
                "answer_count": result["answer_count"],
                "output_dir": str(config.output_dir),
                "logs_dir": str(config.logs_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
