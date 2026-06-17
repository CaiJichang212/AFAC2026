from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class AgentConfig:
    domain: str = "insurance"
    split: str = "A"
    raw_root: Path = Path("data/public_dataset_upload/raw")
    questions_root: Path = Path("data/public_dataset_upload/questions")
    processed_root: Path = Path("data/processed_data")
    pageindex_root: Path = Path("open_projects/PageIndex")
    output_root: Path = Path("outputs")
    inference_model: str = "dashscope/qwen3.6-plus"
    dev_model: str | None = None
    toc_check_page_num: int = 20
    max_page_num_each_node: int = 8
    max_token_num_each_node: int = 20000
    max_docs_per_question: int = 4
    max_nodes_per_doc: int = 5
    max_pages_per_doc: int = 8
    max_evidence_per_option: int = 3
    max_retry_per_question: int = 1

    @property
    def split_lower(self) -> str:
        return self.split.lower()

    @property
    def raw_dir(self) -> Path:
        return self.raw_root / self.domain

    @property
    def questions_path(self) -> Path:
        return self.questions_root / f"group_{self.split_lower}" / f"{self.domain}_questions.json"

    @property
    def output_dir(self) -> Path:
        return self.output_root / f"{self.domain}_{self.split_lower}"

    @property
    def logs_dir(self) -> Path:
        return self.output_dir / "logs"

    @property
    def pages_dir(self) -> Path:
        return self.processed_root / "pages" / self.domain

    @property
    def markdown_dir(self) -> Path:
        return self.processed_root / "markdown" / self.domain

    @property
    def pageindex_dir(self) -> Path:
        return self.processed_root / "pageindex" / self.domain

    @property
    def quality_dir(self) -> Path:
        return self.processed_root / "quality"

    @property
    def catalog_path(self) -> Path:
        return self.processed_root / "catalog" / "doc_catalog.jsonl"

    @property
    def evidence_path(self) -> Path:
        return self.output_dir / "evidence.jsonl"

    @property
    def answers_path(self) -> Path:
        return self.output_dir / "answer.csv"

    @property
    def pageindex_build_options(self) -> dict[str, Any]:
        return {
            "model": self.inference_model,
            "toc_check_page_num": self.toc_check_page_num,
            "max_page_num_each_node": self.max_page_num_each_node,
            "max_token_num_each_node": self.max_token_num_each_node,
            "if_add_node_summary": "no",
            "if_add_doc_description": "no",
            "if_add_node_text": "no",
            "if_add_node_id": "yes",
        }

    @property
    def retrieval_budget(self) -> dict[str, int]:
        return {
            "max_docs_per_question": self.max_docs_per_question,
            "max_nodes_per_doc": self.max_nodes_per_doc,
            "max_pages_per_doc": self.max_pages_per_doc,
            "max_evidence_per_option": self.max_evidence_per_option,
            "max_retry_per_question": self.max_retry_per_question,
        }

    @classmethod
    def from_args(cls, argv: Sequence[str] | None = None) -> "AgentConfig":
        parser = build_arg_parser()
        namespace = parser.parse_args(argv)
        return cls(
            domain=namespace.domain,
            split=namespace.split,
            raw_root=Path(namespace.raw_root),
            questions_root=Path(namespace.questions_root),
            processed_root=Path(namespace.processed_root),
            pageindex_root=Path(namespace.pageindex_root),
            output_root=Path(namespace.output_root),
            inference_model=namespace.inference_model,
            dev_model=namespace.dev_model,
            toc_check_page_num=namespace.toc_check_page_num,
            max_page_num_each_node=namespace.max_page_num_each_node,
            max_token_num_each_node=namespace.max_token_num_each_node,
            max_docs_per_question=namespace.max_docs_per_question,
            max_nodes_per_doc=namespace.max_nodes_per_doc,
            max_pages_per_doc=namespace.max_pages_per_doc,
            max_evidence_per_option=namespace.max_evidence_per_option,
            max_retry_per_question=namespace.max_retry_per_question,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Insurance PageIndex pipeline config")
    parser.add_argument("--domain", default=os.getenv("AFAC_DOMAIN", "insurance"))
    parser.add_argument("--split", default=os.getenv("AFAC_SPLIT", "A"))
    parser.add_argument(
        "--raw-root",
        default=os.getenv("AFAC_RAW_ROOT", "data/public_dataset_upload/raw"),
    )
    parser.add_argument(
        "--questions-root",
        default=os.getenv("AFAC_QUESTIONS_ROOT", "data/public_dataset_upload/questions"),
    )
    parser.add_argument(
        "--processed-root",
        default=os.getenv("AFAC_PROCESSED_ROOT", "data/processed_data"),
    )
    parser.add_argument(
        "--pageindex-root",
        default=os.getenv("AFAC_PAGEINDEX_ROOT", "open_projects/PageIndex"),
    )
    parser.add_argument(
        "--output-root",
        default=os.getenv("AFAC_OUTPUT_ROOT", "outputs"),
    )
    parser.add_argument(
        "--inference-model",
        default=os.getenv("AFAC_INFERENCE_MODEL", "dashscope/qwen3.6-plus"),
    )
    parser.add_argument("--dev-model", default=os.getenv("AFAC_DEV_MODEL"))
    parser.add_argument("--toc-check-page-num", type=int, default=20)
    parser.add_argument("--max-page-num-each-node", type=int, default=8)
    parser.add_argument("--max-token-num-each-node", type=int, default=20000)
    parser.add_argument("--max-docs-per-question", type=int, default=4)
    parser.add_argument("--max-nodes-per-doc", type=int, default=5)
    parser.add_argument("--max-pages-per-doc", type=int, default=8)
    parser.add_argument("--max-evidence-per-option", type=int, default=3)
    parser.add_argument("--max-retry-per-question", type=int, default=1)
    return parser
