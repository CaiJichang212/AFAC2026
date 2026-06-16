"""Agent configuration: paths, models, budgets, and CLI integration."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AgentConfig:
    """Immutable configuration for the insurance QA pipeline.

    All paths are derived from domain and split via properties; nothing is
    hard-coded for a specific domain/split pair.
    """

    # -- core identifiers ---------------------------------------------------
    domain: str = "insurance"
    split: str = "A"

    # -- root directories (can be overridden via env / CLI) ------------------
    raw_root: Path = field(default_factory=lambda: Path("data/public_dataset_upload/raw"))
    questions_root: Path = field(default_factory=lambda: Path("data/public_dataset_upload/questions"))
    processed_root: Path = field(default_factory=lambda: Path("data/processed_data"))
    pageindex_root: Path = field(default_factory=lambda: Path("open_projects/PageIndex"))
    output_root: Path = field(default_factory=lambda: Path("outputs"))

    # -- model selection -----------------------------------------------------
    # DEV: uses the Volcengine ARK coding plan (ark-code-latest).
    # FINAL SUBMISSION: must regenerate results with a real Qwen model
    # (e.g. dashscope/qwen3.6-plus) and update inference_model accordingly.
    inference_model: str = "ark-code-latest"
    dev_model: str | None = None  # cheaper model for dev / dry-run

    # -- inference endpoint ---------------------------------------------------
    inference_base_url: str = "https://ark.cn-beijing.volces.com/api/coding/v3"
    inference_api_key_env: str = "ARK_API_KEY"  # name of env var holding the key

    # -- PageIndex build parameters (section 4.1) ----------------------------
    toc_check_page_num: int = 20
    max_page_num_each_node: int = 8
    max_token_num_each_node: int = 20000

    # -- online retrieval budget parameters ----------------------------------
    max_docs_per_question: int = 4
    max_nodes_per_doc: int = 5
    max_pages_per_doc: int = 8
    max_evidence_per_option: int = 3
    max_retry_per_question: int = 1

    # -- per-question LLM-call budget (Phase B) --------------------------------
    max_llm_calls_per_question: int = 12

    # -- max_tokens for LLM stages (Phase C) ----------------------------------
    # IMPORTANT: ark-code-latest is a REASONING model.  It needs max_tokens
    # large enough to cover both the reasoning budget (~4500 tokens) and the
    # content output — hence the 8192 / 4096 defaults.  For a NON-reasoning
    # Qwen model at final submission these can be lowered (e.g. 1536 / 1024)
    # via config or env.  We are NOT lowering below the reasoning budget for
    # ark — the compactness win comes from shorter OUTPUT (achieved via the
    # evidence prompt), not by starving the reasoning budget.
    evidence_max_tokens: int = 8192
    tree_max_tokens: int = 4096

    # ------------------------------------------------------------------------
    # Derived paths (properties)
    # ------------------------------------------------------------------------

    @property
    def raw_dir(self) -> Path:
        """Directory containing raw PDFs for this domain."""
        return self.raw_root / self.domain

    @property
    def questions_path(self) -> Path:
        """Path to the questions JSON file for this domain + split."""
        subdir = f"group_{self.split.lower()}"
        filename = f"{self.domain}_questions.json"
        return self.questions_root / subdir / filename

    @property
    def output_dir(self) -> Path:
        """Output directory: outputs/<domain_lower>_<split_lower>."""
        return self.output_root / f"{self.domain.lower()}_{self.split.lower()}"

    @property
    def pages_dir(self) -> Path:
        """Cached per-page PDF extracts."""
        return self.processed_root / "pages" / self.domain

    @property
    def markdown_dir(self) -> Path:
        """Markdown-converted pages."""
        return self.processed_root / "markdown" / self.domain

    @property
    def pageindex_dir(self) -> Path:
        """Serialized PageIndex trees."""
        return self.processed_root / "pageindex" / self.domain

    @property
    def catalog_path(self) -> Path:
        """Document catalog (domain-agnostic JSONL)."""
        return self.processed_root / "catalog" / "doc_catalog.jsonl"

    @property
    def logs_dir(self) -> Path:
        """Run logs directory."""
        return self.output_dir / "logs"

    @property
    def quality_dir(self) -> Path:
        """Parse quality logs directory."""
        return self.processed_root / "quality"

    @property
    def quality_path(self) -> Path:
        """Parse quality JSONL file for this domain."""
        return self.quality_dir / f"{self.domain}_parse_quality.jsonl"

    @property
    def index_quality_path(self) -> Path:
        """Index quality JSONL file for this domain."""
        return self.quality_dir / f"{self.domain}_index_quality.jsonl"

    @property
    def inference_api_key(self) -> str | None:
        """Read the API key from the configured environment variable.

        Returns None when the env var is not set, so callers can fall back
        to a mock client.
        """
        return os.environ.get(self.inference_api_key_env)

    # ------------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------------

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "AgentConfig":
        """Build an AgentConfig from parsed CLI arguments."""
        kwargs: dict = {}
        for field_name in (
            "domain",
            "split",
            "raw_root",
            "questions_root",
            "processed_root",
            "pageindex_root",
            "output_root",
            "inference_model",
            "dev_model",
            "inference_base_url",
            "inference_api_key_env",
            "toc_check_page_num",
            "max_page_num_each_node",
            "max_token_num_each_node",
            "max_docs_per_question",
            "max_nodes_per_doc",
            "max_pages_per_doc",
            "max_evidence_per_option",
            "max_retry_per_question",
            "max_llm_calls_per_question",
            "evidence_max_tokens",
            "tree_max_tokens",
        ):
            value = getattr(args, field_name, None)
            if value is not None:
                # Convert string paths to Path objects for path-typed fields
                if field_name.endswith("_root"):
                    kwargs[field_name] = Path(value)
                else:
                    kwargs[field_name] = value
        return cls(**kwargs)

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Build an AgentConfig from environment variables.

        Every dataclass field is covered: the field's default is used unless
        an AFAC_* env var is set, in which case the value is parsed according
        to the field type (int / float / Path / str).

        Environment variable names follow the pattern AFAC_<FIELD_NAME>.
        """
        import typing

        hints = typing.get_type_hints(cls)
        kwargs: dict[str, object] = {}

        for name, ftype in hints.items():
            env_var = f"AFAC_{name.upper()}"
            value = os.environ.get(env_var)
            if value is None:
                continue  # keep the dataclass default

            # Unwrap Optional[X] -> X for parsing
            origin = typing.get_origin(ftype)
            if origin is not None:
                args = typing.get_args(ftype)
                non_none = [a for a in args if a is not type(None)]
                if non_none:
                    ftype = non_none[0]

            if ftype is int:
                kwargs[name] = int(value)
            elif ftype is float:
                kwargs[name] = float(value)
            elif ftype is Path:
                kwargs[name] = Path(value)
            else:
                kwargs[name] = value

        return cls(**kwargs)


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """Register standard agent arguments on an argparse parser.

    Call this in each script so they all share the same CLI surface.
    """
    parser.add_argument("--domain", default="insurance", help="Task domain (default: insurance)")
    parser.add_argument("--split", default="A", help="Data split: A or B (default: A)")
    parser.add_argument("--raw-root", default="data/public_dataset_upload/raw", help="Raw PDF root")
    parser.add_argument("--questions-root", default="data/public_dataset_upload/questions", help="Questions root")
    parser.add_argument("--processed-root", default="data/processed_data", help="Processed data root")
    parser.add_argument("--pageindex-root", default="open_projects/PageIndex", help="PageIndex source root")
    parser.add_argument("--output-root", default="outputs", help="Output root")
    parser.add_argument("--inference-model", default="ark-code-latest", help="Inference model name")
    parser.add_argument("--dev-model", default=None, help="Dev/dry-run model name")
    parser.add_argument("--inference-base-url", default="https://ark.cn-beijing.volces.com/api/coding/v3", help="Inference endpoint base URL")
    parser.add_argument("--inference-api-key-env", default="ARK_API_KEY", help="Env var name for the inference API key")
    parser.add_argument("--toc-check-page-num", type=int, default=20, help="TOC check page count")
    parser.add_argument("--max-page-num-each-node", type=int, default=8, help="Max pages per index node")
    parser.add_argument("--max-token-num-each-node", type=int, default=20000, help="Max tokens per index node")
    parser.add_argument("--max-docs-per-question", type=int, default=4, help="Max docs per question")
    parser.add_argument("--max-nodes-per-doc", type=int, default=None, help="Max nodes per doc")
    parser.add_argument("--max-pages-per-doc", type=int, default=None, help="Max pages per doc")
    parser.add_argument("--max-evidence-per-option", type=int, default=3, help="Max evidence per option")
    parser.add_argument("--max-retry-per-question", type=int, default=None, help="Max retries per question")
    parser.add_argument("--max-llm-calls-per-question", type=int, default=None, help="Max LLM calls per question")
