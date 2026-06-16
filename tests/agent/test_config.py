"""Tests for agent/config.py — path derivation, defaults, and CLI parsing."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from agent.config import AgentConfig, add_cli_arguments


# ---------------------------------------------------------------------------
# Test 1: --domain insurance --split A derives correct paths
# ---------------------------------------------------------------------------


class TestPathDerivation:
    """Verify that every derived path matches section 4.1 of the plan."""

    @pytest.fixture
    def config(self) -> AgentConfig:
        return AgentConfig(domain="insurance", split="A")

    def test_raw_dir(self, config: AgentConfig) -> None:
        assert config.raw_dir == Path("data/public_dataset_upload/raw/insurance")

    def test_questions_path(self, config: AgentConfig) -> None:
        assert config.questions_path == Path(
            "data/public_dataset_upload/questions/group_a/insurance_questions.json"
        )

    def test_output_dir_is_lowercased(self, config: AgentConfig) -> None:
        """Split output dir is lowercased: outputs/insurance_a."""
        assert config.output_dir == Path("outputs/insurance_a")
        # Explicitly verify the split portion is lowercased
        assert config.output_dir.name == "insurance_a"

    def test_pages_dir(self, config: AgentConfig) -> None:
        assert config.pages_dir == Path("data/processed_data/pages/insurance")

    def test_markdown_dir(self, config: AgentConfig) -> None:
        assert config.markdown_dir == Path("data/processed_data/markdown/insurance")

    def test_pageindex_dir(self, config: AgentConfig) -> None:
        assert config.pageindex_dir == Path("data/processed_data/pageindex/insurance")

    def test_catalog_path(self, config: AgentConfig) -> None:
        assert config.catalog_path == Path("data/processed_data/catalog/doc_catalog.jsonl")

    def test_logs_dir(self, config: AgentConfig) -> None:
        assert config.logs_dir == Path("outputs/insurance_a/logs")


# ---------------------------------------------------------------------------
# Test 2: PageIndex build params and online retrieval budget params exist
#         with defaults matching section 4.1
# ---------------------------------------------------------------------------


class TestPageIndexBuildParams:
    """Verify PageIndex build parameters exist and have the expected defaults."""

    @pytest.fixture
    def config(self) -> AgentConfig:
        return AgentConfig()

    def test_toc_check_page_num_default(self, config: AgentConfig) -> None:
        assert config.toc_check_page_num == 20

    def test_max_page_num_each_node_default(self, config: AgentConfig) -> None:
        assert config.max_page_num_each_node == 8

    def test_max_token_num_each_node_default(self, config: AgentConfig) -> None:
        assert config.max_token_num_each_node == 20000


class TestOnlineRetrievalBudgetParams:
    """Verify online retrieval budget parameters exist and have expected defaults."""

    @pytest.fixture
    def config(self) -> AgentConfig:
        return AgentConfig()

    def test_max_docs_per_question_default(self, config: AgentConfig) -> None:
        assert config.max_docs_per_question == 4

    def test_max_nodes_per_doc_default(self, config: AgentConfig) -> None:
        assert config.max_nodes_per_doc == 5

    def test_max_pages_per_doc_default(self, config: AgentConfig) -> None:
        assert config.max_pages_per_doc == 8

    def test_max_evidence_per_option_default(self, config: AgentConfig) -> None:
        assert config.max_evidence_per_option == 3

    def test_max_retry_per_question_default(self, config: AgentConfig) -> None:
        assert config.max_retry_per_question == 1


# ---------------------------------------------------------------------------
# Test 3: lowercased split in output dir (already covered above, but kept
#         as a focused test per the task requirements)
# ---------------------------------------------------------------------------


class TestLowercasedOutputDir:
    """Explicitly verify lowercased split in output path."""

    def test_insurance_a(self) -> None:
        cfg = AgentConfig(domain="insurance", split="A")
        assert cfg.output_dir == Path("outputs/insurance_a")

    def test_financial_reports_B(self) -> None:
        """Cross-check: a different domain + uppercase split."""
        cfg = AgentConfig(domain="financial_reports", split="B")
        assert cfg.output_dir == Path("outputs/financial_reports_b")

    def test_split_mixed_case(self) -> None:
        """Even mixed-case input is lowercased."""
        cfg = AgentConfig(domain="Insurance", split="a")
        assert cfg.output_dir == Path("outputs/insurance_a")
        # Note: domain is NOT lowercased by AgentConfig itself — it is used
        # as-is.  If lowercasing the domain were desired it would be done at
        # the property level.  Currently only split is lowercased (as
        # specified in the plan).


# ---------------------------------------------------------------------------
# Test: CLI argument parsing builds correct config
# ---------------------------------------------------------------------------


class TestCLIIntegration:
    """Verify add_cli_arguments + AgentConfig.from_args integration."""

    def test_default_args_produce_default_config(self) -> None:
        parser = argparse.ArgumentParser()
        add_cli_arguments(parser)
        args = parser.parse_args([])
        config = AgentConfig.from_args(args)
        assert config.domain == "insurance"
        assert config.split == "A"
        assert isinstance(config.raw_root, Path)

    def test_domain_and_split_from_cli(self) -> None:
        parser = argparse.ArgumentParser()
        add_cli_arguments(parser)
        args = parser.parse_args(["--domain", "financial_reports", "--split", "B"])
        config = AgentConfig.from_args(args)
        assert config.domain == "financial_reports"
        assert config.split == "B"
        assert config.questions_path == Path(
            "data/public_dataset_upload/questions/group_b/financial_reports_questions.json"
        )
        assert config.output_dir == Path("outputs/financial_reports_b")


# ---------------------------------------------------------------------------
# Test: model config
# ---------------------------------------------------------------------------


class TestModelConfig:
    """Verify model selection fields."""

    def test_default_inference_model(self) -> None:
        cfg = AgentConfig()
        assert cfg.inference_model == "dashscope/qwen3.6-plus"

    def test_dev_model_defaults_to_none(self) -> None:
        cfg = AgentConfig()
        assert cfg.dev_model is None

    def test_can_override_dev_model(self) -> None:
        cfg = AgentConfig(dev_model="dashscope/qwen-turbo")
        assert cfg.dev_model == "dashscope/qwen-turbo"


# ---------------------------------------------------------------------------
# Test: from_env
# ---------------------------------------------------------------------------


class TestFromEnv:
    """Verify AgentConfig.from_env() reads environment variables."""

    def test_from_env_defaults(self, monkeypatch) -> None:
        """Without env vars set, from_env() gives defaults."""
        # Clear relevant env vars
        for key in (
            "AFAC_DOMAIN",
            "AFAC_SPLIT",
            "AFAC_RAW_ROOT",
            "AFAC_QUESTIONS_ROOT",
            "AFAC_PROCESSED_ROOT",
            "AFAC_PAGEINDEX_ROOT",
            "AFAC_OUTPUT_ROOT",
            "AFAC_INFERENCE_MODEL",
            "AFAC_DEV_MODEL",
        ):
            monkeypatch.delenv(key, raising=False)
        cfg = AgentConfig.from_env()
        assert cfg.domain == "insurance"
        assert cfg.split == "A"

    def test_from_env_custom_domain(self, monkeypatch) -> None:
        monkeypatch.setenv("AFAC_DOMAIN", "regulatory")
        monkeypatch.setenv("AFAC_SPLIT", "B")
        monkeypatch.delenv("AFAC_DEV_MODEL", raising=False)
        cfg = AgentConfig.from_env()
        assert cfg.domain == "regulatory"
        assert cfg.split == "B"
