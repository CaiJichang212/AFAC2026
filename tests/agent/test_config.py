from pathlib import Path

from agent.config import AgentConfig


def test_agent_config_derives_current_insurance_a_paths() -> None:
    config = AgentConfig.from_args(["--domain", "insurance", "--split", "A"])

    assert config.domain == "insurance"
    assert config.split == "A"
    assert config.raw_dir == Path("data/public_dataset_upload/raw/insurance")
    assert config.questions_path == Path(
        "data/public_dataset_upload/questions/group_a/insurance_questions.json"
    )
    assert config.output_dir == Path("outputs/insurance_a")
    assert config.logs_dir == Path("outputs/insurance_a/logs")


def test_agent_config_exposes_pageindex_and_retrieval_defaults() -> None:
    config = AgentConfig()

    assert config.inference_model == "dashscope/qwen3.6-plus"
    assert config.toc_check_page_num == 20
    assert config.max_page_num_each_node == 8
    assert config.max_token_num_each_node == 20000
    assert config.max_docs_per_question == 4
    assert config.max_nodes_per_doc == 5
    assert config.max_pages_per_doc == 8
    assert config.max_evidence_per_option == 3
    assert config.max_retry_per_question == 1
    assert config.pageindex_build_options == {
        "model": "dashscope/qwen3.6-plus",
        "toc_check_page_num": 20,
        "max_page_num_each_node": 8,
        "max_token_num_each_node": 20000,
        "if_add_node_summary": "no",
        "if_add_doc_description": "no",
        "if_add_node_text": "no",
        "if_add_node_id": "yes",
    }
    assert config.retrieval_budget == {
        "max_docs_per_question": 4,
        "max_nodes_per_doc": 5,
        "max_pages_per_doc": 8,
        "max_evidence_per_option": 3,
        "max_retry_per_question": 1,
    }


def test_output_dir_uses_lowercase_split_suffix() -> None:
    config = AgentConfig(split="A")

    assert config.output_dir == Path("outputs/insurance_a")
