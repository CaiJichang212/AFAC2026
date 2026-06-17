from agent.config import AgentConfig
from agent.domain_profiles import get_domain_profile
from agent.index_store import IndexStore
from agent.question_parser import QuestionParser
from agent.tree_retriever import TreeRetriever


def test_tree_retriever_returns_bounded_candidate_nodes() -> None:
    config = AgentConfig()
    profile = get_domain_profile("insurance")
    parsed = QuestionParser(profile).parse(
        {
            "qid": "demo",
            "domain": "insurance",
            "split": "A",
            "question": "平安智盈金生的身故保险金如何计算？",
            "options": {"A": "A", "B": "B", "C": "C", "D": "D"},
            "answer_format": "mcq",
            "type": "推理判断",
            "doc_ids": ["1"],
        }
    )

    store = IndexStore(config)
    retriever = TreeRetriever(store, profile, max_nodes_per_doc=5, max_pages_per_doc=8)
    candidates = retriever.retrieve(parsed, "1")

    assert candidates
    assert len(candidates) <= 5
    assert all(candidate.doc_id == "1" for candidate in candidates)
    assert all(candidate.needs_page_fetch is True for candidate in candidates)
    assert all(candidate.page_range for candidate in candidates)
    assert all(candidate.reason for candidate in candidates)
    assert not any("support" in candidate.reason.lower() for candidate in candidates)
