from agent.config import AgentConfig
from agent.domain_profiles import get_domain_profile
from agent.evidence_extractor import EvidenceExtractor
from agent.index_store import IndexStore
from agent.question_parser import QuestionParser
from agent.tree_retriever import TreeRetriever


def test_evidence_extractor_keeps_traceable_quote() -> None:
    config = AgentConfig()
    profile = get_domain_profile("insurance")
    parsed = QuestionParser(profile).parse(
        {
            "qid": "demo",
            "domain": "insurance",
            "split": "A",
            "question": "平安智盈金生的身故保险金如何计算？",
            "options": {"A": "身故保险金", "B": "无关选项"},
            "answer_format": "mcq",
            "type": "事实查询",
            "doc_ids": ["1"],
        }
    )
    store = IndexStore(config)
    candidates = TreeRetriever(store, profile).retrieve(parsed, "1")[:1]

    evidence = EvidenceExtractor(store).extract(parsed, candidates)

    assert evidence
    first = evidence[0]
    assert first.doc_id == "1"
    assert first.node_id
    assert first.pages
    assert first.quote
