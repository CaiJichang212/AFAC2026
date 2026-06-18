from agent.config import AgentConfig
from agent.domain_profiles import get_domain_profile
from agent.evidence_extractor import EvidenceExtractor
from agent.index_store import IndexStore
from agent.question_parser import QuestionParser
from agent.schemas import CandidateNode
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


def test_evidence_extractor_does_not_support_unrelated_option_with_generic_question_term() -> None:
    config = AgentConfig()
    profile = get_domain_profile("insurance")
    parsed = QuestionParser(profile).parse(
        {
            "qid": "demo",
            "domain": "insurance",
            "split": "A",
            "question": "平安智盈金生的身故保险金如何按保单账户价值计算？",
            "options": {
                "A": "领取日前身故按保单账户价值给付",
                "B": "无关选项",
            },
            "answer_format": "mcq",
            "type": "事实查询",
            "doc_ids": ["1"],
        }
    )
    store = IndexStore(config)
    candidate = CandidateNode(
        doc_id="1",
        node_id="page-4",
        title="页文本关键词: 身故保险金",
        page_range="4-4",
        matched_signals=["身故保险金", "保单账户价值"],
        reason="页文本关键词补召回",
    )

    evidence = EvidenceExtractor(store).extract(parsed, [candidate])
    by_option = {record.option: record for record in evidence}

    assert by_option["A"].evidence_type == "support"
    assert "保单账户价值" in by_option["A"].quote
    assert by_option["B"].evidence_type == "unclear"


def test_evidence_extractor_refutes_negative_option_when_quote_allows_action() -> None:
    config = AgentConfig()
    profile = get_domain_profile("insurance")
    parsed = QuestionParser(profile).parse(
        {
            "qid": "demo",
            "domain": "insurance",
            "split": "A",
            "question": "关于平安富鸿金生的保单贷款，哪些说法正确？",
            "options": {
                "C": "若按个人养老金制度投保，则不允许保单贷款",
                "D": "无论何种投保方式，均不允许保单贷款",
            },
            "answer_format": "multi",
            "type": "推理判断",
            "doc_ids": ["16"],
        }
    )
    store = IndexStore(config)
    candidate = CandidateNode(
        doc_id="16",
        node_id="page-9",
        title="页文本关键词: 保单贷款、个人养老金制度",
        page_range="9-9",
        matched_signals=["保单贷款", "个人养老金制度"],
        reason="页文本关键词补召回",
    )

    evidence = EvidenceExtractor(store).extract(parsed, [candidate])
    by_option = {record.option: record for record in evidence}

    assert by_option["C"].evidence_type == "support"
    assert by_option["D"].evidence_type == "refute"
