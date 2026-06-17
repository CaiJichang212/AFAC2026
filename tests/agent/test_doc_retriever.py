import json

from agent.config import AgentConfig
from agent.domain_profiles import get_domain_profile
from agent.doc_retriever import DocRetriever
from agent.question_parser import QuestionParser


def test_a_split_doc_retriever_preserves_question_doc_ids() -> None:
    config = AgentConfig()
    profile = get_domain_profile("insurance")
    parser = QuestionParser(profile)
    retriever = DocRetriever()
    raw_questions = json.loads(config.questions_path.read_text(encoding="utf-8"))

    for raw in raw_questions:
        parsed = parser.parse(raw)
        assert retriever.retrieve(parsed) == raw["doc_ids"]
