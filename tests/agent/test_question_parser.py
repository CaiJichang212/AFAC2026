import json

from agent.config import AgentConfig
from agent.domain_profiles import get_domain_profile
from agent.question_parser import QuestionParser


def test_question_parser_parses_all_insurance_questions() -> None:
    config = AgentConfig()
    profile = get_domain_profile("insurance")
    parser = QuestionParser(profile)
    raw_questions = json.loads(config.questions_path.read_text(encoding="utf-8"))

    parsed = [parser.parse(raw) for raw in raw_questions]

    assert len(parsed) == 20
    assert parsed[0].qid == "ins_a_001"
    assert parsed[0].answer_format == "mcq"
    assert parsed[0].doc_ids == ["1", "2", "15", "16"]
    assert "平安智盈金生专属商业养老保险" in parsed[0].mentioned_products
    assert parsed[2].signals["amounts"]


def test_question_parser_extracts_option_keywords() -> None:
    config = AgentConfig()
    profile = get_domain_profile("insurance")
    parser = QuestionParser(profile)
    raw = json.loads(config.questions_path.read_text(encoding="utf-8"))[6]

    parsed = parser.parse(raw)

    assert parsed.qid == "ins_a_007"
    assert "保单贷款" in parsed.signals["keywords"]
    assert parsed.options["A"].startswith("平安智盈金生")
