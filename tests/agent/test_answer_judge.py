from agent.answer_judge import AnswerJudge
from agent.schemas import EvidenceRecord, ParsedQuestion


def test_answer_judge_legalizes_mcq_and_multi_answers() -> None:
    judge = AnswerJudge()
    mcq = ParsedQuestion("q1", "insurance", "A", "Q", {"A": "a", "B": "b"}, "mcq", "事实查询", ["1"])
    multi = ParsedQuestion(
        "q2", "insurance", "A", "Q", {"A": "a", "B": "b", "C": "c"}, "multi", "事实查询", ["1"]
    )
    evidence = [
        EvidenceRecord("q1", "1", "0001", "1-1", "B", "support", "quote", "fact", [], "high"),
        EvidenceRecord("q2", "1", "0001", "1-1", "C", "support", "quote", "fact", [], "high"),
        EvidenceRecord("q2", "1", "0001", "1-1", "A", "support", "quote", "fact", [], "high"),
    ]

    assert judge.judge(mcq, evidence, []).answer == "B"
    assert judge.judge(multi, evidence, []).answer == "AC"
