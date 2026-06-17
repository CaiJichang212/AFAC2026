from __future__ import annotations

from agent.schemas import AnswerRecord, CalculationRecord, EvidenceRecord, ParsedQuestion


class AnswerJudge:
    def judge(
        self,
        parsed: ParsedQuestion,
        evidence: list[EvidenceRecord],
        calculations: list[CalculationRecord],
    ) -> AnswerRecord:
        support_options = sorted(
            {
                record.option
                for record in evidence
                if record.qid == parsed.qid and record.evidence_type == "support"
            }
        )
        valid_options = set(parsed.options)
        support_options = [option for option in support_options if option in valid_options]

        if parsed.answer_format in {"mcq", "tf"}:
            answer = support_options[0] if support_options else sorted(valid_options)[0]
        elif parsed.answer_format == "multi":
            answer = "".join(support_options) if support_options else sorted(valid_options)[0]
        else:
            answer = sorted(valid_options)[0]

        return AnswerRecord(
            qid=parsed.qid,
            answer=answer,
            option_judgements={
                option: ("support" if option in support_options else "unclear")
                for option in sorted(valid_options)
            },
            warnings=[] if support_options else ["no_support_evidence"],
        )
