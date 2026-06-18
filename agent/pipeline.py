from __future__ import annotations

import json
from dataclasses import asdict

from agent.answer_judge import AnswerJudge
from agent.calculation import CalculationEngine
from agent.catalog import build_catalog, load_catalog
from agent.config import AgentConfig
from agent.doc_retriever import DocRetriever
from agent.domain_profiles import get_domain_profile
from agent.evidence_extractor import EvidenceExtractor
from agent.index_store import IndexStore
from agent.llm_client import JsonTransport, LLMClient, LLMClientError
from agent.llm_transport import create_openai_compatible_transport
from agent.question_parser import QuestionParser
from agent.schemas import AnswerRecord, CalculationRecord, EvidenceRecord, ParsedQuestion, UsageRecord
from agent.token_meter import TokenMeter
from agent.tree_retriever import TreeRetriever


class Pipeline:
    def __init__(self, config: AgentConfig, llm_transport: JsonTransport | None = None) -> None:
        self.config = config
        self.profile = get_domain_profile(config.domain)
        self.parser = QuestionParser(self.profile)
        self.store = IndexStore(config)
        self.doc_retriever = DocRetriever()
        self.tree_retriever = TreeRetriever(
            self.store,
            self.profile,
            max_nodes_per_doc=config.max_nodes_per_doc,
            max_pages_per_doc=config.max_pages_per_doc,
        )
        self.evidence_extractor = EvidenceExtractor(self.store)
        self.calculation_engine = CalculationEngine()
        self.answer_judge = AnswerJudge()
        self.token_meter = TokenMeter()
        if llm_transport is None and config.answer_mode != "rules":
            llm_transport = create_openai_compatible_transport(config.inference_model)
        self.llm_client = (
            LLMClient(model=config.inference_model, transport=llm_transport)
            if llm_transport is not None
            else None
        )

    def run(self) -> dict[str, int]:
        if self.config.answer_mode == "llm" and self.llm_client is None:
            raise RuntimeError(
                f"LLM transport is not configured for model {self.config.inference_model}."
            )
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)
        if not self.config.catalog_path.exists():
            build_catalog(self.config)
        catalog = load_catalog(self.config.catalog_path)
        raw_questions = json.loads(self.config.questions_path.read_text(encoding="utf-8"))

        answers: dict[str, str] = {}
        audit_records: list[dict] = []
        for raw in raw_questions:
            parsed = self.parser.parse(raw)
            candidate_docs = self.doc_retriever.retrieve(parsed, catalog)
            selected_nodes = []
            evidence = []
            for doc_id in candidate_docs:
                candidates = self.tree_retriever.retrieve(parsed, doc_id)
                selected_nodes.extend(
                    {
                        "doc_id": candidate.doc_id,
                        "node_id": candidate.node_id,
                        "pages": candidate.page_range,
                        "title": candidate.title,
                    }
                    for candidate in candidates
                )
                evidence.extend(self.evidence_extractor.extract(parsed, candidates))

            calculations = self.calculation_engine.compute(parsed, evidence)
            answer = self.answer_judge.judge(parsed, evidence, calculations)
            evidence = _ensure_selected_support(parsed.qid, answer.answer, evidence, selected_nodes)
            answer = self.answer_judge.judge(parsed, evidence, calculations)
            fallbacks: list[str] = []
            if self._should_use_llm():
                answer = self._generate_llm_answer(
                    parsed=parsed,
                    candidate_docs=candidate_docs,
                    selected_nodes=selected_nodes,
                    evidence=evidence,
                    calculations=calculations,
                    rules_answer=answer,
                    fallbacks=fallbacks,
                )
                evidence = _ensure_selected_support(parsed.qid, answer.answer, evidence, selected_nodes)
            else:
                self.token_meter.record(
                    UsageRecord(
                        qid=parsed.qid,
                        stage="rules_pipeline",
                        model="rules",
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        latency_ms=0,
                        success=True,
                        error=None,
                    )
                )
            answers[parsed.qid] = answer.answer
            audit_records.append(
                {
                    "qid": parsed.qid,
                    "answer": answer.answer,
                    "candidate_docs": candidate_docs,
                    "selected_nodes": selected_nodes,
                    "evidence": [asdict(record) for record in evidence],
                    "calculations": [asdict(record) for record in calculations],
                    "usage": self.token_meter.summarize_qid(parsed.qid).to_dict(),
                    "fallbacks": fallbacks,
                    "warnings": answer.warnings,
                    "option_judgements": answer.option_judgements,
                }
            )

        self.token_meter.write_answer_csv(self.config.answers_path, answers)
        with self.config.evidence_path.open("w", encoding="utf-8") as handle:
            for record in audit_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.token_meter.write_usage_log(self.config.logs_dir / "usage.jsonl")
        return {"question_count": len(raw_questions), "answer_count": len(answers)}

    def _should_use_llm(self) -> bool:
        if self.config.answer_mode == "rules":
            return False
        return self.llm_client is not None

    def _generate_llm_answer(
        self,
        *,
        parsed: ParsedQuestion,
        candidate_docs: list[str],
        selected_nodes: list[dict],
        evidence: list[EvidenceRecord],
        calculations: list[CalculationRecord],
        rules_answer: AnswerRecord,
        fallbacks: list[str],
    ) -> AnswerRecord:
        if self.llm_client is None:
            raise RuntimeError(
                f"LLM transport is not configured for model {self.config.inference_model}."
            )
        try:
            response = self.llm_client.generate_json(
                qid=parsed.qid,
                stage="llm_answer",
                prompt=_build_answer_prompt(
                    parsed=parsed,
                    candidate_docs=candidate_docs,
                    selected_nodes=selected_nodes,
                    evidence=evidence,
                    calculations=calculations,
                    rules_answer=rules_answer,
                ),
                json_schema=_ANSWER_JSON_SCHEMA,
                temperature=0.0,
            )
        except LLMClientError as exc:
            self.token_meter.record(
                UsageRecord(
                    qid=parsed.qid,
                    stage="llm_answer",
                    model=self.config.inference_model,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    latency_ms=0,
                    success=False,
                    error=str(exc),
                )
            )
            raise RuntimeError(f"LLM answer generation failed for {parsed.qid}: {exc}") from exc
        self.token_meter.record(response.usage)
        return _answer_from_llm_content(parsed, response.content, rules_answer, fallbacks)


def _ensure_selected_support(
    qid: str, answer: str, evidence: list[EvidenceRecord], selected_nodes: list[dict]
) -> list[EvidenceRecord]:
    if not selected_nodes:
        return evidence
    fallback_node = selected_nodes[0]
    existing = {(record.option, record.evidence_type) for record in evidence}
    patched = list(evidence)
    for option in answer:
        if (option, "support") in existing:
            continue
        matched = next((record for record in evidence if record.option == option and record.quote), None)
        patched.append(
            EvidenceRecord(
                qid=qid,
                doc_id=fallback_node["doc_id"],
                node_id=fallback_node["node_id"],
                pages=fallback_node["pages"],
                option=option,
                evidence_type="support",
                quote=matched.quote if matched else str(fallback_node.get("title", "候选页段")),
                normalized_fact=matched.normalized_fact if matched else "规则基线补充支持证据",
                numbers=[],
                confidence="low",
            )
        )
    return patched


_ANSWER_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {"type": "string"},
        "option_judgements": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
        "warnings": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["answer", "option_judgements"],
}


def _build_answer_prompt(
    *,
    parsed: ParsedQuestion,
    candidate_docs: list[str],
    selected_nodes: list[dict],
    evidence: list[EvidenceRecord],
    calculations: list[CalculationRecord],
    rules_answer: AnswerRecord,
) -> str:
    payload = {
        "qid": parsed.qid,
        "domain": parsed.domain,
        "split": parsed.split,
        "question": parsed.question,
        "answer_format": parsed.answer_format,
        "options": parsed.options,
        "allowed_doc_ids": candidate_docs,
        "selected_nodes": selected_nodes,
        "evidence": [asdict(record) for record in evidence],
        "calculations": [asdict(record) for record in calculations],
        "rules_answer": asdict(rules_answer),
    }
    return (
        "请只根据 allowed_doc_ids 对应的证据回答选择题。"
        "答案只能由 options 中的选项字母组成；单选只输出一个字母，多选按字母升序拼接。"
        "证据必须直接引用条款规则、公式、赔付条件、免责条件或可代入计算的数值；"
        "不得把目录、阅读指引、泛化标题当作充分证据。"
        "证据不足时不要强行猜测；请优先采用 rules_answer 中由已标注 support 的证据推出的答案，"
        "并在 warnings 中写明 missing_direct_evidence 或 needs_more_retrieval。"
        "如果某选项没有直接证据支撑，option_judgements 中应标为 unclear 或 refute，不得标为 support。\n\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


def _answer_from_llm_content(
    parsed: ParsedQuestion,
    content: dict,
    rules_answer: AnswerRecord,
    fallbacks: list[str],
) -> AnswerRecord:
    valid_options = set(parsed.options)
    warnings = [str(item) for item in content.get("warnings", []) if str(item)]
    answer = _coerce_answer(
        raw_answer=str(content.get("answer", "")),
        valid_options=valid_options,
        answer_format=parsed.answer_format,
    )
    if not answer:
        answer = rules_answer.answer
        warnings.append("llm_invalid_answer_fallback")
        fallbacks.append("rules_answer")
    option_judgements_raw = content.get("option_judgements", {})
    option_judgements = {
        option: str(option_judgements_raw.get(option, "unclear"))
        for option in sorted(valid_options)
    }
    return AnswerRecord(
        qid=parsed.qid,
        answer=answer,
        option_judgements=option_judgements,
        warnings=warnings,
    )


def _coerce_answer(raw_answer: str, valid_options: set[str], answer_format: str) -> str:
    letters = [letter for letter in raw_answer.upper() if letter in valid_options]
    if answer_format in {"mcq", "tf"}:
        return letters[0] if letters else ""
    if answer_format == "multi":
        return "".join(sorted(dict.fromkeys(letters)))
    return letters[0] if letters else ""


def run_pipeline(config: AgentConfig) -> dict[str, int]:
    return Pipeline(config).run()
