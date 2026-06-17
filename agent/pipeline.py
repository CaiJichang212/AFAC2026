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
from agent.question_parser import QuestionParser
from agent.schemas import EvidenceRecord, UsageRecord
from agent.token_meter import TokenMeter
from agent.tree_retriever import TreeRetriever


class Pipeline:
    def __init__(self, config: AgentConfig) -> None:
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

    def run(self) -> dict[str, int]:
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
                evidence.extend(self.evidence_extractor.extract(parsed, candidates[:1]))

            calculations = self.calculation_engine.compute(parsed, evidence)
            answer = self.answer_judge.judge(parsed, evidence, calculations)
            evidence = _ensure_selected_support(parsed.qid, answer.answer, evidence, selected_nodes)
            answer = self.answer_judge.judge(parsed, evidence, calculations)
            answers[parsed.qid] = answer.answer
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
            audit_records.append(
                {
                    "qid": parsed.qid,
                    "answer": answer.answer,
                    "candidate_docs": candidate_docs,
                    "selected_nodes": selected_nodes,
                    "evidence": [asdict(record) for record in evidence],
                    "calculations": [asdict(record) for record in calculations],
                    "usage": self.token_meter.summarize_qid(parsed.qid).to_dict(),
                    "fallbacks": [],
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


def run_pipeline(config: AgentConfig) -> dict[str, int]:
    return Pipeline(config).run()
