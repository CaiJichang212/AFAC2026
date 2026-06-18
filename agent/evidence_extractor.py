from __future__ import annotations

import re

from agent.index_store import IndexStore
from agent.schemas import CandidateNode, EvidenceRecord, ParsedQuestion

IMPORTANT_TERMS = (
    "保单账户价值",
    "个人账户价值",
    "身故保险金额",
    "身故保险金",
    "身故给付比例",
    "基本保险金额",
    "现金价值",
    "退保费用",
    "解除合同",
    "全部保险费",
    "保险责任",
    "责任免除",
    "不承担保险责任",
    "免赔额",
    "免赔额余额",
    "赔付比例",
    "保险金计算方法",
    "费用补偿",
    "医疗保险金",
    "等待期",
    "犹豫期",
    "保单贷款",
    "个人养老金制度",
    "指定药店",
    "处方审核",
    "效力中止期间",
    "施救费用",
    "最高不超过保险金额",
    "酒后驾驶",
    "艾滋病",
    "自杀",
    "故意自伤",
)
SUBSTANTIVE_MARKERS = (
    "按",
    "给付",
    "等于",
    "扣除",
    "比例",
    "不承担",
    "退还",
    "不得",
    "最高",
    "计算",
    "赔付",
    "费用",
    "责任",
    "%",
)
GENERIC_TOC_MARKERS = ("条款目录", "阅读指引", "……………………", "........")
AMOUNT_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*(?:万元|万|元|%)")
NEGATIVE_OPTION_MARKERS = ("不允许", "不承担", "不赔", "不得", "不能", "无", "未")
QUOTE_NEGATIVE_MARKERS = ("不接受", "不承担", "不予", "不得", "不能", "不允许")
QUOTE_ALLOW_MARKERS = ("可申请", "可办理", "可以", "允许", "承担", "负责赔偿", "给付")


class EvidenceExtractor:
    def __init__(self, store: IndexStore) -> None:
        self.store = store

    def extract(
        self, parsed: ParsedQuestion, candidates: list[CandidateNode]
    ) -> list[EvidenceRecord]:
        records: list[EvidenceRecord] = []
        seen: set[tuple[str, str, str, str]] = set()
        for candidate in candidates:
            pages = self.store.get_page_content(candidate.doc_id, candidate.page_range)
            page_text = "\n".join(page["text"] for page in pages)
            for option, option_text in parsed.options.items():
                quote, matched_option_terms = _find_quote(page_text, option_text, parsed.question)
                evidence_type = _classify_evidence(option_text, quote, matched_option_terms)
                if not quote:
                    quote = page_text[:120].strip()
                key = (candidate.doc_id, candidate.page_range, quote, option)
                if key in seen:
                    continue
                seen.add(key)
                records.append(
                    EvidenceRecord(
                        qid=parsed.qid,
                        doc_id=candidate.doc_id,
                        node_id=candidate.node_id,
                        pages=candidate.page_range,
                        option=option,
                        evidence_type=evidence_type,
                        quote=quote,
                        normalized_fact=quote[:200],
                        numbers=_extract_numbers(quote),
                        confidence=_confidence(evidence_type, quote, matched_option_terms),
                    )
                )
        return records


def _find_quote(page_text: str, option_text: str, question: str) -> tuple[str, list[str]]:
    option_terms = _option_terms(option_text)
    question_terms = _question_terms(question)
    terms = sorted(set(option_terms + question_terms), key=len, reverse=True)
    best_quote = ""
    best_matched_option_terms: list[str] = []
    for term in terms:
        index = page_text.find(term)
        if index >= 0:
            start = max(0, index - 90)
            end = min(len(page_text), index + 260)
            quote = " ".join(page_text[start:end].split())
            matched_option_terms = [candidate for candidate in option_terms if candidate in quote]
            if matched_option_terms:
                return quote, matched_option_terms
            if not best_quote:
                best_quote = quote
    return best_quote, []


def _option_terms(option_text: str) -> list[str]:
    terms = [term for term in IMPORTANT_TERMS if term in option_text]
    terms.extend(match.group(0) for match in AMOUNT_PATTERN.finditer(option_text))
    chunks = [
        chunk.strip()
        for chunk in re.split(r"[，。；：、（）()<>《》\s>＝=+-]+", option_text)
        if len(chunk.strip()) >= 4
    ]
    terms.extend(chunk for chunk in chunks if chunk not in {"无关选项", "无法确定"})
    return list(dict.fromkeys(terms))


def _question_terms(question: str) -> list[str]:
    terms = [term for term in IMPORTANT_TERMS if term in question]
    terms.extend(match.group(0) for match in AMOUNT_PATTERN.finditer(question))
    return list(dict.fromkeys(terms))


def _is_substantive_quote(quote: str, matched_option_terms: list[str]) -> bool:
    if any(marker in quote for marker in GENERIC_TOC_MARKERS):
        return False
    if not matched_option_terms:
        return False
    return any(marker in quote for marker in SUBSTANTIVE_MARKERS)


def _classify_evidence(
    option_text: str,
    quote: str,
    matched_option_terms: list[str],
) -> str:
    if not quote or not matched_option_terms:
        return "unclear"
    if _is_negative_option(option_text):
        if _negative_claim_supported(option_text, quote, matched_option_terms):
            return "support"
        if _quote_allows_action(quote):
            return "refute"
    if _is_substantive_quote(quote, matched_option_terms):
        return "support"
    return "unclear"


def _is_negative_option(option_text: str) -> bool:
    return any(marker in option_text for marker in NEGATIVE_OPTION_MARKERS)


def _negative_claim_supported(
    option_text: str,
    quote: str,
    matched_option_terms: list[str],
) -> bool:
    if "无论" in option_text or "均不" in option_text:
        return False
    return bool(matched_option_terms) and any(marker in quote for marker in QUOTE_NEGATIVE_MARKERS)


def _quote_allows_action(quote: str) -> bool:
    return any(marker in quote for marker in QUOTE_ALLOW_MARKERS)


def _extract_numbers(quote: str) -> list[dict[str, str | float]]:
    numbers: list[dict[str, str | float]] = []
    for match in AMOUNT_PATTERN.finditer(quote):
        raw = match.group(0)
        unit = re.search(r"(万元|万|元|%)$", raw)
        value = re.match(r"\d+(?:\.\d+)?", raw)
        if value is None:
            continue
        numbers.append(
            {
                "raw": raw,
                "value": float(value.group(0)),
                "unit": unit.group(1) if unit else "",
            }
        )
    return numbers


def _confidence(
    evidence_type: str,
    quote: str,
    matched_option_terms: list[str],
) -> str:
    if evidence_type != "support":
        return "low"
    if len(matched_option_terms) >= 2 or len(quote) >= 80:
        return "high"
    return "medium"
