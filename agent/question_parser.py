from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from agent.domain_profiles import DomainProfile
from agent.schemas import ParsedQuestion

AMOUNT_PATTERN = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>万元|万|元|%)")


class QuestionParser:
    def __init__(self, profile: DomainProfile) -> None:
        self.profile = profile

    def parse(self, raw_question: dict[str, Any]) -> ParsedQuestion:
        question = raw_question["question"]
        options = dict(raw_question["options"])
        combined_text = question + "\n" + "\n".join(options.values())
        mentioned_products = self._extract_products(combined_text)
        signals = {
            "keywords": self._extract_keywords(combined_text),
            "amounts": self._extract_amounts(combined_text),
        }
        return ParsedQuestion(
            qid=raw_question["qid"],
            domain=raw_question.get("domain", self.profile.name),
            split=raw_question.get("split", "A"),
            question=question,
            options=options,
            answer_format=raw_question["answer_format"],
            type=raw_question["type"],
            doc_ids=list(raw_question["doc_ids"]),
            mentioned_products=mentioned_products,
            signals=signals,
        )

    def _extract_products(self, text: str) -> list[str]:
        found: list[str] = []
        mapping = self.profile.alias_to_canonical()
        for alias, canonical in mapping.items():
            if alias and alias in text.lower() and canonical not in found:
                found.append(canonical)
        return found

    def _extract_keywords(self, text: str) -> list[str]:
        return [keyword for keyword in self.profile.keywords if keyword in text]

    def _extract_amounts(self, text: str) -> list[dict[str, str | float]]:
        amounts: list[dict[str, str | float]] = []
        for match in AMOUNT_PATTERN.finditer(text):
            amounts.append(
                {
                    "value": float(match.group("value")),
                    "unit": match.group("unit"),
                    "raw": match.group(0),
                }
            )
        return amounts
