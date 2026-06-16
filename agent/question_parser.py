"""Question parser: extract structured signals from insurance QA questions.

Task 7: Parses raw question JSON dicts into ``ParsedQuestion`` dataclass
instances with rule-based signal extraction (no LLM).

Signals extracted:
  - **mentioned_products**: canonical product names found via alias matching
  - **doc_product_map**: doc_id -> canonical product name for each doc
  - **liability_signals**: which liability terms appear in the question text
  - **number_conditions**: structured numeric conditions (amounts, ratios,
    ages, deductibles, account values, cash values, etc.)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agent.domain_profiles import DomainProfile
from agent.schemas import ParsedQuestion

# ---------------------------------------------------------------------------
# Regex patterns for number extraction
# ---------------------------------------------------------------------------

# Ordered from most specific to least specific to avoid double-matching.
# Each tuple: (compiled_regex, multiplier_to_base_unit, base_unit)
_NUMBER_PATTERNS: list[tuple[re.Pattern[str], float, str]] = [
    # "100万元"  ->  100 * 10000 = 1,000,000 元
    (re.compile(r"(\d+(?:\.\d+)?)\s*万\s*元"), 10000.0, "元"),
    # "144万" (bare 万, not followed by 元)  ->  144 * 10000 = 1,440,000 元
    (re.compile(r"(\d+(?:\.\d+)?)\s*万(?!\s*元)"), 10000.0, "元"),
    # Percentages: "80%"  ->  80 %
    (re.compile(r"(\d+(?:\.\d+)?)\s*%"), 1.0, "%"),
    # Age: "40岁"  ->  40 岁
    (re.compile(r"(?<!\d)(\d+)\s*岁"), 1.0, "岁"),
    # Plain 元 amounts (not 万元, not followed by /年/人/天/次/月/周 etc.)
    (re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*元(?![\/年人天次月周])"), 1.0, "元"),
]

# ---------------------------------------------------------------------------
# Context classifiers: (regex, kind, subject_label)
# ---------------------------------------------------------------------------

_CLASSIFIERS: list[tuple[str, str, str]] = [
    (r"已交保费", "premium_paid", "已交保费"),
    (r"累计所交保费", "premium_paid", "累计所交保费"),
    (r"现金价值", "cash_value", "现金价值"),
    (r"基本保额|基本保险金额", "basic_sum_insured", "基本保额"),
    (r"保单账户价值|个人账户价值|账户价值", "account_value", "账户价值"),
    (r"保单账户累计收益", "account_return", "保单账户累计收益"),
    (r"免赔额", "deductible", "免赔额"),
    (r"年度免赔额", "deductible", "年度免赔额"),
    (r"已领养老年金|已领年金", "annuity_received", "已领养老年金"),
    (r"给付比例", "ratio", "给付比例"),
    (r"退保费用", "surrender_charge", "退保费用"),
    (r"医保报销", "medical_reimbursement", "医保报销"),
    (r"总费用", "total_expense", "总费用"),
    (r"自费", "out_of_pocket", "自费"),
    (r"最高限额|赔偿限额", "payout_limit", "最高限额"),
    (r"保险金额(?!.*免赔)", "sum_insured", "保险金额"),
    (r"医疗费用", "medical_expense", "医疗费用"),
    (r"住院费用", "hospitalization_expense", "住院费用"),
    (r"赔付(?!比例|上限|限额)", "payout", "赔付"),
    (r"累计.*?保费", "premium_paid", "累计保费"),
    (r"保险费", "premium", "保险费"),
    (r"退保所得", "surrender_value", "退保所得"),
    (r"赔偿金", "payout", "赔偿金"),
]

# Context window (characters before the number) used for classification
_CONTEXT_WINDOW = 25


class QuestionParser:
    """Parse raw question dicts into ``ParsedQuestion`` with structured signals.

    All signal extraction is rule-based (regex + keyword matching) — no LLM
    calls.  The parser is stateless; ``parse()`` is a pure function.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, raw: dict[str, Any], profile: DomainProfile) -> ParsedQuestion:
        """Parse a single raw question dict.

        Args:
            raw: A dict with keys ``qid``, ``domain``, ``split``, ``question``,
                ``options``, ``answer_format``, ``type``, ``doc_ids``.
            profile: The domain profile (insurance) providing aliases, liability
                terms, and doc_product_map.

        Returns:
            A populated ``ParsedQuestion`` with all basic fields and extracted
            signals.
        """
        qid: str = raw["qid"]
        domain: str = raw["domain"]
        split: str = raw["split"]
        question: str = raw["question"]
        options: dict[str, str] = dict(raw.get("options", {}))
        answer_format: str = raw.get("answer_format", "")
        qtype: str = raw.get("type", "")
        doc_ids: list[str] = list(raw.get("doc_ids", []))

        # Combine question stem + all option texts for signal extraction
        full_text = question + " " + " ".join(options.values())

        mentioned_products = self._extract_mentioned_products(full_text, profile)
        doc_product_map = self._build_doc_product_map(doc_ids, profile)
        liability_signals = self._extract_liability_signals(full_text, profile)
        number_conditions = self._extract_number_conditions(full_text)

        return ParsedQuestion(
            qid=qid,
            domain=domain,
            split=split,
            question=question,
            options=options,
            answer_format=answer_format,
            type=qtype,
            doc_ids=doc_ids,
            mentioned_products=mentioned_products,
            doc_product_map=doc_product_map,
            liability_signals=liability_signals,
            number_conditions=number_conditions,
        )

    def parse_questions(
        self, questions_path: str | Path, profile: DomainProfile
    ) -> list[ParsedQuestion]:
        """Parse all questions from a JSON file.

        Args:
            questions_path: Path to a JSON file containing a list of raw
                question dicts.
            profile: The domain profile.

        Returns:
            A list of ``ParsedQuestion``, one per JSON object.
        """
        path = Path(questions_path)
        raw_list: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        return [self.parse(raw, profile) for raw in raw_list]

    # ------------------------------------------------------------------
    # Signal extraction helpers
    # ------------------------------------------------------------------

    def _extract_mentioned_products(
        self, text: str, profile: DomainProfile
    ) -> list[str]:
        """Find canonical product names mentioned in *text*.

        Matches against ``profile.product_aliases`` (both short aliases and
        canonical names).  Uses longest-match-first to avoid false substring
        matches (e.g. "平安智盈金生" is preferred over "智盈金生" when both
        appear at the same position).
        """
        # Sort aliases descending by length for greedy longest-match-first.
        aliases_sorted = sorted(profile.product_aliases.keys(), key=len, reverse=True)

        # canonical_name -> first occurrence position in text
        found: dict[str, int] = {}

        for alias in aliases_sorted:
            canonical = profile.product_aliases[alias]
            idx = text.find(alias)
            if idx >= 0 and canonical not in found:
                found[canonical] = idx

        # Return canonical names ordered by first occurrence
        return sorted(found.keys(), key=lambda k: found[k])

    @staticmethod
    def _build_doc_product_map(
        doc_ids: list[str], profile: DomainProfile
    ) -> dict[str, str]:
        """Build doc_id -> canonical product name mapping.

        Uses ``profile.doc_product_map`` (int key) to look up the canonical
        product name for each doc_id in the question.
        """
        result: dict[str, str] = {}
        for doc_id in doc_ids:
            try:
                doc_id_int = int(doc_id)
            except ValueError:
                continue
            if doc_id_int in profile.doc_product_map:
                result[doc_id] = profile.doc_product_map[doc_id_int]
        return result

    @staticmethod
    def _extract_liability_signals(
        text: str, profile: DomainProfile
    ) -> list[str]:
        """Find liability terms from the profile that appear in *text*.

        Returns terms in the order they first appear (by scanning each term
        and recording its first position).
        """
        positions: list[tuple[int, str]] = []
        for term in profile.liability_terms:
            idx = text.find(term)
            if idx >= 0:
                positions.append((idx, term))
        positions.sort(key=lambda x: x[0])
        return [term for _, term in positions]

    # ------------------------------------------------------------------
    # Number condition extraction
    # ------------------------------------------------------------------

    def _extract_number_conditions(self, text: str) -> list[dict[str, Any]]:
        """Extract structured numeric conditions from *text*.

        Scans for numbers with Chinese units (万元, 万, 元, %, 岁), classifies
        each by the surrounding context, and returns a list of records.

        Each record has:
            kind: str      — category (premium_paid, cash_value, amount, ratio, …)
            value: int|float — numeric value normalised to base unit
            unit: str      — base unit (元, %, 岁)
            subject: str   — what the number refers to (e.g. "已交保费")
            snippet: str   — raw text around the match for inspection
        """
        conditions: list[dict[str, Any]] = []
        seen: set[tuple[int, int]] = set()  # (start, end) spans already captured

        for pattern, multiplier, unit in _NUMBER_PATTERNS:
            for m in pattern.finditer(text):
                span = (m.start(), m.end())
                if span in seen:
                    continue
                seen.add(span)

                raw_str = m.group(1)  # the digit group
                try:
                    raw_val = float(raw_str)
                except ValueError:
                    continue
                value = raw_val * multiplier
                if value == int(value):
                    value = int(value)

                # Classify from context before the match
                ctx_start = max(0, m.start() - _CONTEXT_WINDOW)
                prefix = text[ctx_start : m.start()]
                kind, subject = self._classify_number(prefix, unit)

                # Build a short snippet for inspection
                snippet_start = max(0, m.start() - 10)
                snippet_end = min(len(text), m.end() + 5)
                snippet = text[snippet_start:snippet_end].strip()

                conditions.append(
                    {
                        "kind": kind,
                        "value": value,
                        "unit": unit,
                        "subject": subject,
                        "snippet": snippet,
                    }
                )

        return conditions

    @staticmethod
    def _classify_number(prefix: str, unit: str) -> tuple[str, str]:
        """Classify a number based on the text immediately before it.

        Finds ALL classifier keywords in the prefix and picks the one whose
        *end position* is closest to the number.  This avoids false matches
        with keywords that appear farther away in the same prefix window
        (e.g. "已交保费" appearing before "现金价值" when the number is
        actually attached to "现金价值").

        Args:
            prefix: The *n* characters immediately before the number match.
            unit: The unit already inferred (元, %, 岁).

        Returns:
            A ``(kind, subject)`` pair.
        """
        best_dist: int = 999999
        best_kind: str | None = None
        best_subject: str = ""

        # Age is unambiguous by unit — skip monetary classifiers
        if unit == "岁":
            return "age", ""

        for pattern_str, kind, subject in _CLASSIFIERS:
            for m in re.finditer(pattern_str, prefix):
                # Distance from end of keyword to the number (end of prefix)
                dist = len(prefix) - m.end()
                if dist < best_dist:
                    best_dist = dist
                    best_kind = kind
                    best_subject = subject

        if best_kind is not None:
            return best_kind, best_subject

        # Fallback: classify by unit alone
        if unit == "%":
            return "ratio", ""
        elif unit == "岁":
            return "age", ""
        else:
            return "amount", ""


def parse_questions(
    questions_path: str | Path, profile: DomainProfile
) -> list[ParsedQuestion]:
    """Convenience function to parse all questions from a JSON file.

    Args:
        questions_path: Path to the questions JSON file (list of question dicts).
        profile: The domain profile.

    Returns:
        List of ``ParsedQuestion`` instances.
    """
    parser = QuestionParser()
    return parser.parse_questions(questions_path, profile)
