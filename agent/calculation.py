"""Deterministic insurance calculation engine.

Task 10: Performs arithmetic on question-stem numbers and evidence-supplied
formulas/parameters WITHOUT any LLM calls or network access.  All functions
are pure: same inputs always produce the same outputs.

Phase D adds fact-matrix computation: for calculation/ranking questions,
structured per-product facts extracted from documents are evaluated against
stem conditions to produce numeric results and rankings — then verdicts are
derived from those results instead of per-page A/B/C/D evidence.

Amount normalisation
--------------------
All monetary amounts are normalised to **元** before arithmetic:

* ``元`` → ×1
* ``万元`` → ×10,000
* bare ``万`` → ×10,000

Ratios (percentages)
--------------------
Percentages are stored as a float with unit ``"%"`` (e.g. 80.0 means 80 %).
The helper ``percent_to_fraction`` converts them to a fraction (e.g. 0.8) for
use in multiplication.  ``fraction_to_percent`` does the reverse.

Evidence convention for formulas
--------------------------------
Evidence records that carry formula information use these conventions:

``evidence_type``
    Set to ``"formula"``.

``normalized_fact``
    A string key identifying the formula, e.g.:

    * ``"death_benefit_account_value"``
    * ``"death_benefit_multiplied"``
    * ``"death_benefit_premium_minus_annuity"``
    * ``"medical_payout"``
    * ``"surrender_value"``
    * ``"ratio_payout"``

``numbers``
    A list of dicts, each with ``kind``, ``value``, ``unit`` (optional).

Fact-matrix expression evaluator
--------------------------------
The fact-matrix path uses a small expression evaluator over the field
vocabulary {保单账户价值, 基本保额, 已交保费, 已领养老年金, 现金价值,
总费用, 医保报销} with operators ``*`` (multiplier) and ``-`` (subtract).
Field names are mapped to stem-condition ``kind`` values for lookup.
"""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any

from agent.schemas import CalculationRecord, EvidenceRecord, FactRecord, ParsedQuestion


# ---------------------------------------------------------------------------
# Field-name vocabulary: Chinese field name -> stem-condition kind
# ---------------------------------------------------------------------------

_FIELD_TO_STEM_KIND: dict[str, str] = {
    "保单账户价值": "account_value",
    "个人账户价值": "account_value",
    "账户价值": "account_value",
    "基本保额": "basic_sum_insured",
    "基本保险金额": "basic_sum_insured",
    "已交保费": "premium_paid",
    "累计所交保费": "premium_paid",
    "已领养老年金": "annuity_received",
    "已领年金": "annuity_received",
    "现金价值": "cash_value",
    "总费用": "total_expense",
    "医保报销": "medical_reimbursement",
    "免赔额": "deductible",
}


# ---------------------------------------------------------------------------
# Unit normalisation
# ---------------------------------------------------------------------------

def normalize_amount(value: float, unit: str) -> float:
    """Convert *value* and *unit* into a base-元 amount.

    >>> normalize_amount(100, "元")
    100.0
    >>> normalize_amount(100, "万元")
    1000000.0
    >>> normalize_amount(144, "万")
    1440000.0
    >>> normalize_amount(80.0, "%")
    80.0
    """
    if unit in ("万元", "万"):
        return value * 10000.0
    # 元 and anything else (%, 岁, "") pass through
    return float(value)


def percent_to_fraction(percent: float) -> float:
    """Convert a percent value to a fraction (e.g. 80.0 → 0.8)."""
    return percent / 100.0


def fraction_to_percent(fraction: float) -> float:
    """Convert a fraction to a percent value (e.g. 0.8 → 80.0)."""
    return fraction * 100.0


# ---------------------------------------------------------------------------
# Deterministic calculation functions
# ---------------------------------------------------------------------------

def calc_death_benefit_account_value(account_value: float) -> tuple[float, str]:
    """Death benefit equals the policy account value.

    Example: 平安智盈金生 (领取日前) = 保单账户价值.
    """
    formula = f"保单账户价值 = {_fmt(account_value)}元"
    return account_value, formula


def calc_death_benefit_multiplied(base: float, multiplier: float) -> tuple[float, str]:
    """Death benefit = base × multiplier.

    Example: 国寿增益宝 = 基本保额 × 1.6.
    """
    result = base * multiplier
    if result == int(result):
        result = int(result)
    else:
        result = round(result, 2)
    formula = (
        f"基本保额 × {multiplier} = {_fmt(base)} × {multiplier}"
        f" = {_fmt(result)}元"
    )
    return result, formula


def calc_death_benefit_premium_minus_annuity(
    premium_paid: float, annuity_received: float
) -> tuple[float, str]:
    """Death benefit = premiums paid − annuity already received.

    Example: 国寿鑫享添盈 = 100万 − 20万 = 80万.
    """
    result = premium_paid - annuity_received
    formula = (
        f"已交保费 − 已领养老年金 = {_fmt(premium_paid)} − {_fmt(annuity_received)}"
        f" = {_fmt(result)}元"
    )
    return result, formula


def calc_surrender_value(
    cash_value: float, surrender_charge: float = 0.0, surrender_charge_pct: float = 0.0
) -> tuple[float, str]:
    """Surrender value = cash value minus surrender charge.

    The charge can be given as a fixed amount (*surrender_charge*, in 元) or as
    a percentage (*surrender_charge_pct*) which is treated as a fraction
    (e.g. 2.0 → 2 % → subtract 0.02 × cash_value).

    Returns ``(result, formula_string)``.
    """
    if surrender_charge:
        result = cash_value - surrender_charge
        formula = (
            f"现金价值 − 退保费用 = {_fmt(cash_value)} − {_fmt(surrender_charge)}"
            f" = {_fmt(result)}元"
        )
    elif surrender_charge_pct:
        charge_amount = cash_value * percent_to_fraction(surrender_charge_pct)
        formula = (
            f"现金价值 × (1 − {surrender_charge_pct}%)"
            f" = {_fmt(cash_value)} × {1 - percent_to_fraction(surrender_charge_pct)}"
            f" = {_fmt(cash_value - charge_amount)}元"
        )
        result = cash_value - charge_amount
    else:
        result = cash_value
        formula = f"现金价值 = {_fmt(result)}元"

    if result == int(result):
        result = int(result)
    else:
        result = round(result, 2)
    return result, formula


def calc_medical_payout(
    total_expense: float,
    reimbursement: float,
    deductible: float,
    payout_ratio_fraction: float,
    cap: float | None = None,
) -> tuple[float, str]:
    """Medical reimbursement after deductions.

    Formula: max(0, total − reimbursement − deductible) × ratio,
    then capped at *cap* if provided.

    *payout_ratio_fraction* is a fraction (e.g. 1.0 for 100%, 0.8 for 80%).

    Returns ``(result, formula_string)``.
    """
    eligible = max(0.0, total_expense - reimbursement - deductible)
    result = eligible * payout_ratio_fraction

    cap_str = ""
    if cap is not None and result > cap:
        result = cap
        cap_str = f", cap at {_fmt(cap)}"

    if result == int(result):
        result = int(result)
    else:
        result = round(result, 2)

    ratio_pct = fraction_to_percent(payout_ratio_fraction)
    formula = (
        f"max(0, {_fmt(total_expense)} − {_fmt(reimbursement)} − {_fmt(deductible)})"
        f" × {ratio_pct}%{cap_str} = {_fmt(eligible)} × {ratio_pct}%"
        f" = {_fmt(result)}元"
    )
    return result, formula


def calc_ratio_payout(
    amount: float,
    ratio_fraction: float,
    cap: float | None = None,
) -> tuple[float, str]:
    """Apply a payout ratio to an amount, optionally capped.

    *ratio_fraction* is a fraction (e.g. 0.6 for 60%).

    Returns ``(result, formula_string)``.
    """
    result = amount * ratio_fraction
    cap_str = ""
    if cap is not None and result > cap:
        result = cap
        cap_str = f", capped at {_fmt(cap)}"

    if result == int(result):
        result = int(result)
    else:
        result = round(result, 2)

    ratio_pct = fraction_to_percent(ratio_fraction)
    formula = (
        f"{_fmt(amount)} × {ratio_pct}%{cap_str} = {_fmt(result)}元"
    )
    return result, formula


def calc_rank_descending(
    items: list[tuple[str, float]],
) -> tuple[list[str], str]:
    """Sort items by value descending, return ordered keys.

    Args:
        items: List of ``(key, value)`` pairs, e.g.
               ``[("产品A", 1440000), ("产品B", 900000)]``.

    Returns:
        ``(ordered_keys, formula_string)`` where *ordered_keys* is the keys
        sorted by descending value.
    """
    sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
    keys = [k for k, _ in sorted_items]
    detail = " > ".join(f"{k}({_fmt_financial(v)})" for k, v in sorted_items)
    formula = f"降序排列: {detail}"
    return keys, formula


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(value: float) -> str:
    """Format a numeric value for display in a formula string."""
    if value == int(value):
        return str(int(value))
    return f"{value:.2f}"


def _fmt_financial(value: float) -> str:
    """Format a monetary value in 万元 for display."""
    wan = value / 10000.0
    if wan == int(wan):
        return f"{int(wan)}万"
    return f"{wan:.1f}万"


# ---------------------------------------------------------------------------
# Product name resolution
# ---------------------------------------------------------------------------

def _resolve_product(doc_id: str, doc_product_map: dict[int, str]) -> str:
    """Resolve a doc_id to a canonical product name.

    Falls back to ``f"doc_{doc_id}"`` when the doc_id is unknown.
    """
    try:
        return doc_product_map[int(doc_id)]
    except (ValueError, KeyError):
        return f"doc_{doc_id}"


# ---------------------------------------------------------------------------
# CalculationEngine
# ---------------------------------------------------------------------------

class CalculationEngine:
    """Deterministic engine for insurance arithmetic.

    Uses rule-based pattern detection on the question text plus
    evidence-supplied formulas/numbers.  No LLM calls, no network.
    """

    def __init__(self, doc_product_map: dict[int, str] | None = None):
        """Initialise the engine.

        Args:
            doc_product_map: Optional doc_id → canonical product name mapping
                used to resolve evidence doc_ids to product names.
        """
        self._doc_product_map = doc_product_map or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        parsed: ParsedQuestion,
        evidence: list[EvidenceRecord],
    ) -> list[CalculationRecord]:
        """Apply deterministic calculations to a parsed question.

        Detects the calculation pattern from the question text and dispatches
        to the appropriate handler.  Uses *parsed.stem_number_conditions* for
        global inputs and *evidence* for per-product formulas/parameters.

        Returns an empty list when no recognised calculation pattern is
        detected.
        """
        question = parsed.question

        # --- pattern detection ---
        has_ranking = "排序" in question
        has_medical = any(
            kw in question for kw in ("免赔额", "医保报销", "医疗费用")
        )
        has_surrender = "退保" in question

        records: list[CalculationRecord] = []

        if has_ranking:
            records.extend(self._compute_ranking(parsed, evidence))
        if has_medical:
            records.extend(self._compute_medical(parsed, evidence))
        if has_surrender:
            records.extend(self._compute_surrender(parsed, evidence))

        # If no pattern matched but evidence carries formulas, try best-effort
        if not records:
            records.extend(self._compute_generic(parsed, evidence))

        return records

    # ------------------------------------------------------------------
    # Pattern: ranking / sorting (e.g. ins_a_001)
    # ------------------------------------------------------------------

    def _compute_ranking(
        self,
        parsed: ParsedQuestion,
        evidence: list[EvidenceRecord],
    ) -> list[CalculationRecord]:
        """Compute per-product death benefits and rank them.

        Evidence records with ``evidence_type == "formula"`` supply the
        per-product formula identifier in ``normalized_fact`` and the numeric
        inputs in ``numbers``.
        """
        # Group formula evidence by doc_id
        formula_evidence = [
            ev for ev in evidence if ev.evidence_type == "formula"
        ]

        if not formula_evidence:
            return []

        product_values: dict[str, float] = {}
        source_ids: list[str] = []
        inputs_summary: dict[str, Any] = {}
        formula_details: list[str] = []

        # Build a quick lookup from stem conditions for global values
        stem_lookup: dict[str, float] = {}
        for cond in parsed.stem_number_conditions:
            stem_lookup[cond["kind"]] = cond["value"]

        for ev in formula_evidence:
            product = _resolve_product(ev.doc_id, self._doc_product_map)
            formula_key = ev.normalized_fact
            numbers = {n["kind"]: n["value"] for n in ev.numbers}

            sid = f"{ev.doc_id}/{ev.node_id}"
            source_ids.append(sid)

            value, formula_str = self._apply_death_benefit_formula(
                formula_key, numbers, stem_lookup
            )
            if value is not None:
                product_values[product] = value
                inputs_summary[product] = {
                    "formula_key": formula_key,
                    "numbers": numbers,
                }
                formula_details.append(f"{product}: {formula_str}")

        if not product_values:
            return []

        # Rank
        ranked_keys, rank_formula = calc_rank_descending(
            list(product_values.items())
        )

        record = CalculationRecord(
            qid=parsed.qid,
            calc_type="ranking",
            inputs={
                "product_values": {
                    k: _fmt(v) for k, v in product_values.items()
                },
                "per_product": inputs_summary,
            },
            formula=(
                "\n".join(formula_details) + "\n" + rank_formula
            ),
            result=0.0,  # ranking has no single scalar result
            unit="",
            source_evidence_ids=source_ids,
        )
        # Store the ranking order in a structured way
        record.result = 0.0  # keep as-is; the formula string carries the order
        # Attach ranking metadata
        record.inputs["ranked_order"] = ranked_keys
        record.inputs["ranked_values"] = {
            k: _fmt(v) for k, v in sorted(
                product_values.items(), key=lambda x: x[1], reverse=True
            )
        }

        return [record]

    @staticmethod
    def _apply_death_benefit_formula(
        formula_key: str,
        numbers: dict[str, float],
        stem_lookup: dict[str, float],
    ) -> tuple[float | None, str]:
        """Apply a death-benefit formula given its key and inputs.

        Returns ``(value, formula_string)`` or ``(None, "")`` if the formula
        key is unrecognised.
        """
        if formula_key == "death_benefit_account_value":
            av = numbers.get("account_value", 0)
            return calc_death_benefit_account_value(av)

        elif formula_key == "death_benefit_multiplied":
            base = numbers.get("basic_sum_insured", 0)
            multiplier = numbers.get("multiplier", 1.0)
            return calc_death_benefit_multiplied(base, multiplier)

        elif formula_key == "death_benefit_premium_minus_annuity":
            pp = numbers.get(
                "premium_paid", stem_lookup.get("premium_paid", 0)
            )
            ar = numbers.get(
                "annuity_received", stem_lookup.get("annuity_received", 0)
            )
            return calc_death_benefit_premium_minus_annuity(pp, ar)

        else:
            return None, ""

    # ------------------------------------------------------------------
    # Pattern: medical deduction (e.g. ins_a_003)
    # ------------------------------------------------------------------

    def _compute_medical(
        self,
        parsed: ParsedQuestion,
        evidence: list[EvidenceRecord],
    ) -> list[CalculationRecord]:
        """Compute medical reimbursement for each product/insurer.

        Extracts total expense and social-insurance reimbursement from the
        stem.  Per-product deductibles / payout ratios / caps come from
        evidence records.
        """
        stem = {c["kind"]: c["value"] for c in parsed.stem_number_conditions}

        total_expense = stem.get("total_expense", 0)
        reimbursement = stem.get("medical_reimbursement", 0)

        formula_evidence = [
            ev for ev in evidence if ev.evidence_type == "formula"
        ]
        if not formula_evidence:
            return []

        records: list[CalculationRecord] = []
        for ev in formula_evidence:
            product = _resolve_product(ev.doc_id, self._doc_product_map)
            numbers = {n["kind"]: n["value"] for n in ev.numbers}

            deductible = numbers.get("deductible", 0)
            payout_ratio_pct = numbers.get("payout_ratio", 100.0)
            payout_ratio_fraction = percent_to_fraction(payout_ratio_pct)
            cap = numbers.get("cap", None)

            value, formula_str = calc_medical_payout(
                total_expense=total_expense,
                reimbursement=reimbursement,
                deductible=deductible,
                payout_ratio_fraction=payout_ratio_fraction,
                cap=cap,
            )

            records.append(
                CalculationRecord(
                    qid=parsed.qid,
                    calc_type="medical_payout",
                    inputs={
                        "product": product,
                        "total_expense": total_expense,
                        "reimbursement": reimbursement,
                        "deductible": deductible,
                        "payout_ratio": payout_ratio_pct,
                        "payout_ratio_fraction": payout_ratio_fraction,
                        "cap": cap,
                    },
                    formula=f"{product}: {formula_str}",
                    result=value,
                    unit="元",
                    source_evidence_ids=[f"{ev.doc_id}/{ev.node_id}"],
                )
            )

        return records

    # ------------------------------------------------------------------
    # Pattern: surrender value
    # ------------------------------------------------------------------

    def _compute_surrender(
        self,
        parsed: ParsedQuestion,
        evidence: list[EvidenceRecord],
    ) -> list[CalculationRecord]:
        """Compute surrender values for each product."""
        formula_evidence = [
            ev for ev in evidence if ev.evidence_type == "formula"
        ]
        if not formula_evidence:
            return []

        records: list[CalculationRecord] = []
        for ev in formula_evidence:
            product = _resolve_product(ev.doc_id, self._doc_product_map)
            numbers = {n["kind"]: n["value"] for n in ev.numbers}

            cash_value = numbers.get("cash_value", 0)
            surrender_charge = numbers.get("surrender_charge", 0)
            surrender_charge_pct = numbers.get("surrender_charge_pct", 0)

            value, formula_str = calc_surrender_value(
                cash_value, surrender_charge, surrender_charge_pct
            )

            records.append(
                CalculationRecord(
                    qid=parsed.qid,
                    calc_type="surrender_value",
                    inputs={
                        "product": product,
                        "cash_value": cash_value,
                        "surrender_charge": surrender_charge,
                        "surrender_charge_pct": surrender_charge_pct,
                    },
                    formula=f"{product}: {formula_str}",
                    result=value,
                    unit="元",
                    source_evidence_ids=[f"{ev.doc_id}/{ev.node_id}"],
                )
            )

        return records

    # ------------------------------------------------------------------
    # Fallback: generic formula application
    # ------------------------------------------------------------------

    def _compute_generic(
        self,
        parsed: ParsedQuestion,
        evidence: list[EvidenceRecord],
    ) -> list[CalculationRecord]:
        """Best-effort: apply any formula evidence even without a recognised
        top-level pattern."""
        formula_evidence = [
            ev for ev in evidence if ev.evidence_type == "formula"
        ]
        if not formula_evidence:
            return []

        records: list[CalculationRecord] = []
        for ev in formula_evidence:
            product = _resolve_product(ev.doc_id, self._doc_product_map)
            numbers = {n["kind"]: n["value"] for n in ev.numbers}

            # Try medical_payout
            if "deductible" in numbers:
                total = numbers.get("total_expense", 0)
                reimb = numbers.get("medical_reimbursement", 0)
                deductible = numbers.get("deductible", 0)
                ratio_pct = numbers.get("payout_ratio", 100.0)
                cap = numbers.get("cap")
                value, formula_str = calc_medical_payout(
                    total, reimb, deductible,
                    percent_to_fraction(ratio_pct), cap,
                )
                records.append(
                    CalculationRecord(
                        qid=parsed.qid,
                        calc_type="medical_payout",
                        inputs={
                            "product": product,
                            "total_expense": total,
                            "reimbursement": reimb,
                            "deductible": deductible,
                            "payout_ratio": ratio_pct,
                            "cap": cap,
                        },
                        formula=f"{product}: {formula_str}",
                        result=value,
                        unit="元",
                        source_evidence_ids=[f"{ev.doc_id}/{ev.node_id}"],
                    )
                )
                continue

            # Try ratio_payout
            if "amount" in numbers and "ratio" in numbers:
                amount = numbers["amount"]
                ratio_pct = numbers["ratio"]
                cap = numbers.get("cap")
                value, formula_str = calc_ratio_payout(
                    amount, percent_to_fraction(ratio_pct), cap,
                )
                records.append(
                    CalculationRecord(
                        qid=parsed.qid,
                        calc_type="ratio_payout",
                        inputs={
                            "product": product,
                            "amount": amount,
                            "ratio": ratio_pct,
                            "cap": cap,
                        },
                        formula=f"{product}: {formula_str}",
                        result=value,
                        unit="元",
                        source_evidence_ids=[f"{ev.doc_id}/{ev.node_id}"],
                    )
                )
                continue

        return records

    # ------------------------------------------------------------------
    # Phase D: Fact-matrix computation
    # ------------------------------------------------------------------

    def compute_from_facts(
        self,
        parsed: ParsedQuestion,
        facts: list[FactRecord],
    ) -> list[CalculationRecord]:
        """Compute results from extracted fact matrix (Phase D).

        Consumes structured per-product facts and stem number conditions
        to produce deterministic CalculationRecords.  Supports ranking
        (death-benefit comparison) and medical-payout patterns.

        Returns an empty list when no computation-relevant facts are found.
        """
        if not facts:
            return []

        # Build stem-values lookup keyed by English kind
        stem_values: dict[str, float] = {}
        for cond in parsed.stem_number_conditions:
            stem_values[cond["kind"]] = cond["value"]

        # Group facts by product
        product_facts: dict[str, list[FactRecord]] = {}
        for f in facts:
            product_facts.setdefault(f.product, []).append(f)

        question = parsed.question
        has_ranking = "排序" in question or any(
            kw in question for kw in ("身故保险金",)
        )
        has_medical = any(
            kw in question for kw in ("免赔额", "医保报销", "医疗费用")
        )
        has_surrender = "退保" in question

        records: list[CalculationRecord] = []

        if has_ranking:
            records.extend(
                self._compute_ranking_from_facts(
                    parsed, product_facts, stem_values
                )
            )

        if has_medical:
            records.extend(
                self._compute_medical_from_facts(
                    parsed, product_facts, stem_values
                )
            )

        if has_surrender:
            records.extend(
                self._compute_surrender_from_facts(
                    parsed, product_facts, stem_values
                )
            )

        # If no pattern matched, try generic fact-based computation
        if not records:
            records.extend(
                self._compute_generic_from_facts(
                    parsed, product_facts, stem_values
                )
            )

        return records

    def _compute_ranking_from_facts(
        self,
        parsed: ParsedQuestion,
        product_facts: dict[str, list[FactRecord]],
        stem_values: dict[str, float],
    ) -> list[CalculationRecord]:
        """Compute per-product death benefits from facts and rank them."""
        product_values: dict[str, float] = {}
        formula_details: list[str] = []
        source_ids: list[str] = []
        inputs_per_product: dict[str, Any] = {}
        supporting_facts: list[dict[str, Any]] = []

        # Build per-product stem lookup from parser-tagged conditions
        per_product_stem = _build_per_product_stem_lookup(parsed)

        for product, pfacts in product_facts.items():
            # Find the death-benefit rule fact for this product
            db_fact = _find_fact(pfacts, "身故保险金")
            if db_fact is None:
                continue

            # Build per-product value overrides from facts: any fact with a
            # plain numeric formula_or_value contributes a per-product value
            # that overrides the global stem value (e.g. product-specific
            # 已领养老年金 differs from the stem-wide default).
            product_fact_overrides: dict[str, float] = {}
            for pf in pfacts:
                if pf.field == "身故保险金":
                    continue  # skip the rule itself
                try:
                    product_fact_overrides[_FIELD_TO_STEM_KIND.get(pf.field, pf.field)] = float(pf.formula_or_value)
                except (ValueError, TypeError):
                    pass

            # Build per-product lookup for evaluate_formula:
            # Priority: product_fact_overrides > per_product_stem > (global fallback)
            pv: dict[str, float] = {}
            # Start with per-product stem values for this product
            p_stem = per_product_stem.get(product, {})
            pv.update(p_stem)
            # Per-product fact overrides take highest priority
            pv.update(product_fact_overrides)

            value = evaluate_formula(
                db_fact.formula_or_value, stem_values, product_values=pv
            )
            if value is None:
                continue

            product_values[product] = value
            sid = f"{db_fact.source_doc_id}/{db_fact.source_node_id}"
            source_ids.append(sid)
            inputs_per_product[product] = {
                "formula_or_value": db_fact.formula_or_value,
                "value": value,
            }
            formula_details.append(
                f"{product}: {db_fact.formula_or_value} = {_fmt(value)}元"
            )
            supporting_facts.append(asdict(db_fact))

        if not product_values:
            return []

        ranked_keys, rank_formula = calc_rank_descending(
            list(product_values.items())
        )

        record = CalculationRecord(
            qid=parsed.qid,
            calc_type="ranking",
            inputs={
                "product_values": {
                    k: _fmt(v) for k, v in product_values.items()
                },
                "per_product": inputs_per_product,
                "ranked_order": ranked_keys,
                "ranked_values": {
                    k: _fmt(v) for k, v in sorted(
                        product_values.items(), key=lambda x: x[1], reverse=True
                    )
                },
                "supporting_facts": supporting_facts,
            },
            formula="\n".join(formula_details) + "\n" + rank_formula,
            result=0.0,
            unit="",
            source_evidence_ids=source_ids,
        )

        return [record]

    def _compute_medical_from_facts(
        self,
        parsed: ParsedQuestion,
        product_facts: dict[str, list[FactRecord]],
        stem_values: dict[str, float],
    ) -> list[CalculationRecord]:
        """Compute medical payouts from per-product deductible/ratio/cap facts."""
        total_expense = stem_values.get("total_expense", 0)
        reimbursement = stem_values.get("medical_reimbursement", 0)

        records: list[CalculationRecord] = []
        for product, pfacts in product_facts.items():
            deductible_fact = _find_fact(pfacts, "免赔额")
            if deductible_fact is None:
                continue

            try:
                deductible = float(deductible_fact.formula_or_value)
            except (ValueError, TypeError):
                continue

            # Optional: payout ratio
            ratio_fact = _find_fact(pfacts, "给付比例")
            payout_ratio_pct: float = 100.0
            if ratio_fact is not None:
                try:
                    payout_ratio_pct = float(ratio_fact.formula_or_value)
                except (ValueError, TypeError):
                    payout_ratio_pct = 100.0

            # Optional: cap
            cap_fact = _find_fact(pfacts, "最高限额")
            cap: float | None = None
            if cap_fact is not None:
                try:
                    cap = float(cap_fact.formula_or_value)
                except (ValueError, TypeError):
                    cap = None

            payout_ratio_fraction = percent_to_fraction(payout_ratio_pct)
            value, formula_str = calc_medical_payout(
                total_expense=total_expense,
                reimbursement=reimbursement,
                deductible=deductible,
                payout_ratio_fraction=payout_ratio_fraction,
                cap=cap,
            )

            supporting_facts: list[dict[str, Any]] = [asdict(deductible_fact)]
            if ratio_fact:
                supporting_facts.append(asdict(ratio_fact))
            if cap_fact:
                supporting_facts.append(asdict(cap_fact))

            sid = f"{deductible_fact.source_doc_id}/{deductible_fact.source_node_id}"

            records.append(
                CalculationRecord(
                    qid=parsed.qid,
                    calc_type="medical_payout",
                    inputs={
                        "product": product,
                        "total_expense": total_expense,
                        "reimbursement": reimbursement,
                        "deductible": deductible,
                        "payout_ratio": payout_ratio_pct,
                        "payout_ratio_fraction": payout_ratio_fraction,
                        "cap": cap,
                        "supporting_facts": supporting_facts,
                    },
                    formula=f"{product}: {formula_str}",
                    result=value,
                    unit="元",
                    source_evidence_ids=[sid],
                )
            )

        return records

    def _compute_surrender_from_facts(
        self,
        parsed: ParsedQuestion,
        product_facts: dict[str, list[FactRecord]],
        stem_values: dict[str, float],
    ) -> list[CalculationRecord]:
        """Compute per-product surrender values from facts and rank them.

        Looks for 现金价值 (cash value) facts per product.  If a surrender-charge
        percentage or deductible is present as a fact on the same product it is
        applied.  Results are ranked descending.
        """
        product_values: dict[str, float] = {}
        formula_details: list[str] = []
        source_ids: list[str] = []
        inputs_per_product: dict[str, Any] = {}
        supporting_facts: list[dict[str, Any]] = []

        per_product_stem = _build_per_product_stem_lookup(parsed)

        for product, pfacts in product_facts.items():
            # Build per-product value overrides from facts
            product_fact_overrides: dict[str, float] = {}
            for pf in pfacts:
                if pf.field == "现金价值":
                    continue
                try:
                    product_fact_overrides[_FIELD_TO_STEM_KIND.get(pf.field, pf.field)] = float(pf.formula_or_value)
                except (ValueError, TypeError):
                    pass

            # Build per-product lookup for evaluate_formula
            pv: dict[str, float] = {}
            p_stem = per_product_stem.get(product, {})
            pv.update(p_stem)
            pv.update(product_fact_overrides)

            # Look for a cash-value fact for this product
            cv_fact = _find_fact(pfacts, "现金价值")
            if cv_fact is None:
                continue

            # Try to parse the cash value as a number or evaluate it
            cv_value: float | None = None
            try:
                cv_value = float(cv_fact.formula_or_value)
            except (ValueError, TypeError):
                # Try expression evaluation with per-product context
                cv_value = evaluate_formula(
                    cv_fact.formula_or_value, stem_values, product_values=pv
                )

            if cv_value is None:
                continue

            # Look for surrender-charge rate (optional)
            charge_rate: float = 0.0
            charge_fact = _find_fact(pfacts, "给付比例")
            if charge_fact is not None:
                try:
                    charge_rate = float(charge_fact.formula_or_value)
                except (ValueError, TypeError):
                    pass

            surrender_value = cv_value * (charge_rate / 100.0) if charge_rate else cv_value
            if surrender_value == int(surrender_value):
                surrender_value = int(surrender_value)
            else:
                surrender_value = round(surrender_value, 2)

            product_values[product] = surrender_value
            sid = f"{cv_fact.source_doc_id}/{cv_fact.source_node_id}"
            source_ids.append(sid)
            inputs_per_product[product] = {
                "cash_value": cv_value,
                "charge_rate_pct": charge_rate,
                "surrender_value": surrender_value,
            }
            charge_detail = f" × {charge_rate}%" if charge_rate else ""
            formula_details.append(
                f"{product}: 现金价值{charge_detail} = {_fmt(surrender_value)}元"
            )
            supporting_facts.append(asdict(cv_fact))
            if charge_fact:
                supporting_facts.append(asdict(charge_fact))

        if not product_values:
            return []

        ranked_keys, rank_formula = calc_rank_descending(
            list(product_values.items())
        )

        record = CalculationRecord(
            qid=parsed.qid,
            calc_type="ranking",
            inputs={
                "product_values": {
                    k: _fmt(v) for k, v in product_values.items()
                },
                "per_product": inputs_per_product,
                "ranked_order": ranked_keys,
                "ranked_values": {
                    k: _fmt(v) for k, v in sorted(
                        product_values.items(), key=lambda x: x[1], reverse=True
                    )
                },
                "supporting_facts": supporting_facts,
            },
            formula="\n".join(formula_details) + "\n" + rank_formula,
            result=0.0,
            unit="",
            source_evidence_ids=source_ids,
        )

        return [record]

    def _compute_generic_from_facts(
        self,
        parsed: ParsedQuestion,
        product_facts: dict[str, list[FactRecord]],
        stem_values: dict[str, float],
    ) -> list[CalculationRecord]:
        """Best-effort fact-based computation for unrecognised patterns."""
        # For now, return empty — can be extended later.
        return []


# ---------------------------------------------------------------------------
# Phase D: Routing helper
# ---------------------------------------------------------------------------


def is_computation_question(parsed: ParsedQuestion) -> bool:
    """Determine whether *parsed* should use the fact-matrix computation path.

    Uses two signals (OR logic):
    1. Keyword-based: the question text contains computation-relevant keywords
       (排序, 免赔额, 退保, 医保报销, 医疗费用).  These indicate questions where
       deterministic arithmetic on extracted facts can produce a ranked or
       numeric answer.
    2. Type-based: ``parsed.type`` is 计算题.

    Note: "身故保险金" alone is NOT a computation keyword — it is too
    common in fact-query questions.  ``推理判断`` type alone also does NOT
    trigger (the keyword check gates it; ins_a_001 has both "排序" and
    推理判断, so it routes correctly).

    Returns True when either signal fires, routing the question to
    ``extract_fact_matrix`` → ``compute_from_facts`` → computation-derived
    verdicts instead of per-page A/B/C/D evidence.
    """
    question = parsed.question
    calc_keywords = [
        "排序", "免赔额", "退保", "医保报销", "医疗费用",
    ]
    has_keyword = any(kw in question for kw in calc_keywords)
    is_calc_type = parsed.type in {"计算题"}
    return has_keyword or is_calc_type


# ---------------------------------------------------------------------------
# Phase D: Mini expression evaluator for fact formulas
# ---------------------------------------------------------------------------


def evaluate_formula(
    formula_or_value: str,
    stem_values: dict[str, float],
    *,
    product_values: dict[str, float] | None = None,
) -> float | None:
    """Evaluate a fact ``formula_or_value`` against stem condition values.

    Supports three expression forms:

    * Plain number: ``"5000"`` → ``5000.0``
    * Field reference: ``"保单账户价值"`` → stem value of account_value
    * Field × multiplier: ``"基本保额*1.6"`` → stem[basic_sum_insured] × 1.6
    * Field − field: ``"已交保费-已领养老年金"`` →
      stem[premium_paid] − stem[annuity_received]

    Resolution priority for field references (highest first):

    1. *product_values* — per-product overrides (fact values + per-product stem)
    2. *stem_values* — global fallback

    Returns the numeric result, or None when the expression cannot be
    evaluated (unknown field, missing stem value).

    >>> evaluate_formula("900000", {})
    900000.0
    >>> evaluate_formula("基本保额*1.6", {"basic_sum_insured": 900000})
    1440000.0
    >>> evaluate_formula("已交保费-已领养老年金", {"premium_paid": 1000000, "annuity_received": 200000})
    800000.0
    """
    expr = formula_or_value.strip()

    if not expr:
        return None

    # Case 1: plain number
    try:
        return float(expr)
    except ValueError:
        pass

    # Helper: resolve a field-kind to a numeric value
    def _resolve(kind: str) -> float | None:
        # Priority 1: per-product overrides
        if product_values is not None and kind in product_values:
            return product_values[kind]
        # Priority 2: global stem fallback
        return stem_values.get(kind)

    # Case 2: field * multiplier  (e.g. "基本保额*1.6")
    m_mult = re.match(r'^(.+?)\*([\d.]+)$', expr)
    if m_mult:
        field_name = m_mult.group(1).strip()
        multiplier = float(m_mult.group(2))
        kind = _FIELD_TO_STEM_KIND.get(field_name)
        if kind is None:
            return None
        val = _resolve(kind)
        if val is None:
            return None
        return val * multiplier

    # Case 3: field - field  (e.g. "已交保费-已领养老年金")
    m_sub = re.match(r'^(.+?)-(.+)$', expr)
    if m_sub:
        left_name = m_sub.group(1).strip()
        right_name = m_sub.group(2).strip()
        left_kind = _FIELD_TO_STEM_KIND.get(left_name)
        right_kind = _FIELD_TO_STEM_KIND.get(right_name)
        if left_kind is None or right_kind is None:
            return None
        left_val = _resolve(left_kind)
        right_val = _resolve(right_kind)
        if left_val is None or right_val is None:
            return None
        return left_val - right_val

    # Case 4: plain field reference (e.g. "保单账户价值")
    kind = _FIELD_TO_STEM_KIND.get(expr)
    if kind is not None:
        val = _resolve(kind)
        if val is not None:
            return val

    return None


# ---------------------------------------------------------------------------
# Phase D: Fact lookup helper
# ---------------------------------------------------------------------------


def _find_fact(
    facts: list[FactRecord], field: str
) -> FactRecord | None:
    """Return the first fact whose ``.field`` matches *field*."""
    for f in facts:
        if f.field == field:
            return f
    return None


# ---------------------------------------------------------------------------
# Phase D: Per-product stem lookup
# ---------------------------------------------------------------------------


def _build_per_product_stem_lookup(
    parsed: ParsedQuestion,
) -> dict[str, dict[str, float]]:
    """Build ``{product_name: {kind: value}}`` from parser-tagged stem conditions.

    Only conditions that carry a non-None ``product`` field are included.
    Returns an empty dict when no conditions have product tags.
    """
    result: dict[str, dict[str, float]] = {}
    for cond in parsed.stem_number_conditions:
        product = cond.get("product")
        if product is not None and isinstance(product, str) and product:
            result.setdefault(product, {})[cond["kind"]] = cond["value"]
    return result
