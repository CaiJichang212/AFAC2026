"""Deterministic insurance calculation engine.

Task 10: Performs arithmetic on question-stem numbers and evidence-supplied
formulas/parameters WITHOUT any LLM calls or network access.  All functions
are pure: same inputs always produce the same outputs.

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
"""

from __future__ import annotations

from typing import Any

from agent.schemas import CalculationRecord, EvidenceRecord, ParsedQuestion


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
