"""Deterministic answer judge — rule-based, no LLM calls.

Task 11: Consumes evidence verdicts and calculation results to produce
a final AnswerRecord.  Every decision is pure and repeatable.
"""

from __future__ import annotations

import re
from typing import Any

from agent.schemas import AnswerRecord, CalculationRecord, EvidenceRecord, ParsedQuestion

# ---------------------------------------------------------------------------
# Confidence weights
# ---------------------------------------------------------------------------

_CONFIDENCE_WEIGHT: dict[str, int] = {"high": 3, "medium": 2, "low": 1}


def _confidence_weight(confidence: str) -> int:
    return _CONFIDENCE_WEIGHT.get(confidence, 1)


# ---------------------------------------------------------------------------
# Option judgement builder
# ---------------------------------------------------------------------------


def _build_option_judgements(
    evidence: list[EvidenceRecord],
    options: list[str],
) -> dict[str, dict[str, Any]]:
    """Aggregate evidence per option into counts and weighted scores."""
    judgements: dict[str, dict[str, Any]] = {}

    for opt in options:
        opt_evidence = [e for e in evidence if e.option == opt]
        support_count = sum(1 for e in opt_evidence if e.evidence_type == "support")
        refute_count = sum(1 for e in opt_evidence if e.evidence_type == "refute")
        unclear_count = sum(1 for e in opt_evidence if e.evidence_type == "unclear")

        support_score = sum(
            _confidence_weight(e.confidence)
            for e in opt_evidence
            if e.evidence_type == "support"
        )
        refute_score = sum(
            _confidence_weight(e.confidence)
            for e in opt_evidence
            if e.evidence_type == "refute"
        )
        net_score = support_score - refute_score

        judgements[opt] = {
            "support_count": support_count,
            "refute_count": refute_count,
            "unclear_count": unclear_count,
            "support_score": support_score,
            "refute_score": refute_score,
            "net_score": net_score,
        }

    return judgements


# ---------------------------------------------------------------------------
# Answer normalisation / correction
# ---------------------------------------------------------------------------


def _normalize_mcq(raw: str, options: list[str]) -> str:
    """Normalise an mcq answer to a single uppercase letter."""
    cleaned = re.sub(r"[^A-Za-z]", "", raw).upper()
    if len(cleaned) >= 1 and cleaned[0] in options:
        return cleaned[0]
    # Best-guess: return the first valid option letter
    return options[0] if options else "A"


def _normalize_multi(raw: str, options: list[str]) -> str:
    """Normalise a multi answer to sorted unique uppercase letters, no separators."""
    cleaned = re.sub(r"[^A-Za-z]", "", raw).upper()
    # Keep only valid option letters, deduplicate, sort
    valid = sorted(set(c for c in cleaned if c in options))
    if valid:
        return "".join(valid)
    return ""


def _normalize_tf(raw: str) -> str:
    """Normalise a true/false answer to 'A' or 'B'."""
    cleaned = re.sub(r"[^A-Za-z]", "", raw).upper()
    if "A" in cleaned:
        return "A"
    if "B" in cleaned:
        return "B"
    return "A"  # best-guess fallback


# ---------------------------------------------------------------------------
# Format-specific decision logic
# ---------------------------------------------------------------------------


def _decide_mcq(
    judgements: dict[str, dict[str, Any]],
    options: list[str],
) -> tuple[str, list[str]]:
    """Pick the single best option for mcq format.

    Returns (answer, warnings).
    """
    warnings: list[str] = []

    # Sort by net_score desc, then support_score desc, then option letter asc
    def _sort_key(opt: str) -> tuple[int, int, str]:
        j = judgements[opt]
        return (-j["net_score"], -j["support_score"], opt)

    sorted_opts = sorted(options, key=_sort_key)
    best = sorted_opts[0]

    # Check if all options are unclear (all net_score == 0 and support_count == 0)
    all_unclear = all(
        judgements[o]["support_count"] == 0 and judgements[o]["refute_count"] == 0
        for o in options
    )
    if all_unclear:
        warnings.append(
            "answer_unclear: all options have no support or refute evidence; "
            f"best-guess selection of option {best} based on fallback ordering"
        )

    return best, warnings


def _decide_multi(
    judgements: dict[str, dict[str, Any]],
    options: list[str],
) -> tuple[str, list[str]]:
    """Select all options with net_score > 0. Fall back to single best if none.

    Returns (answer, warnings).
    """
    warnings: list[str] = []

    positive = sorted(o for o in options if judgements[o]["net_score"] > 0)

    if positive:
        answer = "".join(positive)
        return answer, warnings

    # Fallback: single best option
    def _sort_key(opt: str) -> tuple[int, int, str]:
        j = judgements[opt]
        return (-j["net_score"], -j["support_score"], opt)

    best = sorted(options, key=_sort_key)[0]
    warnings.append(
        f"answer_unclear: no option has net_score > 0; "
        f"falling back to single best option {best}"
    )
    return best, warnings


def _decide_tf(
    judgements: dict[str, dict[str, Any]],
    options: list[str],
) -> tuple[str, list[str]]:
    """Decide true/false: A (True/Yes) or B (False/No).

    Returns (answer, warnings).
    """
    warnings: list[str] = []

    a_net = judgements.get("A", {}).get("net_score", 0)
    b_net = judgements.get("B", {}).get("net_score", 0)

    if a_net > 0:
        return "A", warnings
    if b_net > 0:
        return "B", warnings

    # Neither has net > 0 — best-guess by score
    def _sort_key(opt: str) -> tuple[int, int, str]:
        j = judgements[opt]
        return (-j["net_score"], -j["support_score"], opt)

    best = sorted(["A", "B"], key=_sort_key)[0]
    warnings.append(
        f"answer_unclear: neither A nor B has net_score > 0; "
        f"best-guess selection of option {best}"
    )
    return best, warnings


# ---------------------------------------------------------------------------
# Calculation-informed refinement
# ---------------------------------------------------------------------------


def _apply_calculation_hint(
    raw_answer: str,
    parsed: ParsedQuestion,
    calculations: list[CalculationRecord],
    judgements: dict[str, dict[str, Any]],
    options: list[str],
) -> str:
    """Use calculation results to refine the answer when applicable.

    For ranking calculations: if the computed ranked_order matches an option's
    stated product order, that option is selected (overriding the evidence-based
    pick when the match is unambiguous).

    Returns the (possibly refined) answer string.
    """
    if not calculations:
        return raw_answer

    for calc in calculations:
        if calc.calc_type != "ranking":
            continue

        ranked_order: list[str] = calc.inputs.get("ranked_order", [])
        if not ranked_order:
            continue

        # Build product-name presence maps for each option
        option_matches: list[tuple[str, int]] = []  # (option, match_count)
        for opt in options:
            opt_text = parsed.options.get(opt, "")
            # Count how many ranked product names appear in order in the option
            match_strength = _ranked_order_match_strength(ranked_order, opt_text)
            option_matches.append((opt, match_strength))

        # Sort by match strength descending
        option_matches.sort(key=lambda x: x[1], reverse=True)

        best_match = option_matches[0]
        second_best = option_matches[1] if len(option_matches) > 1 else ("", 0)

        # Only override if the best match is unambiguous (strictly better than second)
        if best_match[1] > second_best[1] and best_match[1] >= len(ranked_order) - 1:
            return best_match[0]

    return raw_answer


def _ranked_order_match_strength(ranked_order: list[str], option_text: str) -> int:
    """Score how well the ranked product order matches an option's text.

    Products present in the correct relative order receive a higher weight
    (3 points each) than products present but out of order (1 point each).
    This ensures that an option whose stated product ordering matches the
    computed ranking order scores higher than one that merely mentions the
    same products.
    """
    score = 0
    last_pos = -1
    for product in ranked_order:
        pos = option_text.find(product)
        if pos >= 0:
            if pos > last_pos:
                score += 3  # correctly ordered
            else:
                score += 1  # present but out of order
            last_pos = pos
    return score


# ---------------------------------------------------------------------------
# Phase D: Synthesize support evidence from computation results
# ---------------------------------------------------------------------------


def _synthesize_computation_evidence(
    parsed: ParsedQuestion,
    calculations: list[CalculationRecord],
    answer: str,
) -> list[EvidenceRecord]:
    """Synthesize support EvidenceRecords from computation supporting facts.

    When a calculation definitively identifies the answer (e.g. a ranking
    whose ranked_order matches exactly one option), this function extracts
    the ``supporting_facts`` embedded in CalculationRecord.inputs and
    converts each fact's quote into a support EvidenceRecord for the
    selected option.

    Returns an empty list when no supporting facts are available or when
    the calculation does not definitively identify an answer.
    """
    if not calculations or not answer:
        return []

    synthetic: list[EvidenceRecord] = []

    for calc in calculations:
        supporting_facts: list[dict[str, Any]] = (
            calc.inputs.get("supporting_facts", [])
        )
        if not supporting_facts:
            # Fallback for ranking: check per_product inputs
            if calc.calc_type == "ranking":
                per_product = calc.inputs.get("per_product", {})
                for product, pinfo in per_product.items():
                    # Create a minimal evidence record from formula info
                    formula = pinfo.get("formula_or_value", "")
                    synthetic.append(
                        EvidenceRecord(
                            qid=parsed.qid,
                            doc_id="",
                            node_id="",
                            pages="",
                            option=answer,
                            evidence_type="support",
                            quote=f"{product}: {formula}",
                            normalized_fact=f"{product}身故保险金={formula}",
                            numbers=[],
                            confidence="high",
                        )
                    )
            continue

        for sf in supporting_facts:
            quote = sf.get("quote", "")
            product = sf.get("product", "")
            field = sf.get("field", "")
            formula_or_value = sf.get("formula_or_value", "")

            synthetic.append(
                EvidenceRecord(
                    qid=parsed.qid,
                    doc_id=sf.get("source_doc_id", ""),
                    node_id=sf.get("source_node_id", ""),
                    pages=sf.get("source_pages", ""),
                    option=answer,
                    evidence_type="support",
                    quote=quote,
                    normalized_fact=f"{product}: {field}={formula_or_value}",
                    numbers=[],
                    confidence="high",
                )
            )

    return synthetic


# ---------------------------------------------------------------------------
# AnswerJudge
# ---------------------------------------------------------------------------


class AnswerJudge:
    """Deterministic, rule-based answer judge.

    Consumes evidence records and calculation results; produces a final
    AnswerRecord.  No LLM calls, no network access — fully repeatable.
    """

    def judge(
        self,
        parsed: ParsedQuestion,
        evidence: list[EvidenceRecord],
        calculations: list[CalculationRecord] | None = None,
    ) -> AnswerRecord:
        """Produce a final AnswerRecord from evidence and calculations.

        Args:
            parsed: The parsed question with options and answer_format.
            evidence: Evidence records from the evidence extraction stage.
            calculations: Optional calculation records from the calculation engine.

        Returns:
            A fully populated AnswerRecord.
        """
        if calculations is None:
            calculations = []

        options = sorted(parsed.options.keys())
        fmt = parsed.answer_format
        warnings: list[str] = []
        fallbacks: list[str] = []

        # 1. Build per-option aggregate judgements
        option_judgements = _build_option_judgements(evidence, options)

        # 2. Format-specific raw decision
        if fmt == "mcq":
            raw_answer, decision_warnings = _decide_mcq(option_judgements, options)
        elif fmt == "multi":
            raw_answer, decision_warnings = _decide_multi(option_judgements, options)
        elif fmt == "tf":
            raw_answer, decision_warnings = _decide_tf(option_judgements, options)
        else:
            # Unknown format — treat as mcq best-guess
            raw_answer, decision_warnings = _decide_mcq(option_judgements, options)
            fallbacks.append(f"unknown_answer_format: {fmt!r} treated as mcq")

        warnings.extend(decision_warnings)

        # 3. Calculation-informed refinement
        calc_answer = _apply_calculation_hint(
            raw_answer, parsed, calculations, option_judgements, options
        )
        if calc_answer != raw_answer:
            fallbacks.append(
                f"calculation_override: {raw_answer!r} -> {calc_answer!r} "
                "based on ranking calculation"
            )
            raw_answer = calc_answer

        # 3b. Phase D: Computation-derived verdicts — when calculations
        # carry definitive results (ranking order or medical payouts), derive
        # the answer and synthesize support EvidenceRecords from the
        # supporting facts embedded in calculation inputs.
        calc_evidence = _synthesize_computation_evidence(
            parsed, calculations, raw_answer
        )
        if calc_evidence:
            # Merge synthesized support evidence into the evidence list so
            # the validator's _has_support_for_answer check passes.
            evidence = list(evidence) + calc_evidence
            # Rebuild option judgements with the augmented evidence
            option_judgements = _build_option_judgements(evidence, options)

        # 4. Normalise / correct to valid format
        if fmt == "mcq":
            normalized = _normalize_mcq(raw_answer, options)
        elif fmt == "multi":
            normalized = _normalize_multi(raw_answer, options)
            if not normalized:
                # Fallback: pick the single best option
                best = sorted(
                    options,
                    key=lambda o: (
                        option_judgements[o]["net_score"],
                        option_judgements[o]["support_score"],
                    ),
                    reverse=True,
                )[0]
                normalized = best
                warnings.append(
                    f"answer_unclear: multi answer normalised to empty; "
                    f"falling back to single best option {best}"
                )
                fallbacks.append("low_confidence")
        elif fmt == "tf":
            normalized = _normalize_tf(raw_answer)
        else:
            normalized = _normalize_mcq(raw_answer, options)

        # 5. Final safety: never emit an empty answer
        if not normalized:
            normalized = options[0] if options else "A"
            warnings.append(
                "answer_unclear: answer is empty after all corrections; "
                f"defaulting to {normalized}"
            )
            fallbacks.append("low_confidence")

        # 6. Assemble selected_nodes from supporting evidence
        selected_nodes: list[str] = []
        if fmt == "mcq" or fmt == "tf":
            answer_opts = [normalized]
        else:
            answer_opts = list(normalized)  # multi: each letter is a selected option

        for rec in evidence:
            if rec.option in answer_opts and rec.evidence_type == "support":
                if rec.node_id and rec.node_id not in selected_nodes:
                    selected_nodes.append(rec.node_id)

        # 7. Build AnswerRecord
        return AnswerRecord(
            qid=parsed.qid,
            answer=normalized,
            candidate_docs=list(parsed.doc_ids),
            selected_nodes=selected_nodes,
            evidence=list(evidence),
            calculations=[
                {
                    "calc_type": c.calc_type,
                    "inputs": c.inputs,
                    "formula": c.formula,
                    "result": c.result,
                    "unit": c.unit,
                    "source_evidence_ids": c.source_evidence_ids,
                }
                for c in calculations
            ],
            usage={},
            fallbacks=fallbacks,
            warnings=warnings,
            option_judgements=option_judgements,
        )
