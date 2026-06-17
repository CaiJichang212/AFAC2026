from __future__ import annotations

import re

from agent.schemas import CalculationRecord


def normalize_amount(raw: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)(万元|万|元)", raw)
    if not match:
        raise ValueError(f"Unsupported amount: {raw}")
    value = float(match.group(1))
    unit = match.group(2)
    if unit in {"万元", "万"}:
        return value * 10000
    return value


class CalculationEngine:
    def compute(self, parsed, evidence) -> list[CalculationRecord]:
        return []

    def medical_reimbursement(
        self, *, total: float, reimbursed: float, deductible: float
    ) -> CalculationRecord:
        result = max(total - reimbursed - deductible, 0)
        return CalculationRecord(
            name="medical_reimbursement",
            inputs={"total": total, "reimbursed": reimbursed, "deductible": deductible},
            formula="(total - reimbursed - deductible)",
            result=result,
            unit="元",
            source_evidence_ids=[],
        )
