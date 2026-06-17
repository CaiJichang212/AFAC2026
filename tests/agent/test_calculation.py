from agent.calculation import CalculationEngine, normalize_amount


def test_normalize_amount_supports_yuan_and_wan() -> None:
    assert normalize_amount("1.5万元") == 15000
    assert normalize_amount("5000元") == 5000


def test_medical_reimbursement_deducts_reimbursed_and_deductible() -> None:
    engine = CalculationEngine()

    result = engine.medical_reimbursement(total=80000, reimbursed=30000, deductible=5000)

    assert result.result == 45000
    assert result.formula == "(total - reimbursed - deductible)"
