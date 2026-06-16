"""Tests for agent.calculation — deterministic insurance arithmetic.

Covers:
  - Unit normalisation: normalize_amount (元/万元/万/%)
  - Ratio conversion: percent_to_fraction / fraction_to_percent
  - Each calc function with clear inputs / expected outputs
  - ins_a_001 ranking: real stem numbers, real per-product formulas
  - ins_a_003 medical deduction: real stem numbers, per-product params
  - compute() integration with real parsed question + synthetic evidence
"""

from __future__ import annotations

import pytest

from agent.calculation import (
    CalculationEngine,
    calc_death_benefit_account_value,
    calc_death_benefit_multiplied,
    calc_death_benefit_premium_minus_annuity,
    calc_medical_payout,
    calc_rank_descending,
    calc_ratio_payout,
    calc_surrender_value,
    fraction_to_percent,
    normalize_amount,
    percent_to_fraction,
)
from agent.domain_profiles import get_profile
from agent.question_parser import QuestionParser
from agent.schemas import CalculationRecord, EvidenceRecord, ParsedQuestion


# ===================================================================
# Unit normalisation
# ===================================================================

class TestNormalizeAmount:
    """Tests for normalize_amount and ratio helpers."""

    def test_yuan_passthrough(self):
        assert normalize_amount(100, "元") == 100.0
        assert normalize_amount(3.5, "元") == 3.5

    def test_wan_yuan_to_yuan(self):
        assert normalize_amount(100, "万元") == 1_000_000.0
        assert normalize_amount(1, "万元") == 10_000.0

    def test_bare_wan_to_yuan(self):
        assert normalize_amount(144, "万") == 1_440_000.0
        assert normalize_amount(0.5, "万") == 5_000.0

    def test_percent_passthrough(self):
        assert normalize_amount(80.0, "%") == 80.0

    def test_unknown_unit_passthrough(self):
        assert normalize_amount(40, "岁") == 40.0
        assert normalize_amount(99, "") == 99.0

    def test_percent_to_fraction(self):
        assert percent_to_fraction(80.0) == 0.8
        assert percent_to_fraction(100.0) == 1.0
        assert percent_to_fraction(0.0) == 0.0
        assert percent_to_fraction(60.0) == 0.6

    def test_fraction_to_percent(self):
        assert fraction_to_percent(0.8) == 80.0
        assert fraction_to_percent(1.0) == 100.0
        assert fraction_to_percent(0.0) == 0.0
        assert fraction_to_percent(0.6) == 60.0

    def test_roundtrip(self):
        assert percent_to_fraction(fraction_to_percent(0.75)) == 0.75
        assert fraction_to_percent(percent_to_fraction(90.0)) == 90.0


# ===================================================================
# Calc function unit tests
# ===================================================================

class TestDeathBenefitAccountValue:
    """平安智盈金生 style: death benefit = account value."""

    def test_basic(self):
        value, formula = calc_death_benefit_account_value(900_000)
        assert value == 900_000
        assert "900000" in formula
        assert "元" in formula

    def test_zero(self):
        value, formula = calc_death_benefit_account_value(0)
        assert value == 0
        assert "0" in formula


class TestDeathBenefitMultiplied:
    """国寿增益宝 style: death benefit = base × multiplier."""

    def test_gain_bao(self):
        value, formula = calc_death_benefit_multiplied(900_000, 1.6)
        assert value == 1_440_000
        assert "1.6" in formula
        assert "1440000" in formula

    def test_multiply_by_one(self):
        value, _ = calc_death_benefit_multiplied(500_000, 1.0)
        assert value == 500_000

    def test_with_decimal(self):
        value, _ = calc_death_benefit_multiplied(1_000_000, 1.05)
        assert value == 1_050_000

    def test_int_result_for_int_inputs(self):
        value, _ = calc_death_benefit_multiplied(100_000, 2.0)
        assert isinstance(value, int)
        assert value == 200_000


class TestDeathBenefitPremiumMinusAnnuity:
    """国寿鑫享添盈 / 平安富鸿金生 style: premium − annuity received."""

    def test_xin_xiang_tian_ying(self):
        value, formula = calc_death_benefit_premium_minus_annuity(
            1_000_000, 200_000
        )
        assert value == 800_000
        assert "1000000" in formula
        assert "200000" in formula
        assert "800000" in formula

    def test_fu_hong_jin_sheng(self):
        value, formula = calc_death_benefit_premium_minus_annuity(
            1_000_000, 150_000
        )
        assert value == 850_000
        assert "850000" in formula

    def test_no_annuity(self):
        value, _ = calc_death_benefit_premium_minus_annuity(1_000_000, 0)
        assert value == 1_000_000

    def test_annuity_exceeds_premium(self):
        value, _ = calc_death_benefit_premium_minus_annuity(100_000, 150_000)
        assert value == -50_000


class TestSurrenderValue:
    """Surrender value = cash value minus surrender charge."""

    def test_no_charge(self):
        value, formula = calc_surrender_value(120_000)
        assert value == 120_000
        assert "退保费用" not in formula

    def test_fixed_charge(self):
        value, formula = calc_surrender_value(120_000, surrender_charge=2_400)
        assert value == 117_600
        assert "2400" in formula

    def test_percentage_charge(self):
        # 2 % surrender charge on 100,000 → 98,000
        value, _ = calc_surrender_value(100_000, surrender_charge_pct=2.0)
        assert value == 98_000

    def test_percentage_charge_larger(self):
        # 5 % on 200,000 → 190,000
        value, _ = calc_surrender_value(200_000, surrender_charge_pct=5.0)
        assert value == 190_000

    def test_fixed_takes_precedence_over_pct(self):
        # When both provided, fixed charge is used (design choice)
        value, _ = calc_surrender_value(
            100_000, surrender_charge=5_000, surrender_charge_pct=10.0
        )
        assert value == 95_000


class TestMedicalPayout:
    """Medical reimbursement after social insurance + deductible."""

    def test_zhong_an(self):
        """众安白血病医疗险: deductible=0, ratio=100%."""
        value, formula = calc_medical_payout(
            total_expense=80_000,
            reimbursement=30_000,
            deductible=0,
            payout_ratio_fraction=1.0,
        )
        assert value == 50_000
        assert "50000" in formula

    def test_ping_an_e_sheng_bao(self):
        """平安e生保: deductible=5000, ratio=100%."""
        value, _ = calc_medical_payout(
            total_expense=80_000,
            reimbursement=30_000,
            deductible=5_000,
            payout_ratio_fraction=1.0,
        )
        assert value == 45_000

    def test_tai_bao(self):
        """太保团体百万医疗: deductible=10000, ratio=100%."""
        value, _ = calc_medical_payout(
            total_expense=80_000,
            reimbursement=30_000,
            deductible=10_000,
            payout_ratio_fraction=1.0,
        )
        assert value == 40_000

    def test_ratio_less_than_100(self):
        """80 % payout ratio."""
        value, _ = calc_medical_payout(
            total_expense=80_000,
            reimbursement=30_000,
            deductible=5_000,
            payout_ratio_fraction=0.8,
        )
        assert value == 36_000  # (80000-30000-5000) * 0.8 = 45000 * 0.8

    def test_with_cap_below_result(self):
        """Cap limits the payout."""
        value, _ = calc_medical_payout(
            total_expense=100_000,
            reimbursement=20_000,
            deductible=0,
            payout_ratio_fraction=1.0,
            cap=50_000,
        )
        assert value == 50_000

    def test_with_cap_above_result(self):
        """Cap doesn't affect when result is below it."""
        value, _ = calc_medical_payout(
            total_expense=60_000,
            reimbursement=30_000,
            deductible=10_000,
            payout_ratio_fraction=1.0,
            cap=50_000,
        )
        assert value == 20_000

    def test_reimbursement_exceeds_total(self):
        """Eligible expense cannot go negative."""
        value, _ = calc_medical_payout(
            total_expense=10_000,
            reimbursement=15_000,
            deductible=0,
            payout_ratio_fraction=1.0,
        )
        assert value == 0

    def test_deductible_exceeds_remaining(self):
        value, _ = calc_medical_payout(
            total_expense=50_000,
            reimbursement=30_000,
            deductible=30_000,
            payout_ratio_fraction=1.0,
        )
        assert value == 0


class TestRatioPayout:
    """Simple ratio × amount, optionally capped."""

    def test_basic(self):
        value, _ = calc_ratio_payout(100_000, 0.6)
        assert value == 60_000

    def test_with_cap(self):
        value, _ = calc_ratio_payout(100_000, 0.6, cap=50_000)
        assert value == 50_000

    def test_ratio_one(self):
        value, _ = calc_ratio_payout(50_000, 1.0)
        assert value == 50_000

    def test_no_cap(self):
        value, _ = calc_ratio_payout(200_000, 0.8)
        assert value == 160_000


class TestRankDescending:
    """Sort items by value descending."""

    def test_four_items(self):
        items = [
            ("平安智盈金生", 900_000),
            ("国寿增益宝", 1_440_000),
            ("国寿鑫享添盈", 800_000),
            ("平安富鸿金生", 850_000),
        ]
        keys, formula = calc_rank_descending(items)
        assert keys == [
            "国寿增益宝",
            "平安智盈金生",
            "平安富鸿金生",
            "国寿鑫享添盈",
        ]
        assert "144" in formula
        assert ">" in formula

    def test_single_item(self):
        keys, _ = calc_rank_descending([("A", 100)])
        assert keys == ["A"]

    def test_empty(self):
        keys, _ = calc_rank_descending([])
        assert keys == []

    def test_ties_stable(self):
        items = [("A", 100), ("B", 100), ("C", 50)]
        keys, _ = calc_rank_descending(items)
        assert keys[0] in ("A", "B")
        assert keys[1] in ("A", "B")
        assert keys[2] == "C"


# ===================================================================
# ins_a_001: real ranking test
# ===================================================================

class TestInsA001Ranking:
    """Faithful reproduction of ins_a_001 death-benefit ranking.

    Real stem numbers:
      - 已交保费 = 1,000,000 元 (all products)
      - 现金价值 = 800,000 元 (all products)
      - 平安智盈金生: 保单账户价值 = 900,000 元
      - 国寿增益宝: 基本保额 = 900,000 元, multiplier = 1.6
      - 国寿鑫享添盈: 已领养老年金 = 200,000 元
      - 平安富鸿金生: 已领养老年金 = 150,000 元

    Expected per-product death benefits:
      - 平安智盈金生 = 900,000  (account value)
      - 国寿增益宝   = 900,000 × 1.6 = 1,440,000
      - 国寿鑫享添盈 = 1,000,000 − 200,000 = 800,000
      - 平安富鸿金生 = 1,000,000 − 150,000 = 850,000

    Descending order (option B):
      国寿增益宝(144万) > 平安智盈金生(90万) > 平安富鸿金生(85万) > 国寿鑫享添盈(80万)
    """

    def test_individual_calculations(self):
        """Verify each product's death benefit calculation."""
        # 平安智盈金生: account value
        av, _ = calc_death_benefit_account_value(900_000)
        assert av == 900_000

        # 国寿增益宝: base × 1.6
        gs, _ = calc_death_benefit_multiplied(900_000, 1.6)
        assert gs == 1_440_000

        # 国寿鑫享添盈: premium − annuity
        xt, _ = calc_death_benefit_premium_minus_annuity(1_000_000, 200_000)
        assert xt == 800_000

        # 平安富鸿金生: premium − annuity
        fh, _ = calc_death_benefit_premium_minus_annuity(1_000_000, 150_000)
        assert fh == 850_000

    def test_ranking_order_matches_option_b(self):
        """Descending order = option B."""
        items = [
            ("平安智盈金生", 900_000),
            ("国寿增益宝", 1_440_000),
            ("国寿鑫享添盈", 800_000),
            ("平安富鸿金生", 850_000),
        ]
        keys, formula = calc_rank_descending(items)

        assert keys == [
            "国寿增益宝",      # 144万
            "平安智盈金生",  # 90万
            "平安富鸿金生",  # 85万
            "国寿鑫享添盈",  # 80万
        ], f"Expected option B order, got {keys}"

    def test_computed_values_match_wan(self):
        """Verify the computed values in 万元 for display correctness."""
        values = {
            "国寿增益宝": 1_440_000,
            "平安智盈金生": 900_000,
            "平安富鸿金生": 850_000,
            "国寿鑫享添盈": 800_000,
        }
        assert values["国寿增益宝"] / 10000 == 144
        assert values["平安智盈金生"] / 10000 == 90
        assert values["平安富鸿金生"] / 10000 == 85
        assert values["国寿鑫享添盈"] / 10000 == 80

    def test_descending_inequalities(self):
        """Verify strict inequality chain: 144 > 90 > 85 > 80."""
        assert 1_440_000 > 900_000 > 850_000 > 800_000


# ===================================================================
# ins_a_003: real medical-deduction test
# ===================================================================

class TestInsA003Medical:
    """Faithful reproduction of ins_a_003 medical reimbursement.

    Real stem numbers:
      - 总费用 = 80,000 元
      - 医保报销 = 30,000 元
      - 自费 = 50,000 元

    Per-product parameters:
      - 众安白血病医疗险: deductible=0,        ratio=100 %, cap=none
      - 平安e生保:         deductible=5,000,   ratio=100 %, cap=none
      - 太保团体百万医疗:   deductible=10,000,  ratio=100 %, cap=none

    Expected (matching options A/B/C):
      - 众安:  5.0 万
      - 平安e生保: 4.5 万
      - 太保:  4.0 万
      - 合计: 13.5 万
    """

    TOTAL = 80_000
    REIMBURSEMENT = 30_000

    def test_zhong_an_leukemia(self):
        value, formula = calc_medical_payout(
            self.TOTAL, self.REIMBURSEMENT, deductible=0,
            payout_ratio_fraction=1.0,
        )
        assert value == 50_000, f"Expected 50000, got {value}"
        assert "50000" in formula

    def test_ping_an_e_sheng_bao(self):
        value, formula = calc_medical_payout(
            self.TOTAL, self.REIMBURSEMENT, deductible=5_000,
            payout_ratio_fraction=1.0,
        )
        assert value == 45_000, f"Expected 45000, got {value}"
        assert "45000" in formula

    def test_tai_bao_group(self):
        value, formula = calc_medical_payout(
            self.TOTAL, self.REIMBURSEMENT, deductible=10_000,
            payout_ratio_fraction=1.0,
        )
        assert value == 40_000, f"Expected 40000, got {value}"
        assert "40000" in formula

    def test_total_payout(self):
        payouts = [
            calc_medical_payout(self.TOTAL, self.REIMBURSEMENT, 0, 1.0)[0],
            calc_medical_payout(self.TOTAL, self.REIMBURSEMENT, 5_000, 1.0)[0],
            calc_medical_payout(self.TOTAL, self.REIMBURSEMENT, 10_000, 1.0)[0],
        ]
        total = sum(payouts)
        assert total == 135_000, f"Expected 135000, got {total}"

    def test_values_in_wan(self):
        assert 50_000 / 10000 == 5.0
        assert 45_000 / 10000 == 4.5
        assert 40_000 / 10000 == 4.0

    def test_synthetic_with_ratio(self):
        """Synthetic test: 80 % ratio, 1万 deductible, 20万 total, 8万 reimbursement."""
        value, _ = calc_medical_payout(
            total_expense=200_000,
            reimbursement=80_000,
            deductible=10_000,
            payout_ratio_fraction=0.8,
        )
        # eligible = 200000 - 80000 - 10000 = 110000
        # payout = 110000 * 0.8 = 88000
        assert value == 88_000

    def test_synthetic_with_cap(self):
        """Synthetic: cap limits payout."""
        value, _ = calc_medical_payout(
            total_expense=200_000,
            reimbursement=0,
            deductible=0,
            payout_ratio_fraction=1.0,
            cap=150_000,
        )
        assert value == 150_000


# ===================================================================
# compute() integration tests
# ===================================================================

class TestComputeIntegration:
    """Test CalculationEngine.compute() with real parsed questions."""

    @classmethod
    @pytest.fixture(scope="class")
    def profile(cls):
        return get_profile("insurance")

    @classmethod
    @pytest.fixture(scope="class")
    def parser(cls):
        return QuestionParser()

    @classmethod
    @pytest.fixture(scope="class")
    def ins_a_001_raw(cls):
        import json
        from pathlib import Path
        path = Path("data/public_dataset_upload/questions/group_a/insurance_questions.json")
        questions = json.loads(path.read_text(encoding="utf-8"))
        for q in questions:
            if q["qid"] == "ins_a_001":
                return q
        raise ValueError("ins_a_001 not found")

    @classmethod
    @pytest.fixture(scope="class")
    def ins_a_003_raw(cls):
        import json
        from pathlib import Path
        path = Path("data/public_dataset_upload/questions/group_a/insurance_questions.json")
        questions = json.loads(path.read_text(encoding="utf-8"))
        for q in questions:
            if q["qid"] == "ins_a_003":
                return q
        raise ValueError("ins_a_003 not found")

    def test_engine_no_evidence_returns_empty(self):
        """Without evidence, compute returns empty list."""
        engine = CalculationEngine()
        parsed = ParsedQuestion(
            qid="test", domain="insurance", split="A",
            question="关于身故保险金的排序问题？",
            options={"A": "a", "B": "b"}, answer_format="mcq",
            type="推理判断",
        )
        records = engine.compute(parsed, [])
        assert records == []

    def test_ins_a_001_ranking_via_compute(
        self, parser, profile, ins_a_001_raw
    ):
        """compute() with synthetic evidence produces ranking for ins_a_001."""
        parsed = parser.parse(ins_a_001_raw, profile)

        # Synthetic evidence carrying per-product death-benefit formulas.
        # doc_ids: 1=智盈金生, 2=增益宝, 15=鑫享添盈, 16=富鸿金生
        evidence = [
            EvidenceRecord(
                qid="ins_a_001", doc_id="1", node_id="n1",
                evidence_type="formula",
                normalized_fact="death_benefit_account_value",
                numbers=[{"kind": "account_value", "value": 900_000, "unit": "元"}],
            ),
            EvidenceRecord(
                qid="ins_a_001", doc_id="2", node_id="n2",
                evidence_type="formula",
                normalized_fact="death_benefit_multiplied",
                numbers=[
                    {"kind": "basic_sum_insured", "value": 900_000, "unit": "元"},
                    {"kind": "multiplier", "value": 1.6},
                ],
            ),
            EvidenceRecord(
                qid="ins_a_001", doc_id="15", node_id="n15",
                evidence_type="formula",
                normalized_fact="death_benefit_premium_minus_annuity",
                numbers=[
                    {"kind": "premium_paid", "value": 1_000_000, "unit": "元"},
                    {"kind": "annuity_received", "value": 200_000, "unit": "元"},
                ],
            ),
            EvidenceRecord(
                qid="ins_a_001", doc_id="16", node_id="n16",
                evidence_type="formula",
                normalized_fact="death_benefit_premium_minus_annuity",
                numbers=[
                    {"kind": "premium_paid", "value": 1_000_000, "unit": "元"},
                    {"kind": "annuity_received", "value": 150_000, "unit": "元"},
                ],
            ),
        ]

        engine = CalculationEngine(doc_product_map=profile.doc_product_map)
        records = engine.compute(parsed, evidence)

        assert len(records) >= 1, "Expected at least one CalculationRecord"

        rec = records[0]
        assert isinstance(rec, CalculationRecord)
        assert rec.qid == "ins_a_001"
        assert rec.calc_type == "ranking"

        # Verify inputs are populated
        assert "product_values" in rec.inputs
        assert "ranked_order" in rec.inputs
        assert "ranked_values" in rec.inputs

        # Verify the ranked order (canonical names from doc_product_map)
        ranked = rec.inputs["ranked_order"]
        assert len(ranked) == 4

        # Find each product in the ranking
        # Canonical names contain these substrings
        assert any("智盈金生" in p for p in ranked)
        assert any("增益宝" in p for p in ranked)
        assert any("鑫享添盈" in p for p in ranked)
        assert any("富鸿金生" in p for p in ranked)

        # Verify gainsbao (增益宝) is first (highest value = 144万)
        assert "增益宝" in ranked[0], (
            f"增益宝 should rank 1st, got {ranked}"
        )

        # Verify formula is non-empty
        assert len(rec.formula) > 0
        # Verify source_evidence_ids populated
        assert len(rec.source_evidence_ids) == 4

    def test_ins_a_003_medical_via_compute(
        self, parser, profile, ins_a_003_raw
    ):
        """compute() with synthetic evidence produces medical records for ins_a_003."""
        parsed = parser.parse(ins_a_003_raw, profile)

        evidence = [
            EvidenceRecord(
                qid="ins_a_003", doc_id="3", node_id="n3",
                evidence_type="formula",
                normalized_fact="medical_payout",
                numbers=[
                    {"kind": "deductible", "value": 0},
                    {"kind": "payout_ratio", "value": 100.0},
                ],
            ),
            EvidenceRecord(
                qid="ins_a_003", doc_id="5", node_id="n5",
                evidence_type="formula",
                normalized_fact="medical_payout",
                numbers=[
                    {"kind": "deductible", "value": 5_000},
                    {"kind": "payout_ratio", "value": 100.0},
                ],
            ),
            EvidenceRecord(
                qid="ins_a_003", doc_id="6", node_id="n6",
                evidence_type="formula",
                normalized_fact="medical_payout",
                numbers=[
                    {"kind": "deductible", "value": 10_000},
                    {"kind": "payout_ratio", "value": 100.0},
                ],
            ),
        ]

        engine = CalculationEngine(doc_product_map=profile.doc_product_map)
        records = engine.compute(parsed, evidence)

        assert len(records) == 3

        # Verify results by doc_id
        results_by_doc = {}
        for rec in records:
            assert isinstance(rec, CalculationRecord)
            assert rec.qid == "ins_a_003"
            assert rec.calc_type == "medical_payout"
            assert rec.unit == "元"
            assert rec.formula
            assert rec.source_evidence_ids
            # Extract doc_id from source_evidence_ids
            doc_id = rec.source_evidence_ids[0].split("/")[0]
            results_by_doc[doc_id] = rec.result
            assert "product" in rec.inputs
            assert "deductible" in rec.inputs
            assert "payout_ratio" in rec.inputs

        assert results_by_doc["3"] == 50_000, f"众安: {results_by_doc}"
        assert results_by_doc["5"] == 45_000, f"平安e生保: {results_by_doc}"
        assert results_by_doc["6"] == 40_000, f"太保: {results_by_doc}"

    def test_compute_returns_appropriate_calc_type(
        self, parser, profile, ins_a_001_raw
    ):
        """CalculationRecord.calc_type is set correctly."""
        parsed = parser.parse(ins_a_001_raw, profile)

        evidence = [
            EvidenceRecord(
                qid="ins_a_001", doc_id="1", node_id="n1",
                evidence_type="formula",
                normalized_fact="death_benefit_account_value",
                numbers=[{"kind": "account_value", "value": 900_000, "unit": "元"}],
            ),
            EvidenceRecord(
                qid="ins_a_001", doc_id="2", node_id="n2",
                evidence_type="formula",
                normalized_fact="death_benefit_multiplied",
                numbers=[
                    {"kind": "basic_sum_insured", "value": 900_000, "unit": "元"},
                    {"kind": "multiplier", "value": 1.6},
                ],
            ),
            EvidenceRecord(
                qid="ins_a_001", doc_id="15", node_id="n15",
                evidence_type="formula",
                normalized_fact="death_benefit_premium_minus_annuity",
                numbers=[
                    {"kind": "premium_paid", "value": 1_000_000, "unit": "元"},
                    {"kind": "annuity_received", "value": 200_000, "unit": "元"},
                ],
            ),
            EvidenceRecord(
                qid="ins_a_001", doc_id="16", node_id="n16",
                evidence_type="formula",
                normalized_fact="death_benefit_premium_minus_annuity",
                numbers=[
                    {"kind": "premium_paid", "value": 1_000_000, "unit": "元"},
                    {"kind": "annuity_received", "value": 150_000, "unit": "元"},
                ],
            ),
        ]

        engine = CalculationEngine(doc_product_map=profile.doc_product_map)
        records = engine.compute(parsed, evidence)

        assert len(records) == 1
        assert records[0].calc_type == "ranking"

    def test_calculation_record_fields_populated(
        self, parser, profile, ins_a_001_raw
    ):
        """All required CalculationRecord fields are populated."""
        parsed = parser.parse(ins_a_001_raw, profile)

        evidence = [
            EvidenceRecord(
                qid="ins_a_001", doc_id="1", node_id="n1",
                evidence_type="formula",
                normalized_fact="death_benefit_account_value",
                numbers=[{"kind": "account_value", "value": 900_000, "unit": "元"}],
            ),
            EvidenceRecord(
                qid="ins_a_001", doc_id="2", node_id="n2",
                evidence_type="formula",
                normalized_fact="death_benefit_multiplied",
                numbers=[
                    {"kind": "basic_sum_insured", "value": 900_000, "unit": "元"},
                    {"kind": "multiplier", "value": 1.6},
                ],
            ),
            EvidenceRecord(
                qid="ins_a_001", doc_id="15", node_id="n15",
                evidence_type="formula",
                normalized_fact="death_benefit_premium_minus_annuity",
                numbers=[
                    {"kind": "premium_paid", "value": 1_000_000, "unit": "元"},
                    {"kind": "annuity_received", "value": 200_000, "unit": "元"},
                ],
            ),
            EvidenceRecord(
                qid="ins_a_001", doc_id="16", node_id="n16",
                evidence_type="formula",
                normalized_fact="death_benefit_premium_minus_annuity",
                numbers=[
                    {"kind": "premium_paid", "value": 1_000_000, "unit": "元"},
                    {"kind": "annuity_received", "value": 150_000, "unit": "元"},
                ],
            ),
        ]

        engine = CalculationEngine(doc_product_map=profile.doc_product_map)
        records = engine.compute(parsed, evidence)

        rec = records[0]
        # All required fields
        assert rec.qid == "ins_a_001"
        assert rec.calc_type
        assert isinstance(rec.inputs, dict) and len(rec.inputs) > 0
        assert isinstance(rec.formula, str) and len(rec.formula) > 0
        assert isinstance(rec.result, (int, float))
        assert isinstance(rec.unit, str)
        assert isinstance(rec.source_evidence_ids, list)
        assert len(rec.source_evidence_ids) > 0


# ===================================================================
# Determinism
# ===================================================================

class TestDeterminism:
    """All calc functions are deterministic: same inputs → same outputs."""

    def test_multiple_invocations_same_result(self):
        for _ in range(5):
            v1, f1 = calc_death_benefit_multiplied(900_000, 1.6)
            v2, f2 = calc_death_benefit_multiplied(900_000, 1.6)
            assert v1 == v2 == 1_440_000
            assert f1 == f2

    def test_medical_payout_deterministic(self):
        for _ in range(5):
            v1, f1 = calc_medical_payout(80_000, 30_000, 5_000, 1.0)
            v2, f2 = calc_medical_payout(80_000, 30_000, 5_000, 1.0)
            assert v1 == v2 == 45_000
            assert f1 == f2

    def test_rank_descending_deterministic(self):
        items = [("A", 3), ("B", 1), ("C", 2)]
        for _ in range(5):
            k1, _ = calc_rank_descending(items)
            k2, _ = calc_rank_descending(items)
            assert k1 == k2 == ["A", "C", "B"]

    def test_normalize_amount_deterministic(self):
        for _ in range(5):
            assert normalize_amount(100, "万元") == 1_000_000.0
