"""Tests for agent.question_parser — rule-based signal extraction.

Covers:
  - All 20 real A-split insurance questions parse without error
  - in_a_001 signal checks: mentioned_products, doc_product_map,
    liability_signals, number_conditions
  - Synthetic edge cases: no numbers, percentages, 万元 normalisation
"""

from __future__ import annotations

import pytest

from agent.domain_profiles import get_profile
from agent.question_parser import QuestionParser, parse_questions

# ---------------------------------------------------------------------------
# Path to the real questions file
# ---------------------------------------------------------------------------

QUESTIONS_PATH = "data/public_dataset_upload/questions/group_a/insurance_questions.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def profile():
    return get_profile("insurance")


@pytest.fixture(scope="module")
def parser():
    return QuestionParser()


@pytest.fixture(scope="module")
def all_parsed(parser, profile):
    return parser.parse_questions(QUESTIONS_PATH, profile)


# ---------------------------------------------------------------------------
# Basic parse checks — all 20 real questions
# ---------------------------------------------------------------------------

def test_all_20_questions_parse(all_parsed):
    """All 20 real questions parse without error."""
    assert len(all_parsed) == 20


def test_all_questions_have_required_fields(all_parsed):
    """Every parsed question has the required basic fields."""
    for q in all_parsed:
        assert q.qid
        assert q.qid.startswith("ins_a_")
        assert q.domain == "insurance"
        assert q.split == "A"
        assert isinstance(q.question, str) and len(q.question) > 0
        assert isinstance(q.options, dict)
        assert len(q.options) > 0
        assert q.answer_format in ("mcq", "multi", "tf")
        assert isinstance(q.type, str) and len(q.type) > 0
        assert isinstance(q.doc_ids, list)
        assert len(q.doc_ids) > 0
        assert all(isinstance(d, str) for d in q.doc_ids)


def test_all_questions_have_signal_fields(all_parsed):
    """Every parsed question has the signal fields (may be empty)."""
    for q in all_parsed:
        assert isinstance(q.mentioned_products, list)
        assert isinstance(q.doc_product_map, dict)
        assert isinstance(q.liability_signals, list)
        assert isinstance(q.number_conditions, list)


# ---------------------------------------------------------------------------
# ins_a_001 detailed signal checks
# ---------------------------------------------------------------------------

def test_ins_a_001_mentioned_products(all_parsed):
    """ins_a_001 mentions the four expected products (canonicalised)."""
    q = all_parsed[0]
    assert q.qid == "ins_a_001"

    # All four products should appear
    products = set(q.mentioned_products)
    assert "平安智盈金生专属商业养老保险" in products
    assert "国寿增益宝终身寿险（万能型）（2025版）" in products
    assert "国寿鑫享添盈养老年金保险（互联网专属）" in products
    assert "平安富鸿金生（悦享版）养老年金保险（分红型）" in products


def test_ins_a_001_doc_product_map(all_parsed):
    """ins_a_001 doc_product_map maps each doc_id to its canonical product name."""
    q = all_parsed[0]
    assert q.qid == "ins_a_001"

    assert q.doc_product_map["1"] == "平安智盈金生专属商业养老保险"
    assert q.doc_product_map["2"] == "国寿增益宝终身寿险（万能型）（2025版）"
    assert q.doc_product_map["15"] == "国寿鑫享添盈养老年金保险（互联网专属）"
    assert q.doc_product_map["16"] == "平安富鸿金生（悦享版）养老年金保险（分红型）"
    assert len(q.doc_product_map) == 4


def test_ins_a_001_liability_signals(all_parsed):
    """ins_a_001 has 身故保险金 as a liability signal."""
    q = all_parsed[0]
    assert q.qid == "ins_a_001"
    assert "身故保险金" in q.liability_signals


def test_ins_a_001_number_conditions_premium_and_cash_value(all_parsed):
    """ins_a_001 captures 已交保费=100万 and 现金价值=80万 with correct 元 values."""
    q = all_parsed[0]
    assert q.qid == "ins_a_001"

    # Find the 已交保费 condition
    premium_conds = [c for c in q.number_conditions if c["subject"] == "已交保费"]
    assert len(premium_conds) >= 1
    premium_cond = premium_conds[0]
    assert premium_cond["value"] == 1_000_000  # 100万元 -> 1,000,000 元
    assert premium_cond["unit"] == "元"
    assert premium_cond["kind"] == "premium_paid"

    # Find the 现金价值 condition
    cv_conds = [c for c in q.number_conditions if c["subject"] == "现金价值"]
    assert len(cv_conds) >= 1
    cv_cond = cv_conds[0]
    assert cv_cond["value"] == 800_000  # 80万元 -> 800,000 元
    assert cv_cond["unit"] == "元"
    assert cv_cond["kind"] == "cash_value"


def test_ins_a_001_number_conditions_source_tagging(all_parsed):
    """ins_a_001 stem conditions have source=='stem'; option numbers have source starting with 'option'."""
    q = all_parsed[0]
    assert q.qid == "ins_a_001"

    # --- Stem conditions: 已交保费 and 现金价值 must be tagged 'stem' ---
    stem_conds = [c for c in q.number_conditions if c.get("source") == "stem"]
    # There should be at least the 已交保费 and 现金价值 conditions from the stem
    stem_subjects = {c["subject"] for c in stem_conds}
    assert "已交保费" in stem_subjects
    assert "现金价值" in stem_subjects

    # Verify 已交保费=1,000,000 and 现金价值=800,000 are among stem conditions
    premium_stem = [
        c for c in stem_conds
        if c["subject"] == "已交保费" and c["value"] == 1_000_000
    ]
    assert len(premium_stem) >= 1
    cv_stem = [
        c for c in stem_conds
        if c["subject"] == "现金价值" and c["value"] == 800_000
    ]
    assert len(cv_stem) >= 1

    # --- Option conditions: 90万/144万 values appear in option text ---
    option_conds = [c for c in q.number_conditions if c.get("source", "").startswith("option")]
    option_values = {c["value"] for c in option_conds}
    assert 900_000 in option_values   # 90万
    assert 1_440_000 in option_values  # 144万

    # All option sources should follow pattern "option_<LETTER>"
    for c in option_conds:
        src = c.get("source", "")
        assert src.startswith("option_"), f"Expected source like 'option_A', got {src!r}"

    # --- Overall: at least some stem conditions and some option conditions ---
    assert len(stem_conds) > 0
    assert len(option_conds) > 0


def test_ins_a_007_percent_values_are_ratio_kinds(all_parsed):
    """ins_a_007: 80% values must be ratio-like kinds (NOT cash_value), unit='%'."""
    q = all_parsed[6]
    assert q.qid == "ins_a_007"

    # Monetary kinds that must NOT appear for % values
    _MONETARY_KINDS = {
        "cash_value", "account_value", "basic_sum_insured", "premium_paid",
        "annuity_received", "amount", "deductible", "total_expense",
        "sum_insured", "premium", "account_return", "medical_reimbursement",
        "out_of_pocket", "payout_limit", "medical_expense",
        "hospitalization_expense", "payout", "surrender_value",
    }
    # Ratio-compatible kinds allowed for %
    _ALLOWED_PCT_KINDS = {"ratio", "surrender_charge", "payout_ratio"}

    pct_conds = [c for c in q.number_conditions if c["unit"] == "%"]
    assert len(pct_conds) >= 2  # at least the two 80% from options A and B

    for nc in pct_conds:
        assert nc["unit"] == "%"
        assert nc["kind"] not in _MONETARY_KINDS, \
            f"% unit got monetary kind {nc['kind']!r} — should be ratio-like"
        assert nc["kind"] in _ALLOWED_PCT_KINDS, \
            f"% unit got kind {nc['kind']!r}, expected one of {_ALLOWED_PCT_KINDS}"
        assert nc["value"] == 80
        # Source should be 'option' (not stem, since stem has no %)
        assert nc.get("source", "").startswith("option"), \
            f"Expected option source for ins_a_007 80%, got {nc.get('source')!r}"


def test_stem_number_conditions_property(parser, profile):
    """ParsedQuestion.stem_number_conditions filters to source=='stem' only."""
    raw = {
        "qid": "synth_011",
        "domain": "insurance",
        "split": "A",
        "question": "已交保费100万元，现金价值80万元。",
        "options": {"A": "赔付90万元", "B": "赔付144万元"},
        "answer_format": "mcq",
        "type": "计算题",
        "doc_ids": ["1"],
    }
    pq = parser.parse(raw, profile)

    # Full number_conditions includes both stem and option
    all_sources = {c.get("source") for c in pq.number_conditions}
    assert "stem" in all_sources
    assert any(s.startswith("option") for s in all_sources)

    # stem_number_conditions property filters to stem only
    stem_only = pq.stem_number_conditions
    assert len(stem_only) >= 2  # 已交保费 + 现金价值
    for nc in stem_only:
        assert nc.get("source") == "stem"

    # Verify the stem values are correct
    stem_values = {(c["subject"], c["value"]) for c in stem_only}
    assert ("已交保费", 1_000_000) in stem_values
    assert ("现金价值", 800_000) in stem_values


def test_ins_a_001_number_conditions_has_extra_amounts(all_parsed):
    """ins_a_001 also captures other amounts from stem + options."""
    q = all_parsed[0]
    # We expect at least several number conditions (amounts from stem + options)
    assert len(q.number_conditions) >= 4  # at least the stem amounts
    # We should capture some amounts like 90万, 144万 etc.
    values = [c["value"] for c in q.number_conditions]
    assert 900_000 in values  # 90万 for 保单账户价值
    # 144万 should appear from options
    assert 1_440_000 in values


# ---------------------------------------------------------------------------
# Synthetic / edge-case tests
# ---------------------------------------------------------------------------

def test_synthetic_no_numbers(parser, profile):
    """A question with no numeric amounts yields empty number_conditions."""
    raw = {
        "qid": "synth_001",
        "domain": "insurance",
        "split": "A",
        "question": "这是否为保险合同？",
        "options": {"A": "是", "B": "否"},
        "answer_format": "tf",
        "type": "事实查询",
        "doc_ids": ["1"],
    }
    pq = parser.parse(raw, profile)
    assert pq.number_conditions == []


def test_synthetic_percentage(parser, profile):
    """A percentage is captured correctly."""
    raw = {
        "qid": "synth_002",
        "domain": "insurance",
        "split": "A",
        "question": "给付比例为80%，测试用例。",
        "options": {"A": "对", "B": "错"},
        "answer_format": "tf",
        "type": "事实查询",
        "doc_ids": ["1"],
    }
    pq = parser.parse(raw, profile)
    ratio_conds = [c for c in pq.number_conditions if c["unit"] == "%"]
    assert len(ratio_conds) >= 1
    rc = ratio_conds[0]
    assert rc["value"] == 80
    assert rc["unit"] == "%"


def test_synthetic_wan_yuan_normalisation(parser, profile):
    """万元 amounts are normalised to 元 (x10000)."""
    raw = {
        "qid": "synth_003",
        "domain": "insurance",
        "split": "A",
        "question": "已交保费100万元，现金价值80万元。",
        "options": {"A": "对", "B": "错"},
        "answer_format": "tf",
        "type": "事实查询",
        "doc_ids": ["1"],
    }
    pq = parser.parse(raw, profile)
    # 100万元 -> 1,000,000
    premium = [c for c in pq.number_conditions if c["subject"] == "已交保费"]
    assert len(premium) >= 1
    assert premium[0]["value"] == 1_000_000
    assert premium[0]["unit"] == "元"

    # 80万元 -> 800,000
    cv = [c for c in pq.number_conditions if c["subject"] == "现金价值"]
    assert len(cv) >= 1
    assert cv[0]["value"] == 800_000
    assert cv[0]["unit"] == "元"


def test_synthetic_bare_wan(parser, profile):
    """Bare 万 (without 元) is also normalised to 元."""
    raw = {
        "qid": "synth_004",
        "domain": "insurance",
        "split": "A",
        "question": "基本保额90万。",
        "options": {"A": "对", "B": "错"},
        "answer_format": "tf",
        "type": "事实查询",
        "doc_ids": ["1"],
    }
    pq = parser.parse(raw, profile)
    bsi = [c for c in pq.number_conditions if c["kind"] == "basic_sum_insured"]
    assert len(bsi) >= 1
    assert bsi[0]["value"] == 900_000


def test_synthetic_deductible(parser, profile):
    """免赔额 is captured with correct kind."""
    raw = {
        "qid": "synth_005",
        "domain": "insurance",
        "split": "A",
        "question": "免赔额5000元。",
        "options": {"A": "对", "B": "错"},
        "answer_format": "tf",
        "type": "事实查询",
        "doc_ids": ["1"],
    }
    pq = parser.parse(raw, profile)
    ded = [c for c in pq.number_conditions if c["kind"] == "deductible"]
    assert len(ded) >= 1
    assert ded[0]["value"] == 5000
    assert ded[0]["unit"] == "元"


def test_synthetic_account_value(parser, profile):
    """保单账户价值 is captured with account_value kind."""
    raw = {
        "qid": "synth_006",
        "domain": "insurance",
        "split": "A",
        "question": "保单账户价值90万元。",
        "options": {"A": "对", "B": "错"},
        "answer_format": "tf",
        "type": "事实查询",
        "doc_ids": ["1"],
    }
    pq = parser.parse(raw, profile)
    av = [c for c in pq.number_conditions if c["kind"] == "account_value"]
    assert len(av) >= 1
    assert av[0]["value"] == 900_000


def test_synthetic_annuity_received(parser, profile):
    """已领养老年金 is captured."""
    raw = {
        "qid": "synth_007",
        "domain": "insurance",
        "split": "A",
        "question": "已领养老年金20万元。",
        "options": {"A": "对", "B": "错"},
        "answer_format": "tf",
        "type": "事实查询",
        "doc_ids": ["1"],
    }
    pq = parser.parse(raw, profile)
    ar = [c for c in pq.number_conditions if c["kind"] == "annuity_received"]
    assert len(ar) >= 1
    assert ar[0]["value"] == 200_000


def test_synthetic_age(parser, profile):
    """Age (岁) is captured."""
    raw = {
        "qid": "synth_008",
        "domain": "insurance",
        "split": "A",
        "question": "被保险人40岁。",
        "options": {"A": "对", "B": "错"},
        "answer_format": "tf",
        "type": "事实查询",
        "doc_ids": ["1"],
    }
    pq = parser.parse(raw, profile)
    ages = [c for c in pq.number_conditions if c["kind"] == "age"]
    assert len(ages) >= 1
    assert ages[0]["value"] == 40
    assert ages[0]["unit"] == "岁"


def test_synthetic_mixed_products(parser, profile):
    """Mentioned products are canonicalised and ordered by first occurrence."""
    raw = {
        "qid": "synth_009",
        "domain": "insurance",
        "split": "A",
        "question": "智盈金生和增益宝相比，安佑福重疾险的保障范围如何？",
        "options": {"A": "对", "B": "错"},
        "answer_format": "tf",
        "type": "事实查询",
        "doc_ids": ["1", "2", "4"],
    }
    pq = parser.parse(raw, profile)
    # 智盈金生 -> 平安智盈金生专属商业养老保险
    # 增益宝 -> 国寿增益宝终身寿险（万能型）（2025版）
    # 安佑福重疾险 -> 平安安佑福重大疾病保险
    assert pq.mentioned_products[0] == "平安智盈金生专属商业养老保险"
    assert pq.mentioned_products[1] == "国寿增益宝终身寿险（万能型）（2025版）"
    assert pq.mentioned_products[2] == "平安安佑福重大疾病保险"


def test_convenience_parse_questions_function(profile):
    """The convenience function parse_questions works."""
    result = parse_questions(QUESTIONS_PATH, profile)
    assert len(result) == 20
    assert all(hasattr(q, "qid") for q in result)


def test_number_conditions_have_required_keys(parser, profile):
    """Every number_condition record has kind, value, unit, subject, snippet, source."""
    raw = {
        "qid": "synth_010",
        "domain": "insurance",
        "split": "A",
        "question": "已交保费100万元，现金价值80万元，给付比例80%。",
        "options": {"A": "对", "B": "错"},
        "answer_format": "tf",
        "type": "事实查询",
        "doc_ids": ["1"],
    }
    pq = parser.parse(raw, profile)
    for nc in pq.number_conditions:
        assert "kind" in nc
        assert "value" in nc
        assert "unit" in nc
        assert "subject" in nc
        assert "snippet" in nc
        assert "source" in nc


def test_ins_a_002_number_conditions(all_parsed):
    """ins_a_002 (退保计算题) captures relevant amounts."""
    q = all_parsed[1]
    assert q.qid == "ins_a_002"
    # Should have 累计所交保费=10万, 账户价值=12万, 现金价值=9万 etc.
    kinds = {c["kind"] for c in q.number_conditions}
    # At minimum we expect some kinds
    assert len(q.number_conditions) > 0


def test_ins_a_003_number_conditions(all_parsed):
    """ins_a_003 (医疗费用计算) captures deductible and medical amounts."""
    q = all_parsed[2]
    assert q.qid == "ins_a_003"
    kinds = {c["kind"] for c in q.number_conditions}
    # Should include deductible amounts
    assert "deductible" in kinds or "amount" in kinds


def test_split_a_is_A_in_all_questions(all_parsed):
    """Every real question has split='A'."""
    for q in all_parsed:
        assert q.split == "A"


def test_parser_is_stateless(parser, profile):
    """parse_questions can be called twice with same result."""
    first = parser.parse_questions(QUESTIONS_PATH, profile)
    second = parser.parse_questions(QUESTIONS_PATH, profile)
    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert a.qid == b.qid
        assert a.mentioned_products == b.mentioned_products
        assert a.number_conditions == b.number_conditions
