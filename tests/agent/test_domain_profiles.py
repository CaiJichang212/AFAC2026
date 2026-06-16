"""Tests for domain profiles (Task 2)."""

from __future__ import annotations

import pytest

from agent.domain_profiles import DomainProfile, INSURANCE_PROFILE, get_profile, register_profile


# ---------------------------------------------------------------------------
# DomainProfile dataclass
# ---------------------------------------------------------------------------

class TestDomainProfileDataclass:
    """Verify the DomainProfile dataclass behaves as expected."""

    def test_frozen(self) -> None:
        """DomainProfile instances must be immutable."""
        profile = DomainProfile(name="test")
        with pytest.raises(Exception):
            profile.name = "changed"  # type: ignore[misc]

    def test_defaults(self) -> None:
        """All collection fields default to empty."""
        profile = DomainProfile(name="test")
        assert profile.keywords == []
        assert profile.product_aliases == {}
        assert profile.insurer_aliases == {}
        assert profile.liability_terms == []
        assert profile.calculation_patterns == []
        assert profile.quality_thresholds == {}

    def test_full_profile_construction(self) -> None:
        """All fields can be set at construction time."""
        profile = DomainProfile(
            name="demo",
            keywords=["kw1", "kw2"],
            product_aliases={"short": "canonical"},
            insurer_aliases={"abbr": "Full Insurer Name"},
            liability_terms=["death_benefit"],
            calculation_patterns=[{"id": "calc1", "label": "Test"}],
            quality_thresholds={"min_title_count": 5},
        )
        assert profile.name == "demo"
        assert len(profile.keywords) == 2
        assert profile.product_aliases["short"] == "canonical"
        assert profile.quality_thresholds["min_title_count"] == 5


# ---------------------------------------------------------------------------
# Registry: get_profile / register_profile
# ---------------------------------------------------------------------------

class TestRegistry:
    """Verify the domain profile registry."""

    def test_get_insurance_profile(self) -> None:
        profile = get_profile("insurance")
        assert profile is INSURANCE_PROFILE
        assert profile.name == "insurance"

    def test_get_unknown_domain_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown domain"):
            get_profile("nonexistent_domain")

    def test_register_profile(self) -> None:
        new_profile = DomainProfile(name="test_domain", keywords=["a", "b"])
        register_profile(new_profile)
        assert get_profile("test_domain") is new_profile


# ---------------------------------------------------------------------------
# Insurance profile content
# ---------------------------------------------------------------------------

class TestInsuranceProfile:
    """Verify the insurance DomainProfile is populated correctly."""

    def test_non_empty_keywords(self) -> None:
        assert len(INSURANCE_PROFILE.keywords) > 20

    def test_non_empty_liability_terms(self) -> None:
        assert len(INSURANCE_PROFILE.liability_terms) > 5

    def test_non_empty_calculation_patterns(self) -> None:
        assert len(INSURANCE_PROFILE.calculation_patterns) > 3

    def test_non_empty_quality_thresholds(self) -> None:
        thresholds = INSURANCE_PROFILE.quality_thresholds
        assert "min_title_count" in thresholds
        assert "max_empty_title_ratio" in thresholds
        assert "min_page_mapping_coverage" in thresholds

    def test_non_empty_product_aliases(self) -> None:
        """We must have at least the 16 canonical entries plus short aliases."""
        aliases = INSURANCE_PROFILE.product_aliases
        assert len(aliases) >= 32  # at least canonical + short form per doc

    def test_non_empty_insurer_aliases(self) -> None:
        assert len(INSURANCE_PROFILE.insurer_aliases) >= 4


# ---------------------------------------------------------------------------
# Product alias resolution (question-stem short names)
# ---------------------------------------------------------------------------

class TestProductAliasResolution:
    """Verify that short names from A-split question stems resolve correctly."""

    CANONICAL_ZHOUYING = "平安智盈金生专属商业养老保险"
    CANONICAL_ZENGYIBAO = "国寿增益宝终身寿险（万能型）（2025版）"
    CANONICAL_XINXIANG = "国寿鑫享添盈养老年金保险（互联网专属）"
    CANONICAL_FUHONG = "平安富鸿金生（悦享版）养老年金保险（分红型）"
    CANONICAL_ESB = "平安e生保住院7.0医疗保险A款"
    CANONICAL_BAIXUEBING = "众安个人急性白血病复发医疗保险（互联网2026版A款）"
    CANONICAL_TUANBAI = "太保团体百万医疗保险（2022版）"
    CANONICAL_ANYOUFU = "平安安佑福重大疾病保险"
    CANONICAL_JIACAI = "平安产险家庭财产保险（家庭版）（2025版）"
    CANONICAL_SHIJI = "众安食品安全责任保险（互联网2026版）"

    @pytest.mark.parametrize(
        "short_name,expected_canonical",
        [
            # Core short names from question stems
            ("智盈金生", CANONICAL_ZHOUYING),
            ("平安智盈金生", CANONICAL_ZHOUYING),
            ("增益宝", CANONICAL_ZENGYIBAO),
            ("国寿增益宝", CANONICAL_ZENGYIBAO),
            ("鑫享添盈", CANONICAL_XINXIANG),
            ("国寿鑫享添盈", CANONICAL_XINXIANG),
            ("富鸿金生", CANONICAL_FUHONG),
            ("平安富鸿金生", CANONICAL_FUHONG),
            ("e生保", CANONICAL_ESB),
            ("平安e生保", CANONICAL_ESB),
            ("众安白血病医疗险", CANONICAL_BAIXUEBING),
            ("白血病医疗险", CANONICAL_BAIXUEBING),
            ("太保团体百万医疗", CANONICAL_TUANBAI),
            ("团体百万医疗", CANONICAL_TUANBAI),
            ("平安安佑福重疾险", CANONICAL_ANYOUFU),
            ("平安安佑福", CANONICAL_ANYOUFU),
            ("安佑福", CANONICAL_ANYOUFU),
            ("平安家财险", CANONICAL_JIACAI),
            ("众安食责险", CANONICAL_SHIJI),
            ("众安食品安全责任险", CANONICAL_SHIJI),
            # Canonical names resolve to themselves
            (CANONICAL_ZHOUYING, CANONICAL_ZHOUYING),
            (CANONICAL_ZENGYIBAO, CANONICAL_ZENGYIBAO),
            (CANONICAL_XINXIANG, CANONICAL_XINXIANG),
            (CANONICAL_FUHONG, CANONICAL_FUHONG),
        ],
    )
    def test_short_name_resolves_to_canonical(
        self, short_name: str, expected_canonical: str
    ) -> None:
        aliases = INSURANCE_PROFILE.product_aliases
        assert short_name in aliases, (
            f"Expected {short_name!r} to be a key in product_aliases"
        )
        assert aliases[short_name] == expected_canonical, (
            f"{short_name!r} resolved to {aliases[short_name]!r}, "
            f"expected {expected_canonical!r}"
        )

    def test_all_canonical_forms_are_self_mapping(self) -> None:
        """Every canonical name should map to itself."""
        aliases = INSURANCE_PROFILE.product_aliases
        canonical_names = {
            v for v in aliases.values()
        }
        for cn in canonical_names:
            assert aliases.get(cn) == cn, (
                f"Canonical name {cn!r} does not self-map"
            )

    def test_unique_canonical_count(self) -> None:
        """There should be exactly 16 distinct canonical product names."""
        aliases = INSURANCE_PROFILE.product_aliases
        canonicals = set(aliases.values())
        assert len(canonicals) == 16, (
            f"Expected 16 distinct canonical names, got {len(canonicals)}: {sorted(canonicals)}"
        )


# ---------------------------------------------------------------------------
# Insurer alias resolution
# ---------------------------------------------------------------------------

class TestInsurerAliases:
    """Verify insurer short names resolve."""

    def test_guoshou_resolves(self) -> None:
        assert INSURANCE_PROFILE.insurer_aliases["国寿"] == "中国人寿保险股份有限公司"

    def test_pingan_resolves(self) -> None:
        assert INSURANCE_PROFILE.insurer_aliases["平安"] == "中国平安保险（集团）股份有限公司"

    def test_taibao_resolves(self) -> None:
        assert INSURANCE_PROFILE.insurer_aliases["太保"] == "中国太平洋保险（集团）股份有限公司"

    def test_zhongan_resolves(self) -> None:
        assert INSURANCE_PROFILE.insurer_aliases["众安"] == "众安在线财产保险股份有限公司"


# ---------------------------------------------------------------------------
# Keywords contain expected insurance vocabulary
# ---------------------------------------------------------------------------

class TestKeywordsCoverage:
    """Verify key insurance vocabulary is present."""

    def test_structural_keywords_present(self) -> None:
        keywords = INSURANCE_PROFILE.keywords
        assert "保险责任" in keywords
        assert "责任免除" in keywords
        assert "释义" in keywords
        assert "现金价值" in keywords

    def test_benefit_keywords_present(self) -> None:
        keywords = INSURANCE_PROFILE.keywords
        assert "身故保险金" in keywords
        assert "养老保险金" in keywords
        assert "满期生存保险金" in keywords

    def test_procedural_keywords_present(self) -> None:
        keywords = INSURANCE_PROFILE.keywords
        assert "犹豫期" in keywords
        assert "宽限期" in keywords
        assert "等待期" in keywords
        assert "免赔额" in keywords
        assert "给付比例" in keywords


# ---------------------------------------------------------------------------
# Calculation pattern structure
# ---------------------------------------------------------------------------

class TestCalculationPatterns:
    """Verify each calculation pattern has required fields."""

    REQUIRED_KEYS = {"id", "label", "description", "typical_formula"}

    def test_all_patterns_have_required_keys(self) -> None:
        for pattern in INSURANCE_PROFILE.calculation_patterns:
            missing = self.REQUIRED_KEYS - set(pattern.keys())
            assert not missing, (
                f"Pattern {pattern.get('id', '?')} missing keys: {missing}"
            )

    def test_pattern_ids_are_unique(self) -> None:
        ids = [p["id"] for p in INSURANCE_PROFILE.calculation_patterns]
        assert len(ids) == len(set(ids)), f"Duplicate pattern ids: {ids}"
