from agent.domain_profiles import get_domain_profile


def test_insurance_profile_maps_product_aliases() -> None:
    profile = get_domain_profile("insurance")

    assert profile.resolve_product_alias("智盈金生") == "平安智盈金生专属商业养老保险"
    assert profile.resolve_product_alias("国寿增益宝") == "国寿增益宝终身寿险（万能型）（2025版）"
    assert profile.resolve_product_alias("e生保") == "平安e生保住院医疗保险（A款）"
    assert profile.resolve_product_alias("太保团体百万医疗") == "太保团体百万医疗保险（2022版）"
    assert profile.resolve_product_alias("特种车险") in {
        "中国平安特种车商业保险示范条款（2020版）",
        "众安特种车商业保险示范条款（2020版）",
    }


def test_insurance_profile_exposes_keywords_and_thresholds() -> None:
    profile = get_domain_profile("insurance")

    assert "保险责任" in profile.keywords
    assert "责任免除" in profile.liability_terms
    assert "medical_reimbursement" in profile.calculation_patterns
    assert profile.quality_thresholds["min_keyword_title_hits"] >= 1
