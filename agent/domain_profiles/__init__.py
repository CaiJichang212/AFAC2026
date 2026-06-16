"""Domain profile registry.

Each domain (insurance, financial_contracts, etc.) provides a ``DomainProfile``
that describes its vocabulary, product aliases, liability terms, calculation
patterns, and quality thresholds.  Downstream tasks (question parsing, catalog
building, retrieval) use these profiles to normalise product names and guide
extraction heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.domain_profiles import insurance as _insurance


@dataclass(frozen=True)
class DomainProfile:
    """Static knowledge about a domain used by the QA pipeline.

    All fields are frozen so the profile can be shared safely across concurrent
    stages.  Every domain that the pipeline supports must define one instance.

    Attributes:
        name: Short domain identifier (e.g. ``"insurance"``).
        keywords: Broad vocabulary useful for title-recovery and node matching.
        product_aliases: Mapping from short name / alias to canonical product
            name.  Both directions should typically be present so that lookups
            from question stems and from document titles both resolve.
        insurer_aliases: Mapping from short insurer name to full insurer name.
        liability_terms: Standardised liability category names.
        calculation_patterns: Descriptions of the calculation types the engine
            can handle (death-benefit comparison, surrender value, medical-fee
            deduction, payout-ratio-plus-cap, ranking/sorting, annuity offset).
        quality_thresholds: Tunable thresholds for parse/index quality decisions.
    """

    name: str
    keywords: list[str] = field(default_factory=list)
    product_aliases: dict[str, str] = field(default_factory=dict)
    insurer_aliases: dict[str, str] = field(default_factory=dict)
    liability_terms: list[str] = field(default_factory=list)
    calculation_patterns: list[dict[str, Any]] = field(default_factory=list)
    quality_thresholds: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Build the insurance profile from raw data (avoids circular imports)
# ---------------------------------------------------------------------------

INSURANCE_PROFILE = DomainProfile(
    name="insurance",
    keywords=_insurance.KEYWORDS,
    product_aliases=_insurance.PRODUCT_ALIASES,
    insurer_aliases=_insurance.INSURER_ALIASES,
    liability_terms=_insurance.LIABILITY_TERMS,
    calculation_patterns=_insurance.CALCULATION_PATTERNS,
    quality_thresholds=_insurance.QUALITY_THRESHOLDS,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_PROFILES: dict[str, DomainProfile] = {
    "insurance": INSURANCE_PROFILE,
}


def get_profile(domain: str) -> DomainProfile:
    """Look up a domain profile by name.

    Raises:
        KeyError: If *domain* is not registered.
    """
    if domain not in _PROFILES:
        raise KeyError(
            f"Unknown domain {domain!r}. Registered domains: {list(_PROFILES)}"
        )
    return _PROFILES[domain]


def register_profile(profile: DomainProfile) -> None:
    """Register an additional domain profile at runtime (for extensibility)."""
    _PROFILES[profile.name] = profile


__all__ = ["DomainProfile", "get_profile", "register_profile", "INSURANCE_PROFILE"]
