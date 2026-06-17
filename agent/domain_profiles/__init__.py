from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DomainProfile:
    name: str
    keywords: tuple[str, ...]
    product_aliases: dict[str, tuple[str, ...]]
    liability_terms: tuple[str, ...]
    calculation_patterns: tuple[str, ...]
    quality_thresholds: dict[str, int | float]

    def resolve_product_alias(self, alias: str) -> str | None:
        alias_norm = alias.strip().lower()
        for canonical, aliases in self.product_aliases.items():
            if alias_norm == canonical.lower():
                return canonical
            if any(alias_norm == candidate.lower() for candidate in aliases):
                return canonical
        return None

    def alias_to_canonical(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for canonical, aliases in self.product_aliases.items():
            mapping[canonical.lower()] = canonical
            for alias in aliases:
                mapping[alias.lower()] = canonical
        return mapping


def get_domain_profile(domain: str) -> DomainProfile:
    if domain == "insurance":
        from agent.domain_profiles.insurance import INSURANCE_PROFILE

        return INSURANCE_PROFILE
    raise ValueError(f"Unsupported domain profile: {domain}")


__all__ = ["DomainProfile", "get_domain_profile"]
