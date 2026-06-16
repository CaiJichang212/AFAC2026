"""Document catalog: build and query a JSONL catalogue of insurance documents.

Task 6: Produces ``data/processed_data/catalog/doc_catalog.jsonl`` with one
JSON object per line (16 lines for the insurance domain, doc_id 1..16).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Classifier helpers
# ---------------------------------------------------------------------------

def _detect_insurer(product_name: str, insurer_aliases: dict[str, str]) -> str:
    """Detect the canonical insurer from a product name.

    Uses keyword signals on the product name.  Falls back to *insurer_aliases*
    lookup when no keyword matches.
    """
    # Ordered keyword signals (first match wins)
    signals: list[tuple[str, str]] = [
        ("太保", "中国太平洋保险"),
        ("太平洋", "中国太平洋保险"),
        ("平安", "平安"),
        ("国寿", "中国人寿"),
        ("中国人寿", "中国人寿"),
        ("众安", "众安"),
    ]
    for keyword, label in signals:
        if keyword in product_name:
            # Use insurer_aliases to canonicalize if the label is itself an alias
            if label in insurer_aliases:
                # Try to find a shorter canonical form from aliases
                for alias, canonical in insurer_aliases.items():
                    if canonical == insurer_aliases.get(label):
                        continue
                return label
            return label
    return "其他"


def _classify_insurance_type(product_name: str) -> str:
    """Classify a product name into one of the standard insurance types."""
    # Ordered checks — more specific patterns checked first
    checks: list[tuple[str, str]] = [
        ("养老年金", "养老年金保险"),
        ("专属商业养老保险", "养老年金保险"),
        ("养老保险", "养老年金保险"),
        ("终身寿险", "终身寿险"),
        ("重大疾病保险", "重大疾病保险"),
        ("重大疾病", "重大疾病保险"),
        ("医疗保险", "医疗保险"),
        ("意外伤害保险", "意外伤害保险"),
        ("意外伤害", "意外伤害保险"),
        ("特种车", "车险"),
        ("家庭财产保险", "家庭财产保险"),
        ("家庭财产", "家庭财产保险"),
        ("食品安全责任保险", "责任保险"),
        ("食品安全责任险", "责任保险"),
        ("责任保险", "责任保险"),
    ]
    for keyword, itype in checks:
        if keyword in product_name:
            return itype
    return "(其他)"


# ---------------------------------------------------------------------------
# Heading extraction
# ---------------------------------------------------------------------------

def _extract_top_titles(markdown_path: Path, top_n: int = 8) -> list[str]:
    """Extract up to *top_n* deduplicated heading strings from a markdown file.

    Only lines that start with ``#`` (one or more) are considered.  Leading
    hash characters and surrounding whitespace are stripped.
    """
    if not markdown_path.exists():
        return []

    titles: list[str] = []
    seen: set[str] = set()

    try:
        text = markdown_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("#"):
            continue
        # Strip leading #'s and whitespace
        title = line.lstrip("#").strip()
        if not title:
            continue
        if title not in seen:
            seen.add(title)
            titles.append(title)
            if len(titles) >= top_n:
                break

    return titles


# ---------------------------------------------------------------------------
# Catalog row builder
# ---------------------------------------------------------------------------

def build_catalog_row(
    doc_id: str,
    profile: Any,  # DomainProfile
    markdown_dir: Path,
    index_quality: dict[str, str],
) -> dict[str, Any]:
    """Build a single catalog row for *doc_id*.

    Parameters:
        doc_id: String document identifier (``"1"``..``"16"``).
        profile: A ``DomainProfile`` with ``.product_aliases``,
            ``.insurer_aliases``, and ``.doc_product_map``.
        markdown_dir: Directory containing ``{doc_id}.md`` files.
        index_quality: Mapping ``doc_id`` (str) -> ``index_source`` string
            read from the index quality JSONL.
    """
    # Canonical product name
    product_name: str
    doc_id_int = int(doc_id)
    if doc_id_int in profile.doc_product_map:
        product_name = profile.doc_product_map[doc_id_int]
    else:
        # Fallback: try to read first heading from markdown
        titles = _extract_top_titles(markdown_dir / f"{doc_id}.md", top_n=1)
        product_name = titles[0] if titles else f"doc_{doc_id}"

    # Aliases: invert product_aliases for this product
    aliases: list[str] = []
    for short, canonical in profile.product_aliases.items():
        if canonical == product_name:
            aliases.append(short)
    # Deduplicate while preserving order; ensure canonical is included
    seen: set[str] = set()
    deduped: list[str] = []
    for a in aliases:
        if a not in seen:
            seen.add(a)
            deduped.append(a)
    if product_name not in seen:
        deduped.insert(0, product_name)
        seen.add(product_name)
    aliases = deduped

    # Insurer
    insurer = _detect_insurer(product_name, profile.insurer_aliases)

    # Insurance type
    insurance_type = _classify_insurance_type(product_name)

    # Source PDF path
    source_pdf = f"data/public_dataset_upload/raw/insurance/{doc_id}.pdf"

    # Top titles from markdown
    top_titles = _extract_top_titles(markdown_dir / f"{doc_id}.md")

    # Primary index route
    primary_index_route = index_quality.get(doc_id, "unknown")

    return {
        "doc_id": doc_id,
        "product_name": product_name,
        "aliases": aliases,
        "insurer": insurer,
        "insurance_type": insurance_type,
        "source_pdf": source_pdf,
        "top_titles": top_titles,
        "primary_index_route": primary_index_route,
    }


def build_catalog(
    profile: Any,
    markdown_dir: Path,
    index_quality_path: Path,
) -> list[dict[str, Any]]:
    """Build the full catalog for all documents in the profile.

    Returns a list of catalog rows sorted by doc_id numerically.
    """
    # Load index quality
    index_quality: dict[str, str] = {}
    if index_quality_path.exists():
        for line in index_quality_path.read_text(encoding="utf-8").strip().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            index_quality[str(row["doc_id"])] = row.get("index_source", "unknown")

    rows: list[dict[str, Any]] = []
    for doc_id_int in sorted(profile.doc_product_map.keys()):
        doc_id = str(doc_id_int)
        row = build_catalog_row(
            doc_id=doc_id,
            profile=profile,
            markdown_dir=markdown_dir,
            index_quality=index_quality,
        )
        rows.append(row)

    return rows


def write_catalog(rows: list[dict[str, Any]], path: Path) -> None:
    """Write catalog rows to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# DocCatalog
# ---------------------------------------------------------------------------

class DocCatalog:
    """A simple, dependency-free document catalogue.

    Load from a JSONL file and query by doc_id.
    """

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows: dict[str, dict[str, Any]] = {}
        for row in rows:
            self._rows[row["doc_id"]] = row

    @classmethod
    def load(cls, catalog_path: Path) -> "DocCatalog":
        """Load a DocCatalog from a JSONL file."""
        rows: list[dict[str, Any]] = []
        if catalog_path.exists():
            for line in catalog_path.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        return cls(rows)

    def get(self, doc_id: str) -> dict[str, Any]:
        """Return the catalog row for *doc_id*."""
        return self._rows[doc_id]

    def contains(self, doc_id: str) -> bool:
        """Check whether *doc_id* is in the catalog."""
        return doc_id in self._rows

    def validate_coverage(self, required_doc_ids: list[str]) -> set[str]:
        """Return the set of *required_doc_ids* that are missing from the catalog."""
        return {d for d in required_doc_ids if d not in self._rows}
