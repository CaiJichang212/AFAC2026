"""Tests for agent/catalog.py (Task 6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.catalog import (
    DocCatalog,
    _classify_insurance_type,
    _detect_insurer,
    _extract_top_titles,
    build_catalog,
    build_catalog_row,
    write_catalog,
)
from agent.domain_profiles import DomainProfile, INSURANCE_PROFILE


# ---------------------------------------------------------------------------
# Helpers for synthetic data
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = [
    "doc_id",
    "product_name",
    "aliases",
    "insurer",
    "insurance_type",
    "source_pdf",
    "top_titles",
    "primary_index_route",
]


def _make_synthetic_profile() -> DomainProfile:
    """Build a minimal synthetic profile mirroring the insurance structure."""
    product_aliases: dict[str, str] = {
        # Doc 1
        "平安智盈金生专属商业养老保险": "平安智盈金生专属商业养老保险",
        "平安智盈金生": "平安智盈金生专属商业养老保险",
        "智盈金生": "平安智盈金生专属商业养老保险",
        # Doc 2
        "国寿增益宝终身寿险（万能型）（2025版）": "国寿增益宝终身寿险（万能型）（2025版）",
        "国寿增益宝": "国寿增益宝终身寿险（万能型）（2025版）",
        "增益宝": "国寿增益宝终身寿险（万能型）（2025版）",
        # Doc 3
        "众安个人急性白血病复发医疗保险（互联网2026版A款）": "众安个人急性白血病复发医疗保险（互联网2026版A款）",
        "众安白血病医疗险": "众安个人急性白血病复发医疗保险（互联网2026版A款）",
        # Doc 6
        "太保团体百万医疗保险（2022版）": "太保团体百万医疗保险（2022版）",
        "太保团体百万医疗": "太保团体百万医疗保险（2022版）",
    }
    insurer_aliases: dict[str, str] = {
        "平安": "中国平安保险（集团）股份有限公司",
        "国寿": "中国人寿保险股份有限公司",
        "太保": "中国太平洋保险（集团）股份有限公司",
        "众安": "众安在线财产保险股份有限公司",
    }
    doc_product_map: dict[int, str] = {
        1: "平安智盈金生专属商业养老保险",
        2: "国寿增益宝终身寿险（万能型）（2025版）",
        3: "众安个人急性白血病复发医疗保险（互联网2026版A款）",
        4: "平安安佑福重大疾病保险",
        5: "平安e生保住院7.0医疗保险A款",
        6: "太保团体百万医疗保险（2022版）",
        7: "平安产险预防接种意外伤害保险（E款）（互联网版）",
        8: "众安营运交通工具团体意外伤害保险（互联网版2025A款）",
        9: "平安特种车商业保险示范条款（2020版）",
        10: "众安特种车商业保险示范条款（2020版）",
        11: "平安产险家庭财产保险（家庭版）（2025版）",
        12: "众安家庭财产综合保险（互联网2023版）",
        13: "众安食品安全责任保险（互联网2026版）",
        14: "平安产险食品安全责任保险（2021版）",
        15: "国寿鑫享添盈养老年金保险（互联网专属）",
        16: "平安富鸿金生（悦享版）养老年金保险（分红型）",
    }
    return DomainProfile(
        name="insurance",
        product_aliases=product_aliases,
        insurer_aliases=insurer_aliases,
        doc_product_map=doc_product_map,
    )


def _write_synthetic_markdown(md_dir: Path, doc_id: str, titles: list[str]) -> None:
    """Write a markdown file with headings for testing top_titles extraction."""
    content = "\n".join(f"### {t}" for t in titles)
    (md_dir / f"{doc_id}.md").write_text(content, encoding="utf-8")


def _write_synthetic_index_quality(path: Path, entries: list[dict]) -> None:
    """Write a synthetic index quality JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Insurer detection tests
# ---------------------------------------------------------------------------

class TestInsurerDetection:
    """Verify insurer detection from product names."""

    INSURER_ALIASES = INSURANCE_PROFILE.insurer_aliases

    @pytest.mark.parametrize(
        "product_name,expected",
        [
            ("平安智盈金生专属商业养老保险", "平安"),
            ("国寿增益宝终身寿险（万能型）（2025版）", "中国人寿"),
            ("众安个人急性白血病复发医疗保险（互联网2026版A款）", "众安"),
            ("太保团体百万医疗保险（2022版）", "中国太平洋保险"),
            ("平安e生保住院7.0医疗保险A款", "平安"),
            ("国寿鑫享添盈养老年金保险（互联网专属）", "中国人寿"),
            ("众安食品安全责任保险（互联网2026版）", "众安"),
        ],
    )
    def test_detect_insurer_from_product_name(
        self, product_name: str, expected: str
    ) -> None:
        assert _detect_insurer(product_name, self.INSURER_ALIASES) == expected

    def test_unknown_insurer_fallback(self) -> None:
        """A product name without any known insurer signals returns '其他'."""
        assert _detect_insurer("测试产品名称", self.INSURER_ALIASES) == "其他"


# ---------------------------------------------------------------------------
# Insurance-type classification tests
# ---------------------------------------------------------------------------

class TestInsuranceTypeClassification:
    """Verify insurance type classification from product names."""

    @pytest.mark.parametrize(
        "product_name,expected",
        [
            # 养老年金保险
            ("平安智盈金生专属商业养老保险", "养老年金保险"),
            ("国寿鑫享添盈养老年金保险（互联网专属）", "养老年金保险"),
            ("平安富鸿金生（悦享版）养老年金保险（分红型）", "养老年金保险"),
            # 终身寿险
            ("国寿增益宝终身寿险（万能型）（2025版）", "终身寿险"),
            # 医疗保险
            ("众安个人急性白血病复发医疗保险（互联网2026版A款）", "医疗保险"),
            ("平安e生保住院7.0医疗保险A款", "医疗保险"),
            ("太保团体百万医疗保险（2022版）", "医疗保险"),
            # 重大疾病保险
            ("平安安佑福重大疾病保险", "重大疾病保险"),
            # 意外伤害保险
            ("平安产险预防接种意外伤害保险（E款）（互联网版）", "意外伤害保险"),
            ("众安营运交通工具团体意外伤害保险（互联网版2025A款）", "意外伤害保险"),
            # 车险
            ("平安特种车商业保险示范条款（2020版）", "车险"),
            ("众安特种车商业保险示范条款（2020版）", "车险"),
            # 家庭财产保险
            ("平安产险家庭财产保险（家庭版）（2025版）", "家庭财产保险"),
            ("众安家庭财产综合保险（互联网2023版）", "家庭财产保险"),
            # 责任保险
            ("众安食品安全责任保险（互联网2026版）", "责任保险"),
            ("平安产险食品安全责任保险（2021版）", "责任保险"),
        ],
    )
    def test_classify_insurance_type(
        self, product_name: str, expected: str
    ) -> None:
        assert _classify_insurance_type(product_name) == expected

    def test_unknown_type_fallback(self) -> None:
        """An unrecognized product name returns '(其他)'."""
        assert _classify_insurance_type("某未知产品名") == "(其他)"


# ---------------------------------------------------------------------------
# Top-titles extraction tests
# ---------------------------------------------------------------------------

class TestTopTitlesExtraction:
    """Verify heading extraction from markdown files."""

    def test_extract_headings(self, tmp_path: Path) -> None:
        md_file = tmp_path / "1.md"
        content = "\n".join(
            [
                "### 保险责任",
                "Some body text",
                "## 责任免除",
                "### 犹豫期",
                "# 总则",
                "### 保险责任",  # duplicate
            ]
        )
        md_file.write_text(content, encoding="utf-8")
        titles = _extract_top_titles(md_file, top_n=8)
        # Expected: deduplicated, order-preserved, stripped of #'s
        assert titles == [
            "保险责任",
            "责任免除",
            "犹豫期",
            "总则",
        ]

    def test_caps_at_top_n(self, tmp_path: Path) -> None:
        md_file = tmp_path / "1.md"
        titles_in = [f"Title {i}" for i in range(20)]
        content = "\n".join(f"### {t}" for t in titles_in)
        md_file.write_text(content, encoding="utf-8")
        titles = _extract_top_titles(md_file, top_n=8)
        assert len(titles) == 8
        assert titles == titles_in[:8]

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        titles = _extract_top_titles(tmp_path / "nonexistent.md")
        assert titles == []

    def test_no_headings_returns_empty(self, tmp_path: Path) -> None:
        md_file = tmp_path / "1.md"
        md_file.write_text("Just body text.\nNo headings here.\n", encoding="utf-8")
        titles = _extract_top_titles(md_file)
        assert titles == []


# ---------------------------------------------------------------------------
# Synthetic catalog build tests (core logic, artifact-free)
# ---------------------------------------------------------------------------

class TestBuildCatalogRowSynthetic:
    """Test build_catalog_row with synthetic inputs in tmp_path."""

    def test_row_has_all_required_fields(self, tmp_path: Path) -> None:
        profile = _make_synthetic_profile()
        md_dir = tmp_path / "markdown"
        md_dir.mkdir()
        _write_synthetic_markdown(
            md_dir, "1", ["阅读指引", "保险责任", "责任免除"]
        )

        row = build_catalog_row(
            doc_id="1",
            profile=profile,
            markdown_dir=md_dir,
            index_quality={"1": "markdown"},
        )

        for field in REQUIRED_FIELDS:
            assert field in row, f"Missing field {field}"
        assert row["doc_id"] == "1"
        assert len(row["aliases"]) >= 1
        assert row["insurance_type"] != ""
        assert row["insurer"] != ""
        assert row["primary_index_route"] == "markdown"
        assert len(row["top_titles"]) == 3

    def test_aliases_include_canonical_and_short_forms(self, tmp_path: Path) -> None:
        profile = _make_synthetic_profile()
        md_dir = tmp_path / "markdown"
        md_dir.mkdir()
        _write_synthetic_markdown(md_dir, "1", ["阅"])

        row = build_catalog_row(
            doc_id="1",
            profile=profile,
            markdown_dir=md_dir,
            index_quality={"1": "markdown"},
        )

        assert "平安智盈金生专属商业养老保险" in row["aliases"]
        assert "平安智盈金生" in row["aliases"]
        assert "智盈金生" in row["aliases"]


class TestBuildCatalogSynthetic:
    """Test build_catalog with synthetic inputs."""

    def test_build_catalog_16_rows(self, tmp_path: Path) -> None:
        profile = _make_synthetic_profile()
        md_dir = tmp_path / "markdown"
        md_dir.mkdir()

        for doc_id_int in profile.doc_product_map:
            _write_synthetic_markdown(
                md_dir, str(doc_id_int), [f"Title for doc {doc_id_int}"]
            )

        iq_path = tmp_path / "index_quality.jsonl"
        entries = [
            {"doc_id": i, "index_source": "markdown", "status": "ok"}
            for i in profile.doc_product_map
        ]
        _write_synthetic_index_quality(iq_path, entries)

        rows = build_catalog(profile, md_dir, iq_path)

        assert len(rows) == 16
        doc_ids = [r["doc_id"] for r in rows]
        assert doc_ids == [str(i) for i in range(1, 17)]
        assert len(set(doc_ids)) == 16

    def test_all_rows_have_required_fields_nonempty(self, tmp_path: Path) -> None:
        profile = _make_synthetic_profile()
        md_dir = tmp_path / "markdown"
        md_dir.mkdir()

        for doc_id_int in profile.doc_product_map:
            _write_synthetic_markdown(
                md_dir, str(doc_id_int), [f"Title for doc {doc_id_int}"]
            )

        iq_path = tmp_path / "index_quality.jsonl"
        entries = [
            {"doc_id": i, "index_source": "markdown", "status": "ok"}
            for i in profile.doc_product_map
        ]
        _write_synthetic_index_quality(iq_path, entries)

        rows = build_catalog(profile, md_dir, iq_path)

        for row in rows:
            for field in REQUIRED_FIELDS:
                value = row[field]
                assert value is not None, (
                    f"Doc {row['doc_id']}: field {field} is None"
                )
                if isinstance(value, list):
                    assert len(value) >= 1, (
                        f"Doc {row['doc_id']}: field {field} is empty list"
                    )
                else:
                    assert value != "", (
                        f"Doc {row['doc_id']}: field {field} is empty string"
                    )

    def test_primary_index_route_from_quality_log(self, tmp_path: Path) -> None:
        profile = _make_synthetic_profile()
        md_dir = tmp_path / "markdown"
        md_dir.mkdir()

        for doc_id_int in profile.doc_product_map:
            _write_synthetic_markdown(
                md_dir, str(doc_id_int), [f"Title for doc {doc_id_int}"]
            )

        iq_path = tmp_path / "index_quality.jsonl"
        entries = [
            {"doc_id": 1, "index_source": "markdown", "status": "ok"},
            {"doc_id": 2, "index_source": "pdf_fallback", "status": "ok"},
            {"doc_id": 3, "index_source": "page_keyword", "status": "ok"},
        ] + [
            {"doc_id": i, "index_source": "markdown", "status": "ok"}
            for i in range(4, 17)
        ]
        _write_synthetic_index_quality(iq_path, entries)

        rows = build_catalog(profile, md_dir, iq_path)
        by_id = {r["doc_id"]: r for r in rows}

        assert by_id["1"]["primary_index_route"] == "markdown"
        assert by_id["2"]["primary_index_route"] == "pdf_fallback"
        assert by_id["3"]["primary_index_route"] == "page_keyword"


# ---------------------------------------------------------------------------
# DocCatalog class tests
# ---------------------------------------------------------------------------

class TestDocCatalog:
    """Verify the DocCatalog query class."""

    def _sample_rows(self) -> list[dict]:
        return [
            {
                "doc_id": "1",
                "product_name": "Test Product A",
                "aliases": ["Test Product A", "prod a"],
                "insurer": "Test Insurer",
                "insurance_type": "医疗保险",
                "source_pdf": "data/public_dataset_upload/raw/insurance/1.pdf",
                "top_titles": ["Title 1"],
                "primary_index_route": "markdown",
            },
            {
                "doc_id": "2",
                "product_name": "Test Product B",
                "aliases": ["Test Product B"],
                "insurer": "Another Insurer",
                "insurance_type": "养老年金保险",
                "source_pdf": "data/public_dataset_upload/raw/insurance/2.pdf",
                "top_titles": ["Title 2"],
                "primary_index_route": "pdf_fallback",
            },
        ]

    def test_load_and_get(self, tmp_path: Path) -> None:
        path = tmp_path / "catalog.jsonl"
        write_catalog(self._sample_rows(), path)

        catalog = DocCatalog.load(path)
        assert catalog.contains("1")
        assert catalog.contains("2")
        assert not catalog.contains("3")

        row = catalog.get("1")
        assert row["product_name"] == "Test Product A"
        assert row["insurance_type"] == "医疗保险"
        assert row["primary_index_route"] == "markdown"

    def test_get_missing_raises_keyerror(self, tmp_path: Path) -> None:
        path = tmp_path / "catalog.jsonl"
        write_catalog(self._sample_rows(), path)
        catalog = DocCatalog.load(path)
        with pytest.raises(KeyError):
            catalog.get("999")

    def test_validate_coverage_all_covered(self, tmp_path: Path) -> None:
        path = tmp_path / "catalog.jsonl"
        write_catalog(self._sample_rows(), path)
        catalog = DocCatalog.load(path)
        missing = catalog.validate_coverage(["1", "2"])
        assert missing == set()

    def test_validate_coverage_missing(self, tmp_path: Path) -> None:
        path = tmp_path / "catalog.jsonl"
        write_catalog(self._sample_rows(), path)
        catalog = DocCatalog.load(path)
        missing = catalog.validate_coverage(["1", "2", "15", "16"])
        assert missing == {"15", "16"}

    def test_validate_coverage_from_question_doc_ids(self, tmp_path: Path) -> None:
        """Question-derived doc_id set should be fully covered by synthetic catalog."""
        rows = []
        for i in range(1, 17):
            rows.append(
                {
                    "doc_id": str(i),
                    "product_name": f"Product {i}",
                    "aliases": [f"Product {i}"],
                    "insurer": "Insurer",
                    "insurance_type": "医疗保险",
                    "source_pdf": f"data/public_dataset_upload/raw/insurance/{i}.pdf",
                    "top_titles": [f"Title {i}"],
                    "primary_index_route": "markdown",
                }
            )
        path = tmp_path / "catalog.jsonl"
        write_catalog(rows, path)
        catalog = DocCatalog.load(path)

        # A synthetic question set
        required = ["1", "2", "15", "16"]
        missing = catalog.validate_coverage(required)
        assert missing == set()

    def test_load_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        catalog = DocCatalog.load(path)
        assert catalog.validate_coverage(["1"]) == {"1"}


# ---------------------------------------------------------------------------
# Write and re-read round-trip
# ---------------------------------------------------------------------------

class TestCatalogRoundTrip:
    """Verify JSONL write/read preserves all fields."""

    def test_round_trip(self, tmp_path: Path) -> None:
        rows = [
            {
                "doc_id": "1",
                "product_name": "Test",
                "aliases": ["Test", "T"],
                "insurer": "平安",
                "insurance_type": "医疗保险",
                "source_pdf": "data/public_dataset_upload/raw/insurance/1.pdf",
                "top_titles": ["保险责任", "责任免除", "犹豫期"],
                "primary_index_route": "markdown",
            }
        ]
        path = tmp_path / "catalog.jsonl"
        write_catalog(rows, path)

        catalog = DocCatalog.load(path)
        loaded = catalog.get("1")
        assert loaded == rows[0]


# ---------------------------------------------------------------------------
# Integration test (real markdown + index quality)
# ---------------------------------------------------------------------------

class TestCatalogIntegration:
    """Light integration tests using the real markdown directory and index quality log."""

    def test_build_with_real_data_16_rows(self) -> None:
        """Build catalog from real markdown + index quality log and verify 16 rows."""
        config_path = Path("data/processed_data")
        markdown_dir = config_path / "markdown" / "insurance"
        iq_path = config_path / "quality" / "insurance_index_quality.jsonl"

        if not markdown_dir.exists() or not iq_path.exists():
            pytest.skip("Real data not available")

        profile = INSURANCE_PROFILE
        rows = build_catalog(profile, markdown_dir, iq_path)

        assert len(rows) == 16
        doc_ids = [r["doc_id"] for r in rows]
        expected_ids = [str(i) for i in range(1, 17)]
        assert doc_ids == expected_ids

    def test_product_names_match_profile(self) -> None:
        """Every row's product_name matches the profile's doc_product_map."""
        config_path = Path("data/processed_data")
        markdown_dir = config_path / "markdown" / "insurance"
        iq_path = config_path / "quality" / "insurance_index_quality.jsonl"

        if not markdown_dir.exists() or not iq_path.exists():
            pytest.skip("Real data not available")

        profile = INSURANCE_PROFILE
        rows = build_catalog(profile, markdown_dir, iq_path)

        for row in rows:
            doc_id_int = int(row["doc_id"])
            expected_name = profile.doc_product_map.get(doc_id_int)
            if expected_name is not None:
                assert row["product_name"] == expected_name, (
                    f"Doc {row['doc_id']}: expected {expected_name!r}, "
                    f"got {row['product_name']!r}"
                )

    def test_questions_union_covered(self) -> None:
        """The union of all doc_ids from the 20 questions must be fully covered."""
        config_path = Path("data/processed_data")
        markdown_dir = config_path / "markdown" / "insurance"
        iq_path = config_path / "quality" / "insurance_index_quality.jsonl"
        questions_path = Path(
            "data/public_dataset_upload/questions/group_a/insurance_questions.json"
        )

        if (
            not markdown_dir.exists()
            or not iq_path.exists()
            or not questions_path.exists()
        ):
            pytest.skip("Real data not available")

        profile = INSURANCE_PROFILE
        rows = build_catalog(profile, markdown_dir, iq_path)
        catalog = DocCatalog(rows)

        # Gather all doc_ids from questions
        questions = json.loads(questions_path.read_text(encoding="utf-8"))
        all_q_doc_ids: set[str] = set()
        for q in questions:
            all_q_doc_ids.update(q["doc_ids"])

        missing = catalog.validate_coverage(sorted(all_q_doc_ids))
        assert missing == set(), f"Missing doc_ids: {missing}"
