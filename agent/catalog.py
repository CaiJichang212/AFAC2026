from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fitz

from agent.config import AgentConfig
from agent.domain_profiles import get_domain_profile

DOC_PRODUCT_MAP = {
    "1": "平安智盈金生专属商业养老保险",
    "2": "国寿增益宝终身寿险（万能型）（2025版）",
    "3": "众安个人急性白血病复发医疗保险（互联网2026版A款）",
    "4": "平安安佑福重大疾病保险",
    "5": "平安e生保住院医疗保险（A款）",
    "6": "太保团体百万医疗保险（2022版）",
    "7": "平安产险预防接种意外伤害保险（E款）（互联网版）",
    "8": "众安营运交通工具团体意外伤害保险（互联网版2025A款）",
    "9": "中国平安特种车商业保险示范条款（2020版）",
    "10": "众安特种车商业保险示范条款（2020版）",
    "11": "平安产险家庭财产保险（家庭版）（2025版）",
    "12": "众安家庭财产综合保险（互联网2023版）",
    "13": "众安食品安全责任保险（互联网2026版）",
    "14": "平安产险食品安全责任保险（2021版）",
    "15": "国寿鑫享添盈年金保险",
    "16": "平安富鸿金生（悦享版）养老年金保险（分红型）",
}


def _infer_insurer(product_name: str) -> str:
    if product_name.startswith("平安"):
        return "平安"
    if product_name.startswith("国寿"):
        return "中国人寿"
    if product_name.startswith("众安"):
        return "众安"
    if product_name.startswith("太保"):
        return "太平洋健康"
    return ""


def _infer_type(product_name: str) -> str:
    for keyword in ("养老", "寿险", "医疗", "重大疾病", "意外", "特种车", "家庭财产", "食品安全"):
        if keyword in product_name:
            return keyword
    return "保险"


def build_catalog(config: AgentConfig) -> Path:
    profile = get_domain_profile(config.domain)
    config.catalog_path.parent.mkdir(parents=True, exist_ok=True)
    with config.catalog_path.open("w", encoding="utf-8") as handle:
        for pdf_path in sorted(config.raw_dir.glob("*.pdf"), key=lambda path: int(path.stem)):
            doc_id = pdf_path.stem
            product_name = DOC_PRODUCT_MAP.get(doc_id, doc_id)
            aliases = list(profile.product_aliases.get(product_name, ()))
            top_titles = []
            try:
                with fitz.open(pdf_path) as doc:
                    first_text = doc[0].get_text("text").strip().splitlines()
                    top_titles = [line.strip() for line in first_text if line.strip()][:3]
            except Exception:
                top_titles = []
            record = {
                "doc_id": doc_id,
                "product_name": product_name,
                "aliases": aliases,
                "insurer": _infer_insurer(product_name),
                "insurance_type": _infer_type(product_name),
                "source_pdf": str(pdf_path),
                "top_titles": top_titles,
                "primary_index_route": "markdown",
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return config.catalog_path


def load_catalog(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        records[record["doc_id"]] = record
    return records
