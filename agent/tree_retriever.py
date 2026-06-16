"""TreeRetriever: candidate-node selection from compact PageIndex trees.

Task 8: Given a parsed question, a document's compact tree, and a domain
profile, apply a rule-based prescreen then optionally an LLM selection step to
pick the most relevant nodes (capped by budget).  Falls back deterministically
when no LLM client is available or the LLM call fails.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent.config import AgentConfig
from agent.domain_profiles import DomainProfile
from agent.index_store import _flatten_tree, _parse_page_range
from agent.llm_client import LLMClient, NonRetryableError
from agent.schemas import CandidateNode, ParsedQuestion, UsageRecord
from agent.token_meter import TokenMeter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Threshold for "small" vs "large" tree
# ---------------------------------------------------------------------------

_SMALL_TREE_NODE_LIMIT: int = 12

# ---------------------------------------------------------------------------
# Amount / ratio token helpers
# ---------------------------------------------------------------------------

# Simple pattern for Chinese number + unit tokens commonly found in questions.
_AMOUNT_RATIO_TOKENS: list[str] = [
    "万元", "万", "元", "%", "岁",
]


def _extract_amount_ratio_tokens(text: str) -> list[str]:
    """Return short snippets from *text* that look like amounts/ratios."""
    tokens: list[str] = []
    import re

    # Match patterns like "100万元", "80%", "40岁", etc.
    for token_str in _AMOUNT_RATIO_TOKENS:
        # Find digit sequences followed by the token
        for m in re.finditer(rf"(\d+(?:\.\d+)?)\s*{re.escape(token_str)}", text):
            tokens.append(m.group(0).strip())
    return tokens


# ---------------------------------------------------------------------------
# LLM prompt helpers
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT: str = (
    "你是一个保险条款检索专家。你的任务是根据用户的问题，从文档节点树中选择最相关的节点。"
    "返回一个 JSON 对象，其中包含一个 ``nodes`` 列表，每个元素有 ``node_id`` 和 ``reason`` 两个字段。"
    "最多选择 {max_nodes} 个节点，按相关性从高到低排列。"
)


def _build_compact_tree_for_llm(
    nodes: list[dict],
) -> list[dict]:
    """Build a LLM-friendly subset of the tree (titles + page_ranges only)."""
    result: list[dict] = []
    for node in nodes:
        entry: dict[str, Any] = {
            "node_id": node.get("node_id", ""),
            "title": node.get("title", ""),
            "page_range": node.get("page_range", ""),
        }
        children = node.get("nodes") or node.get("structure") or []
        if children:
            entry["nodes"] = _build_compact_tree_for_llm(children)
        result.append(entry)
    return result


def _render_compact_tree_for_prompt(tree: list[dict], indent: int = 0) -> str:
    """Render a compact tree as indented text lines for the LLM prompt."""
    lines: list[str] = []
    prefix = "  " * indent
    for node in tree:
        nid = node.get("node_id", "?")
        title = node.get("title", "")
        pr = node.get("page_range", "")
        lines.append(f"{prefix}- [{nid}] {title}  (页码: {pr})")
        children = node.get("nodes") or []
        if children:
            lines.append(_render_compact_tree_for_prompt(children, indent + 1))
    return "\n".join(lines)


def _build_node_selection_prompt(
    question: str,
    options: dict[str, str],
    compact_tree: list[dict],
    max_nodes: int,
) -> str:
    """Build the user prompt for LLM node selection."""
    tree_text = _render_compact_tree_for_prompt(compact_tree)
    options_text = "\n".join(f"  {k}. {v}" for k, v in options.items())
    return (
        f"# 问题\n{question}\n\n"
        f"# 选项\n{options_text}\n\n"
        f"# 文档节点树\n{tree_text}\n\n"
        f"请从上述节点树中选择最多 {max_nodes} 个与问题最相关的节点。"
    )


_LLM_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["node_id", "reason"],
                "additionalProperties": False,
            },
            "maxItems": 5,
        },
    },
    "required": ["nodes"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Page-range helpers
# ---------------------------------------------------------------------------


def _count_pages_in_range(page_range: str) -> int:
    """Return the number of pages covered by a page-range string."""
    if not page_range:
        return 0
    return len(_parse_page_range(page_range))


# ---------------------------------------------------------------------------
# TreeRetriever
# ---------------------------------------------------------------------------


class TreeRetriever:
    """Select candidate nodes from a single document's compact tree.

    Combines deterministic rule-based prescreening with optional LLM
    refinement, then enforces node and page budgets.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        parsed: ParsedQuestion,
        doc_id: str,
        compact_tree: list[dict],
        config: AgentConfig,
        profile: DomainProfile,
        *,
        llm_client: LLMClient | None = None,
        token_meter: TokenMeter | None = None,
    ) -> list[CandidateNode]:
        """Retrieve candidate nodes for *parsed* question from *compact_tree*.

        Args:
            parsed: The parsed question with signals.
            doc_id: The document identifier (string).
            compact_tree: Output of ``IndexStore.get_document_structure(doc_id)``.
            config: Agent configuration (budgets).
            profile: Domain profile (keywords, liability terms, product aliases).
            llm_client: Optional LLM client for refinement. If None or failing,
                the method falls back to rule-based selection.

        Returns:
            Up to ``config.max_nodes_per_doc`` CandidateNode instances, with
            total pages across them capped at ``config.max_pages_per_doc``.
        """
        warnings: list[str] = []

        # 1. Flatten the compact tree
        flat_nodes = _flatten_tree(compact_tree)

        # 2. Rule-based prescreen: score every node
        scored = self._prescreen_nodes(parsed, flat_nodes, profile)
        # scored is list of (node_dict, score, matched_signals)

        # Build a lookup: node_id -> node dict
        node_lookup: dict[str, dict] = {n["node_id"]: n for n in flat_nodes}

        max_nodes = config.max_nodes_per_doc
        max_pages = config.max_pages_per_doc
        small_tree = len(flat_nodes) <= _SMALL_TREE_NODE_LIMIT

        # 3. LLM selection (if available)
        selected_node_ids: list[str] = []
        reasons: dict[str, str] = {}
        llm_used: bool = False

        if llm_client is not None and scored:
            try:
                if small_tree:
                    # Send the whole compact tree
                    llm_input = _build_compact_tree_for_llm(compact_tree)
                else:
                    # Send only the prescreened top-K
                    top_k = self._top_k_by_score(scored, _SMALL_TREE_NODE_LIMIT * 2)
                    llm_input = _build_compact_tree_for_llm(
                        [n for n, _, _ in top_k]
                    )

                prompt = _build_node_selection_prompt(
                    parsed.question, parsed.options, llm_input, max_nodes
                )
                response = llm_client.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": _LLM_SYSTEM_PROMPT.format(max_nodes=max_nodes),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    json_schema=_LLM_JSON_SCHEMA,
                    temperature=0.6,
                    max_tokens=2048,
                )
                # Record usage if a meter is provided
                if token_meter is not None:
                    token_meter.record(
                        UsageRecord(
                            qid=parsed.qid,
                            stage="tree_retrieval",
                            model=response.model or config.inference_model,
                            prompt_tokens=response.prompt_tokens,
                            completion_tokens=response.completion_tokens,
                            total_tokens=response.total_tokens,
                            latency_ms=response.latency_ms,
                            success=True,
                        )
                    )
                parsed_response = json.loads(response.content)
                llm_nodes = parsed_response.get("nodes", [])
                for item in llm_nodes:
                    nid = item.get("node_id", "")
                    if nid and nid in node_lookup:
                        selected_node_ids.append(nid)
                        reasons[nid] = item.get("reason", "")
                llm_used = True
            except Exception as exc:
                logger.warning("LLM node selection failed: %s; falling back to rule prescreen", exc)
                # Fall through to fallback

        # 4. Fallback: use rule-prescreen ranking
        if not llm_used:
            for node, _score, _signals in self._top_k_by_score(scored, max_nodes):
                nid = node["node_id"]
                selected_node_ids.append(nid)
                reasons[nid] = "rule prescreen"

        # 5. Deduplicate and limit to max_nodes
        seen: set[str] = set()
        deduped: list[str] = []
        for nid in selected_node_ids:
            if nid not in seen:
                seen.add(nid)
                deduped.append(nid)
        selected_node_ids = deduped[:max_nodes]

        # 6. Build CandidateNode list with page budget enforcement
        candidates: list[CandidateNode] = []
        total_pages: int = 0

        # Build signal lookup from scored
        signal_lookup: dict[str, list[str]] = {
            n["node_id"]: sigs for n, _, sigs in scored
        }

        for nid in selected_node_ids:
            node = node_lookup.get(nid)
            if node is None:
                continue
            page_range = node.get("page_range", "")
            page_count = _count_pages_in_range(page_range)

            if total_pages + page_count > max_pages:
                warnings.append(
                    f"Node {nid!r} ({node.get('title', '')[:30]}) skipped: "
                    f"would exceed page budget ({total_pages}+{page_count}>{max_pages})"
                )
                continue

            total_pages += page_count
            candidates.append(
                CandidateNode(
                    doc_id=doc_id,
                    node_id=nid,
                    title=node.get("title", ""),
                    page_range=page_range,
                    matched_signals=signal_lookup.get(nid, []),
                    reason=reasons.get(nid, "rule prescreen"),
                    needs_page_fetch=True,
                )
            )

        if warnings:
            logger.warning(
                "TreeRetriever page-budget warnings for doc %s: %s",
                doc_id,
                "; ".join(warnings),
            )

        return candidates

    # ------------------------------------------------------------------
    # Rule-based prescreening
    # ------------------------------------------------------------------

    def _prescreen_nodes(
        self,
        parsed: ParsedQuestion,
        flat_nodes: list[dict],
        profile: DomainProfile,
    ) -> list[tuple[dict, int, list[str]]]:
        """Score every node by signal match and return sorted (node, score, signals).

        Scoring signals:
          - Product name mention (+3 per match)
          - Liability term mention (+3 per match)
          - Insurance keyword match (+1 per match)
          - Amount/ratio token match (+1 per match)

        Nodes with score 0 are excluded.
        """
        # Collect match targets from the parsed question
        product_names: list[str] = list(parsed.mentioned_products)

        # Build alias list for each mentioned product (canonical + all aliases)
        product_aliases_to_check: list[str] = []
        for canonical in product_names:
            product_aliases_to_check.append(canonical)
            # Also check all aliases that map to this canonical name
            for alias, cname in profile.product_aliases.items():
                if cname == canonical and alias not in product_aliases_to_check:
                    product_aliases_to_check.append(alias)

        liability_signals: list[str] = list(parsed.liability_signals)
        keywords = profile.keywords

        # Amount / ratio tokens from question + options
        full_text = parsed.question + " " + " ".join(parsed.options.values())
        amount_tokens = _extract_amount_ratio_tokens(full_text)

        scored: list[tuple[dict, int, list[str]]] = []

        for node in flat_nodes:
            title = node.get("title", "")
            score = 0
            signals: list[str] = []

            # Product name match (check all aliases of mentioned products)
            for pn in product_aliases_to_check:
                if pn and pn in title:
                    score += 3
                    signals.append(f"product:{pn}")
                    break  # count once per node

            # Liability term match
            for lt in liability_signals:
                if lt and lt in title:
                    score += 3
                    signals.append(f"liability:{lt}")

            # Keyword match (lower weight)
            for kw in keywords:
                if kw and kw in title:
                    score += 1
                    signals.append(f"keyword:{kw}")

            # Amount/ratio token match
            for at in amount_tokens:
                if at and at in title:
                    score += 1
                    signals.append(f"amount:{at}")

            if score > 0:
                scored.append((node, score, signals))

        # Sort descending by score, then by title stability
        scored.sort(key=lambda x: (-x[1], x[0].get("title", "")))
        return scored

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _top_k_by_score(
        scored: list[tuple[dict, int, list[str]]], k: int
    ) -> list[tuple[dict, int, list[str]]]:
        """Return the top-K entries from (node, score, signals) list.

        When fewer than *k* entries exist, returns all of them.
        """
        return scored[:k]
