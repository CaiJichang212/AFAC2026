"""Document retriever: A-split passthrough + B-split stub.

Task 7: For A-split questions the retriever returns the question's ``doc_ids``
verbatim (same strings, same order).  This is the authoritative A-split
passthrough rule.  B-split candidate recall is stubbed out and will be
implemented in a later task.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.schemas import ParsedQuestion

logger = logging.getLogger(__name__)


class DocRetriever:
    """Retrieve candidate document IDs for a parsed question.

    Split-A: passthrough — return ``parsed.doc_ids`` exactly.
    Split-B: not yet implemented (raises ``NotImplementedError``).
    """

    def retrieve(
        self,
        parsed: ParsedQuestion,
        catalog: Any,  # DocCatalog — not imported to keep coupling low
    ) -> list[str]:
        """Return candidate document IDs for *parsed*.

        Args:
            parsed: A ``ParsedQuestion`` with ``.split`` and ``.doc_ids``.
            catalog: A ``DocCatalog`` instance (used for validation warnings).

        Returns:
            A copy of ``parsed.doc_ids`` (same strings, same order).

        Raises:
            NotImplementedError: If ``parsed.split != "A"`` (B-split not yet
                implemented).
        """
        if parsed.split != "A":
            raise NotImplementedError(
                f"DocRetriever.retrieve() only supports split='A' "
                f"(got split={parsed.split!r}). B-split retrieval is not "
                f"yet implemented."
            )

        doc_ids = list(parsed.doc_ids)  # defensive copy

        # Warn about doc_ids missing from the catalog, but still return them.
        # A-split passthrough is authoritative — the question's doc_ids are
        # assumed correct even if the catalog is incomplete.
        for doc_id in doc_ids:
            if not catalog.contains(doc_id):
                logger.warning(
                    "Question %s: doc_id %r is missing from the catalog "
                    "(A-split passthrough — still returning it).",
                    parsed.qid,
                    doc_id,
                )

        return doc_ids

    def retrieve_all(
        self,
        parsed_questions: list[ParsedQuestion],
        catalog: Any,
    ) -> dict[str, list[str]]:
        """Convenience: retrieve candidates for multiple questions.

        Args:
            parsed_questions: List of parsed questions.
            catalog: A ``DocCatalog`` instance.

        Returns:
            Dict mapping ``qid`` -> ``list[str]`` of candidate doc_ids.
        """
        result: dict[str, list[str]] = {}
        for pq in parsed_questions:
            result[pq.qid] = self.retrieve(pq, catalog)
        return result
