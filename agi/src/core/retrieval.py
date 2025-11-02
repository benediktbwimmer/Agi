from __future__ import annotations

"""Utilities for retrieving contextual memory snippets.

The :mod:`agi.src.core.memory` module provides the durable storage and
indexing primitives for episodic memory.  To make that information easier to
consume by higher level agents we expose a light-weight retriever that can
blend claim-specific lookups with lexical semantic search.  The resulting
snippets are designed to be serialisable and compact so they can be forwarded
directly to language models or planning components without additional
normalisation.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from .memory import MemoryStore


def _summarise_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Condense a raw memory record into a portable summary."""

    summary: Dict[str, Any] = {"type": record.get("type")}
    for key in (
        "time",
        "tool",
        "call_id",
        "ok",
        "stdout",
        "summary",
        "notes",
        "plan_id",
        "status",
        "amendments",
        "issues",
        "key_findings",
        "provenance",
        "artifacts",
    ):
        value = record.get(key)
        if value is not None and value != "":
            summary[key] = value

    claim = record.get("claim")
    if isinstance(claim, Mapping):
        claim_payload: MutableMapping[str, Any] = {}
        if claim.get("id"):
            claim_payload["id"] = claim["id"]
        if claim.get("text"):
            claim_payload["text"] = claim["text"]
        if claim_payload:
            summary["claim"] = dict(claim_payload)

    return summary


def _compose_query_text(goal: str, context: Optional[Mapping[str, Any]]) -> str:
    """Construct a lexical search query from the caller provided context."""

    parts: List[str] = [goal]
    if context:
        for value in context.values():
            if value is None:
                continue
            if isinstance(value, str):
                parts.append(value)
            else:
                parts.append(str(value))
    return "\n".join(part for part in parts if part)


@dataclass
class MemoryRetriever:
    """Blend claim lookups and semantic search for contextual memory."""

    memory: MemoryStore
    semantic_types: Sequence[str] = field(
        default_factory=lambda: ("episode", "reflection", "critique", "semantic")
    )
    per_claim_limit: int = 5
    semantic_limit: int = 5

    def retrieve(
        self,
        *,
        goal: str,
        context: Optional[Mapping[str, Any]] = None,
        claim_ids: Iterable[str] | None = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return memory snippets relevant to the provided goal."""

        claim_ids = list(claim_ids or [])
        effective_limit = limit if limit is not None else self.per_claim_limit + self.semantic_limit
        snippets: List[Dict[str, Any]] = []
        seen: set[str] = set()

        def _add_snippet(payload: Dict[str, Any]) -> None:
            fingerprint_payload = {k: v for k, v in payload.items() if k != "source"}
            fingerprint = json.dumps(fingerprint_payload, sort_keys=True)
            if fingerprint in seen:
                return
            seen.add(fingerprint)
            snippets.append(payload)

        for claim_id in claim_ids:
            episodes = self.memory.query_by_claim(claim_id)
            if not episodes:
                continue
            for record in episodes[-self.per_claim_limit :]:
                summary = _summarise_record(record)
                summary["source"] = "claim"
                summary["claim_id"] = claim_id
                _add_snippet(summary)
                if len(snippets) >= effective_limit:
                    return snippets[:effective_limit]

        query = _compose_query_text(goal, context)
        if query:
            semantic_records = self.memory.semantic_search(
                query,
                limit=self.semantic_limit,
                types=self.semantic_types,
            )
            for record in semantic_records:
                summary = _summarise_record(record)
                summary["source"] = "semantic"
                _add_snippet(summary)
                if len(snippets) >= effective_limit:
                    return snippets[:effective_limit]

        return snippets[:effective_limit]


__all__ = [
    "MemoryRetriever",
]

