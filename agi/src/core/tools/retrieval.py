from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Sequence

from . import ToolCapability, ToolParameter, ToolSpec
from ..memory import MemoryStore
from ..types import RunContext, Source, ToolResult


def _normalise_limit(value: Any, default: int, maximum: int) -> int:
    try:
        limit = int(value)
    except (TypeError, ValueError):
        return default
    if limit <= 0:
        return default
    return min(limit, maximum)


def _parse_types(value: Any) -> Iterable[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        filtered = [part for part in parts if part]
        return filtered or None
    if isinstance(value, Sequence):
        filtered = [str(item).strip() for item in value if str(item).strip()]
        return filtered or None
    return None


def _normalise_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return _normalise_datetime(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        text = text.replace("Z", "+00:00")
        try:
            return _normalise_datetime(datetime.fromisoformat(text))
        except ValueError:
            return None
    return None


def _record_hash(record: Dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, default=str)


@dataclass
class RetrievalTool:
    """Memory-backed retrieval tool providing semantic and temporal search."""

    memory: MemoryStore
    name: str = "retrieval"
    safety: str = "T0"
    default_limit: int = 5
    max_limit: int = 25

    async def run(self, args: Dict[str, Any], ctx: RunContext) -> ToolResult:
        query = (args.get("query") or "").strip()
        limit = _normalise_limit(args.get("limit"), self.default_limit, self.max_limit)
        type_filter = _parse_types(args.get("types"))
        since = _parse_time(args.get("since") or args.get("start"))
        until = _parse_time(args.get("until") or args.get("end"))
        claim_id = args.get("claim_id")

        candidates: List[Dict[str, Any]] = []
        if query:
            candidates.extend(
                self.memory.semantic_search(query, limit=self.max_limit, types=type_filter)
            )
        if claim_id:
            candidates.extend(self.memory.query_by_claim(str(claim_id)))
        if since or until:
            start_dt = since or datetime.min.replace(tzinfo=timezone.utc)
            end_dt = until or datetime.max.replace(tzinfo=timezone.utc)
            start_text = _normalise_datetime(start_dt).isoformat()
            end_text = _normalise_datetime(end_dt).isoformat()
            candidates.extend(self.memory.query_by_time(start_text, end_text))

        if not candidates:
            start_text = _normalise_datetime(datetime.min.replace(tzinfo=timezone.utc)).isoformat()
            end_text = _normalise_datetime(datetime.max.replace(tzinfo=timezone.utc)).isoformat()
            candidates.extend(self.memory.query_by_time(start_text, end_text))

        filtered: List[Dict[str, Any]] = []
        seen = set()
        for record in candidates:
            record_hash = _record_hash(record)
            if record_hash in seen:
                continue
            seen.add(record_hash)
            if claim_id:
                record_claim = None
                if isinstance(record.get("claim"), dict):
                    record_claim = record["claim"].get("id")
                elif "claim_id" in record:
                    record_claim = record.get("claim_id")
                if str(record_claim) != str(claim_id):
                    continue
            if type_filter is not None:
                record_type = record.get("type")
                if record_type not in type_filter:
                    continue
            record_time = _parse_time(record.get("time"))
            if since and (record_time is None or record_time < since):
                continue
            if until and (record_time is None or record_time > until):
                continue
            filtered.append(record)
            if len(filtered) >= limit:
                break

        stdout = (
            f"retrieved {len(filtered)} record(s) from memory"
            if filtered
            else "no matching memory records"
        )
        provenance_note = (
            f"retrieval:{len(filtered)} results for query '{query}'"
            if query
            else f"retrieval:{len(filtered)} results"
        )
        return ToolResult(
            call_id=args.get("id", "retrieval"),
            ok=True,
            stdout=stdout,
            data={"records": filtered},
            provenance=[
                Source(
                    kind="memory",
                    ref=str(self.memory.path),
                    note=provenance_note,
                )
            ],
        )

    def describe(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description="Query episodic memory by semantic similarity, claim association, or time range.",
            safety_tier=self.safety,
            metadata={"default_limit": self.default_limit, "max_limit": self.max_limit},
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Free-form query text for lexical search."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": self.max_limit},
                    "types": {
                        "oneOf": [
                            {"type": "string"},
                            {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        ],
                        "description": "Optional record type filter.",
                    },
                    "since": {"type": "string", "format": "date-time"},
                    "until": {"type": "string", "format": "date-time"},
                    "claim_id": {"type": "string"},
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "object",
                        "properties": {
                            "records": {"type": "array", "items": {"type": "object"}},
                        },
                    },
                    "stdout": {"type": "string"},
                },
            },
            capabilities=(
                ToolCapability(
                    name="retrieve_memory",
                    description="Return recent or relevant episodic memory entries for planner context.",
                    safety_tier=self.safety,
                    parameters=(
                        ToolParameter(
                            name="query",
                            description="Text query used for semantic search.",
                            required=False,
                            schema={"type": "string"},
                        ),
                        ToolParameter(
                            name="claim_id",
                            description="Restrict results to a specific claim identifier.",
                            required=False,
                            schema={"type": "string"},
                        ),
                        ToolParameter(
                            name="since",
                            description="ISO timestamp lower bound for record time.",
                            required=False,
                            schema={"type": "string", "format": "date-time"},
                        ),
                        ToolParameter(
                            name="until",
                            description="ISO timestamp upper bound for record time.",
                            required=False,
                            schema={"type": "string", "format": "date-time"},
                        ),
                    ),
                    outputs=("data.records",),
                ),
            ),
        )
