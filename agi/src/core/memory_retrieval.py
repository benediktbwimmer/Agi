from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import re
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .memory import MemoryStore, _extract_text, _normalise_time


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _tokenise(text: str) -> set[str]:
    return {match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)}


def _confidence_for_record(query_tokens: set[str], record: Mapping[str, Any]) -> float:
    if not query_tokens:
        return 0.0
    record_tokens = _tokenise(_extract_text(record))
    if not record_tokens:
        return 0.0
    overlap = len(query_tokens.intersection(record_tokens))
    if overlap == 0:
        return 0.0
    coverage = overlap / len(query_tokens)
    density = overlap / len(record_tokens)
    score = (coverage * 0.7) + (density * 0.3)
    return max(0.0, min(1.0, round(score, 4)))


def _summarise_confidence(confidences: Sequence[float]) -> Dict[str, float]:
    if not confidences:
        return {}
    summary = {
        "min": round(min(confidences), 4),
        "max": round(max(confidences), 4),
        "mean": round(mean(confidences), 4),
        "count": len(confidences),
    }
    nonzero = sum(1 for value in confidences if value > 0)
    summary["nonzero_fraction"] = round(nonzero / len(confidences), 4)
    return summary


def _annotate_confidence(query: str, records: List[Dict[str, Any]]) -> Dict[str, float]:
    tokens = _tokenise(query)
    confidences: List[float] = []
    for record in records:
        score = _confidence_for_record(tokens, record)
        record["confidence"] = score
        confidences.append(score)
    return _summarise_confidence(confidences)


def _normalise_limit(value: Optional[int], default: int) -> int:
    if value is None:
        return default
    if value <= 0:
        return 0
    return value


def _record_tools(record: Mapping[str, Any]) -> set[str]:
    tools: set[str] = set()
    tool_name = record.get("tool")
    if isinstance(tool_name, str):
        tools.add(tool_name)
    trace = record.get("trace")
    if isinstance(trace, list):
        for step in trace:
            if isinstance(step, Mapping):
                trace_tool = step.get("tool")
                if isinstance(trace_tool, str):
                    tools.add(trace_tool)
    return tools


def _compute_coverage(records: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    coverage: Dict[str, int] = {}
    for record in records:
        record_type = str(record.get("type", "unknown"))
        coverage[record_type] = coverage.get(record_type, 0) + 1
    return coverage


def _filter_by_time(
    records: Iterable[Dict[str, Any]],
    *,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for record in records:
        timestamp = record.get("time")
        if timestamp is None:
            continue
        try:
            parsed = _normalise_time(timestamp)
        except ValueError:
            continue
        if start is not None and parsed < start:
            continue
        if end is not None and parsed > end:
            continue
        filtered.append(record)
    return filtered


@dataclass
class MemorySlice:
    """Semantic slice of persistent memory used for planner context."""

    query: str
    matches: List[Dict[str, Any]]
    coverage: Dict[str, int] = field(default_factory=dict)
    window: Dict[str, Optional[str]] | None = None
    confidence_summary: Dict[str, float] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "query": self.query,
            "matches": self.matches,
            "coverage": dict(self.coverage),
        }
        if self.window:
            payload["window"] = dict(self.window)
        if self.confidence_summary:
            payload["confidence"] = dict(self.confidence_summary)
        return payload


@dataclass
class TemporalWindow:
    """Chronological view over memory for temporal grounding."""

    start: Optional[str]
    end: Optional[str]
    records: List[Dict[str, Any]]
    coverage: Dict[str, int] = field(default_factory=dict)
    filters: Dict[str, Any] | None = None

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "start": self.start,
            "end": self.end,
            "records": self.records,
            "coverage": dict(self.coverage),
        }
        if self.filters:
            payload["filters"] = dict(self.filters)
        return payload


@dataclass
class PlanContext:
    """All memory artefacts associated with a specific plan run."""

    plan_id: str
    episodes: List[Dict[str, Any]]
    reflections: List[Dict[str, Any]]
    coverage: Dict[str, int] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "episodes": self.episodes,
            "reflections": self.reflections,
            "coverage": dict(self.coverage),
        }


@dataclass
class MemoryRetriever:
    """High-level retrieval API combining semantic and temporal search."""

    memory: MemoryStore
    default_limit: int = 5

    def search(
        self,
        query: str,
        *,
        limit: Optional[int] = None,
        types: Sequence[str] | None = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        tools: Sequence[str] | None = None,
    ) -> MemorySlice:
        limit_value = _normalise_limit(limit, self.default_limit)
        if limit_value == 0:
            return MemorySlice(query=query, matches=[])

        semantic_matches = self.memory.semantic_search(
            query,
            limit=max(limit_value * 3, limit_value),
            types=types,
        )

        start = _normalise_time(since) if since else None
        end = _normalise_time(until) if until else None
        tool_filter = {t for t in tools} if tools is not None else None

        filtered: List[Dict[str, Any]] = []
        for record in semantic_matches:
            if start or end:
                in_window = _filter_by_time([record], start=start, end=end)
                if not in_window:
                    continue
            if tool_filter is not None:
                if not _record_tools(record).intersection(tool_filter):
                    continue
            filtered.append(record)
            if len(filtered) >= limit_value:
                break

        coverage = _compute_coverage(filtered)
        confidence_summary = _annotate_confidence(query, filtered)
        window_meta: Dict[str, Optional[str]] | None = None
        if since or until:
            window_meta = {"since": since, "until": until}
        return MemorySlice(
            query=query,
            matches=filtered,
            coverage=coverage,
            window=window_meta,
            confidence_summary=confidence_summary,
        )

    def timeline(
        self,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: Optional[int] = None,
        types: Sequence[str] | None = None,
        tools: Sequence[str] | None = None,
    ) -> TemporalWindow:
        limit_value = _normalise_limit(limit, self.default_limit)
        if limit_value == 0:
            return TemporalWindow(start=start, end=end, records=[], coverage={})

        tool_filter = {t for t in tools} if tools is not None else None
        requested_types = list(types) if types is not None else None

        if start or end:
            start_bound = start or "1970-01-01T00:00:00+00:00"
            end_bound = end or "9999-12-31T23:59:59+00:00"
            records = self.memory.query_by_time(start_bound, end_bound)
        else:
            fetch_limit = max(limit_value * 3, limit_value)
            records = self.memory.recent(limit=fetch_limit, types=types)

        if requested_types is not None and (start or end):
            type_set = set(requested_types)
            records = [r for r in records if r.get("type") in type_set]

        if tool_filter is not None:
            filtered_by_tool: List[Dict[str, Any]] = []
            for record in records:
                if _record_tools(record).intersection(tool_filter):
                    filtered_by_tool.append(record)
            records = filtered_by_tool

        sliced = records[-limit_value:]

        coverage = _compute_coverage(sliced)
        filter_meta: Dict[str, Any] | None = None
        if requested_types or tools:
            filter_meta = {}
            if requested_types:
                filter_meta["types"] = list(requested_types)
            if tools:
                filter_meta["tools"] = list(tools)

        return TemporalWindow(start=start, end=end, records=sliced, coverage=coverage, filters=filter_meta)

    def context_for_goal(
        self,
        goal: str,
        *,
        limit: Optional[int] = None,
        recent: Optional[int] = None,
        types: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        semantic_slice: MemorySlice | None = None
        query = goal.strip()
        limit_value = _normalise_limit(limit, self.default_limit)
        if query:
            semantic_slice = self.search(query, limit=limit_value, types=types)

        recent_limit = _normalise_limit(recent, self.default_limit)
        temporal_window = self.timeline(limit=recent_limit, types=types)

        payload: Dict[str, Any] = {"goal": goal}
        if semantic_slice and semantic_slice.matches:
            payload["semantic"] = semantic_slice.to_payload()
        if temporal_window.records:
            payload["recent"] = temporal_window.to_payload()
        return payload

    def plan_context(self, plan_id: str, *, limit: Optional[int] = None) -> PlanContext:
        limit_value = _normalise_limit(limit, self.default_limit)
        if limit_value == 0:
            return PlanContext(plan_id=plan_id, episodes=[], reflections=[], coverage={})

        plan_records = self.memory.query_by_plan(plan_id)
        if not plan_records:
            return PlanContext(plan_id=plan_id, episodes=[], reflections=[], coverage={})

        episodes: List[Dict[str, Any]] = []
        reflections: List[Dict[str, Any]] = []

        for record in plan_records:
            record_type = record.get("type")
            if record_type == "episode":
                episodes.append(record)
            elif record_type == "reflection":
                reflections.append(record)

        episodes = episodes[-limit_value:] if limit_value else []
        reflections = reflections[-limit_value:] if limit_value else []
        coverage = _compute_coverage(episodes + reflections)
        return PlanContext(
            plan_id=plan_id,
            episodes=episodes,
            reflections=reflections,
            coverage=coverage,
        )


__all__ = [
    "MemoryRetriever",
    "MemorySlice",
    "PlanContext",
    "TemporalWindow",
]
