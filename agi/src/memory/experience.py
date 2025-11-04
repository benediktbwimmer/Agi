from __future__ import annotations

"""Experience replay summarisation helpers for Putnam-style continual learning."""

from collections import Counter
from dataclasses import asdict
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from agi.src.core.manifest import RunManifest
from agi.src.core.types import Report
from agi.src.core.reflection import summarise_working_memory


def _goal_text(goal: Any) -> str | None:
    if isinstance(goal, Mapping):
        value = goal.get("goal") or goal.get("description")
        if value is None and "text" in goal:
            value = goal["text"]
        return str(value) if value is not None else None
    if goal is None:
        return None
    return str(goal)


def _truncate(text: Any, *, limit: int = 160) -> str | None:
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    stripped = text.strip()
    if not stripped:
        return None
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 1] + "…"


def _get_field(value: Any, field: str) -> Any:
    if hasattr(value, field):
        return getattr(value, field)
    if isinstance(value, Mapping):
        return value.get(field)
    return None


def _serialise_tool_results(results: Sequence[Any], *, limit: int = 5) -> list[Dict[str, Any]]:
    serialised: list[Dict[str, Any]] = []
    for result in results[:limit]:
        payload = {
            "call_id": _get_field(result, "call_id"),
            "ok": _get_field(result, "ok"),
            "stdout": _truncate(_get_field(result, "stdout")),
            "wall_time_ms": _get_field(result, "wall_time_ms"),
            "cost_tokens": _get_field(result, "cost_tokens"),
        }
        serialised.append({key: value for key, value in payload.items() if value is not None})
    return serialised


def _safety_summary(entries: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    total = 0
    blocked: list[Dict[str, Any]] = []
    for entry in entries:
        total += 1
        if not entry.get("approved", True):
            blocked.append(
                {
                    "plan_id": entry.get("plan_id"),
                    "step_id": entry.get("step_id"),
                    "tool": entry.get("tool_name") or entry.get("tool"),
                    "effective_level": entry.get("effective_level"),
                    "reason": entry.get("reason"),
                }
            )
    return {
        "total": total,
        "blocked": blocked,
    }


def summarise_experience(
    report: Report,
    manifest: RunManifest,
    *,
    working_memory: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Summarise a completed run into an experience replay chunk."""

    goal_text = _goal_text(manifest.goal or report.goal)
    tool_results = list(manifest.tool_results or [])
    safety_audit = [_ensure_mapping(entry) for entry in manifest.safety_audit or []]
    risk_assessments = [_ensure_mapping(entry) for entry in manifest.risk_assessments or []]
    belief_updates: list[Dict[str, Any]] = []
    for delta in manifest.belief_updates or []:
        if hasattr(delta, "model_dump"):
            belief_updates.append(delta.model_dump())
        elif isinstance(delta, Mapping):
            belief_updates.append(dict(delta))
        else:
            belief_updates.append(asdict(delta))
    success = all(result.ok for result in tool_results) if tool_results else True
    working_summary: Dict[str, Any] | None = None
    scratchpad: Dict[str, Any] | None = None
    if working_memory:
        if not isinstance(working_memory, Mapping):
            working_summary = summarise_working_memory(working_memory)
            scratchpad = getattr(working_memory, "scratchpad", None)
        else:
            scratchpad = working_memory.get("scratchpad")

    experience: Dict[str, Any] = {
        "goal": goal_text,
        "summary": report.summary,
        "success": success,
        "created_at": manifest.created_at,
        "tool_invocations": len(tool_results),
        "key_findings": list(report.key_findings),
        "artifacts": list(report.artifacts),
        "tool_results": _serialise_tool_results(tool_results),
        "safety_audit": _safety_summary(safety_audit),
        "risk_assessments": _safety_summary(risk_assessments),
        "belief_updates": belief_updates,
    }
    if working_summary:
        experience["working_memory"] = working_summary
    if scratchpad:
        experience["scratchpad"] = scratchpad
    negotiations = list(manifest.negotiations or [])
    if negotiations:
        agent_pairs: Counter[tuple[str, str]] = Counter()
        kind_counts: Counter[str] = Counter()
        agents: set[str] = set()
        sample: list[Dict[str, Any]] = []
        for message in negotiations:
            sender = _get_field(message, "sender") or "unknown"
            recipient = _get_field(message, "recipient") or "unknown"
            kind_value = _get_field(message, "kind") or "unspecified"
            timestamp = _get_field(message, "timestamp")
            agent_pairs[(sender, recipient)] += 1
            kind_counts[kind_value.lower()] += 1
            agents.update({sender, recipient})
            if len(sample) < 5:
                sample.append(
                    {
                        "time": timestamp,
                        "from": sender,
                        "to": recipient,
                        "kind": kind_value,
                    }
                )
        experience["negotiations"] = {
            "count": len(negotiations),
            "agents": sorted(agent for agent in agents if agent),
            "pairs": [
                {"pair": f"{pair[0]}->{pair[1]}", "count": count}
                for pair, count in agent_pairs.most_common(5)
            ],
            "kinds": [
                {"kind": kind, "count": count} for kind, count in kind_counts.most_common()
            ],
            "sample": sample,
        }
    return experience


def _ensure_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


__all__ = ["summarise_experience"]
