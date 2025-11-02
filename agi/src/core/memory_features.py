from __future__ import annotations

"""Feature extraction helpers for memory contexts used by the planner."""

from collections import Counter
from typing import Iterable, Mapping, Dict, Any


def _context_records(memory_context: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    semantic = memory_context.get("semantic")
    if isinstance(semantic, Mapping):
        for record in semantic.get("matches", []) or []:
            if isinstance(record, Mapping):
                yield record
    recent = memory_context.get("recent")
    if isinstance(recent, Mapping):
        for record in recent.get("records", []) or []:
            if isinstance(record, Mapping):
                yield record
    plan_ctx = memory_context.get("plan_context")
    if isinstance(plan_ctx, Mapping):
        for section in ("episodes", "reflections"):
            for record in plan_ctx.get(section, []) or []:
                if isinstance(record, Mapping):
                    yield record


def extract_memory_features(memory_context: Mapping[str, Any]) -> Dict[str, Any]:
    """Aggregate lightweight feature signals from ``memory_context``."""

    keyword_counts: Counter[str] = Counter()
    modality_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()

    for record in _context_records(memory_context):
        for keyword in record.get("keywords", []) or []:
            if isinstance(keyword, str):
                token = keyword.strip().lower()
                if token:
                    keyword_counts[token] += 1
        sensor = record.get("sensor")
        if isinstance(sensor, Mapping):
            modality = sensor.get("modality")
            if isinstance(modality, str) and modality:
                modality_counts[modality.lower()] += 1
        tier = record.get("safety_tier")
        if isinstance(tier, str) and tier:
            tier_counts[tier.upper()] += 1

    features: Dict[str, Any] = {}
    if keyword_counts:
        features["keywords"] = [word for word, _ in keyword_counts.most_common(5)]
    if modality_counts:
        features["modalities"] = [mod for mod, _ in modality_counts.most_common(3)]
    if tier_counts:
        features["safety_tiers"] = [tier for tier, _ in tier_counts.most_common(3)]
    return features


__all__ = ["extract_memory_features"]
