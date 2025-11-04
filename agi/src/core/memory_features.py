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
    negotiation_pairs: Counter[str] = Counter()
    negotiation_agents: Counter[str] = Counter()
    negotiation_kinds: Counter[str] = Counter()

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
        record_type = record.get("type")
        if record_type == "negotiation":
            sender = record.get("sender")
            recipient = record.get("recipient")
            if isinstance(sender, str) and isinstance(recipient, str) and sender and recipient:
                pair_key = f"{sender}->{recipient}"
                negotiation_pairs[pair_key] += 1
                negotiation_agents[sender] += 1
                negotiation_agents[recipient] += 1
            kind = record.get("kind")
            if isinstance(kind, str) and kind:
                negotiation_kinds[kind.lower()] += 1
        elif record_type == "negotiation_summary":
            for item in record.get("pairs", []) or []:
                if not isinstance(item, Mapping):
                    continue
                pair = item.get("pair")
                count = item.get("count")
                if isinstance(pair, str) and pair:
                    increment = 1
                    if isinstance(count, (int, float)):
                        increment = int(round(float(count)))
                        if increment <= 0:
                            increment = 1
                    negotiation_pairs[pair] += increment
            for agent in record.get("agents", []) or []:
                if isinstance(agent, str) and agent:
                    negotiation_agents[agent] += 1
            for item in record.get("kinds", []) or []:
                if not isinstance(item, Mapping):
                    continue
                kind = item.get("kind")
                count = item.get("count")
                if isinstance(kind, str) and kind:
                    increment = 1
                    if isinstance(count, (int, float)):
                        increment = int(round(float(count)))
                        if increment <= 0:
                            increment = 1
                    negotiation_kinds[kind.lower()] += increment

    features: Dict[str, Any] = {}
    if keyword_counts:
        features["keywords"] = [word for word, _ in keyword_counts.most_common(5)]
    if modality_counts:
        features["modalities"] = [mod for mod, _ in modality_counts.most_common(3)]
    if tier_counts:
        features["safety_tiers"] = [tier for tier, _ in tier_counts.most_common(3)]
    if negotiation_pairs:
        features["negotiation_pairs"] = [pair for pair, _ in negotiation_pairs.most_common(3)]
    if negotiation_agents:
        features["negotiation_agents"] = [agent for agent, _ in negotiation_agents.most_common(3)]
    if negotiation_kinds:
        features["negotiation_kinds"] = [kind for kind, _ in negotiation_kinds.most_common(3)]
    return features


__all__ = ["extract_memory_features"]
