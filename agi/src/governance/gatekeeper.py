from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.world_model import WorldModel

@dataclass(slots=True)
class Gatekeeper:
    """Simple policy-based gatekeeper used in tests.

    The policy dictionary recognises two optional keys:

    ``max_tier``
        The highest tier permitted for all tools.  Defaults to ``"T1"`` which
        keeps potentially destructive T2 operations blocked while still allowing
        low tier experimentation.

    ``tools``
        A mapping of tool name to an overridden ``max_tier``.  This mirrors the
        shape we expect a richer policy engine to expose without committing to a
        particular schema yet.
    """

    policy: Mapping[str, object] = field(default_factory=dict)
    world_model: "WorldModel | None" = None

    _TIER_ORDER = {"T0": 0, "T1": 1, "T2": 2, "T3": 3}

    def _normalise_tier(self, level: str | None) -> str:
        if not level:
            return "T0"
        tier = str(level).upper()
        if tier not in self._TIER_ORDER:
            raise ValueError(f"Unknown safety tier: {tier}")
        return tier

    def _tier_value(self, level: str) -> int:
        return self._TIER_ORDER[self._normalise_tier(level)]

    def _evaluation_rules(self) -> Iterable[Mapping[str, Any]]:
        rules = self.policy.get("evaluation_rules", [])
        if isinstance(rules, Mapping):
            rules = [rules]
        if isinstance(rules, Sequence):
            for rule in rules:
                if isinstance(rule, Mapping):
                    yield rule

    def _apply_evaluation_bias(self, tier: str, tool: str | None) -> str:
        if self.world_model is None:
            return tier
        current_value = self._tier_value(tier)
        for rule in self._evaluation_rules():
            claim = rule.get("claim")
            if not claim:
                continue
            belief = self.world_model.beliefs.get(str(claim))
            if belief is None:
                continue
            try:
                min_credence = float(rule.get("min_credence", 0.6))
            except (TypeError, ValueError):
                min_credence = 0.6
            degrade = belief.credence < min_credence
            max_uncertainty = rule.get("max_uncertainty")
            if max_uncertainty is not None:
                try:
                    threshold = float(max_uncertainty)
                except (TypeError, ValueError):
                    threshold = None
                if threshold is not None and belief.uncertainty > threshold:
                    degrade = True
            max_interval = rule.get("max_confidence_interval")
            if max_interval is not None:
                try:
                    interval_threshold = float(max_interval)
                except (TypeError, ValueError):
                    interval_threshold = None
                if interval_threshold is not None:
                    lower, upper = belief.confidence_interval
                    if (upper - lower) > interval_threshold:
                        degrade = True
            if not degrade:
                continue
            tools = rule.get("tools")
            if tools:
                if isinstance(tools, str):
                    tools = [tools]
                if tool not in {str(item) for item in tools if item is not None}:
                    continue
            allowed_tier = self._normalise_tier(rule.get("max_tier", "T0"))
            allowed_value = self._tier_value(allowed_tier)
            if allowed_value < current_value:
                tier = allowed_tier
                current_value = allowed_value
        return tier

    def _max_allowed_for(self, tool: str | None) -> str:
        default_tier = self._normalise_tier(str(self.policy.get("max_tier", "T1")))
        if not tool:
            return self._apply_evaluation_bias(default_tier, tool)
        tool_policies = self.policy.get("tools", {})
        if isinstance(tool_policies, Mapping):
            override = tool_policies.get(tool)
            if override is not None:
                return self._apply_evaluation_bias(self._normalise_tier(str(override)), tool)
        return self._apply_evaluation_bias(default_tier, tool)

    def review(
        self,
        tier: str,
        *,
        tool: str | None = None,
        context: Mapping[str, Any] | None = None,  # context reserved for subclasses
    ) -> bool:
        requested = self._normalise_tier(tier)
        allowed = self._max_allowed_for(tool)
        return self._tier_value(requested) <= self._tier_value(allowed)
