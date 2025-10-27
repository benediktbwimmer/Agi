from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

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

    def _max_allowed_for(self, tool: str | None) -> str:
        default_tier = self._normalise_tier(str(self.policy.get("max_tier", "T1")))
        if not tool:
            return default_tier
        tool_policies = self.policy.get("tools", {})
        if isinstance(tool_policies, Mapping):
            override = tool_policies.get(tool)
            if override is not None:
                return self._normalise_tier(str(override))
        return default_tier

    def review(self, tier: str, *, tool: str | None = None) -> bool:
        requested = self._normalise_tier(tier)
        allowed = self._max_allowed_for(tool)
        return self._tier_value(requested) <= self._tier_value(allowed)
