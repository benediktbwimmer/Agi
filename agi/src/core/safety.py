from __future__ import annotations

"""Safety utilities for validating tool usage plans.

The safety module centralises logic around reasoning about tool safety tiers
and delegating approval decisions to the governance layer.  The orchestrator
uses these helpers to ensure that every tool invocation complies with the
configured safety policy before any side effects occur.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Mapping, Optional

from ..governance.gatekeeper import Gatekeeper
from .types import Plan, PlanStep, ToolCall


_TIER_ORDER = {
    "T0": 0,
    "T1": 1,
    "T2": 2,
    "T3": 3,
}


def _normalise_tier(level: str | None) -> str:
    if not level:
        return "T0"
    level = level.upper()
    if level not in _TIER_ORDER:
        raise ValueError(f"Unknown safety tier: {level}")
    return level


def _max_tier(*levels: str | None) -> str:
    normalised = [_normalise_tier(level) for level in levels if level is not None]
    if not normalised:
        return "T0"
    return max(normalised, key=lambda lvl: _TIER_ORDER[lvl])


@dataclass(frozen=True)
class SafetyDecision:
    """Decision returned by the gatekeeper for a single plan step."""

    plan_id: str
    step_id: str
    tool_name: str
    requested_level: str
    tool_level: str
    effective_level: str
    approved: bool
    reason: str | None = None

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RiskAssessment:
    """Real-time risk check performed immediately before tool execution."""

    plan_id: str
    step_id: str
    tool_name: str
    requested_level: str
    tool_level: str
    effective_level: str
    approved: bool
    reason: str | None = None

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def _resolve_tool(tools: Mapping[str, object], call: ToolCall | PlanStep) -> object:
    tool = tools.get(call.tool)
    if tool is None:
        raise KeyError(f"Unknown tool {call.tool}")
    if not hasattr(tool, "run"):
        raise TypeError(f"Tool {call.tool} does not implement run()")
    return tool


def enforce_plan_safety(
    plans: Iterable[Plan],
    tools: Mapping[str, object],
    gatekeeper: Gatekeeper,
) -> list[SafetyDecision]:
    """Ensure every step in every plan passes the gatekeeper.

    Args:
        plans: Plans proposed by the planner.
        tools: Mapping of tool name to tool implementation.
        gatekeeper: Safety gatekeeper responsible for the approvals.

    Returns:
        A list of :class:`SafetyDecision` objects, one per tool invocation, that
        were approved by the gatekeeper.

    Raises:
        PermissionError: If any tool invocation fails the safety review.
        KeyError: If a referenced tool is missing.
        TypeError: If a referenced tool does not satisfy the tool interface.
        ValueError: If an unknown safety tier is encountered.
    """

    decisions: list[SafetyDecision] = []
    denied: list[SafetyDecision] = []

    for plan in plans:
        for step in plan.iter_tool_calls():
            tool = _resolve_tool(tools, step)
            tool_level = getattr(tool, "safety", "T0")
            requested_level = step.safety_level or tool_level
            effective = _max_tier(requested_level, tool_level)
            approved = gatekeeper.review(effective, tool=step.tool)
            decision = SafetyDecision(
                plan_id=plan.id,
                step_id=step.id,
                tool_name=step.tool,
                requested_level=_normalise_tier(requested_level),
                tool_level=_normalise_tier(tool_level),
                effective_level=_normalise_tier(effective),
                approved=approved,
                reason=None if approved else f"Gatekeeper denied tier {effective}",
            )
            decisions.append(decision)
            if not approved:
                denied.append(decision)

    if denied:
        summary = "; ".join(
            f"plan={d.plan_id} step={d.step_id} tool={d.tool_name} tier={d.effective_level}"
            for d in denied
        )
        raise PermissionError(f"Safety gatekeeper denied execution for: {summary}")

    return decisions


def assess_step_risk(
    plan: Plan,
    step: PlanStep,
    tools: Mapping[str, object],
    gatekeeper: Gatekeeper,
) -> RiskAssessment:
    """Perform a real-time risk assessment for the given plan step."""

    tool = _resolve_tool(tools, step)
    tool_level = getattr(tool, "safety", "T0")
    requested_level = step.safety_level or tool_level
    effective = _max_tier(requested_level, tool_level)
    approved = gatekeeper.review(effective, tool=step.tool)
    reason: Optional[str] = None if approved else f"Gatekeeper denied tier {effective}"
    return RiskAssessment(
        plan_id=plan.id,
        step_id=step.id,
        tool_name=step.tool or "",
        requested_level=_normalise_tier(requested_level),
        tool_level=_normalise_tier(tool_level),
        effective_level=_normalise_tier(effective),
        approved=approved,
        reason=reason,
    )

