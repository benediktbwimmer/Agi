from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List

from .types import BranchCondition, Plan, PlanBranch, PlanStep, Prediction, Claim


class PlannerError(RuntimeError):
    pass


@dataclass
class Planner:
    llm: Callable[[Dict[str, Any]], str]

    async def plan_from(
        self,
        hypotheses: Iterable[Dict[str, Any]],
        *,
        feedback: Iterable[Dict[str, Any]] | None = None,
    ) -> List[Plan]:
        payload = {"hypotheses": list(hypotheses)}
        feedback_list = list(feedback or [])
        if feedback_list:
            payload["feedback"] = feedback_list
        raw = self.llm(payload)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise PlannerError("Planner LLM returned invalid JSON") from exc

        plans: List[Plan] = []
        for plan_dict in data.get("plans", []):
            plans.append(_parse_plan(plan_dict))
        if not plans:
            raise PlannerError("Planner produced no plans")
        return plans


def _parse_plan(data: Dict[str, Any]) -> Plan:
    return Plan(
        id=data["id"],
        claim_ids=list(data.get("claim_ids", [])),
        steps=[_parse_plan_step(step) for step in data.get("steps", [])],
        expected_cost=data.get("expected_cost", {}),
        risks=list(data.get("risks", [])),
        ablations=list(data.get("ablations", [])),
    )


def _parse_plan_step(data: Dict[str, Any]) -> PlanStep:
    sub_steps = [_parse_plan_step(step) for step in data.get("sub_steps", [])]
    branches = [_parse_plan_branch(branch) for branch in data.get("branches", [])]
    tool = data.get("tool")
    return PlanStep(
        id=data["id"],
        tool=tool,
        args=data.get("args", {}),
        safety_level=data.get("safety_level", "T0"),
        description=data.get("description"),
        goal=data.get("goal"),
        sub_steps=sub_steps,
        branches=branches,
    )


def _parse_plan_branch(data: Dict[str, Any]) -> PlanBranch:
    return PlanBranch(
        condition=BranchCondition.from_raw(data.get("condition")),
        steps=[_parse_plan_step(step) for step in data.get("steps", [])],
    )
