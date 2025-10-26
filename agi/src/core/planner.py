from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List

from .types import Plan, Prediction, Claim, ToolCall


class PlannerError(RuntimeError):
    pass


@dataclass
class Planner:
    llm: Callable[[Dict[str, Any]], str]

    async def plan_from(self, hypotheses: Iterable[Dict[str, Any]]) -> List[Plan]:
        payload = {"hypotheses": list(hypotheses)}
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
    steps = [
        ToolCall(
            id=step["id"],
            tool=step["tool"],
            args=step.get("args", {}),
            safety_level=step.get("safety_level", "T0"),
        )
        for step in data.get("steps", [])
    ]
    return Plan(
        id=data["id"],
        claim_ids=list(data.get("claim_ids", [])),
        steps=steps,
        expected_cost=data.get("expected_cost", {}),
        risks=list(data.get("risks", [])),
        ablations=list(data.get("ablations", [])),
    )
