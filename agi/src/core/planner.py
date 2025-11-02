from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Set

from .types import BranchCondition, Plan, PlanBranch, PlanStep, Prediction, Claim
from .reflection import normalise_insight_records, summarise_reflection_insights
from .memory_features import extract_memory_features


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
        memory_context: Mapping[str, Any] | None = None,
    ) -> List[Plan]:
        payload = {
            "hypotheses": [json.loads(json.dumps(hypothesis)) for hypothesis in hypotheses],
        }
        feedback_list = list(feedback or [])
        if feedback_list:
            payload["feedback"] = feedback_list
        reflective_summary: Dict[str, Any] | None = None
        if memory_context:
            payload["memory_context"] = json.loads(json.dumps(memory_context))
            features = extract_memory_features(memory_context)
            if features:
                payload["memory_features"] = features
            reflective_summary = _augment_payload_with_reflections(payload, memory_context)
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
        if reflective_summary:
            return _apply_reflective_bias(plans, reflective_summary)
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


def _augment_payload_with_reflections(
    payload: Dict[str, Any],
    memory_context: Mapping[str, Any],
) -> Dict[str, Any] | None:
    insight_records = normalise_insight_records(memory_context.get("insights"))
    if not insight_records:
        return None
    summary = summarise_reflection_insights(insight_records)
    payload["reflective_summary"] = summary
    dominant_tags = summary.get("dominant_tags", [])
    caution_score = summary.get("caution_score", 0)
    risk_events = summary.get("risk_events", 0)

    for hypothesis in payload.get("hypotheses", []):
        if not isinstance(hypothesis, dict):
            continue
        reflective_context: Dict[str, Any] = {}
        if dominant_tags:
            reflective_context["dominant_tags"] = dominant_tags
        if caution_score:
            reflective_context["caution_score"] = caution_score
        if risk_events:
            reflective_context["risk_events"] = risk_events
        if not reflective_context:
            continue
        previous = hypothesis.get("reflective_context")
        if isinstance(previous, Mapping):
            merged = dict(previous)
            merged.update(reflective_context)
            hypothesis["reflective_context"] = merged
        else:
            hypothesis["reflective_context"] = reflective_context
    focus_counts = summary.get("hypothesis_focus")
    if isinstance(focus_counts, Mapping):
        for hypothesis in payload.get("hypotheses", []):
            if not isinstance(hypothesis, dict):
                continue
            ident = hypothesis.get("id") or hypothesis.get("claim_id")
            if not ident:
                continue
            count = focus_counts.get(str(ident))
            if not count:
                continue
            context = dict(hypothesis.get("reflective_context") or {})
            context["focus_count"] = count
            hypothesis["reflective_context"] = context
    planner_bias = summary.get("planner_bias")
    if planner_bias:
        payload["planner_bias"] = planner_bias
    return summary


def _apply_reflective_bias(plans: List[Plan], summary: Mapping[str, Any]) -> List[Plan]:
    caution_score = float(summary.get("caution_score") or 0)
    risk_events = float(summary.get("risk_events") or 0)
    dominant_tags = summary.get("dominant_tags", [])
    motif_issue_counts = summary.get("motif_issue_counts", {})
    failure_branches = summary.get("failure_branches", [])
    motif_pressure = float(summary.get("motif_pressure") or 0)
    if not (caution_score or risk_events or dominant_tags):
        return plans

    caution_factor = 1.0 + (0.5 * caution_score) + (0.25 * risk_events) + (0.3 * motif_pressure)
    issue_keywords = {str(name).lower() for name in motif_issue_counts.keys()}
    failure_steps = {
        str(entry.get("step_id"))
        for entry in failure_branches
        if isinstance(entry, Mapping) and entry.get("step_id")
    }
    scored: List[tuple[float, Plan]] = []
    for plan in plans:
        risk = _estimate_plan_risk(plan, caution_factor, dominant_tags, issue_keywords, failure_steps)
        scored.append((risk, plan))
    scored.sort(key=lambda item: (item[0], item[1].id))

    base_threshold = 30.0
    dynamic_threshold = base_threshold / max(1.0, caution_factor)
    filtered = [plan for score, plan in scored if score <= dynamic_threshold]
    if filtered:
        return filtered
    return [plan for _, plan in scored]


_SAFETY_COST = {
    "T0": 0.0,
    "T1": 2.0,
    "T2": 4.0,
    "T3": 6.0,
}


def _estimate_plan_risk(
    plan: Plan,
    caution_factor: float,
    tags: Sequence[str],
    issue_keywords: Set[str],
    failure_steps: Set[str],
) -> float:
    tag_set = {tag.lower() for tag in tags if isinstance(tag, str)}
    keyword_set = {kw for kw in issue_keywords if kw}
    base = 0.0
    for step in plan.iter_tool_calls():
        level = (step.safety_level or "T0").upper()
        base += _SAFETY_COST.get(level, 3.0)
        if "safety" in tag_set and level != "T0":
            base += 1.0
        match_text = " ".join(
            filter(
                None,
                [
                    step.tool or "",
                    step.description or "",
                    step.goal or "",
                ],
            )
        ).lower()
        if keyword_set and any(keyword in match_text for keyword in keyword_set):
            base += 1.5
        if failure_steps and step.id in failure_steps:
            base += 2.5
    base += len(plan.risks) * 2.0
    return base * caution_factor
