from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set

from .critic import Critic
from .manifest import RunManifest
from .memory import MemoryStore
from .memory_retrieval import MemoryRetriever
from ..memory.encoders import summarise_tool_result
from .memory_features import extract_memory_features
from ..memory.experience import summarise_experience
from ..memory.reflection_job import consolidate_reflections
from .planner import Planner
from .safety import (
    RiskAssessment,
    SafetyDecision,
    assess_step_risk,
    enforce_plan_safety,
)
from .tools import ToolSpec, describe_tool
from .telemetry import Telemetry
from .types import BranchCondition, Plan, PlanStep, Report, RunContext, ToolResult
from .world_model import WorldModel
from ..governance.gatekeeper import Gatekeeper


def _truncate_text(value: Any, limit: int = 160) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "\u2026"


def _summarise_memory_records(records: Iterable[Mapping[str, Any]], *, limit: int = 5) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for record in list(records)[:limit]:
        if not isinstance(record, Mapping):
            continue
        entry: Dict[str, Any] = {}
        for key in ("id", "plan_id", "type", "tool", "time"):
            value = record.get(key)
            if isinstance(value, str) and value:
                entry[key] = value
        confidence = record.get("confidence")
        if isinstance(confidence, (int, float)):
            entry["confidence"] = round(float(confidence), 4)
        for key in ("summary", "notes", "stdout"):
            text = _truncate_text(record.get(key))
            if text:
                entry[key] = text
        keywords = record.get("keywords")
        if isinstance(keywords, list) and keywords:
            entry["keywords"] = [str(kw) for kw in keywords[:5] if kw]
        sensor = record.get("sensor")
        if isinstance(sensor, Mapping):
            modality = sensor.get("modality")
            trust = sensor.get("trust")
            if modality or trust:
                entry["sensor"] = {
                    "modality": modality,
                    "trust": trust,
                }
        if entry:
            summary.append(entry)
    return summary


def _summarise_memory_context(context: Mapping[str, Any] | None) -> Dict[str, Any] | None:
    if not context:
        return None
    summary: Dict[str, Any] = {}
    goal = context.get("goal")
    if isinstance(goal, str) and goal.strip():
        summary["goal"] = goal.strip()
    semantic = context.get("semantic")
    if isinstance(semantic, Mapping):
        matches = semantic.get("matches")
        if isinstance(matches, list):
            semantic_summary = _summarise_memory_records(matches)
            if semantic_summary:
                summary["semantic_matches"] = semantic_summary
        query = semantic.get("query")
        if isinstance(query, str) and query.strip():
            summary["semantic_query"] = query.strip()
        confidence = semantic.get("confidence")
        if isinstance(confidence, Mapping):
            summary["semantic_confidence"] = {
                key: float(value)
                for key, value in confidence.items()
                if isinstance(value, (int, float))
            }
    recent = context.get("recent")
    if isinstance(recent, Mapping):
        records = recent.get("records")
        if isinstance(records, list):
            recent_summary = _summarise_memory_records(records)
            if recent_summary:
                summary["recent_records"] = recent_summary
    return summary or None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_input_provenance(
    goal_spec: Mapping[str, Any],
    hypotheses: Iterable[Any],
    memory_context: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    provenance: Dict[str, Any] = {}
    goal = goal_spec.get("goal")
    if isinstance(goal, str) and goal.strip():
        provenance["goal"] = goal.strip()
    hypotheses_list = list(hypotheses)
    if hypotheses_list:
        provenance["hypotheses"] = json.loads(json.dumps(hypotheses_list))
    memory_summary = _summarise_memory_context(memory_context)
    if memory_summary:
        provenance["memory"] = memory_summary
    expected_unit = goal_spec.get("expected_unit")
    if expected_unit:
        provenance["expected_unit"] = expected_unit
    return provenance


@dataclass
class PlanTrace:
    """Snapshot of deliberation information for a single plan."""

    plan_id: str
    claim_ids: List[str]
    step_count: int
    approved: Optional[bool] = None
    execution_succeeded: Optional[bool] = None
    rationale_tags: Set[str] = field(default_factory=set)

    def mark_approved(self) -> None:
        self.approved = True

    def mark_rejected(self) -> None:
        self.approved = False

    def mark_execution(self, succeeded: bool) -> None:
        self.execution_succeeded = succeeded

    def add_rationale_tags(self, tags: Iterable[str]) -> None:
        for tag in tags:
            if tag:
                self.rationale_tags.add(str(tag))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "claim_ids": list(self.claim_ids),
            "step_count": self.step_count,
            "approved": self.approved,
            "execution_succeeded": self.execution_succeeded,
            "rationale_tags": sorted(self.rationale_tags),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanTrace":
        plan_id = str(payload.get("plan_id", ""))
        raw_claims = payload.get("claim_ids", [])
        claim_ids: List[str] = []
        if isinstance(raw_claims, Iterable) and not isinstance(raw_claims, (str, bytes)):
            claim_ids = [str(item) for item in raw_claims]
        raw_step_count = payload.get("step_count")
        try:
            step_count = int(raw_step_count)
        except (TypeError, ValueError):
            step_count = 0
        approved = payload.get("approved")
        if approved is not None:
            approved = bool(approved)
        execution_succeeded = payload.get("execution_succeeded")
        if execution_succeeded is not None:
            execution_succeeded = bool(execution_succeeded)
        raw_tags = payload.get("rationale_tags", [])
        rationale_tags: Set[str] = set()
        if isinstance(raw_tags, Iterable) and not isinstance(raw_tags, (str, bytes)):
            rationale_tags = {str(tag) for tag in raw_tags if tag}
        return cls(
            plan_id=plan_id,
            claim_ids=claim_ids,
            step_count=step_count,
            approved=approved,
            execution_succeeded=execution_succeeded,
            rationale_tags=rationale_tags,
        )


@dataclass
class DeliberationAttempt:
    """Tracks a single orchestration attempt across planning and execution."""

    index: int
    status: str = "planning"
    plans: List[PlanTrace] = field(default_factory=list)
    critiques: List[Dict[str, Any]] = field(default_factory=list)
    execution_feedback: List[Dict[str, Any]] = field(default_factory=list)
    risk_assessments: List[Dict[str, Any]] = field(default_factory=list)
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    branch_log: List[Dict[str, Any]] = field(default_factory=list)
    critic_rationales: List[Dict[str, Any]] = field(default_factory=list)
    _plan_index: Dict[str, PlanTrace] = field(default_factory=dict, init=False, repr=False)

    def register_plan(self, plan: Plan) -> PlanTrace:
        trace = PlanTrace(
            plan_id=plan.id,
            claim_ids=list(plan.claim_ids),
            step_count=len(plan.steps),
        )
        self.plans.append(trace)
        self._plan_index[plan.id] = trace
        return trace

    def set_hypotheses(self, hypotheses: Iterable[Any]) -> None:
        snapshot: List[Dict[str, Any]] = []
        for item in hypotheses:
            if isinstance(item, Mapping):
                snapshot.append(json.loads(json.dumps(item)))
            else:
                snapshot.append({"value": json.loads(json.dumps(item)) if isinstance(item, (dict, list)) else str(item)})
        self.hypotheses = snapshot

    def record_critique(self, plan_id: str, critique: Dict[str, Any]) -> None:
        self.critiques.append(dict(critique))
        plan = self._plan_index.get(plan_id)
        if plan is not None:
            plan.mark_rejected()
            plan.add_rationale_tags(critique.get("rationale_tags", []))
        rationale = {
            "plan_id": plan_id,
            "status": critique.get("status"),
            "summary": critique.get("summary"),
            "issues": critique.get("issues"),
            "rationale_tags": critique.get("rationale_tags"),
        }
        self.critic_rationales.append(
            {k: json.loads(json.dumps(v)) if isinstance(v, Mapping) else v for k, v in rationale.items() if v}
        )

    def record_execution_feedback(self, feedback: Iterable[Dict[str, Any]]) -> None:
        self.execution_feedback = [dict(item) for item in feedback]

    def mark_status(self, status: str) -> None:
        self.status = status

    def update_execution_results(self, result_map: Mapping[str, bool]) -> None:
        for plan_id, succeeded in result_map.items():
            plan = self._plan_index.get(plan_id)
            if plan is not None:
                plan.mark_execution(succeeded)

    def record_risk_assessment(self, assessment: RiskAssessment) -> None:
        self.risk_assessments.append(assessment.as_dict())

    def record_branch_decision(
        self,
        *,
        plan_id: str,
        step_id: str,
        metadata: Mapping[str, Any] | None,
        taken: bool,
    ) -> None:
        entry: Dict[str, Any] = {
            "plan_id": plan_id,
            "step_id": step_id,
            "taken": taken,
        }
        if metadata:
            entry["metadata"] = json.loads(json.dumps(metadata))
        self.branch_log.append(entry)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "status": self.status,
            "plans": [plan.to_dict() for plan in self.plans],
            "critiques": [dict(item) for item in self.critiques],
            "execution_feedback": [dict(item) for item in self.execution_feedback],
            "risk_assessments": [dict(item) for item in self.risk_assessments],
            "hypotheses": json.loads(json.dumps(self.hypotheses)),
            "branch_log": [dict(item) for item in self.branch_log],
            "critic_rationales": [dict(item) for item in self.critic_rationales],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DeliberationAttempt":
        raw_plans = payload.get("plans", [])
        plans: List[PlanTrace] = []
        if isinstance(raw_plans, Iterable) and not isinstance(raw_plans, (str, bytes)):
            for item in raw_plans:
                if isinstance(item, Mapping):
                    plans.append(PlanTrace.from_dict(item))
        critiques = [
            dict(item)
            for item in payload.get("critiques", [])
            if isinstance(item, Mapping)
        ]
        feedback = [
            dict(item)
            for item in payload.get("execution_feedback", [])
            if isinstance(item, Mapping)
        ]
        risks = [
            dict(item)
            for item in payload.get("risk_assessments", [])
            if isinstance(item, Mapping)
        ]
        try:
            index = int(payload.get("index", 0))
        except (TypeError, ValueError):
            index = 0
        status = str(payload.get("status", "planning"))
        attempt = cls(
            index=index,
            status=status,
            plans=plans,
            critiques=critiques,
            execution_feedback=feedback,
            risk_assessments=risks,
        )
        for plan in plans:
            attempt._plan_index[plan.plan_id] = plan
        hyp_payload = payload.get("hypotheses", [])
        if isinstance(hyp_payload, Iterable) and not isinstance(hyp_payload, (str, bytes)):
            attempt.hypotheses = [
                json.loads(json.dumps(item)) if isinstance(item, Mapping) else {"value": item}
                for item in hyp_payload
            ]
        branch_payload = payload.get("branch_log", [])
        if isinstance(branch_payload, Iterable) and not isinstance(branch_payload, (str, bytes)):
            attempt.branch_log = [
                dict(item) for item in branch_payload if isinstance(item, Mapping)
            ]
        rationale_payload = payload.get("critic_rationales", [])
        if isinstance(rationale_payload, Iterable) and not isinstance(rationale_payload, (str, bytes)):
            attempt.critic_rationales = [
                dict(item) for item in rationale_payload if isinstance(item, Mapping)
            ]
        return attempt


@dataclass
class WorkingMemory:
    """Lightweight working memory capturing deliberation history."""

    run_id: str
    goal: Optional[str]
    hypotheses: List[Any]
    attempts: List[DeliberationAttempt] = field(default_factory=list)
    input_provenance: Dict[str, Any] | None = None
    context_features: Dict[str, Any] | None = None

    def start_attempt(self, index: int) -> DeliberationAttempt:
        attempt = DeliberationAttempt(index=index)
        self.attempts.append(attempt)
        return attempt

    @property
    def scratchpad(self) -> Dict[str, Any]:
        attempts_payload: List[Dict[str, Any]] = []
        for attempt in self.attempts:
            entry: Dict[str, Any] = {"index": attempt.index, "status": attempt.status}
            if attempt.hypotheses:
                entry["hypotheses"] = json.loads(json.dumps(attempt.hypotheses))
            if attempt.branch_log:
                entry["branch_log"] = [dict(item) for item in attempt.branch_log]
            if attempt.critic_rationales:
                entry["critic_rationales"] = [dict(item) for item in attempt.critic_rationales]
            if len(entry) > 2:
                attempts_payload.append(entry)
        if attempts_payload:
            return {"attempts": attempts_payload}
        return {}

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "run_id": self.run_id,
            "goal": self.goal,
            "hypotheses": json.loads(json.dumps(self.hypotheses)),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }
        if self.input_provenance is not None:
            payload["input_provenance"] = json.loads(json.dumps(self.input_provenance))
        if self.context_features is not None:
            payload["context_features"] = json.loads(json.dumps(self.context_features))
        scratchpad = self.scratchpad
        if scratchpad:
            payload["scratchpad"] = scratchpad
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorkingMemory":
        run_id = str(payload.get("run_id", ""))
        goal_value = payload.get("goal")
        goal = goal_value if goal_value is None else str(goal_value)
        raw_hypotheses = payload.get("hypotheses", [])
        hypotheses = (
            list(raw_hypotheses)
            if isinstance(raw_hypotheses, Iterable) and not isinstance(raw_hypotheses, (str, bytes))
            else []
        )
        attempts_payload = payload.get("attempts", [])
        attempts: List[DeliberationAttempt] = []
        if isinstance(attempts_payload, Iterable) and not isinstance(attempts_payload, (str, bytes)):
            for item in attempts_payload:
                if isinstance(item, Mapping):
                    attempts.append(DeliberationAttempt.from_dict(item))
        instance = cls(
            run_id=run_id,
            goal=goal,
            hypotheses=json.loads(json.dumps(hypotheses)),
            attempts=attempts,
        )
        provenance = payload.get("input_provenance")
        if isinstance(provenance, Mapping):
            instance.input_provenance = json.loads(json.dumps(provenance))
        features = payload.get("context_features")
        if isinstance(features, Mapping):
            instance.context_features = json.loads(json.dumps(features))
        return instance


@dataclass
class Orchestrator:
    planner: Planner
    critic: Critic
    tools: Mapping[str, Any]
    memory: MemoryStore
    world_model: WorldModel
    gatekeeper: Gatekeeper | None = None
    working_dir: Path = Path("artifacts")
    telemetry: Telemetry | None = None
    max_replans: int = 2
    memory_retriever: MemoryRetriever | None = None

    _working_memory: WorkingMemory | None = field(init=False, default=None, repr=False)
    _input_provenance: Dict[str, Any] | None = field(init=False, default=None, repr=False)
    _tool_specs: Dict[str, ToolSpec] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.memory_retriever is None:
            self.memory_retriever = MemoryRetriever(self.memory)

    def _emit(self, event: str, **payload: Any) -> None:
        if self.telemetry is not None:
            self.telemetry.emit(event, **payload)

    @property
    def working_memory(self) -> WorkingMemory | None:
        """Return the latest working memory snapshot."""

        return self._working_memory

    def _tool_event_provenance(self, plan: Plan, step: PlanStep) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self._input_provenance:
            payload["inputs"] = json.loads(json.dumps(self._input_provenance))
        payload["plan"] = {
            "id": plan.id,
            "claims": list(plan.claim_ids),
        }
        step_payload: Dict[str, Any] = {"id": step.id}
        if step.goal:
            step_payload["goal"] = step.goal
        if step.description:
            step_payload["description"] = step.description
        payload["step"] = step_payload
        references: Dict[str, Any] = {}
        inputs = payload.get("inputs") or {}
        hypothesis_refs = []
        for item in inputs.get("hypotheses", []) or []:
            if isinstance(item, Mapping):
                ident = item.get("id") or item.get("claim_id")
                if ident:
                    hypothesis_refs.append(str(ident))
        if hypothesis_refs:
            references["hypotheses"] = hypothesis_refs
        memory_inputs = inputs.get("memory") or {}
        memory_refs: List[str] = []
        for key in ("semantic_matches", "recent_records"):
            for record in memory_inputs.get(key, []) or []:
                if isinstance(record, Mapping):
                    ident = record.get("id") or record.get("call_id")
                    if ident:
                        memory_refs.append(str(ident))
        if memory_refs:
            unique_refs = []
            seen: Set[str] = set()
            for ref in memory_refs:
                if ref not in seen:
                    seen.add(ref)
                    unique_refs.append(ref)
            references["memory_records"] = unique_refs
        if references:
            payload["references"] = references
        return payload

    async def run(self, goal_spec: Dict[str, Any], constraints: Dict[str, Any] | None = None) -> Report:
        constraints = constraints or {}
        run_id = uuid.uuid4().hex
        self._emit("orchestrator.run_started", run_id=run_id, goal=goal_spec.get("goal"))
        run_dir = self.working_dir / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        tool_catalog = [
            describe_tool(tool, override_name=name)
            for name, tool in sorted(self.tools.items(), key=lambda item: item[0])
        ]
        self._tool_specs = {spec.name: spec for spec in tool_catalog}

        raw_hypotheses = goal_spec.get("hypotheses", [])
        if raw_hypotheses is None:
            raw_hypotheses = []
        elif not isinstance(raw_hypotheses, list):
            raw_hypotheses = [raw_hypotheses]
        hypotheses = raw_hypotheses
        feedback: List[Dict[str, Any]] = []
        critiques: List[Dict[str, Any]] = []
        memory_entries: List[Dict[str, Any]] = []
        plans: List[Plan] = []
        tool_results: List[ToolResult] = []
        plan_results: Dict[str, List[ToolResult]] = {}
        safety_audit: List[SafetyDecision] = []
        risk_assessments: List[RiskAssessment] = []

        self._working_memory = WorkingMemory(
            run_id=run_id,
            goal=str(goal_spec.get("goal")) if goal_spec.get("goal") is not None else None,
            hypotheses=json.loads(json.dumps(hypotheses)),
        )
        self._input_provenance = None

        memory_context_payload: Dict[str, Any] | None = None
        if self.memory_retriever is not None:
            goal_text = goal_spec.get("goal")
            if isinstance(goal_text, str):
                context_limit = int(goal_spec.get("memory_context_limit", 0)) or None
                recent_limit = int(goal_spec.get("memory_recent_limit", 0)) or None
                context = self.memory_retriever.context_for_goal(
                    goal_text,
                    limit=context_limit,
                    recent=recent_limit,
                )
                semantic_matches = context.get("semantic", {}).get("matches", []) if context.get("semantic") else []
                recent_records = context.get("recent", {}).get("records", []) if context.get("recent") else []
                if semantic_matches or recent_records:
                    memory_context_payload = context
                    features = extract_memory_features(memory_context_payload)
                    if features:
                        memory_context_payload.setdefault("features", features)
                        if self._working_memory is not None:
                            self._working_memory.context_features = json.loads(json.dumps(features))
                    self._emit(
                        "orchestrator.memory_context_ready",
                        run_id=run_id,
                        semantic_matches=len(semantic_matches),
                        recent_records=len(recent_records),
                        features=features or None,
                    )

        self._input_provenance = _build_input_provenance(
            goal_spec,
            hypotheses,
            memory_context_payload,
        )
        if not self._input_provenance:
            self._input_provenance = None
        if self._working_memory is not None and self._input_provenance is not None:
            self._working_memory.input_provenance = json.loads(
                json.dumps(self._input_provenance)
            )
        if self._input_provenance:
            self._emit(
                "orchestrator.input_provenance_ready",
                run_id=run_id,
                provenance=self._input_provenance,
            )

        for attempt in range(self.max_replans + 1):
            working_attempt = self._working_memory.start_attempt(attempt)
            working_attempt.set_hypotheses(hypotheses)
            self._emit(
                "orchestrator.working_memory.hypotheses",
                run_id=run_id,
                attempt=attempt,
                count=len(hypotheses),
            )
            plans = await self.planner.plan_from(
                hypotheses,
                feedback=feedback,
                memory_context=memory_context_payload,
            )
            plan_critiques: List[Dict[str, Any]] = []
            replan_required = False
            for plan in plans:
                plan_trace = working_attempt.register_plan(plan)
                self._emit(
                    "orchestrator.plan_ready",
                    run_id=run_id,
                    plan_id=plan.id,
                    step_count=len(plan.steps),
                )
                critique = await self.critic.check(_plan_to_dict(plan))
                status = str(critique.get("status", "FAIL")).upper()
                if status == "PASS":
                    plan_trace.mark_approved()
                    continue
                critique_record = _normalise_critique(plan.id, critique)
                critiques.append(critique_record)
                working_attempt.record_critique(plan.id, critique_record)
                self._emit(
                    "orchestrator.working_memory.critique",
                    run_id=run_id,
                    attempt=attempt,
                    plan_id=plan.id,
                    tags=critique_record.get("rationale_tags"),
                )
                memory_entries.append(
                    _critique_memory_record(
                        critique_record,
                        default_time=goal_spec.get(
                            "time", "1970-01-01T00:00:00+00:00"
                        ),
                    )
                )
                if status == "FAIL" and not (
                    critique_record.get("amendments")
                    or critique_record.get("issues")
                ):
                    raise RuntimeError(f"Plan {plan.id} rejected: {critique}")
                plan_critiques.append(critique_record)
                replan_required = True
            if replan_required:
                working_attempt.mark_status("needs_replan")
                feedback.extend(
                    json.loads(json.dumps(record)) for record in plan_critiques
                )
                continue

            if self.gatekeeper is not None:
                safety_audit = enforce_plan_safety(plans, self.tools, self.gatekeeper)

            attempt_tool_results: List[ToolResult] = []
            attempt_plan_results: Dict[str, List[ToolResult]] = {}
            attempt_result_indices: Dict[str, Dict[str, ToolResult]] = {}
            attempt_memory_entries: List[Dict[str, Any]] = []

            working_attempt.mark_status("executing")

            for plan in plans:
                per_plan_results: List[ToolResult] = []
                attempt_plan_results[plan.id] = per_plan_results
                result_index: Dict[str, ToolResult] = {}
                attempt_result_indices[plan.id] = result_index
                await _execute_plan_steps(
                    orchestrator=self,
                    plan=plan,
                    steps=plan.steps,
                    goal_spec=goal_spec,
                    constraints=constraints,
                    run_dir=run_dir,
                    per_plan_results=per_plan_results,
                    all_results=attempt_tool_results,
                    memory_entries=attempt_memory_entries,
                    result_index=result_index,
                    run_id=run_id,
                    risk_assessments=risk_assessments,
                    working_attempt=working_attempt,
                )

            memory_entries.extend(attempt_memory_entries)

            plan_failure_map = {
                plan.id: any(not res.ok for res in attempt_plan_results.get(plan.id, []))
                for plan in plans
            }
            failing_plan_ids = {plan_id for plan_id, failed in plan_failure_map.items() if failed}

            if failing_plan_ids and len(failing_plan_ids) == len(plan_failure_map):
                execution_feedback = _build_execution_feedback(
                    plans, attempt_result_indices, failing_plan_ids
                )
                for item in execution_feedback:
                    self._emit(
                        "orchestrator.plan_execution_failed",
                        run_id=run_id,
                        plan_id=item["plan_id"],
                        failure_count=len(item.get("failures", [])),
                    )
                working_attempt.record_execution_feedback(execution_feedback)
                working_attempt.mark_status("retry")
                feedback.extend(json.loads(json.dumps(item)) for item in execution_feedback)
                continue

            tool_results = attempt_tool_results
            plan_results = attempt_plan_results
            working_attempt.update_execution_results(
                {plan.id: not plan_failure_map.get(plan.id, False) for plan in plans}
            )
            working_attempt.mark_status("complete")
            break
        else:
            raise RuntimeError(
                "Exceeded maximum replanning attempts without approval or execution success"
            )

        for entry in memory_entries:
            self.memory.append(entry)

        update_payloads: List[Dict[str, Any]] = []
        for plan in plans:
            claim_ids = list(plan.claim_ids) or [plan.id]
            plan_ok = all(res.ok for res in plan_results.get(plan.id, []))
            provenance = [
                asdict(src)
                for res in plan_results.get(plan.id, [])
                for src in res.provenance
            ]
            for claim_id in claim_ids:
                update_payloads.append(
                    {
                        "claim_id": claim_id,
                        "passed": plan_ok,
                        "provenance": provenance,
                        "expected_unit": goal_spec.get("expected_unit"),
                        "observed_unit": goal_spec.get(
                            "observed_unit", goal_spec.get("expected_unit")
                        ),
                    }
                )

        updates = self.world_model.update(update_payloads)
        for belief in updates:
            self._emit(
                "orchestrator.belief_updated",
                run_id=run_id,
                claim_id=belief.claim_id,
                credence=belief.credence,
            )

        manifest = RunManifest.build(
            run_id=run_id,
            goal=goal_spec,
            constraints=constraints,
            tool_results=tool_results,
            belief_updates=updates,
            safety_audit=safety_audit,
            risk_assessments=risk_assessments,
            critiques=critiques,
            tool_catalog=tool_catalog,
        )
        manifest.write(run_dir / "manifest.json")
        RunManifest.write_schema(run_dir / "manifest.schema.json")

        consolidate = goal_spec.get("goal")
        reflection_summary = consolidate_reflections(
            self.memory,
            goal=consolidate,
            write_back=True,
            world_model=self.world_model,
            limit=100,
        )
        summary_path: Path | None = None
        if reflection_summary.get("sample_size"):
            summary_path = run_dir / "reflection_summary.json"
            summary_path.write_text(json.dumps(reflection_summary, indent=2), encoding="utf-8")
            self._emit(
                "orchestrator.reflection_consolidated",
                run_id=run_id,
                summary=reflection_summary,
            )

        working_memory_snapshot = None
        working_memory_path = run_dir / "working_memory.json"
        if self._working_memory is not None:
            working_memory_snapshot = self._working_memory.to_dict()
            working_memory_path.write_text(
                json.dumps(working_memory_snapshot, indent=2),
                encoding="utf-8",
            )
            self._emit(
                "orchestrator.working_memory_persisted",
                run_id=run_id,
                path=_safe_relpath(working_memory_path),
            )

        success = all(result.ok for result in tool_results) if tool_results else True
        artifacts = [_safe_relpath(run_dir)]
        if summary_path is not None:
            artifacts.append(_safe_relpath(summary_path))
        if working_memory_snapshot is not None:
            artifacts.append(_safe_relpath(working_memory_path))
        report = Report(
            goal=goal_spec.get("goal", ""),
            summary="Completed run",
            key_findings=[result.stdout or "" for result in tool_results],
            belief_deltas=updates,
            artifacts=artifacts,
        )
        try:
            experience_chunk = summarise_experience(
                report,
                manifest,
                working_memory=self._working_memory,
            )
            experience_entry = {
                "type": "experience_replay",
                "time": manifest.created_at or goal_spec.get("time") or _now_iso(),
                "goal": experience_chunk.get("goal"),
                "summary": experience_chunk,
            }
            self.memory.append(experience_entry)
            self._emit(
                "orchestrator.experience_recorded",
                run_id=run_id,
                goal=experience_entry.get("goal"),
            )
        except Exception:  # pragma: no cover - defensive
            pass
        self._emit(
            "orchestrator.run_completed",
            run_id=run_id,
            success=success,
            tool_invocations=len(tool_results),
        )
        return report


def _plan_to_dict(plan: Plan) -> Dict[str, Any]:
    return {
        "id": plan.id,
        "claim_ids": plan.claim_ids,
        "steps": [_plan_step_to_dict(step) for step in plan.steps],
        "expected_cost": plan.expected_cost,
        "risks": plan.risks,
        "ablations": plan.ablations,
    }


def _normalise_critique(plan_id: str, critique: Mapping[str, Any]) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "plan_id": plan_id,
        "status": str(critique.get("status", "FAIL")).upper(),
        "reviewer": str(critique.get("reviewer", "critic")),
    }
    for key in ("notes", "summary"):
        value = critique.get(key)
        if value:
            record[key] = str(value)
    amendments = critique.get("amendments")
    if amendments:
        record["amendments"] = [str(item) for item in amendments]
    issues = critique.get("issues")
    if issues:
        record["issues"] = [str(item) for item in issues]
    tags = _derive_rationale_tags(critique)
    if tags:
        record["rationale_tags"] = tags
    return record


def _critique_memory_record(
    critique: Mapping[str, Any], *, default_time: str
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "type": "critique",
        "tool": critique.get("reviewer", "critic"),
        "time": default_time,
        "plan_id": critique.get("plan_id"),
        "status": critique.get("status"),
    }
    for key in ("notes", "summary", "amendments", "issues"):
        value = critique.get(key)
        if value:
            entry[key] = value
    if critique.get("rationale_tags"):
        entry["rationale_tags"] = critique["rationale_tags"]
    return entry


def _derive_rationale_tags(critique: Mapping[str, Any]) -> List[str]:
    tags: Set[str] = set()
    for tag in critique.get("rationale_tags", []) or []:
        text = str(tag).strip()
        if text:
            tags.add(text)
    if critique.get("amendments"):
        tags.add("amendment")
    if critique.get("issues"):
        tags.add("issue")
    summary = str(critique.get("summary", "") or "")
    notes = str(critique.get("notes", "") or "")
    text_blob = f"{summary}\n{notes}".lower()
    if "safety" in text_blob:
        tags.add("safety")
    if "alignment" in text_blob:
        tags.add("alignment")
    return sorted(tags)


async def _execute_plan_steps(
    *,
    orchestrator: "Orchestrator",
    plan: Plan,
    steps: Iterable[PlanStep],
    goal_spec: Dict[str, Any],
    constraints: Dict[str, Any],
    run_dir: Path,
    per_plan_results: List[ToolResult],
    all_results: List[ToolResult],
    memory_entries: List[Dict[str, Any]],
    result_index: Dict[str, ToolResult],
    run_id: str,
    risk_assessments: List[RiskAssessment],
    working_attempt: DeliberationAttempt,
) -> None:
    for step in steps:
        await _execute_plan_step(
            orchestrator=orchestrator,
            plan=plan,
            step=step,
            goal_spec=goal_spec,
            constraints=constraints,
            run_dir=run_dir,
            per_plan_results=per_plan_results,
            all_results=all_results,
            memory_entries=memory_entries,
            result_index=result_index,
            run_id=run_id,
            risk_assessments=risk_assessments,
            working_attempt=working_attempt,
        )


async def _execute_plan_step(
    *,
    orchestrator: "Orchestrator",
    plan: Plan,
    step: PlanStep,
    goal_spec: Dict[str, Any],
    constraints: Dict[str, Any],
    run_dir: Path,
    per_plan_results: List[ToolResult],
    all_results: List[ToolResult],
    memory_entries: List[Dict[str, Any]],
    result_index: Dict[str, ToolResult],
    run_id: str,
    risk_assessments: List[RiskAssessment],
    working_attempt: DeliberationAttempt,
) -> None:
    if step.tool:
        tool = orchestrator.tools.get(step.tool)
        if tool is None:
            raise KeyError(f"Unknown tool {step.tool}")
        if orchestrator.gatekeeper is not None:
            assessment = assess_step_risk(plan, step, orchestrator.tools, orchestrator.gatekeeper)
            risk_assessments.append(assessment)
            working_attempt.record_risk_assessment(assessment)
            orchestrator._emit(
                "orchestrator.risk_assessed",
                run_id=run_id,
                plan_id=plan.id,
                step_id=step.id,
                tool=step.tool,
                approved=assessment.approved,
                effective_level=assessment.effective_level,
                reason=assessment.reason,
            )
            risk_entry: Dict[str, Any] = {
                "type": "risk_assessment",
                "plan_id": plan.id,
                "step_id": step.id,
                "tool": step.tool,
                "approved": assessment.approved,
                "requested_level": assessment.requested_level,
                "tool_level": assessment.tool_level,
                "effective_level": assessment.effective_level,
                "time": goal_spec.get("time", "1970-01-01T00:00:00+00:00"),
            }
            if assessment.reason:
                risk_entry["reason"] = assessment.reason
            memory_entries.append(risk_entry)
            if not assessment.approved:
                failure = ToolResult(
                    call_id=step.id,
                    ok=False,
                    stdout=assessment.reason or "Risk assessment blocked execution",
                )
                per_plan_results.append(failure)
                all_results.append(failure)
                result_index[step.id] = failure
                result_index[failure.call_id] = failure
                orchestrator._emit(
                    "orchestrator.tool_blocked",
                    run_id=run_id,
                    plan_id=plan.id,
                    step_id=step.id,
                    tool=step.tool,
                    effective_level=assessment.effective_level,
                    reason=assessment.reason,
                )
                return
        ctx = RunContext(
            working_dir=str(run_dir),
            timeout_s=int(constraints.get("timeout_s", goal_spec.get("timeout_s", 60))),
            env_whitelist=list(goal_spec.get("env", [])),
            network=constraints.get("network", "off"),
            record_provenance=True,
        )
        provenance_payload = orchestrator._tool_event_provenance(plan, step)
        references_payload = provenance_payload.get("references")
        orchestrator._emit(
            "orchestrator.tool_started",
            run_id=run_id,
            plan_id=plan.id,
            step_id=step.id,
            tool=step.tool,
            provenance=provenance_payload,
            references=references_payload,
        )
        result = await tool.run({**step.args, "id": step.id}, ctx)
        per_plan_results.append(result)
        all_results.append(result)
        result_index[step.id] = result
        result_index[result.call_id] = result
        sensor_metadata: Dict[str, Any] | None = None
        safety_tier: str | None = None
        tool_spec = orchestrator._tool_specs.get(step.tool) if step.tool else None
        if tool_spec is not None:
            safety_tier = tool_spec.safety_tier
            profile = tool_spec.sensor_profile
            if profile is not None:
                sensor_metadata = {
                    "modality": profile.modality,
                    "latency_ms": profile.latency_ms,
                    "trust": profile.trust,
                    "description": profile.description,
                }

        orchestrator._emit(
            "orchestrator.tool_completed",
            run_id=run_id,
            plan_id=plan.id,
            step_id=step.id,
            tool=step.tool,
            ok=result.ok,
            wall_time_ms=result.wall_time_ms,
            provenance=provenance_payload,
            sensor=sensor_metadata,
            safety_tier=safety_tier,
            references=references_payload,
        )
        output_summary = summarise_tool_result(result)
        episode_entry: Dict[str, Any] = {
            "type": "episode",
            "plan_id": plan.id,
            "tool": step.tool,
            "call_id": result.call_id,
            "ok": result.ok,
            "stdout": result.stdout,
            "time": goal_spec.get("time", "1970-01-01T00:00:00+00:00"),
            "provenance": [asdict(src) for src in result.provenance],
            "summary": output_summary.get("summary"),
            "tokens": output_summary.get("tokens"),
            "latency_ms": result.wall_time_ms,
            "sensor": sensor_metadata,
            "safety_tier": safety_tier,
            "keywords": output_summary.get("keywords"),
            "structured": output_summary.get("structured"),
            "claims": output_summary.get("claims"),
            "embedding": output_summary.get("embedding"),
        }
        if episode_entry["summary"] is None:
            episode_entry.pop("summary")
        if episode_entry["tokens"] is None:
            episode_entry.pop("tokens")
        if not episode_entry.get("keywords"):
            episode_entry.pop("keywords", None)
        if not episode_entry.get("structured"):
            episode_entry.pop("structured", None)
        if not episode_entry.get("claims"):
            episode_entry.pop("claims", None)
        if not episode_entry.get("embedding"):
            episode_entry.pop("embedding", None)
        memory_entries.append(episode_entry)

    if step.sub_steps:
        await _execute_plan_steps(
            orchestrator=orchestrator,
            plan=plan,
            steps=step.sub_steps,
            goal_spec=goal_spec,
            constraints=constraints,
            run_dir=run_dir,
            per_plan_results=per_plan_results,
            all_results=all_results,
            memory_entries=memory_entries,
            result_index=result_index,
            run_id=run_id,
            risk_assessments=risk_assessments,
            working_attempt=working_attempt,
        )

    for branch_index, branch in enumerate(step.branches):
        condition_payload = branch.condition.to_payload()
        branch_metadata = {
            "index": branch_index,
            "condition": condition_payload,
            "next_steps": [child.id for child in branch.steps],
        }
        branch_taken = _should_execute_branch(branch.condition, result_index)
        working_attempt.record_branch_decision(
            plan_id=plan.id,
            step_id=step.id,
            metadata=branch_metadata,
            taken=branch_taken,
        )
        orchestrator._emit(
            "orchestrator.working_memory.branch",
            run_id=run_id,
            plan_id=plan.id,
            step_id=step.id,
            branch_index=branch_index,
            taken=branch_taken,
            condition=condition_payload,
        )
        if branch_taken:
            await _execute_plan_steps(
                orchestrator=orchestrator,
                plan=plan,
                steps=branch.steps,
                goal_spec=goal_spec,
                constraints=constraints,
                run_dir=run_dir,
                per_plan_results=per_plan_results,
                all_results=all_results,
                memory_entries=memory_entries,
                result_index=result_index,
                run_id=run_id,
                risk_assessments=risk_assessments,
                working_attempt=working_attempt,
            )


def _should_execute_branch(
    condition: BranchCondition, results: Mapping[str, ToolResult]
) -> bool:
    return condition.evaluate(results)


def _plan_step_to_dict(step: PlanStep) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "id": step.id,
        "args": step.args,
        "safety_level": step.safety_level,
    }
    if step.tool is not None:
        payload["tool"] = step.tool
    if step.description is not None:
        payload["description"] = step.description
    if step.goal is not None:
        payload["goal"] = step.goal
    if step.sub_steps:
        payload["sub_steps"] = [_plan_step_to_dict(child) for child in step.sub_steps]
    if step.branches:
        payload["branches"] = [
            {
                "condition": branch.condition.to_payload(),
                "steps": [_plan_step_to_dict(child) for child in branch.steps],
            }
            for branch in step.branches
        ]
    return payload


def _build_execution_feedback(
    plans: Iterable[Plan],
    result_indices: Mapping[str, Mapping[str, ToolResult]],
    failing_plan_ids: Set[str],
) -> List[Dict[str, Any]]:
    feedback: List[Dict[str, Any]] = []
    for plan in plans:
        if plan.id not in failing_plan_ids:
            continue
        plan_failures: List[Dict[str, Any]] = []
        plan_results = result_indices.get(plan.id, {})
        for step in plan.iter_tool_calls():
            result = plan_results.get(step.id)
            if result is None or result.ok:
                continue
            plan_failures.append(
                {
                    "step_id": step.id,
                    "tool": step.tool,
                    "status": "FAILED",
                    "stdout": result.stdout,
                }
            )
        if plan_failures:
            feedback.append(
                {
                    "plan_id": plan.id,
                    "status": "FAILED_EXECUTION",
                    "failures": plan_failures,
                }
            )
    return feedback


def _safe_relpath(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def load_working_memory_snapshot(path: Path | str) -> WorkingMemory:
    """Load a persisted working memory snapshot from disk."""

    snapshot_path = Path(path)
    data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("working memory snapshot must be a JSON object")
    return WorkingMemory.from_dict(data)
