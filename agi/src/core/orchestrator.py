from __future__ import annotations

import json
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

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
from .types import (
    AgentProfile,
    Belief,
    BranchCondition,
    NegotiationMessage,
    Plan,
    PlanStep,
    Report,
    RunContext,
    ToolResult,
)
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


def _negotiation_memory_records(
    messages: Sequence[NegotiationMessage],
    *,
    run_id: str,
    goal: str | None,
) -> List[Dict[str, Any]]:
    if not messages:
        return []
    goal_text = goal.strip() if isinstance(goal, str) else (str(goal) if goal else None)
    pair_counts: Counter[tuple[str, str]] = Counter()
    kind_counts: Counter[str] = Counter()
    entries: List[Dict[str, Any]] = []
    for message in messages:
        payload = message.to_dict()
        timestamp = payload.get("timestamp") or _now_iso()
        sender = payload.get("sender") or "unknown"
        recipient = payload.get("recipient") or "unknown"
        kind = payload.get("kind") or "unspecified"
        pair_counts[(sender, recipient)] += 1
        kind_counts[kind.lower()] += 1
        keywords = [
            "negotiation",
            f"{sender}->{recipient}",
            sender,
            recipient,
            f"kind:{kind}",
        ]
        entry: Dict[str, Any] = {
            "type": "negotiation",
            "time": timestamp,
            "run_id": run_id,
            "goal": goal_text,
            "sender": sender,
            "recipient": recipient,
            "kind": kind,
            "agents": [agent for agent in {sender, recipient} if agent],
            "content": payload.get("content") or {},
            "outcomes": payload.get("outcomes") or {},
            "keywords": [token for token in keywords if token],
        }
        entries.append(entry)
    summary_time = entries[-1]["time"] if entries else _now_iso()
    pair_summary = [
        {"pair": f"{pair[0]}->{pair[1]}", "count": count}
        for pair, count in pair_counts.most_common(5)
    ]
    kind_summary = [
        {"kind": kind, "count": count} for kind, count in kind_counts.most_common()
    ]
    agent_set = sorted({msg.sender for msg in messages} | {msg.recipient for msg in messages})
    summary_keywords = ["negotiation_summary"]
    summary_keywords.extend(pair["pair"] for pair in pair_summary[:3])
    summary_entry: Dict[str, Any] = {
        "type": "negotiation_summary",
        "time": summary_time,
        "run_id": run_id,
        "goal": goal_text,
        "negotiation_count": len(messages),
        "pairs": pair_summary,
        "kinds": kind_summary,
        "agents": agent_set,
        "keywords": summary_keywords,
    }
    entries.append(summary_entry)
    return entries


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
class PlanBranchTrace:
    """Execution metadata for a conditional branch within a plan step."""

    index: int
    condition: Any
    steps: List["PlanStepTrace"] = field(default_factory=list)
    taken: Optional[bool] = None

    def mark_decision(self, taken: bool) -> None:
        self.taken = taken
        if not taken:
            for step in self.steps:
                step.mark_skipped()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "condition": self.condition,
            "taken": self.taken,
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanBranchTrace":
        index = int(payload.get("index", 0))
        condition = payload.get("condition")
        taken = payload.get("taken")
        steps_payload = payload.get("steps", [])
        steps: List["PlanStepTrace"] = []
        if isinstance(steps_payload, Iterable) and not isinstance(steps_payload, (str, bytes)):
            for item in steps_payload:
                if isinstance(item, Mapping):
                    steps.append(PlanStepTrace.from_dict(item))
        instance = cls(index=index, condition=condition, steps=steps, taken=taken)
        if taken is False:
            for step in instance.steps:
                step.status = "skipped"
        return instance


@dataclass
class PlanStepTrace:
    """Runtime trace for a single plan step."""

    step_id: str
    goal: Optional[str]
    tool: Optional[str]
    agent: Optional[str]
    kind: str
    depth: int
    parent_id: Optional[str] = None
    branch_index: Optional[int] = None
    branch_condition: Any = None
    status: str = "pending"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: Optional[float] = None
    failure_reason: Optional[str] = None
    children: List["PlanStepTrace"] = field(default_factory=list)
    branches: List[PlanBranchTrace] = field(default_factory=list)
    _runtime_started: Optional[float] = field(default=None, init=False, repr=False)

    def mark_started(self) -> None:
        if self.status in {"pending", "skipped"}:
            self.status = "running"
            self.started_at = datetime.now(timezone.utc).isoformat()
            self._runtime_started = perf_counter()

    def mark_completed(self, *, succeeded: bool, reason: Optional[str] = None) -> None:
        if self.status == "completed":
            return
        if self.status != "running":
            self.mark_started()
        self.status = "succeeded" if succeeded else "failed"
        self.completed_at = datetime.now(timezone.utc).isoformat()
        if self._runtime_started is not None:
            self.duration_ms = round((perf_counter() - self._runtime_started) * 1000, 3)
        if not succeeded:
            self.failure_reason = reason

    def mark_skipped(self) -> None:
        if self.status == "completed":
            return
        self.status = "skipped"
        self.started_at = self.started_at or datetime.now(timezone.utc).isoformat()
        self.completed_at = self.completed_at or self.started_at
        self.duration_ms = self.duration_ms or 0.0
        for child in self.children:
            child.mark_skipped()
        for branch in self.branches:
            branch.mark_decision(False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.step_id,
            "goal": self.goal,
            "tool": self.tool,
            "agent": self.agent,
            "kind": self.kind,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "branch_index": self.branch_index,
            "branch_condition": self.branch_condition,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "failure_reason": self.failure_reason,
            "children": [child.to_dict() for child in self.children],
            "branches": [branch.to_dict() for branch in self.branches],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanStepTrace":
        instance = cls(
            step_id=str(payload.get("id", "")),
            goal=payload.get("goal"),
            tool=payload.get("tool"),
            agent=payload.get("agent"),
            kind=str(payload.get("kind", "atomic")),
            depth=int(payload.get("depth", 0)),
            parent_id=payload.get("parent_id"),
            branch_index=payload.get("branch_index"),
            branch_condition=payload.get("branch_condition"),
            status=str(payload.get("status", "pending")),
            started_at=payload.get("started_at"),
            completed_at=payload.get("completed_at"),
            duration_ms=payload.get("duration_ms"),
            failure_reason=payload.get("failure_reason"),
        )
        children_payload = payload.get("children", [])
        if isinstance(children_payload, Iterable) and not isinstance(children_payload, (str, bytes)):
            instance.children = [
                PlanStepTrace.from_dict(item)
                for item in children_payload
                if isinstance(item, Mapping)
            ]
        branch_payload = payload.get("branches", [])
        if isinstance(branch_payload, Iterable) and not isinstance(branch_payload, (str, bytes)):
            instance.branches = [
                PlanBranchTrace.from_dict(item)
                for item in branch_payload
                if isinstance(item, Mapping)
            ]
        return instance

    def iter_all(self) -> Iterable["PlanStepTrace"]:
        yield self
        for child in self.children:
            yield from child.iter_all()
        for branch in self.branches:
            for step in branch.steps:
                yield from step.iter_all()

    def get_branch(self, index: int) -> Optional[PlanBranchTrace]:
        for branch in self.branches:
            if branch.index == index:
                return branch
        return None


@dataclass
class PlanTrace:
    """Snapshot of deliberation information for a single plan."""

    plan_id: str
    claim_ids: List[str]
    step_count: int
    approved: Optional[bool] = None
    execution_succeeded: Optional[bool] = None
    rationale_tags: Set[str] = field(default_factory=set)
    steps: List[PlanStepTrace] = field(default_factory=list)
    _step_index: Dict[str, PlanStepTrace] = field(default_factory=dict, init=False, repr=False)

    def attach_structure(self, plan_steps: Sequence[PlanStep]) -> None:
        self.steps = [_build_step_trace(step, depth=0, parent_id=None) for step in plan_steps]
        self._step_index.clear()
        for root in self.steps:
            for trace in root.iter_all():
                self._step_index[trace.step_id] = trace

    def get_step(self, step_id: str) -> Optional[PlanStepTrace]:
        return self._step_index.get(step_id)

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

    def mark_step_started(self, step_id: str) -> None:
        trace = self._step_index.get(step_id)
        if trace is not None:
            trace.mark_started()

    def mark_step_completed(self, step_id: str, *, succeeded: bool, reason: Optional[str] = None) -> None:
        trace = self._step_index.get(step_id)
        if trace is not None:
            trace.mark_completed(succeeded=succeeded, reason=reason)

    def mark_step_skipped(self, step_id: str) -> None:
        trace = self._step_index.get(step_id)
        if trace is not None:
            trace.mark_skipped()

    def record_branch_decision(self, step_id: str, branch_index: int, taken: bool) -> None:
        trace = self._step_index.get(step_id)
        if trace is None:
            return
        branch = trace.get_branch(branch_index)
        if branch is not None:
            branch.mark_decision(taken)
            if taken is False:
                for child in branch.steps:
                    child.mark_skipped()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "claim_ids": list(self.claim_ids),
            "step_count": self.step_count,
            "approved": self.approved,
            "execution_succeeded": self.execution_succeeded,
            "rationale_tags": sorted(self.rationale_tags),
            "steps": [step.to_dict() for step in self.steps],
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
        steps_payload = payload.get("steps", [])
        steps: List[PlanStepTrace] = []
        if isinstance(steps_payload, Iterable) and not isinstance(steps_payload, (str, bytes)):
            for item in steps_payload:
                if isinstance(item, Mapping):
                    steps.append(PlanStepTrace.from_dict(item))
        instance = cls(
            plan_id=plan_id,
            claim_ids=claim_ids,
            step_count=step_count,
            approved=approved,
            execution_succeeded=execution_succeeded,
            rationale_tags=rationale_tags,
            steps=steps,
        )
        for root in instance.steps:
            for trace in root.iter_all():
                instance._step_index[trace.step_id] = trace
        return instance

    def to_manifest(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "claim_ids": list(self.claim_ids),
            "approved": self.approved,
            "execution_succeeded": self.execution_succeeded,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass
class AgentRuntime:
    """Execution context for a collaborating agent."""

    name: str
    tools: Mapping[str, Any]
    memory: MemoryStore


def _infer_step_kind(step: PlanStep) -> str:
    if step.tool and step.sub_steps:
        return "tool_with_substeps"
    if step.tool:
        return "tool"
    if step.sub_steps:
        return "composite"
    if step.branches:
        return "branch"
    return "noop"


def _build_step_trace(
    step: PlanStep,
    *,
    depth: int,
    parent_id: Optional[str],
    branch_index: Optional[int] = None,
    branch_condition: Any = None,
) -> PlanStepTrace:
    trace = PlanStepTrace(
        step_id=step.id,
        goal=step.goal,
        tool=step.tool,
        agent=step.agent,
        kind=_infer_step_kind(step),
        depth=depth,
        parent_id=parent_id,
        branch_index=branch_index,
        branch_condition=branch_condition,
    )
    trace.children = [
        _build_step_trace(child, depth=depth + 1, parent_id=step.id)
        for child in step.sub_steps
    ]
    trace.branches = []
    for idx, branch in enumerate(step.branches):
        branch_payload = branch.condition.to_payload()
        branch_trace = PlanBranchTrace(
            index=idx,
            condition=branch_payload,
            steps=[
                _build_step_trace(
                    child,
                    depth=depth + 1,
                    parent_id=step.id,
                    branch_index=idx,
                    branch_condition=branch_payload,
                )
                for child in branch.steps
            ],
        )
        trace.branches.append(branch_trace)
    return trace


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
    negotiations: List[Dict[str, Any]] = field(default_factory=list)
    _plan_index: Dict[str, PlanTrace] = field(default_factory=dict, init=False, repr=False)

    def register_plan(self, plan: Plan) -> PlanTrace:
        trace = PlanTrace(
            plan_id=plan.id,
            claim_ids=list(plan.claim_ids),
            step_count=sum(1 for _ in plan.iter_tool_calls()) or len(plan.steps),
        )
        trace.attach_structure(plan.steps)
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
        plan = self._plan_index.get(plan_id)
        if plan is not None and metadata is not None:
            branch_index = metadata.get("index")
            if branch_index is not None:
                try:
                    branch_index_int = int(branch_index)
                except (TypeError, ValueError):
                    branch_index_int = None
            else:
                branch_index_int = None
            if branch_index_int is not None:
                plan.record_branch_decision(step_id=step_id, branch_index=branch_index_int, taken=taken)

    def record_negotiation(self, message: NegotiationMessage) -> None:
        self.negotiations.append(message.to_dict())

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
            "negotiations": [dict(item) for item in self.negotiations],
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
        negotiation_payload = payload.get("negotiations", [])
        if isinstance(negotiation_payload, Iterable) and not isinstance(negotiation_payload, (str, bytes)):
            attempt.negotiations = [
                dict(item) for item in negotiation_payload if isinstance(item, Mapping)
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
    negotiations: List[Dict[str, Any]] = field(default_factory=list)

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
            if attempt.negotiations:
                entry["negotiations"] = [dict(item) for item in attempt.negotiations[-5:]]
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
        if self.negotiations:
            payload["negotiations"] = [dict(item) for item in self.negotiations]
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
        negotiations_payload = payload.get("negotiations", [])
        if isinstance(negotiations_payload, Iterable) and not isinstance(negotiations_payload, (str, bytes)):
            instance.negotiations = [
                dict(item) for item in negotiations_payload if isinstance(item, Mapping)
            ]
        return instance


@dataclass
class Orchestrator:
    planner: Planner
    critic: Critic
    tools: Mapping[str, Any]
    memory: MemoryStore
    world_model: WorldModel
    agents: Mapping[str, Mapping[str, Any]] | None = None
    gatekeeper: Gatekeeper | None = None
    working_dir: Path = Path("artifacts")
    telemetry: Telemetry | None = None
    max_replans: int = 2
    memory_retriever: MemoryRetriever | None = None

    _working_memory: WorkingMemory | None = field(init=False, default=None, repr=False)
    _input_provenance: Dict[str, Any] | None = field(init=False, default=None, repr=False)
    _tool_specs: Dict[str, ToolSpec] = field(init=False, default_factory=dict, repr=False)
    _agents: Dict[str, AgentRuntime] = field(init=False, default_factory=dict, repr=False)
    _global_tools: Dict[str, Any] = field(init=False, default_factory=dict, repr=False)
    _agent_profiles: List[AgentProfile] = field(init=False, default_factory=list, repr=False)
    _negotiation_log: List[NegotiationMessage] = field(init=False, default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.memory_retriever is None:
            self.memory_retriever = MemoryRetriever(self.memory)
        self._agents = {}
        self._register_agent("default", tools=self.tools, memory=self.memory)
        for name, spec in (self.agents or {}).items():
            if not name:
                continue
            tools_obj = spec.get("tools") if isinstance(spec, Mapping) else None  # type: ignore[assignment]
            tools_mapping = tools_obj if isinstance(tools_obj, Mapping) else self.tools
            memory_override = spec.get("memory") if isinstance(spec, Mapping) else None  # type: ignore[assignment]
            memory_store = memory_override if isinstance(memory_override, MemoryStore) else self.memory
            self._register_agent(str(name), tools=tools_mapping, memory=memory_store)
        self._recalculate_global_tools()
        self._agent_profiles = []
        self._negotiation_log = []

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
        if step.agent:
            step_payload["agent"] = step.agent
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

    def _record_belief_updates(
        self,
        payloads: Iterable[Mapping[str, Any]],
        *,
        run_id: str,
        context: str,
    ) -> List[Belief]:
        payload_list = [
            json.loads(json.dumps(payload))
            for payload in payloads
            if payload
        ]
        if not payload_list:
            return []
        updates = self.world_model.update(payload_list)
        for belief in updates:
            self._emit(
                "orchestrator.belief_updated",
                run_id=run_id,
                context=context,
                claim_id=belief.claim_id,
                credence=belief.credence,
                support=belief.support,
                conflict=belief.conflict,
                uncertainty=belief.uncertainty,
                confidence_interval=belief.confidence_interval,
            )
        return updates

    def _log_negotiation(
        self,
        *,
        sender: str,
        recipient: str,
        kind: str,
        content: Mapping[str, Any] | None = None,
        outcomes: Mapping[str, Any] | None = None,
    ) -> None:
        message = NegotiationMessage(
            timestamp=_now_iso(),
            sender=sender,
            recipient=recipient,
            kind=kind,
            content=json.loads(json.dumps(content or {})),
            outcomes=json.loads(json.dumps(outcomes or {})),
        )
        self._negotiation_log.append(message)
        if self._working_memory is not None:
            self._working_memory.negotiations.append(message.to_dict())
            if self._working_memory.attempts:
                self._working_memory.attempts[-1].record_negotiation(message)
        self._emit(
            "orchestrator.negotiation",
            sender=sender,
            recipient=recipient,
            kind=kind,
        )

    def _default_agent_name(self) -> str:
        for profile in self._agent_profiles:
            if profile.default:
                return profile.name
        return "default"

    def _ensure_agent_registered(self, agent_name: str) -> None:
        if agent_name in self._agents:
            return
        base = self._agents.get("default")
        if base is None:
            return
        self._register_agent(agent_name, tools=base.tools, memory=base.memory)
        self._recalculate_global_tools()
        self._emit("orchestrator.agent_auto_registered", agent=agent_name)

    def _bootstrap_plan_negotiations(self, plan: Plan) -> None:
        default_agent = self._default_agent_name()
        agent_to_steps: Dict[str, List[str]] = {}
        for step in plan.iter_tool_calls():
            agent = step.agent or default_agent
            agent_to_steps.setdefault(agent, []).append(step.id)
            self._ensure_agent_registered(agent)
        for agent, steps in agent_to_steps.items():
            if agent == default_agent:
                continue
            self._log_negotiation(
                sender=default_agent,
                recipient=agent,
                kind="delegation",
                content={"plan_id": plan.id, "step_ids": steps},
            )

    def _register_agent(
        self,
        name: str,
        *,
        tools: Mapping[str, Any],
        memory: MemoryStore,
    ) -> None:
        self._agents[name] = AgentRuntime(name=name, tools=tools, memory=memory)

    def _recalculate_global_tools(self) -> None:
        merged: Dict[str, Any] = {}
        for agent in self._agents.values():
            merged.update(agent.tools)
        self._global_tools = merged

    def _resolve_agent_context(self, agent_name: Optional[str]) -> AgentRuntime:
        if agent_name and agent_name in self._agents:
            return self._agents[agent_name]
        fallback_ctx = self._agents.get("default")
        if fallback_ctx is None and self._agents:
            fallback_ctx = next(iter(self._agents.values()))
        if fallback_ctx is None:
            raise RuntimeError("No agents registered for orchestrator")
        if agent_name and agent_name not in self._agents:
            self._log_negotiation(
                sender=fallback_ctx.name,
                recipient=agent_name,
                kind="fallback",
                content={"reason": "agent_unknown"},
                outcomes={"assigned_agent": fallback_ctx.name},
            )
            self._emit("orchestrator.agent_fallback", requested=agent_name, fallback=fallback_ctx.name)
        return fallback_ctx

    async def run(self, goal_spec: Dict[str, Any], constraints: Dict[str, Any] | None = None) -> Report:
        constraints = constraints or {}
        run_id = uuid.uuid4().hex
        self._emit("orchestrator.run_started", run_id=run_id, goal=goal_spec.get("goal"))
        run_dir = self.working_dir / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self._agent_profiles = []
        self._negotiation_log = []

        tool_catalog = [
            describe_tool(tool, override_name=name)
            for name, tool in sorted(self._global_tools.items(), key=lambda item: item[0])
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
                context_start = perf_counter()
                context = self.memory_retriever.context_for_goal(
                    goal_text,
                    limit=context_limit,
                    recent=recent_limit,
                )
                context_duration = round((perf_counter() - context_start) * 1000, 3)
                semantic_matches = context.get("semantic", {}).get("matches", []) if context.get("semantic") else []
                recent_records = context.get("recent", {}).get("records", []) if context.get("recent") else []
                insight_count = len(context.get("insights", []) or [])
                features: Dict[str, Any] | None = None
                if semantic_matches or recent_records or insight_count:
                    memory_context_payload = context
                    features = extract_memory_features(memory_context_payload)
                    if features:
                        memory_context_payload.setdefault("features", features)
                        if self._working_memory is not None:
                            self._working_memory.context_features = json.loads(json.dumps(features))
                else:
                    features = None
                self._emit(
                    "orchestrator.memory_context_ready",
                    run_id=run_id,
                    semantic_matches=len(semantic_matches),
                    recent_records=len(recent_records),
                    insights=insight_count,
                    features=features or None,
                    duration_ms=context_duration,
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
            if attempt == 0:
                planner_agents = self.planner.agent_profiles
                if planner_agents:
                    self._agent_profiles = planner_agents
                    default_ctx = self._agents.get("default")
                    if default_ctx is not None:
                        for profile in planner_agents:
                            if profile.default and profile.name not in self._agents:
                                self._register_agent(profile.name, tools=default_ctx.tools, memory=default_ctx.memory)
                    for profile in planner_agents:
                        if profile.name in self._agents:
                            continue
                        if not profile.tool_names:
                            continue
                        tool_subset = {
                            name: self._global_tools.get(name)
                            for name in profile.tool_names
                            if name in self._global_tools
                        }
                        if tool_subset:
                            self._register_agent(profile.name, tools=tool_subset, memory=self.memory)
                    self._recalculate_global_tools()
            plan_critiques: List[Dict[str, Any]] = []
            replan_required = False
            for plan in plans:
                plan_trace = working_attempt.register_plan(plan)
                self._bootstrap_plan_negotiations(plan)
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
                critique_updates = _critique_updates(
                    plan=plan,
                    critique=critique_record,
                    goal_spec=goal_spec,
                )
                self._record_belief_updates(
                    critique_updates,
                    run_id=run_id,
                    context="critic_review",
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
                safety_audit = enforce_plan_safety(plans, self._global_tools, self.gatekeeper)

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
                failure_updates = _execution_failure_updates(
                    plans=plans,
                    feedback=execution_feedback,
                    goal_spec=goal_spec,
                )
                self._record_belief_updates(
                    failure_updates,
                    run_id=run_id,
                    context="execution_failure",
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
        negotiation_entries = _negotiation_memory_records(
            self._negotiation_log,
            run_id=run_id,
            goal=goal_spec.get("goal"),
        )
        for entry in negotiation_entries:
            self.memory.append(entry)

        update_payloads: List[Dict[str, Any]] = []
        for plan in plans:
            claim_ids = list(plan.claim_ids) or [plan.id]
            plan_tool_results = plan_results.get(plan.id, [])
            plan_ok = all(res.ok for res in plan_tool_results)
            provenance = [
                asdict(src)
                for res in plan_tool_results
                for src in res.provenance
            ]
            plan_critiques = [crit for crit in critiques if crit.get("plan_id") == plan.id]
            plan_safety_checks = [decision for decision in safety_audit if decision.plan_id == plan.id]
            plan_risk_reviews = [
                assessment for assessment in risk_assessments if assessment.plan_id == plan.id
            ]
            plan_confidence = _estimate_plan_confidence(
                plan_tool_results,
                plan_critiques,
                plan_ok,
            )
            plan_evidence_payload = _assemble_plan_evidence(
                plan_tool_results,
                plan_critiques,
                plan_safety_checks,
                plan_risk_reviews,
            )
            success_count = sum(1 for result in plan_tool_results if result.ok)
            total_calls = len(plan_tool_results)
            outcome_note = _summarise_plan_outcome(plan.id, success_count, total_calls, plan_ok)
            for claim_id in claim_ids:
                update_payloads.append(
                    {
                        "claim_id": claim_id,
                        "passed": plan_ok,
                        "provenance": provenance,
                        "confidence": plan_confidence,
                        "evidence": plan_evidence_payload,
                        "note": outcome_note,
                        "expected_unit": goal_spec.get("expected_unit"),
                        "observed_unit": goal_spec.get(
                            "observed_unit", goal_spec.get("expected_unit")
                        ),
                    }
                )

        updates = self._record_belief_updates(
            update_payloads,
            run_id=run_id,
            context="plan_outcome",
        )

        if not self._agent_profiles:
            self._agent_profiles = [
                AgentProfile(
                    name=ctx.name,
                    description=None,
                    default=(ctx.name == "default"),
                    tool_names=sorted(ctx.tools.keys()),
                )
                for ctx in self._agents.values()
            ]

        plan_manifest_payload = []
        if working_attempt is not None:
            plan_manifest_payload = [plan_trace.to_manifest() for plan_trace in working_attempt.plans]

        agent_manifest_payload = [
            {
                "name": profile.name,
                "description": profile.description,
                "default": profile.default,
                "tool_names": list(profile.tool_names),
            }
            for profile in self._agent_profiles
        ]

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
            plans=plan_manifest_payload,
            agents=agent_manifest_payload,
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


def _assemble_plan_evidence(
    tool_results: List[ToolResult],
    critiques: List[Mapping[str, Any]],
    safety_checks: List[SafetyDecision],
    risk_reviews: List[RiskAssessment],
) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    for result in tool_results:
        weight = 1.0 if result.ok else 1.25
        confidence = 0.9 if result.ok else 0.75
        note = result.stdout.strip() if result.stdout else None
        evidence.append(
            {
                "source": {
                    "kind": "tool",
                    "ref": result.call_id,
                    "note": note,
                },
                "outcome": "support" if result.ok else "conflict",
                "weight": weight,
                "confidence": confidence,
                "value": result.data if result.data is not None else result.stdout,
                "unit": "observation",
            }
        )
    for critique in critiques:
        status = str(critique.get("status", "FAIL")).upper()
        outcome = "support" if status == "PASS" else "conflict"
        confidence = 0.7 if status == "PASS" else 0.85
        note = critique.get("summary") or critique.get("notes")
        evidence.append(
            {
                "source": {
                    "kind": "critic",
                    "ref": critique.get("reviewer", "critic"),
                    "note": note,
                },
                "outcome": outcome,
                "weight": 0.6,
                "confidence": confidence,
                "value": critique.get("issues") or critique.get("amendments"),
                "unit": "analysis",
            }
        )
    for decision in safety_checks:
        evidence.append(
            {
                "source": {
                    "kind": "gatekeeper",
                    "ref": decision.tool_name,
                    "note": decision.reason,
                },
                "outcome": "support" if decision.approved else "conflict",
                "weight": 0.75,
                "confidence": 0.8,
                "unit": "policy",
            }
        )
    for assessment in risk_reviews:
        evidence.append(
            {
                "source": {
                    "kind": "risk",
                    "ref": assessment.tool_name,
                    "note": assessment.reason,
                },
                "outcome": "support" if assessment.approved else "conflict",
                "weight": 0.5,
                "confidence": 0.7,
                "unit": "policy",
            }
        )
    return evidence


def _estimate_plan_confidence(
    tool_results: List[ToolResult],
    critiques: List[Mapping[str, Any]],
    plan_ok: bool,
) -> float:
    if not tool_results:
        base = 0.5
    else:
        successes = sum(1 for result in tool_results if result.ok)
        base = successes / len(tool_results)
    if any(str(crit.get("status", "")).upper() != "PASS" for crit in critiques):
        base *= 0.65
    if not plan_ok:
        base *= 0.5
    return max(0.05, min(1.0, base))


def _summarise_plan_outcome(
    plan_id: str,
    success_count: int,
    total_calls: int,
    plan_ok: bool,
) -> str:
    if total_calls:
        success_pct = (success_count / total_calls) * 100.0
        stats = f"{success_count}/{total_calls} tool calls succeeded ({success_pct:.0f}%)"
    else:
        stats = "no tool calls executed"
    status = "succeeded" if plan_ok else "encountered failures"
    return f"Plan {plan_id} {status}; {stats}."


def _critique_updates(
    *,
    plan: Plan,
    critique: Mapping[str, Any],
    goal_spec: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    status = str(critique.get("status", "FAIL")).upper()
    outcome = "support" if status == "PASS" else "conflict"
    weight = 0.6
    if status == "REVISION":
        confidence = 0.65
    elif status == "PASS":
        confidence = 0.85
    else:
        confidence = 0.9
    value: Dict[str, Any] = {}
    if critique.get("issues"):
        value["issues"] = list(critique["issues"])
    if critique.get("amendments"):
        value["amendments"] = list(critique["amendments"])
    if critique.get("summary"):
        value["summary"] = critique["summary"]
    note = critique.get("notes") or critique.get("summary")
    reviewer = critique.get("reviewer", "critic")
    claim_ids = list(plan.claim_ids) or [plan.id]
    timestamp = goal_spec.get("time") or _now_iso()
    updates: List[Dict[str, Any]] = []
    for claim_id in claim_ids:
        updates.append(
            {
                "claim_id": claim_id,
                "passed": status == "PASS",
                "weight": weight,
                "confidence": confidence,
                "timestamp": timestamp,
                "note": note,
                "evidence": [
                    {
                        "source": {
                            "kind": "critic",
                            "ref": reviewer,
                            "note": note,
                        },
                        "outcome": outcome,
                        "weight": weight,
                        "confidence": confidence,
                        "unit": "critique",
                        "value": value or None,
                        "observed_at": timestamp,
                    }
                ],
            }
        )
    return updates


def _execution_failure_updates(
    *,
    plans: Sequence[Plan],
    feedback: Iterable[Mapping[str, Any]],
    goal_spec: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    plan_map = {plan.id: plan for plan in plans}
    timestamp = goal_spec.get("time") or _now_iso()
    updates: List[Dict[str, Any]] = []
    for item in feedback:
        plan_id = item.get("plan_id")
        plan = plan_map.get(str(plan_id))
        if plan is None:
            continue
        failures = item.get("failures") or []
        if not failures:
            continue
        claim_ids = list(plan.claim_ids) or [plan.id]
        note = f"Execution failed for plan {plan.id}"
        for claim_id in claim_ids:
            evidence_entries = []
            for failure in failures:
                evidence_entries.append(
                    {
                        "source": {
                            "kind": "execution",
                            "ref": str(failure.get("tool") or failure.get("step_id")),
                            "note": failure.get("stdout"),
                        },
                        "outcome": "conflict",
                        "weight": 0.8,
                        "confidence": 0.6,
                        "unit": "execution",
                        "value": {
                            "step_id": failure.get("step_id"),
                            "stdout": failure.get("stdout"),
                        },
                        "observed_at": timestamp,
                    }
                )
            updates.append(
                {
                    "claim_id": claim_id,
                    "passed": False,
                    "weight": 0.8,
                    "confidence": 0.6,
                    "timestamp": timestamp,
                    "note": note,
                    "evidence": evidence_entries,
                }
            )
    return updates


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
    plan_trace = working_attempt._plan_index.get(plan.id)
    step_trace = plan_trace.get_step(step.id) if plan_trace is not None else None
    agent_ctx = orchestrator._resolve_agent_context(getattr(step, "agent", None))
    agent_name = agent_ctx.name
    default_agent_name = orchestrator._default_agent_name()
    if step_trace is not None:
        step_trace.agent = agent_name
        step_trace.mark_started()
        orchestrator._emit(
            "orchestrator.plan_step_started",
            run_id=run_id,
            plan_id=plan.id,
            step_id=step.id,
            goal=step.goal,
            tool=step.tool,
            agent=agent_name,
            kind=step_trace.kind,
            depth=step_trace.depth,
        )
    if step.tool:
        tool = agent_ctx.tools.get(step.tool)
        if tool is None:
            raise KeyError(f"Unknown tool {step.tool}")
        if orchestrator.gatekeeper is not None:
            assessment = assess_step_risk(plan, step, agent_ctx.tools, orchestrator.gatekeeper)
            risk_assessments.append(assessment)
            working_attempt.record_risk_assessment(assessment)
            orchestrator._emit(
                "orchestrator.risk_assessed",
                run_id=run_id,
                plan_id=plan.id,
                step_id=step.id,
                tool=step.tool,
                agent=agent_name,
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
                "agent": agent_name,
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
                    agent=agent_name,
                )
                if step_trace is not None:
                    step_trace.mark_completed(succeeded=False, reason=assessment.reason)
                orchestrator._emit(
                    "orchestrator.plan_step_completed",
                    run_id=run_id,
                    plan_id=plan.id,
                    step_id=step.id,
                    success=False,
                    reason=assessment.reason,
                    duration_ms=step_trace.duration_ms if step_trace is not None else None,
                    agent=agent_name,
                )
                orchestrator._log_negotiation(
                    sender=agent_name,
                    recipient=default_agent_name,
                    kind="status",
                    content={"plan_id": plan.id, "step_id": step.id},
                    outcomes={"success": False, "reason": assessment.reason or "risk_block"},
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
        call_started = perf_counter()
        result = await tool.run({**step.args, "id": step.id}, ctx)
        duration_ms = round((perf_counter() - call_started) * 1000, 3)
        if result.wall_time_ms is None:
            try:
                result.wall_time_ms = int(duration_ms)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                result.wall_time_ms = None
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
            duration_ms=duration_ms,
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
            "agent": agent_name,
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

        if not result.ok:
            failure_reason = _truncate_text(result.stdout) or "tool failure"
            orchestrator._emit(
                "orchestrator.tool_failed",
                run_id=run_id,
                plan_id=plan.id,
                step_id=step.id,
                tool=step.tool,
                duration_ms=duration_ms,
                stdout=_truncate_text(result.stdout),
                agent=agent_name,
            )
            if step_trace is not None:
                step_trace.mark_completed(succeeded=False, reason=failure_reason)
                orchestrator._emit(
                    "orchestrator.plan_step_completed",
                    run_id=run_id,
                    plan_id=plan.id,
                    step_id=step.id,
                    success=False,
                    reason=failure_reason,
                    duration_ms=step_trace.duration_ms,
                    agent=agent_name,
                )
            orchestrator._log_negotiation(
                sender=agent_name,
                recipient=default_agent_name,
                kind="status",
                content={"plan_id": plan.id, "step_id": step.id},
                outcomes={"success": False, "reason": failure_reason},
            )
            return

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
            "agent": agent_name,
        }
        branch_taken = _should_execute_branch(branch.condition, result_index)
        working_attempt.record_branch_decision(
            plan_id=plan.id,
            step_id=step.id,
            metadata=branch_metadata,
            taken=branch_taken,
        )
        plan_branch_trace = None
        if step_trace is not None:
            plan_branch_trace = step_trace.get_branch(branch_index)
        if plan_branch_trace is not None and branch_taken is False:
            # Branch skipped; emit completion events for the skipped steps.
            for skipped_step in plan_branch_trace.steps:
                orchestrator._emit(
                    "orchestrator.plan_step_completed",
                    run_id=run_id,
                    plan_id=plan.id,
                    step_id=skipped_step.step_id,
                    success=False,
                    reason="branch_not_taken",
                    duration_ms=skipped_step.duration_ms,
                    agent=agent_name,
                )
        orchestrator._emit(
            "orchestrator.working_memory.branch",
            run_id=run_id,
            plan_id=plan.id,
            step_id=step.id,
            branch_index=branch_index,
            taken=branch_taken,
            condition=condition_payload,
            agent=agent_name,
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

    if step_trace is not None and step_trace.status not in {"failed", "skipped"}:
        child_statuses: List[str] = [child.status for child in step_trace.children]
        for branch in step_trace.branches:
            if branch.taken:
                child_statuses.extend(child.status for child in branch.steps)
        failed_child = any(status == "failed" for status in child_statuses)
        pending_child = any(status in {"pending", "running"} for status in child_statuses)
        succeeded = not failed_child and not pending_child
        if step.tool:
            result = result_index.get(step.id)
            if result is not None:
                succeeded = succeeded and result.ok
        step_trace.mark_completed(succeeded=succeeded)
        orchestrator._emit(
            "orchestrator.plan_step_completed",
            run_id=run_id,
            plan_id=plan.id,
            step_id=step.id,
            success=succeeded,
            duration_ms=step_trace.duration_ms,
            agent=step_trace.agent,
        )
        orchestrator._log_negotiation(
            sender=agent_name,
            recipient=default_agent_name,
            kind="status",
            content={"plan_id": plan.id, "step_id": step.id},
            outcomes={"success": succeeded},
        )
    elif step_trace is not None and step_trace.status == "skipped":
        orchestrator._emit(
            "orchestrator.plan_step_completed",
            run_id=run_id,
            plan_id=plan.id,
            step_id=step.id,
            success=False,
            reason=step_trace.failure_reason or "skipped",
            duration_ms=step_trace.duration_ms,
            agent=step_trace.agent,
        )
        orchestrator._log_negotiation(
            sender=agent_name,
            recipient=default_agent_name,
            kind="status",
            content={"plan_id": plan.id, "step_id": step.id},
            outcomes={"success": False, "reason": step_trace.failure_reason or "skipped"},
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
