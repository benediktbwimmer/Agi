from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Set

from .critic import Critic
from .manifest import RunManifest
from .memory import MemoryStore
from .planner import Planner
from .safety import SafetyDecision, enforce_plan_safety
from .telemetry import Telemetry
from .types import BranchCondition, Plan, PlanStep, Report, RunContext, ToolResult
from .world_model import WorldModel
from ..governance.gatekeeper import Gatekeeper


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

    def _emit(self, event: str, **payload: Any) -> None:
        if self.telemetry is not None:
            self.telemetry.emit(event, **payload)

    async def run(self, goal_spec: Dict[str, Any], constraints: Dict[str, Any] | None = None) -> Report:
        constraints = constraints or {}
        run_id = uuid.uuid4().hex
        self._emit("orchestrator.run_started", run_id=run_id, goal=goal_spec.get("goal"))
        run_dir = self.working_dir / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        hypotheses = goal_spec.get("hypotheses", [])
        feedback: List[Dict[str, Any]] = []
        critiques: List[Dict[str, Any]] = []
        memory_entries: List[Dict[str, Any]] = []
        plans: List[Plan] = []
        tool_results: List[ToolResult] = []
        plan_results: Dict[str, List[ToolResult]] = {}
        safety_audit: List[SafetyDecision] = []

        for attempt in range(self.max_replans + 1):
            plans = await self.planner.plan_from(hypotheses, feedback=feedback)
            plan_critiques: List[Dict[str, Any]] = []
            replan_required = False
            for plan in plans:
                self._emit(
                    "orchestrator.plan_ready",
                    run_id=run_id,
                    plan_id=plan.id,
                    step_count=len(plan.steps),
                )
                critique = await self.critic.check(_plan_to_dict(plan))
                status = str(critique.get("status", "FAIL")).upper()
                if status == "PASS":
                    continue
                critique_record = _normalise_critique(plan.id, critique)
                critiques.append(critique_record)
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
                feedback.extend(json.loads(json.dumps(item)) for item in execution_feedback)
                continue

            tool_results = attempt_tool_results
            plan_results = attempt_plan_results
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
            critiques=critiques,
        )
        manifest.write(run_dir / "manifest.json")
        RunManifest.write_schema(run_dir / "manifest.schema.json")

        success = all(result.ok for result in tool_results) if tool_results else True
        report = Report(
            goal=goal_spec.get("goal", ""),
            summary="Completed run",
            key_findings=[result.stdout or "" for result in tool_results],
            belief_deltas=updates,
            artifacts=[_safe_relpath(run_dir)],
        )
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
    return entry


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
) -> None:
    if step.tool:
        tool = orchestrator.tools.get(step.tool)
        if tool is None:
            raise KeyError(f"Unknown tool {step.tool}")
        ctx = RunContext(
            working_dir=str(run_dir),
            timeout_s=int(constraints.get("timeout_s", goal_spec.get("timeout_s", 60))),
            env_whitelist=list(goal_spec.get("env", [])),
            network=constraints.get("network", "off"),
            record_provenance=True,
        )
        orchestrator._emit(
            "orchestrator.tool_started",
            run_id=run_id,
            plan_id=plan.id,
            step_id=step.id,
            tool=step.tool,
        )
        result = await tool.run({**step.args, "id": step.id}, ctx)
        per_plan_results.append(result)
        all_results.append(result)
        result_index[result.call_id] = result
        orchestrator._emit(
            "orchestrator.tool_completed",
            run_id=run_id,
            plan_id=plan.id,
            step_id=step.id,
            tool=step.tool,
            ok=result.ok,
            wall_time_ms=result.wall_time_ms,
        )
        memory_entries.append(
            {
                "type": "episode",
                "plan_id": plan.id,
                "tool": step.tool,
                "call_id": result.call_id,
                "ok": result.ok,
                "stdout": result.stdout,
                "time": goal_spec.get("time", "1970-01-01T00:00:00+00:00"),
                "provenance": [asdict(src) for src in result.provenance],
            }
        )

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
        )

    for branch in step.branches:
        if _should_execute_branch(branch.condition, result_index):
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
