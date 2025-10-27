from __future__ import annotations

import json
import uuid
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Set

from .critic import Critic
from .memory import MemoryStore, WorkingMemory
from .planner import Planner
from .safety import SafetyDecision, enforce_plan_safety
from .types import Belief, Plan, Report, RunContext, Source, ToolResult
from .world_model import WorldModel
from ..governance.gatekeeper import Gatekeeper


@dataclass
class Orchestrator:
    planner: Planner
    critic: Critic
    tools: Mapping[str, Any]
    episodic_memory: MemoryStore
    working_memory: WorkingMemory
    world_model: WorldModel
    gatekeeper: Gatekeeper | None = None
    working_dir: Path = Path("artifacts")
    enable_memory: bool = True

    def __post_init__(self) -> None:
        flag = os.getenv("AGI_ENABLE_MEMORY")
        if flag is not None:
            self.enable_memory = flag.strip().lower() not in {"0", "false", "no"}

    async def run(self, goal_spec: Dict[str, Any], constraints: Dict[str, Any] | None = None) -> Report:
        constraints = constraints or {}
        run_id = uuid.uuid4().hex
        run_dir = self.working_dir / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        hypotheses = goal_spec.get("hypotheses", [])
        plans = await self.planner.plan_from(hypotheses)
        safety_audit: List[SafetyDecision] = []
        if self.gatekeeper is not None:
            safety_audit = enforce_plan_safety(plans, self.tools, self.gatekeeper)
        for plan in plans:
            critique = await self.critic.check(_plan_to_dict(plan))
            if critique.get("status") != "PASS":
                raise RuntimeError(f"Plan {plan.id} rejected: {critique}")

        if self.enable_memory:
            self.working_memory.reset()
            self._hydrate_working_memory(goal_spec, plans)

        tool_results: List[ToolResult] = []
        plan_results: Dict[str, List[ToolResult]] = {}

        for plan in plans:
            per_plan_results: List[ToolResult] = []
            plan_results[plan.id] = per_plan_results
            for step in plan.steps:
                tool = self.tools.get(step.tool)
                if tool is None:
                    raise KeyError(f"Unknown tool {step.tool}")
                ctx = RunContext(
                    working_dir=str(run_dir),
                    timeout_s=int(constraints.get("timeout_s", goal_spec.get("timeout_s", 60))),
                    env_whitelist=list(goal_spec.get("env", [])),
                    network=constraints.get("network", "off"),
                    record_provenance=True,
                    working_memory=self.working_memory.recall(step.tool)
                    if self.enable_memory
                    else [],
                    episodic_memory=self.episodic_memory if self.enable_memory else None,
                )
                result = await tool.run({**step.args, "id": step.id}, ctx)
                per_plan_results.append(result)
                tool_results.append(result)
                if self.enable_memory:
                    episode = {
                        "type": "episode",
                        "plan_id": plan.id,
                        "tool": step.tool,
                        "call_id": result.call_id,
                        "ok": result.ok,
                        "stdout": result.stdout,
                        "time": goal_spec.get("time", "1970-01-01T00:00:00+00:00"),
                        "provenance": [asdict(src) for src in result.provenance],
                        "claim_ids": list(plan.claim_ids),
                        "goal": goal_spec.get("goal"),
                    }
                    self.working_memory.add_episode(episode)
                    if self._is_significant(result):
                        self.episodic_memory.append(episode)

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

        manifest = {
            "run_id": run_id,
            "goal": goal_spec,
            "constraints": constraints,
            "tool_results": [asdict(result) for result in tool_results],
            "belief_updates": [asdict(b) for b in updates],
            "safety_audit": [decision.as_dict() for decision in safety_audit],
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        report = Report(
            goal=goal_spec.get("goal", ""),
            summary="Completed run",
            key_findings=[result.stdout or "" for result in tool_results],
            belief_deltas=updates,
            artifacts=[_safe_relpath(run_dir)],
        )
        return report

    def _hydrate_working_memory(self, goal_spec: Mapping[str, Any], plans: Iterable[Plan]) -> None:
        relevant_tools: Set[str] = set()
        relevant_claims: Set[str] = set(goal_spec.get("claim_ids", []))
        for plan in plans:
            relevant_claims.update(plan.claim_ids)
            for step in plan.steps:
                relevant_tools.add(step.tool)

        if not relevant_claims and not relevant_tools:
            recent = self.episodic_memory.recent(self.working_memory.capacity_global)
            self.working_memory.hydrate(recent)
            return

        for claim_id in sorted(relevant_claims):
            episodes = self.episodic_memory.query_by_claim(claim_id)
            if episodes:
                self.working_memory.hydrate(
                    episodes[-self.working_memory.capacity_per_tool :]
                )

        for tool_name in sorted(relevant_tools):
            episodes = self.episodic_memory.query_by_tool(tool_name)
            if episodes:
                self.working_memory.hydrate(
                    episodes[-self.working_memory.capacity_per_tool :]
                )

    @staticmethod
    def _is_significant(result: ToolResult) -> bool:
        if result.ok:
            return True
        if result.stdout or result.data:
            return True
        return bool(result.provenance)


def _plan_to_dict(plan: Plan) -> Dict[str, Any]:
    return {
        "id": plan.id,
        "claim_ids": plan.claim_ids,
        "steps": [
            {
                "id": step.id,
                "tool": step.tool,
                "args": step.args,
                "safety_level": step.safety_level,
            }
            for step in plan.steps
        ],
        "expected_cost": plan.expected_cost,
        "risks": plan.risks,
        "ablations": plan.ablations,
    }


def _safe_relpath(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)
