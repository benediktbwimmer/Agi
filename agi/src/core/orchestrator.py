from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from .critic import Critic
from .memory import MemoryStore
from .planner import Planner
from .types import Belief, Plan, Report, RunContext, Source, ToolResult
from .world_model import WorldModel


@dataclass
class Orchestrator:
    planner: Planner
    critic: Critic
    tools: Mapping[str, Any]
    memory: MemoryStore
    world_model: WorldModel
    working_dir: Path = Path("artifacts")

    async def run(self, goal_spec: Dict[str, Any], constraints: Dict[str, Any] | None = None) -> Report:
        constraints = constraints or {}
        run_id = uuid.uuid4().hex
        run_dir = self.working_dir / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        hypotheses = goal_spec.get("hypotheses", [])
        plans = await self.planner.plan_from(hypotheses)
        for plan in plans:
            critique = await self.critic.check(_plan_to_dict(plan))
            if critique.get("status") != "PASS":
                raise RuntimeError(f"Plan {plan.id} rejected: {critique}")

        tool_results: List[ToolResult] = []
        memory_entries: List[Dict[str, Any]] = []

        for plan in plans:
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
                )
                result = await tool.run({**step.args, "id": step.id}, ctx)
                tool_results.append(result)
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

        for entry in memory_entries:
            self.memory.append(entry)

        updates = self.world_model.update(
            {
                "claim_id": (plan.claim_ids[0] if plan.claim_ids else plan.id),
                "passed": all(res.ok for res in tool_results),
                "provenance": [asdict(src) for res in tool_results for src in res.provenance],
                "expected_unit": goal_spec.get("expected_unit"),
                "observed_unit": goal_spec.get("observed_unit", goal_spec.get("expected_unit")),
            }
            for plan in plans
        )

        manifest = {
            "run_id": run_id,
            "goal": goal_spec,
            "constraints": constraints,
            "tool_results": [asdict(result) for result in tool_results],
            "belief_updates": [asdict(b) for b in updates],
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
