from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import pytest
import json

from agi.src.core.critic import Critic
from agi.src.core.executive import ExecutiveAgent
from agi.src.core.memory import MemoryStore
from agi.src.core.orchestrator import Orchestrator
from agi.src.core.planner import Planner
from agi.src.core.types import Belief, Report, Source, ToolResult
from agi.src.core.world_model import WorldModel


@dataclass
class DummyOrchestrator:
    captured_goal: Dict[str, Any] | None = None
    captured_constraints: Dict[str, Any] | None = None
    report: Report | None = None

    async def run(self, goal: Mapping[str, Any], constraints: Mapping[str, Any]) -> Report:
        self.captured_goal = dict(goal)
        self.captured_constraints = dict(constraints)
        if self.report is None:
            self.report = Report(
                goal=str(goal.get("goal")),
                summary="ok",
                key_findings=["finding"],
                belief_deltas=[],
                artifacts=["/tmp/artifact"],
            )
        return self.report


def test_executive_agent_enriches_goal(tmp_path: Path) -> None:
    memory_path = tmp_path / "memory.jsonl"
    memory = MemoryStore(memory_path)
    memory.append(
        {
            "type": "episode",
            "plan_id": "plan-1",
            "tool": "calculator",
            "call_id": "step-1",
            "ok": True,
            "stdout": "2",
            "time": "2024-01-01T00:00:00+00:00",
            "provenance": [{"kind": "note", "ref": "calc", "note": "baseline"}],
            "claim": {"id": "claim-123"},
        }
    )

    world = WorldModel()
    world.update(
        [
            {
                "claim_id": "claim-123",
                "passed": True,
                "expected_unit": None,
                "observed_unit": None,
                "provenance": [
                    {"kind": "simulation", "ref": "sim-1", "note": "support"},
                ],
            }
        ]
    )

    orchestrator = DummyOrchestrator()
    agent = ExecutiveAgent(orchestrator=orchestrator, memory=memory, world_model=world)

    report = asyncio.run(
        agent.achieve(
            "optimise process",
            context={"team": "research"},
            claim_ids=["claim-123"],
            metadata={"priority": "high"},
            constraints={"timeout_s": 5},
        )
    )

    assert isinstance(report, Report)
    assert orchestrator.captured_goal is not None
    goal_spec = orchestrator.captured_goal
    assert goal_spec["goal"] == "optimise process"
    assert goal_spec["metadata"]["priority"] == "high"
    assert goal_spec["metadata"]["context"] == {"team": "research"}
    hypotheses = goal_spec["hypotheses"]
    assert hypotheses and hypotheses[0]["id"] == "claim-123"
    assert "belief" in hypotheses[0]
    assert hypotheses[0]["memory"][-1]["tool"] == "calculator"
    contextual_memory = goal_spec["contextual_memory"]
    assert contextual_memory
    assert contextual_memory[0]["source"] == "claim"
    assert contextual_memory[0]["claim_id"] == "claim-123"
    assert contextual_memory[0]["stdout"] == "2"

    stored = memory.query_by_tool("executive_reflection")
    assert stored and stored[-1]["summary"] == "ok"


def test_executive_agent_sync_wrapper(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "mem.jsonl")
    world = WorldModel()
    orchestrator = DummyOrchestrator(
        report=Report(
            goal="sync",
            summary="done",
            key_findings=["kf"],
            belief_deltas=[Belief("c", 0.5, [Source("k", "r")], "now")],
            artifacts=[],
        )
    )
    agent = ExecutiveAgent(orchestrator=orchestrator, memory=memory, world_model=world)

    report = agent.run("sync")
    assert report.summary == "done"
    reflections = memory.query_by_tool("executive_reflection")
    assert reflections[-1]["summary"] == "done"


def test_executive_agent_reflection_includes_insights(tmp_path: Path) -> None:
    plan_response = {
        "plans": [
            {
                "id": "plan-a",
                "claim_ids": ["claim-a"],
                "steps": [
                    {
                        "id": "step-a",
                        "tool": "analysis",
                        "args": {},
                        "safety_level": "T0",
                    }
                ],
                "expected_cost": {},
                "risks": [],
                "ablations": [],
            }
        ]
    }

    planner = Planner(llm=lambda payload: json.dumps(plan_response))
    critic_calls = {"count": 0}

    def critic_llm(plan: Dict[str, Any]) -> str:
        critic_calls["count"] += 1
        if critic_calls["count"] == 1:
            return json.dumps(
                {
                    "status": "FAIL",
                    "summary": "Safety concerns identified",
                    "issues": ["Unsafe preconditions"],
                }
            )
        return json.dumps({"status": "PASS"})

    critic = Critic(llm=critic_llm)
    memory = MemoryStore(tmp_path / "memory.jsonl")
    world_model = WorldModel()

    class AnalysisTool:
        safety = "T0"

        async def run(self, args: Dict[str, Any], ctx: Any):  # pragma: no cover - invoked indirectly
            return ToolResult(
                call_id=args.get("id", "analysis"),
                ok=True,
                stdout="analysis-complete",
                wall_time_ms=2,
                provenance=[],
            )

    orchestrator = Orchestrator(
        planner=planner,
        critic=critic,
        tools={"analysis": AnalysisTool()},
        memory=memory,
        world_model=world_model,
        working_dir=tmp_path,
        max_replans=2,
    )
    agent = ExecutiveAgent(orchestrator=orchestrator, memory=memory, world_model=world_model)

    report = asyncio.run(
        agent.achieve(
            "Assess options",
            metadata={"priority": "medium"},
        )
    )

    assert report.summary == "Completed run"

    reflections = memory.query_by_tool("executive_reflection")
    assert reflections, "reflection record should be stored"
    reflection = reflections[-1]
    insights = reflection.get("insights")
    assert insights is not None
    assert insights["attempt_count"] == 2
    assert insights["final_status"] == "complete"
    assert insights["plan_outcomes"]["approved"] == ["plan-a"]
    assert "safety" in insights.get("critique_tags", [])
    assert insights["attempt_statuses"][0]["status"] == "needs_replan"

    insight_records = memory.recent(types=["reflection_insight"])
    assert insight_records
    latest_insight = insight_records[-1]
    assert latest_insight["goal"] == "Assess options"
    assert latest_insight["insights"]["plan_outcomes"]["approved"] == ["plan-a"]
