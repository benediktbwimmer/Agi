from __future__ import annotations
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from agi.src.core.critic import Critic
from agi.src.core.memory import MemoryStore, WorkingMemory
from agi.src.core.orchestrator import Orchestrator
from agi.src.core.planner import Planner
from agi.src.core.types import ToolResult
from agi.src.core.world_model import WorldModel


class StubTool:
    def __init__(self, ok: bool) -> None:
        self.ok = ok
        self.safety = "T0"

    async def run(self, args: Dict[str, Any], ctx: Any):  # pragma: no cover - exercised via orchestrator
        return ToolResult(
            call_id=args.get("id", "call"),
            ok=self.ok,
            stdout="pass" if self.ok else "fail",
            wall_time_ms=1,
            provenance=[],
        )


class InspectingTool:
    def __init__(self, seen):
        self.safety = "T0"
        self.seen = seen

    async def run(self, args: Dict[str, Any], ctx: Any):
        self.seen.append(
            {
                "working": list(ctx.working_memory),
                "episodic": ctx.recall_from_episodic(tool=args.get("tool_hint")),
            }
        )
        return ToolResult(
            call_id=args.get("id", "inspect"),
            ok=True,
            stdout="observed",
            provenance=[],
        )


def test_orchestrator_tracks_plan_results_independently(tmp_path):
    planner_response = {
        "plans": [
            {
                "id": "plan-1",
                "claim_ids": ["claim-1"],
                "steps": [
                    {
                        "id": "step-1",
                        "tool": "fail_tool",
                        "args": {},
                        "safety_level": "T0",
                    }
                ],
                "expected_cost": {},
                "risks": [],
                "ablations": [],
            },
            {
                "id": "plan-2",
                "claim_ids": ["claim-2"],
                "steps": [
                    {
                        "id": "step-2",
                        "tool": "success_tool",
                        "args": {},
                        "safety_level": "T0",
                    }
                ],
                "expected_cost": {},
                "risks": [],
                "ablations": [],
            },
        ]
    }

    planner = Planner(llm=lambda payload: json.dumps(planner_response))
    critic = Critic(llm=lambda plan: json.dumps({"status": "PASS"}))
    memory = MemoryStore(tmp_path / "memory.jsonl")
    working_memory = WorkingMemory()
    world_model = WorldModel()
    orchestrator = Orchestrator(
        planner=planner,
        critic=critic,
        tools={
            "fail_tool": StubTool(ok=False),
            "success_tool": StubTool(ok=True),
        },
        episodic_memory=memory,
        working_memory=working_memory,
        world_model=world_model,
        working_dir=tmp_path,
    )

    report = asyncio.run(orchestrator.run({"goal": "demo"}, {}))

    assert len(report.belief_deltas) == 2
    credences = {belief.claim_id: belief.credence for belief in report.belief_deltas}
    assert credences["claim-1"] < 0.5
    assert credences["claim-2"] > 0.5

    beliefs = orchestrator.world_model.beliefs
    assert beliefs["claim-1"].credence == pytest.approx(credences["claim-1"])
    assert beliefs["claim-2"].credence == pytest.approx(credences["claim-2"])

    assert memory.query_by_tool("fail_tool")
    assert memory.query_by_tool("success_tool")


def test_orchestrator_updates_all_claims(tmp_path):
    planner_response = {
        "plans": [
            {
                "id": "plan-1",
                "claim_ids": ["claim-1", "claim-2"],
                "steps": [
                    {
                        "id": "step-1",
                        "tool": "success_tool",
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

    planner = Planner(llm=lambda payload: json.dumps(planner_response))
    critic = Critic(llm=lambda plan: json.dumps({"status": "PASS"}))
    memory = MemoryStore(tmp_path / "memory.jsonl")
    working_memory = WorkingMemory()
    world_model = WorldModel()
    orchestrator = Orchestrator(
        planner=planner,
        critic=critic,
        tools={"success_tool": StubTool(ok=True)},
        episodic_memory=memory,
        working_memory=working_memory,
        world_model=world_model,
        working_dir=tmp_path,
    )

    report = asyncio.run(orchestrator.run({"goal": "demo"}, {}))

    assert sorted(b.claim_id for b in report.belief_deltas) == [
        "claim-1",
        "claim-2",
    ]
    assert set(orchestrator.world_model.beliefs.keys()) == {"claim-1", "claim-2"}


def test_orchestrator_hydrates_working_memory(tmp_path):
    baseline_time = "2024-01-01T00:00:00+00:00"
    memory = MemoryStore(tmp_path / "memory.jsonl")
    memory.append(
        {
            "type": "episode",
            "plan_id": "baseline",
            "tool": "inspect_tool",
            "call_id": "baseline-call",
            "ok": True,
            "stdout": "baseline context",
            "time": baseline_time,
            "claim_ids": ["claim-1"],
        }
    )
    working_memory = WorkingMemory()
    planner_response = {
        "plans": [
            {
                "id": "plan-1",
                "claim_ids": ["claim-1"],
                "steps": [
                    {
                        "id": "step-1",
                        "tool": "inspect_tool",
                        "args": {"tool_hint": "inspect_tool"},
                        "safety_level": "T0",
                    }
                ],
                "expected_cost": {},
                "risks": [],
                "ablations": [],
            }
        ]
    }
    planner = Planner(llm=lambda payload: json.dumps(planner_response))
    critic = Critic(llm=lambda plan: json.dumps({"status": "PASS"}))
    seen_contexts: List[Dict[str, Any]] = []
    tools = {"inspect_tool": InspectingTool(seen_contexts)}
    orchestrator = Orchestrator(
        planner=planner,
        critic=critic,
        tools=tools,
        episodic_memory=memory,
        working_memory=working_memory,
        world_model=WorldModel(),
        working_dir=tmp_path,
    )

    goal_spec = {
        "goal": "demo",
        "hypotheses": [{"id": "claim-1"}],
        "claim_ids": ["claim-1"],
        "time": "2024-01-02T00:00:00+00:00",
    }

    asyncio.run(orchestrator.run(goal_spec, {}))
    assert seen_contexts, "expected tool to capture working memory"
    first_context = seen_contexts[0]["working"]
    assert any(ep.get("call_id") == "baseline-call" for ep in first_context)

    stored = memory.query_by_tool("inspect_tool")
    assert any(ep.get("call_id") == "step-1" for ep in stored)

    asyncio.run(orchestrator.run(goal_spec, {}))
    second_context = seen_contexts[1]["working"]
    call_ids = {ep.get("call_id") for ep in second_context}
    assert {"baseline-call", "step-1"}.issubset(call_ids)


def test_orchestrator_instantiates_memory(tmp_path: Path) -> None:
    planner_response = {
        "plans": [
            {
                "id": "plan-1",
                "claim_ids": ["claim-1"],
                "steps": [
                    {
                        "id": "step-1",
                        "tool": "success_tool",
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

    planner = Planner(llm=lambda payload: json.dumps(planner_response))
    critic = Critic(llm=lambda plan: json.dumps({"status": "PASS"}))
    orchestrator = Orchestrator(
        planner=planner,
        critic=critic,
        tools={"success_tool": StubTool(ok=True)},
        world_model=WorldModel(),
        working_dir=tmp_path,
        episodic_memory=None,
        working_memory=None,
        episodic_memory_path=Path("memory.jsonl"),
    )

    goal_spec = {
        "goal": "demo",
        "hypotheses": [{"id": "claim-1"}],
        "claim_ids": ["claim-1"],
        "time": "2024-01-02T00:00:00+00:00",
    }

    asyncio.run(orchestrator.run(goal_spec, {}))

    assert orchestrator.episodic_memory is not None
    assert orchestrator.working_memory is not None

    memory_file = tmp_path / "memory.jsonl"
    assert memory_file.exists()
    contents = memory_file.read_text(encoding="utf-8").strip().splitlines()
    assert contents, "expected episodic memory file to contain entries"
