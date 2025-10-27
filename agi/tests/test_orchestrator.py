from __future__ import annotations
import asyncio
import json
from typing import Any, Dict

import pytest

from agi.src.core.critic import Critic
from agi.src.core.memory import MemoryStore
from agi.src.core.orchestrator import Orchestrator
from agi.src.core.planner import Planner
from agi.src.core.types import ToolResult
from agi.src.core.world_model import WorldModel


class StubTool:
    def __init__(self, ok: bool) -> None:
        self.ok = ok
        self.safety = "T0"
        self.invocations: list[str] = []

    async def run(self, args: Dict[str, Any], ctx: Any):  # pragma: no cover - exercised via orchestrator
        self.invocations.append(args.get("id", "call"))
        return ToolResult(
            call_id=args.get("id", "call"),
            ok=self.ok,
            stdout="pass" if self.ok else "fail",
            wall_time_ms=1,
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
    world_model = WorldModel()
    orchestrator = Orchestrator(
        planner=planner,
        critic=critic,
        tools={
            "fail_tool": StubTool(ok=False),
            "success_tool": StubTool(ok=True),
        },
        memory=memory,
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
    world_model = WorldModel()
    orchestrator = Orchestrator(
        planner=planner,
        critic=critic,
        tools={"success_tool": StubTool(ok=True)},
        memory=memory,
        world_model=world_model,
        working_dir=tmp_path,
    )

    report = asyncio.run(orchestrator.run({"goal": "demo"}, {}))

    assert sorted(b.claim_id for b in report.belief_deltas) == [
        "claim-1",
        "claim-2",
    ]
    assert set(orchestrator.world_model.beliefs.keys()) == {"claim-1", "claim-2"}


def test_orchestrator_executes_hierarchical_plan(tmp_path):
    planner_response = {
        "plans": [
            {
                "id": "plan-hier",
                "claim_ids": ["claim-hier"],
                "steps": [
                    {
                        "id": "root",
                        "tool": "setup",
                        "args": {},
                        "sub_steps": [
                            {
                                "id": "child",
                                "tool": "worker",
                                "args": {},
                            }
                        ],
                        "branches": [
                            {
                                "condition": "on_success(root)",
                                "steps": [
                                    {
                                        "id": "success",
                                        "tool": "finisher",
                                        "args": {},
                                    }
                                ],
                            },
                            {
                                "condition": "on_failure(root)",
                                "steps": [
                                    {
                                        "id": "failure",
                                        "tool": "fallback",
                                        "args": {},
                                    }
                                ],
                            },
                        ],
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
    world_model = WorldModel()
    tools = {
        "setup": StubTool(ok=True),
        "worker": StubTool(ok=True),
        "finisher": StubTool(ok=True),
        "fallback": StubTool(ok=False),
    }
    orchestrator = Orchestrator(
        planner=planner,
        critic=critic,
        tools=tools,
        memory=memory,
        world_model=world_model,
        working_dir=tmp_path,
    )

    report = asyncio.run(orchestrator.run({"goal": "hier"}, {}))

    assert report.summary == "Completed run"
    assert tools["setup"].invocations == ["root"]
    assert tools["worker"].invocations == ["child"]
    assert tools["finisher"].invocations == ["success"]
    assert tools["fallback"].invocations == []

    episodes = memory.query_by_tool("finisher")
    assert episodes and episodes[-1]["call_id"] == "success"
