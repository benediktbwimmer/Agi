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


def test_orchestrator_replans_with_critic_feedback(tmp_path):
    planner_payloads: list[Dict[str, Any]] = []

    class RevisingLLM:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, payload: Dict[str, Any]) -> str:  # pragma: no cover - exercised indirectly
            planner_payloads.append(payload)
            self.calls += 1
            if self.calls == 1:
                return json.dumps(
                    {
                        "plans": [
                            {
                                "id": "plan-risky",
                                "claim_ids": ["claim-risk"],
                                "steps": [
                                    {
                                        "id": "risky-step",
                                        "tool": "risky",
                                        "args": {},
                                    }
                                ],
                                "expected_cost": {},
                                "risks": [],
                                "ablations": [],
                            }
                        ]
                    }
                )
            return json.dumps(
                {
                    "plans": [
                        {
                            "id": "plan-safe",
                            "claim_ids": ["claim-risk"],
                            "steps": [
                                {
                                    "id": "safe-step",
                                    "tool": "safe",
                                    "args": {},
                                }
                            ],
                            "expected_cost": {},
                            "risks": [],
                            "ablations": [],
                        }
                    ]
                }
            )

    planner = Planner(llm=RevisingLLM())
    critic_responses = iter(
        [
            json.dumps(
                {
                    "status": "REVISION",
                    "notes": "Prefer the safe tool",
                    "amendments": ["replace risky"],
                    "issues": ["safety"],
                }
            ),
            json.dumps({"status": "PASS"}),
        ]
    )
    critic = Critic(llm=lambda plan: next(critic_responses))
    memory = MemoryStore(tmp_path / "memory.jsonl")
    world_model = WorldModel()

    class SafeTool(StubTool):
        async def run(self, args: Dict[str, Any], ctx: Any):  # pragma: no cover - exercised via orchestrator
            return await super().run(args, ctx)

    tools = {"safe": SafeTool(ok=True), "risky": StubTool(ok=True)}
    orchestrator = Orchestrator(
        planner=planner,
        critic=critic,
        tools=tools,
        memory=memory,
        world_model=world_model,
        working_dir=tmp_path,
        max_replans=3,
    )

    report = asyncio.run(orchestrator.run({"goal": "safe"}, {}))

    assert report.summary == "Completed run"
    assert len(planner_payloads) == 2
    assert planner_payloads[1]["feedback"][0]["plan_id"] == "plan-risky"
    assert planner_payloads[1]["feedback"][0]["status"] == "REVISION"

    critique_records = memory.query_by_tool("critic")
    assert critique_records and critique_records[-1]["status"] == "REVISION"
    assert "replace risky" in critique_records[-1]["amendments"]

    manifest_path = next(tmp_path.glob("run_*/manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["critiques"][0]["plan_id"] == "plan-risky"
    assert manifest["critiques"][0]["status"] == "REVISION"


def test_orchestrator_recovers_from_execution_failure(tmp_path):
    planner_payloads: list[Dict[str, Any]] = []

    class FallbackPlanner:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, payload: Dict[str, Any]) -> str:  # pragma: no cover - exercised indirectly
            planner_payloads.append(payload)
            self.calls += 1
            if self.calls == 1:
                return json.dumps(
                    {
                        "plans": [
                            {
                                "id": "plan-glitch",
                                "claim_ids": ["claim-unstable"],
                                "steps": [
                                    {
                                        "id": "glitch-step",
                                        "tool": "glitch",  # fails at runtime
                                        "args": {},
                                    }
                                ],
                                "expected_cost": {},
                                "risks": [],
                                "ablations": [],
                            }
                        ]
                    }
                )
            return json.dumps(
                {
                    "plans": [
                        {
                            "id": "plan-stable",
                            "claim_ids": ["claim-unstable"],
                            "steps": [
                                {
                                    "id": "stable-step",
                                    "tool": "stable",
                                    "args": {},
                                }
                            ],
                            "expected_cost": {},
                            "risks": [],
                            "ablations": [],
                        }
                    ]
                }
            )

    planner = Planner(llm=FallbackPlanner())
    critic = Critic(llm=lambda plan: json.dumps({"status": "PASS"}))
    memory = MemoryStore(tmp_path / "memory.jsonl")
    world_model = WorldModel()

    class FailingTool(StubTool):
        async def run(self, args: Dict[str, Any], ctx: Any):  # pragma: no cover - exercised via orchestrator
            return await super().run(args, ctx)

    tools = {"glitch": FailingTool(ok=False), "stable": StubTool(ok=True)}
    orchestrator = Orchestrator(
        planner=planner,
        critic=critic,
        tools=tools,
        memory=memory,
        world_model=world_model,
        working_dir=tmp_path,
        max_replans=2,
    )

    report = asyncio.run(orchestrator.run({"goal": "stabilise"}, {}))

    assert report.summary == "Completed run"
    assert report.belief_deltas and report.belief_deltas[0].credence > 0.5
    assert world_model.beliefs["claim-unstable"].credence > 0.5

    assert len(planner_payloads) == 2
    assert planner_payloads[1]["feedback"][0]["status"] == "FAILED_EXECUTION"
    assert planner_payloads[1]["feedback"][0]["failures"][0]["tool"] == "glitch"

    glitch_records = memory.query_by_tool("glitch")
    assert glitch_records and glitch_records[-1]["ok"] is False

    stable_records = memory.query_by_tool("stable")
    assert stable_records and stable_records[-1]["ok"] is True

    manifest_path = next(tmp_path.glob("run_*/manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    assert [result["call_id"] for result in manifest["tool_results"]] == [
        "stable-step"
    ]


def test_orchestrator_populates_working_memory(tmp_path):
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

    def critic_llm(plan: Dict[str, Any]) -> str:  # pragma: no cover - invoked indirectly
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
    tool = StubTool(ok=True)
    orchestrator = Orchestrator(
        planner=planner,
        critic=critic,
        tools={"analysis": tool},
        memory=memory,
        world_model=world_model,
        working_dir=tmp_path,
        max_replans=2,
    )

    report = asyncio.run(
        orchestrator.run(
            {
                "goal": "Assess options",
                "hypotheses": [
                    {"id": "hyp-1", "statement": "Option A is viable"},
                ],
            },
            {},
        )
    )

    assert report.summary == "Completed run"

    working = orchestrator.working_memory
    assert working is not None
    assert working.goal == "Assess options"
    assert len(working.hypotheses) == 1
    assert len(working.attempts) == 2
    assert working.input_provenance is not None
    assert working.input_provenance.get("goal") == "Assess options"

    first_attempt, second_attempt = working.attempts
    assert first_attempt.status == "needs_replan"
    assert first_attempt.critiques and first_attempt.critiques[0]["status"] == "FAIL"
    first_plan = first_attempt.plans[0]
    assert first_plan.approved is False
    assert "issue" in first_plan.rationale_tags
    assert "safety" in first_plan.rationale_tags
    assert first_plan.execution_succeeded is None

    assert second_attempt.status == "complete"
    second_plan = second_attempt.plans[0]
    assert second_plan.approved is True
    assert second_plan.execution_succeeded is True

    snapshot = working.to_dict()
    assert snapshot["goal"] == "Assess options"
    assert snapshot["attempts"][0]["plans"][0]["rationale_tags"]
    assert snapshot.get("input_provenance", {}).get("goal") == "Assess options"


def test_orchestrator_builds_memory_context_for_planner(tmp_path):
    captured: Dict[str, Any] = {}

    def llm(payload: Dict[str, Any]) -> str:
        captured["payload"] = payload
        return json.dumps(
            {
                "plans": [
                    {
                        "id": "plan-ctx",
                        "claim_ids": ["ctx-1"],
                        "steps": [
                            {
                                "id": "ctx-step",
                                "tool": "context_tool",
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
        )

    planner = Planner(llm=llm)
    critic = Critic(llm=lambda plan: json.dumps({"status": "PASS"}))
    memory = MemoryStore(tmp_path / "memory.jsonl")
    world_model = WorldModel()
    tool = StubTool(ok=True)

    memory.append(
        {
            "type": "reflection",
            "summary": "Analysed prior lunar habitat study",
            "time": "2024-01-01T00:00:00+00:00",
        }
    )
    memory.append(
        {
            "type": "episode",
            "tool": "context_tool",
            "time": "2024-01-01T01:00:00+00:00",
            "stdout": "Evaluated habitat life support",
        }
    )

    orchestrator = Orchestrator(
        planner=planner,
        critic=critic,
        tools={"context_tool": tool},
        memory=memory,
        world_model=world_model,
        working_dir=tmp_path,
    )

    asyncio.run(orchestrator.run({"goal": "Design lunar habitat"}, {}))

    assert "payload" in captured
    memory_context = captured["payload"].get("memory_context")
    assert memory_context is not None
    assert memory_context.get("goal") == "Design lunar habitat"
    semantic = memory_context.get("semantic")
    recent = memory_context.get("recent")
    assert semantic or recent
    if semantic:
        assert semantic["coverage"]["reflection"] >= 1
    if recent:
        assert any(record.get("tool") == "context_tool" for record in recent["records"])
