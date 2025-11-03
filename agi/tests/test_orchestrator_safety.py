from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agi.src.core.critic import Critic
from agi.src.core.memory import MemoryStore
from agi.src.core.orchestrator import Orchestrator
from agi.src.core.planner import Planner
from agi.src.core.types import Belief, Plan, ToolCall, ToolResult, RunContext
from agi.src.core.world_model import WorldModel
from agi.src.governance.gatekeeper import Gatekeeper


class StaticPlanner(Planner):
    def __init__(self, plans: list[Plan]) -> None:
        super().__init__(llm=lambda payload: json.dumps({}))
        self._plans = plans

    async def plan_from(self, hypotheses, *, feedback=None, memory_context=None):  # type: ignore[override]
        return list(self._plans)


class DummyTool:
    name = "dummy"
    safety = "T1"

    async def run(self, args, ctx: RunContext) -> ToolResult:
        return ToolResult(call_id=args.get("id", "dummy"), ok=True, stdout="done")


class AdaptivePlanner(Planner):
    def __init__(self) -> None:
        super().__init__(llm=lambda payload: json.dumps({}))
        self._attempts = 0

    async def plan_from(self, hypotheses, *, feedback=None, memory_context=None):  # type: ignore[override]
        self._attempts += 1
        return [_plan(step_tier="T0")]


class RuntimeGatekeeper(Gatekeeper):
    def __init__(self) -> None:
        super().__init__(policy={})
        self._runtime_block_pending = True
        self._invocation_count = 0

    def review(self, tier: str, *, tool: str | None = None) -> bool:
        self._invocation_count += 1
        if (
            tool == "dummy"
            and self._runtime_block_pending
            and self._invocation_count >= 2
        ):
            self._runtime_block_pending = False
            return False
        return super().review(tier, tool=tool)


def _critic() -> Critic:
    return Critic(llm=lambda plan: json.dumps({"status": "PASS"}))


def _memory(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory.jsonl")


def _world_model() -> WorldModel:
    return WorldModel()


def _plan(step_tier: str) -> Plan:
    return Plan(
        id="plan-1",
        claim_ids=["claim-1"],
        steps=[
            ToolCall(
                id="step-1",
                tool="dummy",
                args={"id": "call-1"},
                safety_level=step_tier,
            )
        ],
        expected_cost={},
        risks=[],
        ablations=[],
    )


def _branch_plan() -> Plan:
    return Plan(
        id="plan-branch",
        claim_ids=["claim-branch"],
        steps=[
            {
                "id": "root",
                "tool": "dummy",
                "args": {"id": "root"},
                "branches": [
                    {
                        "condition": "on_success(root)",
                        "steps": [
                            {
                                "id": "danger",
                                "tool": "dummy",
                                "args": {"id": "danger"},
                                "safety_level": "T3",
                            }
                        ],
                    }
                ],
            }
        ],
        expected_cost={},
        risks=[],
        ablations=[],
    )


def test_orchestrator_denies_disallowed_tier(tmp_path: Path) -> None:
    planner = StaticPlanner([_plan(step_tier="T2")])
    orchestrator = Orchestrator(
        planner=planner,
        critic=_critic(),
        tools={"dummy": DummyTool()},
        memory=_memory(tmp_path),
        world_model=_world_model(),
        gatekeeper=Gatekeeper(policy={}),
        working_dir=tmp_path,
    )

    with pytest.raises(PermissionError):
        asyncio.run(orchestrator.run({"goal": "test", "hypotheses": [{"id": "h1"}]}))


def test_orchestrator_records_safety_audit(tmp_path: Path) -> None:
    planner = StaticPlanner([_plan(step_tier="T0")])
    orchestrator = Orchestrator(
        planner=planner,
        critic=_critic(),
        tools={"dummy": DummyTool()},
        memory=_memory(tmp_path),
        world_model=_world_model(),
        gatekeeper=Gatekeeper(policy={}),
        working_dir=tmp_path,
    )

    report = asyncio.run(orchestrator.run({"goal": "test", "hypotheses": [{"id": "h1"}]}))
    assert report.summary == "Completed run"


def test_gatekeeper_uses_confidence_interval_bias(tmp_path: Path) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    world_model = WorldModel()
    world_model._beliefs["evaluation::wide"] = Belief(
        claim_id="evaluation::wide",
        credence=0.8,
        last_updated=timestamp,
        evidence=[],
        support=0.0,
        conflict=0.0,
        variance=0.25,
    )
    gatekeeper = Gatekeeper(
        policy={
            "max_tier": "T2",
            "evaluation_rules": [
                {
                    "claim": "evaluation::wide",
                    "max_tier": "T0",
                    "max_confidence_interval": 0.3,
                    "max_uncertainty": 0.5,
                    "tools": ["dummy"],
                }
            ],
        },
        world_model=world_model,
    )

    assert gatekeeper.review("T1", tool="dummy") is False

    world_model._beliefs["evaluation::wide"] = Belief(
        claim_id="evaluation::wide",
        credence=0.8,
        last_updated=timestamp,
        evidence=[],
        support=3.0,
        conflict=0.0,
        variance=0.005,
    )

    assert gatekeeper.review("T1", tool="dummy") is True


def test_orchestrator_checks_branch_tiers(tmp_path: Path) -> None:
    planner = StaticPlanner([_branch_plan()])
    orchestrator = Orchestrator(
        planner=planner,
        critic=_critic(),
        tools={"dummy": DummyTool()},
        memory=_memory(tmp_path),
        world_model=_world_model(),
        gatekeeper=Gatekeeper(policy={}),
        working_dir=tmp_path,
    )

    with pytest.raises(PermissionError):
        asyncio.run(orchestrator.run({"goal": "test", "hypotheses": [{"id": "h1"}]}))


def test_orchestrator_records_real_time_risk_assessment(tmp_path: Path) -> None:
    planner = AdaptivePlanner()
    gatekeeper = RuntimeGatekeeper()
    orchestrator = Orchestrator(
        planner=planner,
        critic=_critic(),
        tools={"dummy": DummyTool()},
        memory=_memory(tmp_path),
        world_model=_world_model(),
        gatekeeper=gatekeeper,
        working_dir=tmp_path,
    )

    report = asyncio.run(
        orchestrator.run({"goal": "test", "hypotheses": [{"id": "h1"}]})
    )
    assert report.summary == "Completed run"

    manifests = list(tmp_path.glob("run_*/manifest.json"))
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    risks = manifest.get("risk_assessments", [])
    assert len(risks) >= 2
    assert risks[0]["approved"] is False
    assert any(entry["approved"] for entry in risks)

    working = orchestrator.working_memory
    assert working is not None
    assert working.attempts[0].risk_assessments
    assert working.attempts[0].risk_assessments[0]["approved"] is False


def test_gatekeeper_respects_evaluation_bias(tmp_path: Path) -> None:
    planner = StaticPlanner([_plan(step_tier="T1")])
    world_model = _world_model()
    world_model.update(
        [
            {
                "claim_id": "evaluation::dummy",
                "passed": False,
                "weight": 2.0,
                "provenance": [],
            }
        ]
    )
    gatekeeper = Gatekeeper(
        policy={
            "evaluation_rules": [
                {
                    "claim": "evaluation::dummy",
                    "max_tier": "T0",
                    "min_credence": 0.6,
                    "tools": ["dummy"],
                }
            ]
        },
        world_model=world_model,
    )
    orchestrator = Orchestrator(
        planner=planner,
        critic=_critic(),
        tools={"dummy": DummyTool()},
        memory=_memory(tmp_path),
        world_model=world_model,
        gatekeeper=gatekeeper,
        working_dir=tmp_path,
    )

    with pytest.raises(PermissionError):
        asyncio.run(orchestrator.run({"goal": "test", "hypotheses": [{"id": "h1"}]}))

    world_model.update(
        [
            {
                "claim_id": "evaluation::dummy",
                "passed": True,
                "weight": 4.0,
                "provenance": [],
            }
        ]
    )

    report = asyncio.run(orchestrator.run({"goal": "test", "hypotheses": [{"id": "h1"}]}))
    assert report.summary == "Completed run"
