from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agi.src.core.critic import Critic
from agi.src.core.memory import MemoryStore
from agi.src.core.orchestrator import Orchestrator
from agi.src.core.planner import Planner
from agi.src.core.types import Plan, ToolCall, ToolResult, RunContext
from agi.src.core.world_model import WorldModel
from agi.src.governance.gatekeeper import Gatekeeper


class StaticPlanner(Planner):
    def __init__(self, plans: list[Plan]) -> None:
        super().__init__(llm=lambda payload: json.dumps({}))
        self._plans = plans

    async def plan_from(self, hypotheses, *, feedback=None):  # type: ignore[override]
        return list(self._plans)


class DummyTool:
    name = "dummy"
    safety = "T1"

    async def run(self, args, ctx: RunContext) -> ToolResult:
        return ToolResult(call_id=args.get("id", "dummy"), ok=True, stdout="done")


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

    manifests = list(tmp_path.glob("run_*/manifest.json"))
    assert manifests, "expected manifest to be written"
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    audit = manifest.get("safety_audit")
    assert audit and audit[0]["approved"] is True
    assert audit[0]["effective_level"] == "T1"


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
