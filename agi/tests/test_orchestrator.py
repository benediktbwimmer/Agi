from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from agi.src.core.memory import MemoryStore
from agi.src.core.orchestrator import Orchestrator
from agi.src.core.tools.python_runner import PythonRunner
from agi.src.core.types import Plan, ToolCall
from agi.src.core.world_model import WorldModel


class PlannerStub:
    async def plan_from(self, hypotheses):
        return [
            Plan(
                id="plan-1",
                claim_ids=["claim-1"],
                steps=[
                    ToolCall(
                        id="step-1",
                        tool="python_runner",
                        args={"code": "print('stub')"},
                        safety_level="T0",
                    )
                ],
                expected_cost={},
                risks=[],
                ablations=[],
            )
        ]


class CriticStub:
    async def check(self, plan):
        return {"status": "PASS"}


def test_orchestrator_runs_end_to_end(tmp_path: Path) -> None:
    orchestrator = Orchestrator(
        planner=PlannerStub(),
        critic=CriticStub(),
        tools={"python_runner": PythonRunner(artifacts_root=tmp_path / "artifacts")},
        memory=MemoryStore(tmp_path / "memory.jsonl"),
        world_model=WorldModel(),
        working_dir=tmp_path,
    )
    report = asyncio.run(orchestrator.run({"goal": "demo", "hypotheses": [{}]}))
    assert report.goal == "demo"
    assert report.artifacts
