from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict

from agi.src.core.critic import Critic
from agi.src.core.manifest import MANIFEST_SCHEMA_VERSION, RunManifest
from agi.src.core.memory import MemoryStore
from agi.src.core.orchestrator import Orchestrator
from agi.src.core.planner import Planner
from agi.src.core.types import ToolResult
from agi.src.core.world_model import WorldModel


class StubTool:
    def __init__(self) -> None:
        self.safety = "T0"

    async def run(self, args: Dict[str, Any], ctx: Any):  # pragma: no cover - exercised via orchestrator
        return ToolResult(
            call_id=args.get("id", "call"),
            ok=True,
            stdout="pass",
            wall_time_ms=1,
            provenance=[],
        )


def _basic_orchestrator(tmp_path: Path) -> Orchestrator:
    planner_response = {
        "plans": [
            {
                "id": "plan-1",
                "claim_ids": ["claim-1"],
                "steps": [
                    {
                        "id": "step-1",
                        "tool": "demo_tool",
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
        tools={"demo_tool": StubTool()},
        memory=memory,
        world_model=world_model,
        working_dir=tmp_path,
    )
    return orchestrator


def test_run_manifest_matches_schema(tmp_path):
    orchestrator = _basic_orchestrator(tmp_path)
    asyncio.run(orchestrator.run({"goal": "demo"}, {}))

    run_dirs = list(tmp_path.glob("run_*/manifest.json"))
    assert run_dirs, "manifest.json should be created"

    manifest_path = run_dirs[0]
    manifest_data = json.loads(manifest_path.read_text())
    manifest = RunManifest.model_validate(manifest_data)

    assert manifest.schema_version == MANIFEST_SCHEMA_VERSION
    assert manifest.goal["goal"] == "demo"
    assert manifest.tool_results[0].call_id == "step-1"
    assert manifest.belief_updates[0].claim_id == "claim-1"
    assert manifest.critiques == []
    assert manifest.tool_catalog
    assert manifest.tool_catalog[0].name == "demo_tool"
    assert manifest.tool_catalog[0].safety_tier == "T0"

    created = manifest.created_at
    assert created.endswith("+00:00"), "timestamp should include timezone"

    schema_file = manifest_path.with_name("manifest.schema.json")
    assert schema_file.exists()
    json.loads(schema_file.read_text())  # should be valid JSON


def test_run_manifest_roundtrip(tmp_path):
    orchestrator = _basic_orchestrator(tmp_path)
    asyncio.run(orchestrator.run({"goal": "demo"}, {}))

    manifest_path = next(tmp_path.glob("run_*/manifest.json"))
    manifest = RunManifest.model_validate_json(manifest_path.read_text())

    regenerated = RunManifest.model_validate_json(manifest.model_dump_json())
    assert regenerated.model_dump() == manifest.model_dump()
