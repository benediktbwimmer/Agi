from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agi.src.core.critic import Critic
from agi.src.core.memory import MemoryStore
from agi.src.core.orchestrator import Orchestrator
from agi.src.core.planner import Planner
from agi.src.core.telemetry import InMemorySink, JsonLinesSink, Telemetry
from agi.src.core.types import ToolResult
from agi.src.core.world_model import WorldModel


class _StubTool:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.safety = "T0"

    async def run(self, args, ctx):  # pragma: no cover - exercised indirectly
        return ToolResult(
            call_id=args.get("id", "call"),
            ok=self.ok,
            stdout="done",
            wall_time_ms=5,
            provenance=[],
        )


def _build_orchestrator(tmp_path: Path, sink: InMemorySink) -> Orchestrator:
    planner_payload = {
        "plans": [
            {
                "id": "plan-1",
                "claim_ids": ["claim-1"],
                "steps": [
                    {
                        "id": "step-1",
                        "tool": "worker",
                        "args": {},
                    }
                ],
            }
        ]
    }
    planner = Planner(llm=lambda payload: json.dumps(planner_payload))
    critic = Critic(llm=lambda plan: json.dumps({"status": "PASS"}))
    telemetry = Telemetry(sinks=[sink])
    memory = MemoryStore(tmp_path / "memory.jsonl", telemetry=telemetry)
    world_model = WorldModel()
    orchestrator = Orchestrator(
        planner=planner,
        critic=critic,
        tools={"worker": _StubTool()},
        memory=memory,
        world_model=world_model,
        working_dir=tmp_path,
        telemetry=telemetry,
    )
    return orchestrator


def test_orchestrator_emits_structured_events(tmp_path: Path) -> None:
    sink = InMemorySink()
    orchestrator = _build_orchestrator(tmp_path, sink)

    asyncio.run(orchestrator.run({"goal": "demo"}, {}))

    events = sink.events
    event_types = [event["event"] for event in events]
    assert "orchestrator.run_started" in event_types
    assert "orchestrator.tool_started" in event_types
    assert "orchestrator.tool_completed" in event_types
    assert "memory.append" in event_types
    assert events[-1]["event"] == "orchestrator.run_completed"


def test_json_lines_sink_writes_file(tmp_path: Path) -> None:
    sink = JsonLinesSink(tmp_path / "telemetry" / "events.jsonl")
    telemetry = Telemetry(sinks=[sink])
    telemetry.emit("demo", answer=42)
    telemetry.emit("demo", answer=43)

    data = (tmp_path / "telemetry" / "events.jsonl").read_text().splitlines()
    assert len(data) == 2
    first = json.loads(data[0])
    assert first["event"] == "demo"
    assert first["answer"] == 42
