from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agi.src.core.critic import Critic
from agi.src.core.memory import MemoryStore
from agi.src.core.orchestrator import Orchestrator
from agi.src.core.planner import Planner
from agi.src.core.telemetry import InMemorySink, JsonLinesSink, Telemetry
from agi.src.core.types import ToolResult
from agi.src.core.world_model import WorldModel
from agi.src.governance.gatekeeper import Gatekeeper


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


def _build_orchestrator(
    tmp_path: Path,
    sink: InMemorySink,
    *,
    gatekeeper: Gatekeeper | None = None,
) -> Orchestrator:
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
    memory.append(
        {
            "type": "reflection_insight",
            "goal": "demo",
            "time": "2024-01-01T00:00:00+00:00",
             "id": "reflection-1",
            "summary": "baseline",
            "insights": {
                "final_status": "needs_replan",
                "critique_tags": ["safety"],
                "risk_events": 1,
                "attempt_count": 1,
            },
        }
    )
    world_model = WorldModel()
    orchestrator = Orchestrator(
        planner=planner,
        critic=critic,
        tools={"worker": _StubTool()},
        memory=memory,
        world_model=world_model,
        working_dir=tmp_path,
        telemetry=telemetry,
        gatekeeper=gatekeeper,
    )
    return orchestrator


def test_orchestrator_emits_structured_events(tmp_path: Path) -> None:
    sink = InMemorySink()
    orchestrator = _build_orchestrator(tmp_path, sink)

    asyncio.run(
        orchestrator.run(
            {"goal": "demo", "hypotheses": [{"id": "hyp-1"}], "memory_context_limit": 3},
            {},
        )
    )

    events = sink.events
    event_types = [event["event"] for event in events]
    assert "orchestrator.run_started" in event_types
    assert "orchestrator.tool_started" in event_types
    assert "orchestrator.tool_completed" in event_types
    assert "orchestrator.plan_step_started" in event_types
    assert "orchestrator.plan_step_completed" in event_types
    assert "orchestrator.input_provenance_ready" in event_types
    assert "orchestrator.working_memory_persisted" in event_types
    assert "orchestrator.reflection_consolidated" in event_types
    assert "memory.append" in event_types
    assert "memory.query" in event_types
    assert events[-1]["event"] == "orchestrator.run_completed"

    tool_started = next(event for event in events if event["event"] == "orchestrator.tool_started")
    assert "provenance" in tool_started
    assert tool_started["provenance"]["plan"]["id"]
    assert tool_started.get("references", {}).get("hypotheses") == ["hyp-1"]
    memory_refs = tool_started.get("references", {}).get("memory_records")
    assert memory_refs == ["reflection-1"]

    tool_completed = next(event for event in events if event["event"] == "orchestrator.tool_completed")
    assert tool_completed["duration_ms"] >= 0
    assert tool_completed["wall_time_ms"] is not None

    query_event = next(event for event in events if event["event"] == "memory.query")
    assert query_event["duration_ms"] >= 0


def test_orchestrator_emits_risk_events(tmp_path: Path) -> None:
    sink = InMemorySink()
    gatekeeper = Gatekeeper(policy={})
    orchestrator = _build_orchestrator(tmp_path, sink, gatekeeper=gatekeeper)

    asyncio.run(orchestrator.run({"goal": "demo"}, {}))

    events = [event for event in sink.events if event["event"] == "orchestrator.risk_assessed"]
    assert events, "expected risk assessment event"
    assert events[0]["approved"] is True
    assert events[0]["effective_level"] == "T0"


def test_orchestrator_emits_tool_failure_event(tmp_path: Path) -> None:
    sink = InMemorySink()
    orchestrator = _build_orchestrator(tmp_path, sink)
    orchestrator.tools["worker"] = _StubTool(ok=False)

    with pytest.raises(RuntimeError):
        asyncio.run(orchestrator.run({"goal": "demo"}, {}))

    failures = [event for event in sink.events if event["event"] == "orchestrator.tool_failed"]
    assert failures, "expected tool failure telemetry"
    assert failures[0]["tool"] == "worker"
    assert failures[0]["duration_ms"] >= 0


def test_memory_queries_emit_duration(tmp_path: Path) -> None:
    sink = InMemorySink()
    telemetry = Telemetry(sinks=[sink])
    store = MemoryStore(tmp_path / "records.jsonl", telemetry=telemetry)
    store.append(
        {
            "type": "episode",
            "plan_id": "plan-1",
            "tool": "worker",
            "call_id": "step-1",
            "time": "2024-01-01T00:00:00+00:00",
        }
    )

    results = store.query_by_plan("plan-1")
    assert results

    query_events = [
        event for event in sink.events if event["event"] == "memory.query" and event.get("method") == "plan"
    ]
    assert query_events, "expected query telemetry"
    assert query_events[0]["result_count"] == 1
    assert query_events[0]["duration_ms"] >= 0


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
