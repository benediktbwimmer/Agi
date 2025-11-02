from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from agi.cli import app
from agi.src.core.manifest import RunManifest
from agi.src.core.memory import MemoryStore
from agi.src.core.safety import RiskAssessment, SafetyDecision
from agi.src.core.tools import SensorProfile, ToolCapability, ToolParameter, ToolSpec
from agi.src.core.types import ToolResult


runner = CliRunner()


def _write_manifest(path: Path) -> None:
    manifest = RunManifest.build(
        run_id="run-1",
        goal={"goal": "demo"},
        constraints={},
        tool_results=[
            ToolResult(call_id="call-1", ok=True, stdout="ok", provenance=[]),
            ToolResult(call_id="call-2", ok=False, stdout="fail", provenance=[]),
        ],
        belief_updates=[],
        safety_audit=[
            SafetyDecision(
                plan_id="plan-1",
                step_id="step-1",
                tool_name="calculator",
                requested_level="T0",
                tool_level="T0",
                effective_level="T0",
                approved=True,
            ),
            SafetyDecision(
                plan_id="plan-2",
                step_id="step-2",
                tool_name="python_runner",
                requested_level="T2",
                tool_level="T1",
                effective_level="T2",
                approved=False,
                reason="Denied for test",
            ),
        ],
        risk_assessments=[
            RiskAssessment(
                plan_id="plan-1",
                step_id="step-1",
                tool_name="calculator",
                requested_level="T0",
                tool_level="T0",
                effective_level="T0",
                approved=True,
            ),
            RiskAssessment(
                plan_id="plan-2",
                step_id="step-2",
                tool_name="python_runner",
                requested_level="T2",
                tool_level="T1",
                effective_level="T2",
                approved=False,
                reason="Risk flagged",
            ),
        ],
        tool_catalog=[
            ToolSpec(
                name="calculator",
                description="Deterministic arithmetic evaluator",
                safety_tier="T0",
                capabilities=(
                    ToolCapability(
                        name="evaluate",
                        description="Evaluate arithmetic expressions",
                        safety_tier="T0",
                        parameters=(
                            ToolParameter(
                                name="expression",
                                description="Expression to evaluate",
                                schema={"type": "string"},
                            ),
                        ),
                        outputs=("stdout",),
                    ),
                ),
                metadata={"category": "math"},
                sensor_profile=SensorProfile(
                    modality="symbolic",
                    latency_ms=10,
                    trust="high",
                    description="Pure computation",
                ),
            ),
            ToolSpec(
                name="python_runner",
                description="Sandboxed Python executor",
                safety_tier="T1",
                capabilities=(
                    ToolCapability(
                        name="execute_python",
                        description="Run Python snippets in sandbox",
                        safety_tier="T1",
                        parameters=(
                            ToolParameter(
                                name="code",
                                description="Python code",
                                schema={"type": "string"},
                            ),
                        ),
                        outputs=("stdout",),
                    ),
                ),
                metadata={"category": "execution"},
                sensor_profile=SensorProfile(
                    modality="code_execution",
                    latency_ms=200,
                    trust="medium",
                    description="Hermetic sandbox",
                ),
            ),
        ],
    )
    manifest.write(path)


def test_manifest_validate_command(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)

    result = runner.invoke(app, ["manifest", "validate", str(manifest_path)])

    assert result.exit_code == 0
    assert "conforms to schema version" in result.stdout


def test_manifest_validate_failure(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    result = runner.invoke(app, ["manifest", "validate", str(manifest_path)])

    assert result.exit_code != 0
    assert "Manifest validation failed" in result.stderr


def test_manifest_summary_command(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)

    result = runner.invoke(app, ["manifest", "summary", str(manifest_path), "--sample", "1"])

    assert result.exit_code == 0
    summary = json.loads(result.stdout)
    assert summary["tool_invocations"] == 2
    assert summary["tool_failures"] == 1
    assert summary["safety_decisions"]["blocked"] == 1
    assert summary["risk_assessments"]["blocked"] == 1
    assert len(summary["sample_tool_results"]) == 1


def test_tools_catalog_command(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)

    result = runner.invoke(app, ["tools", "catalog", str(manifest_path)])

    assert result.exit_code == 0
    catalog = json.loads(result.stdout)
    assert {tool["name"] for tool in catalog} == {"calculator", "python_runner"}

    filtered = runner.invoke(app, ["tools", "catalog", str(manifest_path), "--tier", "T1"])
    assert filtered.exit_code == 0
    filtered_catalog = json.loads(filtered.stdout)
    assert len(filtered_catalog) == 1
    assert filtered_catalog[0]["name"] == "python_runner"


def test_memory_recent_command(tmp_path: Path) -> None:
    memory_path = tmp_path / "memory.jsonl"
    store = MemoryStore(memory_path)
    store.append(
        {
            "type": "episode",
            "tool": "calculator",
            "time": "2024-01-01T00:00:00+00:00",
            "stdout": "Computed trajectory",
        }
    )

    result = runner.invoke(app, ["memory", "recent", str(memory_path)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload[0]["tool"] == "calculator"


def test_memory_replay_command(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)

    result = runner.invoke(app, ["memory", "replay", str(manifest_path)])

    assert result.exit_code == 0
    chunk = json.loads(result.stdout)
    assert chunk["goal"] == "demo"
    assert chunk["tool_invocations"] == 2


def test_memory_search_command(tmp_path: Path) -> None:
    memory_path = tmp_path / "memory.jsonl"
    store = MemoryStore(memory_path)
    store.append(
        {
            "type": "reflection",
            "summary": "Analysed lunar samples",
            "time": "2024-01-01T00:00:00+00:00",
        }
    )

    result = runner.invoke(app, ["memory", "search", str(memory_path), "lunar"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload[0]["summary"].startswith("Analysed")


def test_memory_timeline_command(tmp_path: Path) -> None:
    memory_path = tmp_path / "memory.jsonl"
    store = MemoryStore(memory_path)
    store.append(
        {
            "type": "episode",
            "tool": "calculator",
            "time": "2024-01-01T00:00:00+00:00",
            "stdout": "baseline",
        }
    )
    store.append(
        {
            "type": "reflection",
            "summary": "Follow-up reflection",
            "time": "2024-01-01T00:05:00+00:00",
        }
    )
    store.append(
        {
            "type": "episode",
            "tool": "python_runner",
            "time": "2024-01-01T00:10:00+00:00",
            "stdout": "another run",
        }
    )

    result = runner.invoke(app, ["memory", "timeline", str(memory_path), "--limit", "2"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "records" in payload
    assert len(payload["records"]) == 2
    assert payload["coverage"]["episode"] >= 1


def test_memory_reflect_command(tmp_path: Path) -> None:
    memory_path = tmp_path / "memory.jsonl"
    store = MemoryStore(memory_path)
    store.append(
        {
            "type": "reflection_insight",
            "goal": "moon",
            "time": "2024-01-01T00:00:00+00:00",
            "insights": {
                "critique_tags": ["safety"],
                "final_status": "failed",
                "risk_events": 1,
                "attempt_count": 2,
            },
        }
    )

    result = runner.invoke(app, ["memory", "reflect", str(memory_path), "--goal", "moon"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["goal"] == "moon"
    assert payload["total_runs"] == 1
    assert payload["sample_size"] == 1


def test_memory_reflect_write_back(tmp_path: Path) -> None:
    memory_path = tmp_path / "memory.jsonl"
    store = MemoryStore(memory_path)
    store.append(
        {
            "type": "reflection_insight",
            "goal": "mars",
            "time": "2024-01-01T00:00:00+00:00",
            "insights": {
                "critique_tags": [],
                "final_status": "complete",
                "risk_events": 0,
                "attempt_count": 1,
            },
        }
    )

    result = runner.invoke(
        app,
        ["memory", "reflect", str(memory_path), "--goal", "mars", "--write-back"],
    )

    assert result.exit_code == 0
    with memory_path.open("r", encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh if line.strip()]
    assert any(record.get("type") == "reflection_summary" for record in lines)


def test_working_summarize_command(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "working_memory.json"
    payload = {
        "run_id": "run-123",
        "goal": "demo-run",
        "hypotheses": [],
        "attempts": [
            {
                "index": 0,
                "status": "complete",
                "plans": [
                    {
                        "plan_id": "plan-1",
                        "claim_ids": [],
                        "step_count": 1,
                        "approved": True,
                        "execution_succeeded": True,
                        "rationale_tags": ["safety"],
                    }
                ],
                "critiques": [],
                "execution_feedback": [],
                "risk_assessments": [],
            }
        ],
    }
    snapshot_path.write_text(json.dumps(payload), encoding="utf-8")

    result = runner.invoke(app, ["working", "summarize", str(snapshot_path)])

    assert result.exit_code == 0
    summary = json.loads(result.stdout)
    assert summary["run_id"] == "run-123"
    assert summary["final_status"] == "complete"
    assert summary["plan_outcomes"]["approved"] == ["plan-1"]


def test_run_inspect_command(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    manifest_path = run_dir / "manifest.json"
    _write_manifest(manifest_path)

    working_payload = {
        "run_id": "run-1",
        "goal": "demo",
        "hypotheses": [],
        "attempts": [
            {
                "index": 0,
                "status": "complete",
                "plans": [
                    {
                        "plan_id": "plan-1",
                        "claim_ids": [],
                        "step_count": 1,
                        "approved": True,
                        "execution_succeeded": True,
                        "rationale_tags": ["safety"],
                    }
                ],
                "critiques": [],
                "execution_feedback": [],
                "risk_assessments": [],
            }
        ],
    }
    (run_dir / "working_memory.json").write_text(json.dumps(working_payload), encoding="utf-8")
    (run_dir / "reflection_summary.json").write_text(
        json.dumps({"goal": "demo", "sample_size": 1}), encoding="utf-8"
    )

    memory_path = tmp_path / "memory.jsonl"
    store = MemoryStore(memory_path)
    store.append(
        {
            "type": "reflection_insight",
            "goal": "demo",
            "time": "2024-01-01T00:00:00+00:00",
            "insights": {
                "critique_tags": [],
                "final_status": "complete",
                "risk_events": 0,
                "attempt_count": 1,
            },
        }
    )
    store.append(
        {
            "type": "episode",
            "tool": "calculator",
            "time": "2024-01-01T00:10:00+00:00",
            "stdout": "Demo stdout",
        }
    )

    result = runner.invoke(
        app,
        [
            "run",
            "inspect",
            str(run_dir),
            "--memory",
            str(memory_path),
            "--memory-limit",
            "2",
            "--sample",
            "1",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["manifest"]["tool_invocations"] == 2
    assert payload["working_memory"]["run_id"] == "run-1"
    assert payload["reflection_summary"]["goal"] == "demo"
    assert payload["memory_context"]["goal"] == "demo"
