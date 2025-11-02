from __future__ import annotations

from agi.src.core.manifest import RunManifest
from agi.src.core.types import Report, ToolResult, Source
from agi.src.memory.experience import summarise_experience


def _manifest() -> RunManifest:
    return RunManifest.build(
        run_id="run-test",
        goal={"goal": "demo goal"},
        constraints={},
        tool_results=[
            ToolResult(call_id="call-1", ok=True, stdout="first", provenance=[Source("tool", "call-1")]),
            ToolResult(call_id="call-2", ok=False, stdout="second", provenance=[]),
        ],
        belief_updates=[],
        safety_audit=[],
        risk_assessments=[],
        critiques=[],
        tool_catalog=[],
    )


def test_summarise_experience_produces_structured_chunk():
    manifest = _manifest()
    report = Report(
        goal="demo goal",
        summary="Completed run",
        key_findings=["finding"],
        belief_deltas=[],
        artifacts=["artifact"],
    )

    chunk = summarise_experience(report, manifest)

    assert chunk["goal"] == "demo goal"
    assert chunk["success"] is False
    assert chunk["tool_invocations"] == 2
    assert chunk["tool_results"][0]["call_id"] == "call-1"
    assert chunk["key_findings"] == ["finding"]

