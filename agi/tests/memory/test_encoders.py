from __future__ import annotations

from agi.src.core.types import ToolResult
from agi.src.memory.encoders import summarise_tool_result


def test_summarise_tool_result_extracts_keywords_and_tokens() -> None:
    result = ToolResult(
        call_id="call",
        ok=True,
        stdout="Analyzed lunar regolith sample for oxygen content",
        wall_time_ms=42,
        provenance=[],
    )

    summary = summarise_tool_result(result)

    assert summary["summary"].startswith("Analyzed lunar regolith")
    assert summary["tokens"] == 7
    assert "keywords" in summary
    assert "lunar" in summary["keywords"]
    assert summary["claims"]
    assert summary["claims"][0].startswith("Analyzed lunar regolith")
    embedding = summary.get("embedding")
    assert embedding is not None
    assert embedding["dim"] == 512
    assert embedding["values"]


def test_summarise_tool_result_handles_structured_outputs() -> None:
    payload = {"metrics": {"accuracy": 0.9}, "notes": "demo"}
    result = ToolResult(
        call_id="call",
        ok=True,
        stdout='{"accuracy": 0.9, "status": "ok"}',
        data=payload,
        provenance=[],
    )

    summary = summarise_tool_result(result)

    structured = summary.get("structured", {})
    assert set(structured.get("keys", [])) == {"accuracy", "status"}
    assert set(structured.get("data_keys", [])) == {"metrics", "notes"}
    metrics = structured.get("metrics", {})
    assert metrics
    assert metrics["metrics.accuracy"] == 0.9
