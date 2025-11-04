from __future__ import annotations

import json
from pathlib import Path

from agi.src.core.manifest import RunManifest
from agi.src.core.types import ToolResult
from fastapi.testclient import TestClient

from agi.src.oversight.models import ApprovalDecision, ApprovalRequest
from agi.src.oversight.server import create_app
from agi.src.oversight.store import OversightStore


def _write_run(tmp_path: Path) -> tuple[str, Path]:
    run_id = "run-store-test"
    run_dir = tmp_path / f"run_{run_id}"
    run_dir.mkdir()
    manifest = RunManifest.build(
        run_id=run_id,
        goal={"goal": "demo"},
        constraints={},
        tool_results=[ToolResult(call_id="call-1", ok=True, stdout="pass", provenance=[])],
        belief_updates=[],
        safety_audit=[],
        risk_assessments=[],
        critiques=[],
        tool_catalog=[],
        plans=[],
        agents=[],
    )
    manifest.write(run_dir / "manifest.json")
    (run_dir / "working_memory.json").write_text(json.dumps({"attempts": []}), encoding="utf-8")
    (run_dir / "reflection_summary.json").write_text(json.dumps({"goal": "demo"}), encoding="utf-8")
    return run_id, run_dir


def test_store_ingests_run_and_records_metadata(tmp_path: Path) -> None:
    store = OversightStore()
    run_id, run_dir = _write_run(tmp_path)

    stored_id = store.ingest_run_dir(run_dir)
    assert stored_id == run_id

    runs = store.runs()
    assert runs and runs[0]["run_id"] == run_id

    payload = store.get_run(run_id)
    assert payload["manifest"]["run_id"] == run_id
    assert payload["working_memory"]["attempts"] == []
    assert payload["reflection_summary"]["goal"] == "demo"


def test_store_tracks_memory_and_telemetry(tmp_path: Path) -> None:
    store = OversightStore()
    memory_path = tmp_path / "memory.jsonl"
    memory_path.write_text(json.dumps({"id": "rec-1", "type": "episode"}) + "\n", encoding="utf-8")

    store.ingest_memory("default", memory_path)
    logs = store.memory_logs()
    assert "default" in logs
    assert logs["default"][0]["id"] == "rec-1"

    store.record_telemetry({"event": "demo", "value": 1})
    store.record_telemetry({"event": "demo", "value": 2})
    telemetry = store.telemetry()
    assert len(telemetry) == 2
    assert telemetry[-1]["value"] == 2


def test_store_persists_state_with_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "oversight.db"
    store = OversightStore(db_path=db_path)
    run_id, run_dir = _write_run(tmp_path)
    store.ingest_run_dir(run_dir)

    memory_path = tmp_path / "mem.jsonl"
    memory_path.write_text(json.dumps({"id": "persist", "type": "episode"}) + "\n", encoding="utf-8")
    store.ingest_memory("episodic", memory_path)

    request = ApprovalRequest.build(tool="review", tier="T2", requested_by="tester", context={"plan": "p1"})
    ticket = store.create_approval_request(request)
    decision = ApprovalDecision.build(
        approval_id=ticket.request.id,
        approved=True,
        reviewer="human",
        message="ok",
    )
    store.resolve_approval(ticket.request.id, decision)
    store.record_telemetry({"event": "demo"})

    # Re-open store to ensure state loads from disk
    reopened = OversightStore(db_path=db_path)
    runs = reopened.runs()
    assert any(run["run_id"] == run_id for run in runs)
    assert reopened.memory_logs()["episodic"][0]["id"] == "persist"
    decisions = reopened.list_decisions()
    assert decisions and decisions[0].approval_id == ticket.request.id
    telemetry = reopened.telemetry()
    assert telemetry and telemetry[-1]["event"] == "demo"


def test_oversight_server_serves_ui(tmp_path: Path) -> None:
    store = OversightStore(db_path=tmp_path / "oversight.db")
    app = create_app(store)
    client = TestClient(app)

    resp = client.get("/")
    assert resp.status_code == 200
    assert "Oversight Console" in resp.text

    run_id, run_dir = _write_run(tmp_path)
    store.ingest_run_dir(run_dir)

    runs = client.get("/runs").json()
    assert any(run["run_id"] == run_id for run in runs)
