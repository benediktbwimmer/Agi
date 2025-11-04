from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .models import ApprovalDecision
from .store import OversightStore


class IngestRunPayload(BaseModel):
    run_dir: str = Field(..., description="Path to a run_* directory containing a manifest.json")


class IngestMemoryPayload(BaseModel):
    name: str = Field(..., description="Identifier for the memory log")
    path: str = Field(..., description="Path to a JSONL memory file")
    limit: Optional[int] = Field(default=200, description="Maximum records to ingest")


class TelemetryPayload(BaseModel):
    event: Dict[str, Any]


class ApprovalDecisionPayload(BaseModel):
    approved: bool
    reviewer: str
    message: Optional[str] = None


def create_app(store: Optional[OversightStore] = None) -> FastAPI:
    oversight_store = store or OversightStore()

    app = FastAPI(title="AGI Oversight Console", version="0.1.0")

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir, html=False), name="static")

        @app.get("/", response_class=HTMLResponse)
        def index() -> HTMLResponse:
            index_path = static_dir / "index.html"
            if index_path.exists():
                return HTMLResponse(index_path.read_text(encoding="utf-8"))
            return HTMLResponse("<h1>Oversight console</h1>")
    else:

        @app.get("/", response_class=HTMLResponse)
        def fallback_index() -> HTMLResponse:  # pragma: no cover - development fallback
            return HTMLResponse("<h1>Oversight console assets missing</h1>")

    @app.get("/runs")
    def list_runs() -> Any:
        return oversight_store.runs()

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> Any:
        try:
            return oversight_store.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found") from exc

    @app.post("/runs/ingest")
    def ingest_run(payload: IngestRunPayload) -> Any:
        run_path = Path(payload.run_dir)
        run_id = oversight_store.ingest_run_dir(run_path)
        return {"run_id": run_id}

    @app.post("/memory/ingest")
    def ingest_memory(payload: IngestMemoryPayload) -> Any:
        oversight_store.ingest_memory(payload.name, Path(payload.path), limit=payload.limit or 200)
        return {"name": payload.name, "records": len(oversight_store.memory_logs().get(payload.name, []))}

    @app.get("/memory")
    def memory_logs() -> Any:
        return oversight_store.memory_logs()

    @app.get("/telemetry")
    def telemetry(limit: Optional[int] = None) -> Any:
        return oversight_store.telemetry(limit=limit)

    @app.post("/telemetry")
    def append_telemetry(payload: TelemetryPayload) -> Any:
        oversight_store.record_telemetry(payload.event)
        return {"stored": True}

    @app.get("/approvals/pending")
    def pending_approvals() -> Any:
        return [request.__dict__ for request in oversight_store.list_pending_approvals()]

    @app.post("/approvals/{approval_id}")
    def resolve_approval(approval_id: str, payload: ApprovalDecisionPayload) -> Any:
        decision = ApprovalDecision.build(
            approval_id=approval_id,
            approved=payload.approved,
            reviewer=payload.reviewer,
            message=payload.message,
        )
        try:
            oversight_store.resolve_approval(approval_id, decision)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Approval {approval_id} not pending") from exc
        return {"approval_id": approval_id, "approved": payload.approved}

    @app.get("/approvals/decisions")
    def approval_history() -> Any:
        return [decision.__dict__ for decision in oversight_store.list_decisions()]

    return app


__all__ = ["create_app"]
