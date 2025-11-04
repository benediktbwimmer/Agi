# Oversight Console

The oversight console provides a lightweight service for human-in-the-loop review
of orchestrator runs. It ingests manifests, working-memory snapshots, reflection
summaries, telemetry events, and episodic memory slices, and exposes a live
approval queue that integrates with the interactive gatekeeper.

## Launching the service

```bash
agi-cli oversight serve --artifacts artifacts/ --host 0.0.0.0 --port 8080
```

When `--auto-ingest` is left enabled (the default) the server scans the supplied
artifacts directory for `run_*` folders and ingests `manifest.json`,
`working_memory.json`, and `reflection_summary.json` if present. If an episodic
memory log (`memory.jsonl`) exists it is loaded under the `episodic` namespace.

State persists to SQLite (defaults to `ARTIFACTS/oversight.db`) so telemetry,
runs, and approval history survive restarts. Use `--db` to point to a custom
path.

Open `http://localhost:8080/` in a browser for the console UI. The dashboard
lists runs, surfaces run artefacts, streams telemetry, and shows the active
approval queue with approve/deny controls.

The service exposes a FastAPI application with the following endpoints:

- `GET /runs` – list stored runs with high-level metadata.
- `GET /runs/{run_id}` – return the manifest, working-memory snapshot, and
  reflection summary for a specific run.
- `POST /runs/ingest` – ingest a new run directory on demand.
- `GET /memory` / `POST /memory/ingest` – inspect and load episodic memory logs.
- `GET /telemetry` / `POST /telemetry` – stream orchestrator telemetry events into
  the console.
- `GET /approvals/pending` – view the live approval queue for gated tool calls.
- `POST /approvals/{approval_id}` – record a human approval or denial decision.
- `GET /approvals/decisions` – review historical approval outcomes.

Use the auto-generated interactive docs served at `/docs` to explore and invoke
endpoints while the server is running.

## Integrating the interactive gatekeeper

The interactive gatekeeper wraps the baseline policy checks with a manual
approval flow. Instantiate it with the same `OversightStore` that backs the
console:

```python
from agi.src.governance.interactive_gatekeeper import InteractiveGatekeeper
from agi.src.oversight.store import OversightStore

store = OversightStore()
console_app = create_oversight_app(store)  # optional: serve via uvicorn

gatekeeper = InteractiveGatekeeper(
    oversight_store=store,
    policy={"max_tier": "T3"},
    interactive_min_tier="T1",
    timeout_s=300,
)
```

Pass the gatekeeper to the orchestrator. When a plan step requests a tier higher
than the configured interactive threshold the run will pause until a reviewer
records a decision through the console.

### Telemetry integration

Attach the `OversightSink` to the orchestrator telemetry dispatcher to mirror
real-time events into the console:

```python
from agi.src.core.telemetry import Telemetry, OversightSink

telemetry = Telemetry(sinks=[OversightSink(store)])
```

## Approval workflow

Each interactive review generates an `ApprovalRequest` with metadata describing
the plan, step, tool, and requested tier. Reviewers respond with
`ApprovalDecision` objects via the `/approvals/{approval_id}` endpoint. Decisions
are persisted in memory for auditability and future heuristics.

If no decision is recorded before the configured timeout the gatekeeper raises a
`TimeoutError`, causing the orchestrator run to abort. Tune the timeout to align
with your operational expectations.

## Next steps

- Persist the oversight store to durable storage so approvals and telemetry
  survive restarts.
- Add a browser-based UI that consumes the FastAPI endpoints and renders plan
  hierarchies, working-memory timelines, and approval dialogs.
- Extend approval decisions with templated rationale categories to feed future
  planner biasing heuristics.
