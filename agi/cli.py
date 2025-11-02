from __future__ import annotations

"""Developer-facing CLI utilities for the Putnam-inspired AGI stack."""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import typer
from pydantic import ValidationError

from agi.src.core.manifest import RunManifest
from agi.src.core.memory import MemoryStore
from agi.src.core.memory_retrieval import MemoryRetriever as StructuredMemoryRetriever
from agi.src.core.orchestrator import WorkingMemory
from agi.src.core.reflection import summarise_working_memory
from agi.src.core.types import Report
from agi.src.memory.experience import summarise_experience
from agi.src.memory.reflection_job import consolidate_reflections


app = typer.Typer(help="Utility commands for inspecting AGI artefacts.")
manifest_app = typer.Typer(help="Work with run manifests.")
memory_app = typer.Typer(help="Inspect episodic memory logs.")
working_app = typer.Typer(help="Summarise working-memory deliberation snapshots.")
tools_app = typer.Typer(help="List tool catalog metadata and sensor profiles.")
run_app = typer.Typer(help="Inspect orchestrator run directories.")
app.add_typer(manifest_app, name="manifest")
app.add_typer(memory_app, name="memory")
app.add_typer(working_app, name="working")
app.add_typer(tools_app, name="tools")
app.add_typer(run_app, name="run")


def _load_manifest(path: Path) -> RunManifest:
    """Load and validate a run manifest from ``path``."""

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        typer.secho(f"Failed to read manifest: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    try:
        return RunManifest.model_validate_json(content)
    except ValidationError as exc:
        typer.secho("Manifest validation failed:", err=True, fg=typer.colors.RED)
        typer.echo(exc)
        raise typer.Exit(code=1) from exc


def _format_records(records: Iterable[dict]) -> str:
    """Return a compact JSON representation of memory records."""

    serialised: List[dict] = []
    for record in records:
        serialised.append(record)
    return json.dumps(serialised, indent=2, ensure_ascii=True)


def _load_json_file(path: Path) -> Any:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        typer.secho(f"Failed to read {path}: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        typer.secho(f"Invalid JSON in {path}: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc


def _truncate_text(value: Optional[str], limit: int = 160) -> Optional[str]:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _goal_from_manifest(manifest: RunManifest) -> str:
    goal = manifest.goal
    if isinstance(goal, dict):
        for key in ("goal", "description", "text"):
            value = goal.get(key)
            if value:
                return str(value)
        return json.dumps(goal, ensure_ascii=True)
    if goal is None:
        return ""
    return str(goal)


def _summarise_manifest(manifest: RunManifest, *, sample_limit: int = 5) -> Dict[str, Any]:
    tool_results = list(manifest.tool_results or [])
    successes = sum(1 for result in tool_results if result.ok)
    failures = len(tool_results) - successes

    safety_decisions = list(manifest.safety_audit or [])
    safety_counts = Counter("approved" if decision.approved else "blocked" for decision in safety_decisions)
    risk_assessments = list(manifest.risk_assessments or [])
    risk_counts = Counter("approved" if assessment.approved else "blocked" for assessment in risk_assessments)

    sample: List[Dict[str, Any]] = []
    for result in tool_results[:sample_limit]:
        sample.append(
            {
                "call_id": result.call_id,
                "ok": result.ok,
                "cost_tokens": result.cost_tokens,
                "wall_time_ms": result.wall_time_ms,
                "stdout": _truncate_text(result.stdout),
            }
        )

    blocked_tools = [
        {
            "tool": decision.tool_name,
            "plan_id": decision.plan_id,
            "step_id": decision.step_id,
            "reason": decision.reason,
        }
        for decision in safety_decisions
        if not decision.approved
    ][:sample_limit]

    goal_field = manifest.goal
    goal_text = goal_field.get("goal") if isinstance(goal_field, dict) else goal_field

    return {
        "run_id": manifest.run_id,
        "created_at": manifest.created_at,
        "goal": goal_text,
        "tool_invocations": len(tool_results),
        "tool_successes": successes,
        "tool_failures": failures,
        "belief_updates": len(manifest.belief_updates or []),
        "safety_decisions": {
            "total": len(safety_decisions),
            "approved": safety_counts.get("approved", 0),
            "blocked": safety_counts.get("blocked", 0),
            "blocked_samples": blocked_tools,
        },
        "risk_assessments": {
            "total": len(risk_assessments),
            "approved": risk_counts.get("approved", 0),
            "blocked": risk_counts.get("blocked", 0),
        },
        "sample_tool_results": sample,
    }


def _summarise_tool_spec(spec: Any) -> Dict[str, Any]:
    capabilities: List[Dict[str, Any]] = []
    for capability in getattr(spec, "capabilities", []) or []:
        parameters = [param.name for param in getattr(capability, "parameters", []) or []]
        capabilities.append(
            {
                "name": capability.name,
                "description": capability.description,
                "safety_tier": capability.safety_tier,
                "parameters": parameters,
                "outputs": list(getattr(capability, "outputs", []) or []),
            }
        )

    sensor = getattr(spec, "sensor_profile", None)
    if sensor is None:
        sensor_profile = None
    elif isinstance(sensor, dict):
        sensor_profile = sensor
    else:
        sensor_profile = {
            "modality": getattr(sensor, "modality", None),
            "latency_ms": getattr(sensor, "latency_ms", None),
            "trust": getattr(sensor, "trust", None),
            "description": getattr(sensor, "description", None),
        }

    return {
        "name": spec.name,
        "description": getattr(spec, "description", spec.name),
        "safety_tier": spec.safety_tier,
        "version": getattr(spec, "version", None),
        "sensor_profile": sensor_profile,
        "capabilities": capabilities,
        "metadata": getattr(spec, "metadata", {}),
    }


@manifest_app.command("validate")
def validate_manifest(
    path: Path = typer.Argument(..., exists=True, resolve_path=True, help="Path to manifest.json"),
) -> None:
    """Validate ``manifest.json`` files written by the orchestrator."""

    manifest = _load_manifest(path)
    typer.secho(
        f"Manifest {path} conforms to schema version {manifest.schema_version}",
        fg=typer.colors.GREEN,
    )


@manifest_app.command("schema")
def write_manifest_schema(
    output: Path = typer.Argument(
        ..., resolve_path=True, help="Destination path for writing the manifest JSON schema."
    ),
) -> None:
    """Write the JSON schema describing run manifests to ``output``."""

    try:
        RunManifest.write_schema(output)
    except OSError as exc:
        typer.secho(f"Failed to write schema: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.secho(f"Wrote manifest schema to {output}", fg=typer.colors.GREEN)


@manifest_app.command("summary")
def manifest_summary(
    path: Path = typer.Argument(..., exists=True, resolve_path=True, help="Path to manifest.json"),
    sample: int = typer.Option(
        5,
        "--sample",
        "-n",
        min=1,
        help="Number of tool results to include in the sample section.",
    ),
) -> None:
    """Emit a condensed JSON summary of a manifest."""

    manifest = _load_manifest(path)
    summary = _summarise_manifest(manifest, sample_limit=sample)
    typer.echo(json.dumps(summary, indent=2, ensure_ascii=True))


@tools_app.command("catalog")
def tools_catalog(
    path: Path = typer.Argument(..., exists=True, resolve_path=True, help="Path to manifest.json"),
    tier: List[str] | None = typer.Option(
        None,
        "--tier",
        "-t",
        help="Optional safety-tier filter (e.g. T0).",
    ),
) -> None:
    """List tool specifications captured in a run manifest."""

    manifest = _load_manifest(path)
    catalog = list(manifest.tool_catalog or [])
    tiers = {value.upper() for value in tier or [] if value}

    payload: List[Dict[str, Any]] = []
    for spec in catalog:
        summary = _summarise_tool_spec(spec)
        if tiers and summary["safety_tier"].upper() not in tiers:
            continue
        payload.append(summary)

    typer.echo(json.dumps(payload, indent=2, ensure_ascii=True))


@memory_app.command("recent")
def recent_memory(
    path: Path = typer.Argument(
        ..., exists=True, resolve_path=True, help="Path to the memory JSONL file."
    ),
    limit: int = typer.Option(
        5,
        "--limit",
        "-n",
        min=1,
        help="Number of most recent records to display.",
    ),
    types: List[str] | None = typer.Option(
        None,
        "--type",
        "-t",
        help="Optional memory record types to filter by; may be provided multiple times.",
    ),
) -> None:
    """Display the ``limit`` most recent memory entries."""

    try:
        store = MemoryStore(path)
    except OSError as exc:
        typer.secho(f"Failed to open memory store: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    type_filter: Sequence[str] | None = types if types else None
    records = store.recent(limit=limit, types=type_filter)
    if not records:
        typer.echo("No memory records found.")
        return
    typer.echo(_format_records(records))


@memory_app.command("search")
def search_memory(
    path: Path = typer.Argument(
        ..., exists=True, resolve_path=True, help="Path to the memory JSONL file."
    ),
    query: str = typer.Argument(..., help="Lexical query used for semantic search."),
    limit: int = typer.Option(
        5,
        "--limit",
        "-n",
        min=1,
        help="Maximum number of records to return.",
    ),
    types: List[str] | None = typer.Option(
        None,
        "--type",
        "-t",
        help="Optional record types to filter by; may be provided multiple times.",
    ),
) -> None:
    """Run a semantic search over the episodic memory store."""

    try:
        store = MemoryStore(path)
    except OSError as exc:
        typer.secho(f"Failed to open memory store: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    type_filter: Sequence[str] | None = types if types else None
    records = store.semantic_search(query, limit=limit, types=type_filter)
    if not records:
        typer.echo("No matching memory records found.")
        raise typer.Exit(code=0)
    typer.echo(_format_records(records))


@memory_app.command("reflect")
def reflect_memory(
    path: Path = typer.Argument(
        ..., exists=True, resolve_path=True, help="Path to the memory JSONL file."
    ),
    goal: str | None = typer.Option(
        None,
        "--goal",
        "-g",
        help="Optional goal filter when consolidating reflection insights.",
    ),
    limit: int = typer.Option(
        200,
        "--limit",
        "-n",
        min=1,
        help="Maximum number of reflection insights to sample.",
    ),
    write_back: bool = typer.Option(
        False,
        "--write-back/--no-write-back",
        help="Persist the consolidated summary back into memory.",
    ),
) -> None:
    """Aggregate reflection insights stored in memory."""

    try:
        store = MemoryStore(path)
    except OSError as exc:
        typer.secho(f"Failed to open memory store: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    summary = consolidate_reflections(
        store,
        goal=goal,
        limit=limit,
        write_back=write_back,
    )
    typer.echo(json.dumps(summary, indent=2, ensure_ascii=True))


@memory_app.command("timeline")
def memory_timeline(
    path: Path = typer.Argument(
        ..., exists=True, resolve_path=True, help="Path to the memory JSONL file."
    ),
    start: Optional[str] = typer.Option(
        None,
        "--start",
        help="Inclusive ISO-8601 start timestamp for filtering.",
    ),
    end: Optional[str] = typer.Option(
        None,
        "--end",
        help="Inclusive ISO-8601 end timestamp for filtering.",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-n",
        min=1,
        help="Maximum number of records to return.",
    ),
    types: List[str] | None = typer.Option(
        None,
        "--type",
        "-t",
        help="Optional record types to filter by; may be provided multiple times.",
    ),
    tools: List[str] | None = typer.Option(
        None,
        "--tool",
        help="Optional tool names to filter by; may be provided multiple times.",
    ),
) -> None:
    """Return a chronological window of memory entries with coverage metadata."""

    try:
        store = MemoryStore(path)
    except OSError as exc:
        typer.secho(f"Failed to open memory store: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    retriever = StructuredMemoryRetriever(store)
    try:
        window = retriever.timeline(
            start=start,
            end=end,
            limit=limit,
            types=types or None,
            tools=tools or None,
        )
    except ValueError as exc:
        typer.secho(f"Invalid timeline parameters: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo(json.dumps(window.to_payload(), indent=2, ensure_ascii=True))


@memory_app.command("replay")
def memory_replay(
    manifest: Path = typer.Argument(..., exists=True, resolve_path=True, help="Path to manifest.json"),
    working: Optional[Path] = typer.Option(
        None,
        "--working",
        "-w",
        resolve_path=True,
        help="Optional working_memory.json snapshot to enrich the summary.",
    ),
) -> None:
    """Generate an experience replay summary from a run manifest."""

    manifest_model = _load_manifest(manifest)
    working_payload: Any | None = None
    if working is not None:
        working_data = _load_json_file(working)
        try:
            working_payload = WorkingMemory.from_dict(working_data)  # type: ignore[arg-type]
        except Exception:  # pragma: no cover - defensive
            working_payload = working_data

    key_findings = [entry.stdout for entry in manifest_model.tool_results if getattr(entry, "stdout", None)]
    report = Report(
        goal=_goal_from_manifest(manifest_model),
        summary="Experience replay",
        key_findings=key_findings,
        belief_deltas=[],
        artifacts=[],
    )

    chunk = summarise_experience(report, manifest_model, working_memory=working_payload)
    typer.echo(json.dumps(chunk, indent=2, ensure_ascii=True))


@working_app.command("summarize")
def summarize_working_memory(
    path: Path = typer.Argument(
        ..., exists=True, resolve_path=True, help="Path to a working_memory.json snapshot."
    ),
) -> None:
    """Summarise deliberation attempts captured in working-memory snapshots."""

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        typer.secho(f"Failed to read working memory snapshot: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        typer.secho("Snapshot is not valid JSON:", err=True, fg=typer.colors.RED)
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    working_memory = WorkingMemory.from_dict(payload)
    summary = summarise_working_memory(working_memory)
    typer.echo(json.dumps(summary or {}, indent=2, ensure_ascii=True))


@run_app.command("inspect")
def inspect_run(
    path: Path = typer.Argument(..., exists=True, resolve_path=True, help="Path to a run directory."),
    memory: Optional[Path] = typer.Option(
        None,
        "--memory",
        help="Optional path to the global memory JSONL file for contextual snippets.",
    ),
    memory_limit: int = typer.Option(
        5,
        "--memory-limit",
        "-m",
        min=1,
        help="Maximum number of memory entries to include when --memory is provided.",
    ),
    sample: int = typer.Option(
        5,
        "--sample",
        "-n",
        min=1,
        help="Number of tool results to sample when summarising the manifest.",
    ),
) -> None:
    """Summarise manifest, working memory, and optional memory context for a run."""

    manifest_path = path / "manifest.json"
    manifest = _load_manifest(manifest_path)
    payload: Dict[str, Any] = {
        "manifest": _summarise_manifest(manifest, sample_limit=sample),
    }

    working_path = path / "working_memory.json"
    if working_path.exists():
        try:
            working_payload = json.loads(working_path.read_text(encoding="utf-8"))
            working_memory = WorkingMemory.from_dict(working_payload)
            payload["working_memory"] = summarise_working_memory(working_memory)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            typer.secho(f"Failed to summarise working memory: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc
    else:
        payload["working_memory"] = None

    reflection_path = path / "reflection_summary.json"
    if reflection_path.exists():
        try:
            payload["reflection_summary"] = json.loads(reflection_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            typer.secho(f"Failed to load reflection summary: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc

    if memory is not None:
        try:
            store = MemoryStore(memory)
        except OSError as exc:
            typer.secho(f"Failed to open memory store: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc

        retriever = StructuredMemoryRetriever(store)
        goal_text = str(payload["manifest"].get("goal") or "")
        context = retriever.context_for_goal(
            goal_text,
            limit=memory_limit,
            recent=memory_limit,
        )
        payload["memory_context"] = context

    typer.echo(json.dumps(payload, indent=2, ensure_ascii=True))


def main() -> None:
    """Entrypoint for ``python -m agi.cli``."""

    app()


if __name__ == "__main__":  # pragma: no cover
    main()
