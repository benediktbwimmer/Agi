from __future__ import annotations

import asyncio
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

from ..core.reflection import summarise_working_memory
from ..core.types import Report, Source


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_attr(agent: Any, name: str) -> Any:
    value = getattr(agent, name, None)
    if value is not None:
        return value
    orchestrator = getattr(agent, "orchestrator", None)
    if orchestrator is not None:
        return getattr(orchestrator, name, None)
    return None


def _report_succeeded(report: Report) -> bool:
    summary = (report.summary or "").lower()
    if any(token in summary for token in ("fail", "error", "abort")):
        return False
    if any(token in summary for token in ("complete", "success", "pass")):
        return True
    return True


def _record_evaluation_outcome(
    agent: Any,
    *,
    task_id: str,
    goal: Mapping[str, Any] | Any,
    report: Report,
    insight: Mapping[str, Any] | None,
) -> None:
    memory = _resolve_attr(agent, "memory")
    world_model = _resolve_attr(agent, "world_model")
    success = _report_succeeded(report)
    timestamp = _now_iso()
    summary_entry: Dict[str, Any] = {
        "type": "evaluation_insight",
        "task_id": task_id,
        "time": timestamp,
        "summary": report.summary,
        "goal": goal.get("goal") if isinstance(goal, Mapping) else goal,
        "success": success,
        "key_findings": list(report.key_findings),
        "artifacts": list(report.artifacts),
    }
    if insight:
        summary_entry["insights"] = dict(insight)
    if memory is not None:
        memory.append(summary_entry)
    if world_model is not None:
        claim_id = f"evaluation::{task_id}"
        provenance = [
            {
                "kind": "evaluation",
                "ref": str(task_id),
                "note": "Evaluation harness outcome",
            }
        ]
        weight = 1.0
        if insight:
            weight = float(insight.get("attempt_count") or insight.get("risk_events") or 1.0)
        try:
            world_model.update(
                [
                    {
                        "claim_id": claim_id,
                        "passed": success,
                        "weight": max(weight, 1.0),
                        "provenance": provenance,
                        "timestamp": timestamp,
                    }
                ]
            )
        except Exception:  # pragma: no cover - defensive
            pass

def load_tasks(path: Path) -> List[Dict[str, Any]]:
    """Load evaluation tasks from a JSONL file.

    Empty lines or lines beginning with ``#`` are ignored to make it easy to
    add lightweight comments to task files while authoring experiments.
    """

    tasks: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.lstrip().startswith("#"):
                continue
            tasks.append(json.loads(line))
    return tasks


async def run_eval_async(agent, tasks: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Execute the supplied evaluation tasks with the provided ``agent``.

    The ``agent`` is expected to expose a ``run`` method that accepts a goal
    specification and an optional constraints mapping.  The method may be either
    synchronous or asynchronous; any awaitable return values will be awaited.
    """

    scores: MutableMapping[str, Any] = {}
    insights: MutableMapping[str, Any] = {}
    for task in tasks:
        goal = task.get("goal", {})
        constraints = task.get("constraints", {})
        report = agent.run(goal, constraints)
        if inspect.isawaitable(report):
            report = await report
        task_id = task.get("id") or str(len(scores))
        scores[task_id] = report
        insight: Optional[Dict[str, Any]] = None
        orchestrator = getattr(agent, "orchestrator", None)
        if orchestrator is not None:
            working_memory = getattr(orchestrator, "working_memory", None)
            insight = summarise_working_memory(working_memory)
            if insight:
                insights[task_id] = insight
        _record_evaluation_outcome(
            agent,
            task_id=task_id,
            goal=goal,
            report=report,
            insight=insight,
        )
    return {
        "scores": dict(scores),
        "calibration": {},
        "safety": {},
        "efficiency": {},
        "insights": dict(insights),
    }


def run_eval(agent, tasks: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Synchronous wrapper around :func:`run_eval_async`.

    When invoked from a running event loop, the caller should await
    :func:`run_eval_async` directly to avoid nested event loop errors.
    """

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_eval_async(agent, tasks))
    else:  # pragma: no cover - exercised only inside existing loops
        if loop.is_running():
            raise RuntimeError(
                "run_eval cannot be called while an event loop is running; "
                "await run_eval_async instead"
            )
        return loop.run_until_complete(run_eval_async(agent, tasks))


def main() -> None:  # pragma: no cover - CLI helper
    tasks_path = Path(__file__).parent / "tasks" / "math_small.jsonl"
    tasks = load_tasks(tasks_path)
    print(json.dumps({"tasks_loaded": len(tasks)}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
