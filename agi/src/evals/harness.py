from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping


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
    for task in tasks:
        goal = task.get("goal", {})
        constraints = task.get("constraints", {})
        report = agent.run(goal, constraints)
        if inspect.isawaitable(report):
            report = await report
        task_id = task.get("id") or str(len(scores))
        scores[task_id] = report
    return {"scores": dict(scores), "calibration": {}, "safety": {}, "efficiency": {}}


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
