from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_tasks(path: Path) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    return tasks


def run_eval(agent, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {"scores": {}, "calibration": {}, "safety": {}, "efficiency": {}}
    for task in tasks:
        report = agent.run(task["goal"], task.get("constraints", {}))
        out["scores"][task["id"]] = report
    return out


def main() -> None:  # pragma: no cover - CLI helper
    tasks_path = Path(__file__).parent / "tasks" / "math_small.jsonl"
    tasks = load_tasks(tasks_path)
    print(json.dumps({"tasks_loaded": len(tasks)}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
