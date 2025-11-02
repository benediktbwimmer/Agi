from __future__ import annotations

"""Background job utilities for consolidating reflection insights."""

from typing import Any, Dict, Mapping, Optional

from ..core.memory import MemoryStore
from ..core.reflection import normalise_insight_records, summarise_reflection_insights
from ..core.world_model import WorldModel


def consolidate_reflections(
    memory: MemoryStore,
    *,
    goal: Optional[str] = None,
    limit: int = 200,
    write_back: bool = True,
    world_model: Optional[WorldModel] = None,
) -> Dict[str, Any]:
    """Aggregate reflection insight records and optionally persist a summary."""

    records = memory.iter_reflection_insights(goal=goal, limit=limit)
    insights = normalise_insight_records(records)
    summary = summarise_reflection_insights(insights)
    summary["goal"] = goal
    summary["sample_size"] = len(insights)
    planner_bias: Dict[str, Any] = {}
    failure_branches = summary.get("failure_branches") or []
    if failure_branches:
        planner_bias["avoid_steps"] = [
            entry.get("step_id")
            for entry in failure_branches
            if isinstance(entry, Mapping) and entry.get("step_id")
        ]
    motif_issue_counts = summary.get("motif_issue_counts")
    if isinstance(motif_issue_counts, Mapping):
        planner_bias["avoid_issues"] = list(motif_issue_counts.keys())
    if planner_bias:
        summary["planner_bias"] = planner_bias
    if write_back and summary["sample_size"]:
        memory.append(
            {
                "type": "reflection_summary",
                "goal": goal,
                "time": records[-1].get("time") if records else None,
                "summary": summary,
            }
        )
    if world_model is not None and summary["sample_size"]:
        claim_id = _reflection_claim_id(goal)
        passed = bool(summary.get("caution_score") == 0 and summary.get("risk_events") == 0)
        world_model.update(
            [
                {
                    "claim_id": claim_id,
                    "passed": passed,
                    "weight": summary["sample_size"],
                    "provenance": [
                        {
                            "kind": "reflection_summary",
                            "ref": goal or "global",
                            "note": "Aggregated reflective consolidation",
                        }
                    ],
                }
            ]
        )
        summary["world_model_claim_id"] = claim_id
    return summary


def _reflection_claim_id(goal: Optional[str]) -> str:
    target = goal.strip().lower().replace(" ", "_") if goal else "global"
    return f"reflection::{target}::safety"
