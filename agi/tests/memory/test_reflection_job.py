from __future__ import annotations

from datetime import datetime, timedelta, timezone

from agi.src.core.memory import MemoryStore
from agi.src.core.world_model import WorldModel
from agi.src.memory.reflection_job import consolidate_reflections


def _ts(offset: int) -> str:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return (base + timedelta(hours=offset)).isoformat()


def test_consolidate_reflections(tmp_path):
    memory = MemoryStore(tmp_path / "memory.jsonl")
    for idx in range(3):
        memory.append(
            {
                "type": "reflection_insight",
                "goal": "lunar ops",
                "time": _ts(idx),
                "summary": f"run-{idx}",
                "insights": {
                    "final_status": "needs_replan" if idx < 2 else "complete",
                    "critique_tags": ["safety", "latency"] if idx == 0 else ["safety"],
                    "risk_events": idx,
                    "attempt_count": idx + 1,
                    "hypotheses": [{"id": f"hyp-{idx}"}],
                    "failure_motifs": {
                        "issues": {"latency": idx + 1},
                        "failure_branches": [
                            {"plan_id": "plan-risk", "step_id": "hazard", "count": 1}
                        ],
                    },
                },
            }
        )

    world_model = WorldModel()
    summary = consolidate_reflections(
        memory,
        goal="lunar ops",
        write_back=True,
        world_model=world_model,
    )

    assert summary["goal"] == "lunar ops"
    assert summary["total_runs"] == 3
    assert summary["risk_events"] == sum(range(3))
    assert "safety" in summary["dominant_tags"]
    assert summary["caution_score"] >= 2
    assert summary["motif_issue_counts"]["latency"] == 6
    assert summary["failure_branches"][0]["step_id"] == "hazard"
    assert summary["planner_bias"]["avoid_issues"]
    assert summary["hypothesis_focus"]["hyp-0"] == 1

    summaries = memory.recent(types=["reflection_summary"])
    assert summaries
    latest = summaries[-1]
    assert latest["summary"]["total_runs"] == 3
    assert latest["goal"] == "lunar ops"
    claim_id = summary.get("world_model_claim_id")
    assert claim_id is not None
    beliefs = world_model.beliefs
    assert claim_id in beliefs
    assert beliefs[claim_id].credence < 0.5
