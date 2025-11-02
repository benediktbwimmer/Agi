from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agi.src.core.memory import MemoryStore
from agi.src.core.orchestrator import DeliberationAttempt, PlanTrace, WorkingMemory
from agi.src.core.types import Report
from agi.src.core.world_model import WorldModel
from agi.src.evals.harness import run_eval


@dataclass
class _StubOrchestrator:
    working_memory: WorkingMemory


@dataclass
class _StubAgent:
    orchestrator: _StubOrchestrator

    def run(self, goal, constraints):
        return Report(
            goal=str(goal or "test"),
            summary="ok",
            key_findings=["finding"],
            belief_deltas=[],
            artifacts=[],
        )


def _sample_working_memory() -> WorkingMemory:
    wm = WorkingMemory(run_id="run-sample", goal="demo", hypotheses=[])
    attempt = DeliberationAttempt(index=0, status="needs_replan")
    plan_trace = PlanTrace(plan_id="plan-1", claim_ids=["c1"], step_count=1)
    plan_trace.mark_rejected()
    plan_trace.add_rationale_tags(["safety"])
    attempt.plans.append(plan_trace)
    attempt._plan_index[plan_trace.plan_id] = plan_trace
    attempt.critiques.append({"status": "FAIL", "issues": ["gap"], "rationale_tags": ["safety"]})
    retry = DeliberationAttempt(index=1, status="complete")
    final_plan = PlanTrace(plan_id="plan-1", claim_ids=["c1"], step_count=1)
    final_plan.mark_approved()
    final_plan.mark_execution(True)
    retry.plans.append(final_plan)
    retry._plan_index[final_plan.plan_id] = final_plan
    wm.attempts.extend([attempt, retry])
    return wm


def test_run_eval_surfaces_insights():
    wm = _sample_working_memory()
    agent = _StubAgent(orchestrator=_StubOrchestrator(working_memory=wm))
    result = run_eval(agent, [{"id": "task-1", "goal": "demo"}])
    assert "scores" in result
    insights = result.get("insights")
    assert insights is not None
    assert "task-1" in insights
    assert insights["task-1"]["run_id"] == "run-sample"
    assert insights["task-1"]["final_status"] == "complete"


@dataclass
class _RecordingAgent:
    memory: MemoryStore
    world_model: WorldModel
    orchestrator: _StubOrchestrator

    def run(self, goal, constraints):
        return Report(
            goal=str(goal or "test"),
            summary="Completed evaluation",
            key_findings=["finding"],
            belief_deltas=[],
            artifacts=["artifact.txt"],
        )


def test_run_eval_records_memory_and_belief(tmp_path: Path):
    wm = _sample_working_memory()
    memory = MemoryStore(tmp_path / "memory.jsonl")
    world_model = WorldModel()
    agent = _RecordingAgent(memory=memory, world_model=world_model, orchestrator=_StubOrchestrator(working_memory=wm))

    run_eval(agent, [{"id": "eval-1", "goal": {"goal": "demo"}}])

    entries = memory.recent(types=["evaluation_insight"])
    assert entries, "expected evaluation insight to be recorded"
    latest = entries[-1]
    assert latest["task_id"] == "eval-1"
    assert latest["success"] is True
    belief = world_model.beliefs.get("evaluation::eval-1")
    assert belief is not None
    assert belief.credence > 0.5
