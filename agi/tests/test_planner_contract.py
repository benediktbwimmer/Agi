from __future__ import annotations

import asyncio
import json

import pytest

from agi.src.core.planner import Planner, PlannerError
from agi.src.core.types import BranchCondition, PlanStep


class DummyLLM:
    def __call__(self, payload):
        return json.dumps(
            {
                "plans": [
                    {
                        "id": "plan-1",
                        "claim_ids": ["claim-1"],
                        "steps": [
                            {
                                "id": "step-1",
                                "tool": "python_runner",
                                "args": {"code": "print('ok')"},
                                "safety_level": "T0",
                            }
                        ],
                        "expected_cost": {"tokens": 100},
                        "risks": ["latency"],
                        "ablations": ["baseline"],
                    }
                ]
            }
        )


def test_planner_parses_llm_output():
    planner = Planner(llm=DummyLLM())
    plans = asyncio.run(planner.plan_from([{"claim": "c1"}]))
    assert plans[0].steps[0].tool == "python_runner"


def test_planner_parses_hierarchical_steps():
    class HierLLM:
        def __call__(self, payload):
            return json.dumps(
                {
                    "plans": [
                        {
                            "id": "plan-h1",
                            "claim_ids": ["claim-h1"],
                            "steps": [
                                {
                                    "id": "root",
                                    "tool": "setup",
                                    "args": {},
                                    "sub_steps": [
                                        {
                                            "id": "child",
                                            "tool": "child_tool",
                                            "args": {},
                                        }
                                    ],
                                    "branches": [
                                        {
                                            "condition": "on_success(root)",
                                            "steps": [
                                                {
                                                    "id": "success",
                                                    "tool": "success_tool",
                                                    "args": {},
                                                }
                                            ],
                                        },
                                        {
                                            "condition": {"when": "failure", "step": "root"},
                                            "steps": [
                                                {
                                                    "id": "failure",
                                                    "tool": "failure_tool",
                                                    "args": {},
                                                }
                                            ],
                                        },
                                    ],
                                }
                            ],
                            "expected_cost": {},
                            "risks": [],
                            "ablations": [],
                        }
                    ]
                }
            )

    planner = Planner(llm=HierLLM())
    plans = asyncio.run(planner.plan_from([{"claim": "c1"}]))
    step = plans[0].steps[0]
    assert isinstance(step, PlanStep)
    assert step.sub_steps[0].tool == "child_tool"
    assert len(step.branches) == 2
    assert isinstance(step.branches[0].condition, BranchCondition)
    assert step.branches[0].condition.kind == "success"
    assert step.branches[1].condition.kind == "failure"


def test_planner_requires_plan():
    class EmptyLLM:
        def __call__(self, payload):
            return json.dumps({"plans": []})

    planner = Planner(llm=EmptyLLM())
    with pytest.raises(PlannerError):
        asyncio.run(planner.plan_from([]))


def test_planner_prefers_low_risk_plans_with_reflection():
    class ReflectiveLLM:
        def __init__(self) -> None:
            self.payload = None

        def __call__(self, payload):
            self.payload = payload
            return json.dumps(
                {
                    "plans": [
                        {
                            "id": "plan-high-risk",
                            "claim_ids": ["claim-1"],
                            "steps": [
                                {
                                    "id": "hazard",
                                    "tool": "hazardous_tool",
                                    "args": {},
                                    "safety_level": "T3",
                                }
                            ],
                            "expected_cost": {},
                            "risks": ["collision"],
                            "ablations": [],
                        },
                        {
                            "id": "plan-safe",
                            "claim_ids": ["claim-1"],
                            "steps": [
                                {
                                    "id": "safe",
                                    "tool": "safe_tool",
                                    "args": {},
                                    "safety_level": "T0",
                                }
                            ],
                            "expected_cost": {},
                            "risks": [],
                            "ablations": [],
                        },
                    ]
                }
            )

    llm = ReflectiveLLM()
    memory_context = {
        "goal": "hazard mitigation",
        "semantic": {
            "matches": [
                {
                    "summary": "Mitigation report",
                    "keywords": ["hazard", "mitigation"],
                    "sensor": {"modality": "analysis"},
                    "safety_tier": "T1",
                }
            ]
        },
        "insights": [
            {
                "run_id": "prior-1",
                "time": "2024-01-01T00:00:00+00:00",
                "summary": "Risky approach failed",
                "insights": {
                    "final_status": "needs_replan",
                    "critique_tags": ["safety"],
                    "risk_events": 3,
                    "attempt_count": 2,
                    "hypotheses": [{"id": "hyp"}],
                    "failure_motifs": {
                        "issues": {"latency": 2},
                        "failure_branches": [
                            {"plan_id": "plan-high-risk", "step_id": "hazard", "count": 1}
                        ],
                    },
                },
            }
        ],
    }
    planner = Planner(llm=llm)
    plans = asyncio.run(planner.plan_from([{"id": "hyp"}], memory_context=memory_context))

    assert plans[0].id == "plan-safe"
    assert llm.payload is not None
    summary = llm.payload.get("reflective_summary")
    assert summary is not None
    assert summary["caution_score"] >= 2
    assert summary["risk_events"] == 3
    assert all(step.safety_level != "T3" for plan in plans for step in plan.iter_tool_calls())
    features = llm.payload.get("memory_features")
    assert features is not None
    assert "hazard" in features.get("keywords", [])
    planner_bias = llm.payload.get("planner_bias")
    assert planner_bias is not None
    assert "latency" in planner_bias.get("avoid_issues", [])
    hypothesis_context = llm.payload["hypotheses"][0]["reflective_context"]
    assert hypothesis_context["focus_count"] >= 1
    assert "analysis" in features.get("modalities", [])
