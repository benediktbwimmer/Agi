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
