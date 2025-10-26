from __future__ import annotations

import asyncio
import json

import pytest

from agi.src.core.planner import Planner, PlannerError


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


def test_planner_requires_plan():
    class EmptyLLM:
        def __call__(self, payload):
            return json.dumps({"plans": []})

    planner = Planner(llm=EmptyLLM())
    with pytest.raises(PlannerError):
        asyncio.run(planner.plan_from([]))
