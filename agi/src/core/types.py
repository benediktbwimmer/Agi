from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


UID = str


@dataclass
class Prediction:
    id: UID
    metric: str
    expectation: str
    eval_procedure: str


@dataclass
class Source:
    kind: str
    ref: str
    note: Optional[str] = None


@dataclass
class Claim:
    id: UID
    text: str
    predictions: List[Prediction]
    variables: Dict[str, Any]
    provenance: Optional[List[Source]] = None


@dataclass
class ToolCall:
    id: UID
    tool: str
    args: Dict[str, Any]
    safety_level: str


@dataclass
class ToolResult:
    call_id: UID
    ok: bool
    cost_tokens: Optional[int] = None
    wall_time_ms: Optional[int] = None
    stdout: Optional[str] = None
    data: Any = None
    figures: Optional[List[str]] = None
    provenance: List[Source] = field(default_factory=list)


@dataclass
class Plan:
    id: UID
    claim_ids: List[UID]
    steps: List[ToolCall]
    expected_cost: Dict[str, Optional[float]]
    risks: List[str]
    ablations: List[str]


@dataclass
class Belief:
    claim_id: UID
    credence: float
    evidence: List[Source]
    last_updated: str


@dataclass
class Report:
    goal: str
    summary: str
    key_findings: List[str]
    belief_deltas: List[Belief]
    artifacts: List[str]


class Tool:
    name: str
    safety: str

    async def run(self, args: Dict[str, Any], ctx: "RunContext") -> ToolResult:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class RunContext:
    working_dir: str
    timeout_s: int
    env_whitelist: List[str]
    network: str
    record_provenance: bool
