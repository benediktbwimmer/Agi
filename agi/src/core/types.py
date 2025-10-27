"""Shared type definitions for the AGI runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, TypedDict


UID = str


class Prediction(TypedDict, total=False):
    """Prediction metadata attached to a :class:`Claim`."""

    id: UID
    metric: str
    expectation: str
    eval_procedure: str


@dataclass
class Source:
    """Provenance information for beliefs and tool outputs."""

    kind: str
    ref: str
    note: Optional[str] = None


@dataclass
class Claim:
    """A declarative claim the system is attempting to verify."""

    id: UID
    text: str
    predictions: List[Prediction]
    variables: Dict[str, Any]
    provenance: Optional[List[Source]] = None


@dataclass
class ToolCall:
    """Instruction to invoke a tool during a plan."""

    id: UID
    tool: str
    args: Dict[str, Any]
    safety_level: str = "T0"


@dataclass
class ToolResult:
    """Outcome from executing a tool call."""

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
    """Structured plan produced by the planner."""

    id: UID
    claim_ids: List[UID]
    steps: List[ToolCall]
    expected_cost: Dict[str, Optional[float]]
    risks: List[str]
    ablations: List[str]


@dataclass
class Belief:
    """Belief state for a specific claim."""

    claim_id: UID
    credence: float
    evidence: List[Source]
    last_updated: str


@dataclass
class Report:
    """Structured summary returned to callers after orchestration."""

    goal: str
    summary: str
    key_findings: List[str]
    belief_deltas: List[Belief]
    artifacts: List[str]


class Tool(Protocol):
    """Runtime contract for orchestrator tools."""

    name: str
    safety: str

    async def run(self, args: Dict[str, Any], ctx: "RunContext") -> ToolResult:  # pragma: no cover - interface
        ...


@dataclass
class RunContext:
    """Execution context provided to tools during a run."""

    working_dir: str
    timeout_s: int
    env_whitelist: List[str]
    network: str
    record_provenance: bool


__all__ = [
    "Belief",
    "Claim",
    "Plan",
    "Prediction",
    "Report",
    "RunContext",
    "Source",
    "Tool",
    "ToolCall",
    "ToolResult",
    "UID",
]

