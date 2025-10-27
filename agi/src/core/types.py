from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING


if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .memory import MemoryStore

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
    working_memory: List[Dict[str, Any]] = field(default_factory=list)
    episodic_memory: "MemoryStore | None" = None

    def recall_from_episodic(
        self, *, tool: Optional[str] = None, limit: int = 5
    ) -> List[Dict[str, Any]]:
        if not self.episodic_memory:
            return []
        if tool:
            records = self.episodic_memory.query_by_tool(tool)
        else:
            records = self.episodic_memory.recent(limit)
        if limit >= 0:
            records = records[-limit:]
        return [copy.deepcopy(record) for record in records]
