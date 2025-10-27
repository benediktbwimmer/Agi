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
        self,
        *,
        tool: Optional[str] = None,
        limit: int = 5,
        text_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve prior episodes from the episodic store.

        Args:
            tool: When provided, restrict results to the given tool name.
            limit: Maximum number of records to return. ``0`` yields no
                records, while ``-1`` returns all available entries.
            text_query: Optional case-insensitive substring filter applied to
                the episode ``stdout``, ``summary``, ``goal`` or ``claim_ids``.

        Returns:
            Deep copies of the matching episodic records so callers can mutate
            the payload without affecting the store.
        """

        if not self.episodic_memory:
            return []

        if limit == 0:
            return []

        if tool:
            records = self.episodic_memory.query_by_tool(tool)
        else:
            if limit < 0:
                records = self.episodic_memory.all()
            else:
                size = limit if limit > 0 else 5
                records = self.episodic_memory.recent(size)

        if text_query:
            needle = text_query.lower()

            def _matches(record: Dict[str, Any]) -> bool:
                fields = []
                for key in ("stdout", "summary", "goal"):
                    value = record.get(key)
                    if isinstance(value, str):
                        fields.append(value.lower())
                claim_ids = record.get("claim_ids")
                if isinstance(claim_ids, list):
                    fields.extend(str(cid).lower() for cid in claim_ids)
                return any(needle in field for field in fields)

            records = [record for record in records if _matches(record)]

        if limit > 0:
            records = records[-limit:]

        return [copy.deepcopy(record) for record in records]
