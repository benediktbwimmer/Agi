from __future__ import annotations

"""High level executive agent that coordinates the core subsystems.

The :class:`ExecutiveAgent` wraps the low level orchestrator with additional
state hydration and reflection capabilities.  It is responsible for building a
rich goal specification from the caller's request, augmenting it with the
latest belief state and pertinent episodic memories, and recording a structured
reflection after execution.  This makes the rest of the system dramatically
easier to drive from tests or higher level applications while keeping the
behaviour deterministic and inspectable.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

from .memory import MemoryStore
from .orchestrator import Orchestrator
from .types import Belief, Report
from .world_model import WorldModel


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise_context(context: Optional[Any]) -> MutableMapping[str, Any]:
    if context is None:
        return {}
    if isinstance(context, Mapping):
        return dict(context)
    return {"text": str(context)}


def _serialise_belief(belief: Belief) -> Dict[str, Any]:
    serialised = asdict(belief)
    serialised["evidence"] = [asdict(source) for source in belief.evidence]
    return serialised


def _summarise_episode(record: Mapping[str, Any]) -> Dict[str, Any]:
    keys = ("time", "tool", "call_id", "ok", "stdout", "provenance")
    return {key: record.get(key) for key in keys if key in record}


@dataclass
class ExecutiveAgent:
    """High level agent that prepares orchestrator runs.

    Parameters
    ----------
    orchestrator:
        The orchestrator instance responsible for executing plans.
    memory:
        Persistent episodic memory store used for retrieving contextual
        experiences and logging reflections.
    world_model:
        Belief store that tracks claim credences and provenance.
    """

    orchestrator: Orchestrator
    memory: MemoryStore
    world_model: WorldModel
    default_constraints: Mapping[str, Any] = field(default_factory=dict)

    def build_goal_spec(
        self,
        goal: str,
        *,
        context: Optional[Any] = None,
        claim_ids: Iterable[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Construct the full goal specification provided to the orchestrator."""

        claim_ids_list = list(claim_ids or [])
        goal_context = _normalise_context(context)
        hypotheses = self._synthesise_hypotheses(claim_ids_list)
        enriched_metadata: Dict[str, Any] = {"generated_at": _now_iso()}
        if metadata:
            enriched_metadata.update(metadata)
        if goal_context:
            enriched_metadata.setdefault("context", goal_context)

        return {
            "goal": goal,
            "hypotheses": hypotheses,
            "metadata": enriched_metadata,
            "time": enriched_metadata["generated_at"],
            "claim_ids": claim_ids_list,
        }

    async def achieve(
        self,
        goal: str,
        *,
        context: Optional[Any] = None,
        claim_ids: Iterable[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        constraints: Mapping[str, Any] | None = None,
    ) -> Report:
        """Execute a goal via the orchestrator and record a reflection."""

        goal_spec = self.build_goal_spec(
            goal,
            context=context,
            claim_ids=claim_ids,
            metadata=metadata,
        )
        effective_constraints: Dict[str, Any] = dict(self.default_constraints)
        if constraints:
            effective_constraints.update(constraints)

        report = await self.orchestrator.run(goal_spec, effective_constraints)
        self._record_reflection(goal_spec, report)
        return report

    def run(
        self,
        goal: str,
        *,
        context: Optional[Any] = None,
        claim_ids: Iterable[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        constraints: Mapping[str, Any] | None = None,
    ) -> Report:
        """Synchronous wrapper around :meth:`achieve`."""

        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.achieve(
                    goal,
                    context=context,
                    claim_ids=claim_ids,
                    metadata=metadata,
                    constraints=constraints,
                )
            )
        else:  # pragma: no cover - exercised only within existing loops
            if loop.is_running():
                raise RuntimeError(
                    "ExecutiveAgent.run cannot be invoked inside a running event loop; "
                    "await achieve instead"
                )
            return loop.run_until_complete(
                self.achieve(
                    goal,
                    context=context,
                    claim_ids=claim_ids,
                    metadata=metadata,
                    constraints=constraints,
                )
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _synthesise_hypotheses(self, claim_ids: List[str]) -> List[Dict[str, Any]]:
        beliefs = self.world_model.beliefs
        hypotheses: List[Dict[str, Any]] = []
        for claim_id in claim_ids:
            hypothesis: Dict[str, Any] = {"id": claim_id}
            belief = beliefs.get(claim_id)
            if belief:
                hypothesis["belief"] = _serialise_belief(belief)
            episodes = self.memory.query_by_claim(claim_id)
            if episodes:
                hypothesis["memory"] = [_summarise_episode(ep) for ep in episodes[-5:]]
            hypotheses.append(hypothesis)
        if not hypotheses:
            # Provide a default empty hypothesis to keep downstream tooling happy.
            hypotheses.append({"id": "ad-hoc", "memory": []})
        return hypotheses

    def _record_reflection(self, goal_spec: Mapping[str, Any], report: Report) -> None:
        reflection = {
            "type": "reflection",
            "time": _now_iso(),
            "goal": goal_spec.get("goal"),
            "summary": report.summary,
            "key_findings": list(report.key_findings),
            "tool": "executive_reflection",
            "claim_ids": goal_spec.get("claim_ids", []),
            "belief_deltas": [asdict(delta) for delta in report.belief_deltas],
            "artifacts": list(report.artifacts),
        }
        self.memory.append(reflection)

