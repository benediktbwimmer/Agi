from __future__ import annotations

from typing import Any, Mapping, Optional

from .gatekeeper import Gatekeeper
from ..oversight.models import ApprovalDecision, ApprovalRequest
from ..oversight.store import OversightStore


class InteractiveGatekeeper(Gatekeeper):
    """Gatekeeper that escalates high-risk requests to human reviewers.

    The interactive gatekeeper wraps the baseline :class:`Gatekeeper` policy
    checks with an approval workflow backed by :class:`OversightStore`.  Tool
    invocations that exceed the configured tier threshold are paused until a
    human reviewer records a decision via the oversight console.
    """

    def __init__(
        self,
        *,
        policy: Mapping[str, object] | None = None,
        world_model: Any | None = None,
        oversight_store: OversightStore | None = None,
        interactive_min_tier: str = "T1",
        timeout_s: Optional[float] = None,
        requester: str = "orchestrator",
    ) -> None:
        if oversight_store is None:
            raise ValueError("oversight_store is required for InteractiveGatekeeper")
        super().__init__(policy=policy or {}, world_model=world_model)
        self._store = oversight_store
        self._interactive_threshold = self._normalise_tier(interactive_min_tier)
        self._timeout_s = timeout_s
        self._requester = requester

    def review(
        self,
        tier: str,
        *,
        tool: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        allowed = super().review(tier, tool=tool, context=context)
        if not allowed:
            return False

        # Only escalate if the request tier exceeds the configured threshold.
        tier_value = self._tier_value(self._normalise_tier(tier))
        threshold_value = self._tier_value(self._interactive_threshold)
        if tier_value <= threshold_value:
            return True

        request = ApprovalRequest.build(
            tool=tool,
            tier=self._normalise_tier(tier),
            requested_by=self._requester,
            context=context,
        )
        ticket = self._store.create_approval_request(request)
        decision = ticket.wait(timeout=self._timeout_s)
        if decision is None:
            raise TimeoutError(
                f"Interactive approval timed out for request {request.id} (tool={tool}, tier={tier})"
            )
        return decision.approved

    def record_decision(
        self,
        approval_id: str,
        *,
        approved: bool,
        reviewer: str,
        message: str | None = None,
    ) -> None:
        decision = ApprovalDecision.build(
            approval_id=approval_id,
            approved=approved,
            reviewer=reviewer,
            message=message,
        )
        self._store.resolve_approval(approval_id, decision)


__all__ = ["InteractiveGatekeeper"]
