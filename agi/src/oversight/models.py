from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional
import uuid


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _copy_payload(data: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not data:
        return {}
    return {key: value for key, value in data.items()}


@dataclass(frozen=True)
class ApprovalRequest:
    """Represents a pending approval that requires human oversight."""

    id: str
    created_at: str
    tool: Optional[str]
    tier: str
    requested_by: str
    context: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        tool: Optional[str],
        tier: str,
        requested_by: str,
        context: Mapping[str, Any] | None = None,
        ident: Optional[str] = None,
    ) -> "ApprovalRequest":
        return cls(
            id=ident or uuid.uuid4().hex,
            created_at=_now_iso(),
            tool=tool,
            tier=tier,
            requested_by=requested_by,
            context=_copy_payload(context),
        )


@dataclass(frozen=True)
class ApprovalDecision:
    """Decision produced by a reviewer for a pending approval."""

    approval_id: str
    approved: bool
    reviewer: str
    decided_at: str
    message: Optional[str] = None

    @classmethod
    def build(
        cls,
        *,
        approval_id: str,
        approved: bool,
        reviewer: str,
        message: Optional[str] = None,
        decided_at: Optional[str] = None,
    ) -> "ApprovalDecision":
        return cls(
            approval_id=approval_id,
            approved=approved,
            reviewer=reviewer,
            decided_at=decided_at or _now_iso(),
            message=message,
        )


__all__ = [
    "ApprovalDecision",
    "ApprovalRequest",
]
