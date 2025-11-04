from __future__ import annotations

import threading
import time

from agi.src.governance.interactive_gatekeeper import InteractiveGatekeeper
from agi.src.oversight.models import ApprovalDecision
from agi.src.oversight.store import OversightStore


def test_interactive_gatekeeper_auto_allows_low_tier():
    store = OversightStore()
    gatekeeper = InteractiveGatekeeper(
        oversight_store=store,
        interactive_min_tier="T1",
        timeout_s=2.0,
        policy={"max_tier": "T3"},
    )

    assert gatekeeper.review("T0", tool="calculator", context={"plan_id": "p", "step_id": "s"})


def test_interactive_gatekeeper_blocks_until_decision():
    store = OversightStore()
    gatekeeper = InteractiveGatekeeper(
        oversight_store=store,
        interactive_min_tier="T1",
        timeout_s=2.0,
        policy={"max_tier": "T3"},
    )

    result_container: dict[str, bool] = {}

    def resolver() -> None:
        # Spin until the approval request is registered.
        deadline = time.time() + 2.0
        approval_id = None
        while time.time() < deadline and approval_id is None:
            pending = store.list_pending_approvals()
            if pending:
                approval_id = pending[0].id
                decision = ApprovalDecision.build(
                    approval_id=approval_id,
                    approved=True,
                    reviewer="tester",
                    message="Manual override",
                )
                store.resolve_approval(approval_id, decision)
                return
            time.sleep(0.05)
        raise AssertionError("Approval request was not registered")

    thread = threading.Thread(target=resolver, daemon=True)
    thread.start()

    approved = gatekeeper.review(
        "T2",
        tool="danger",
        context={"plan_id": "demo", "step_id": "1"},
    )
    result_container["approved"] = approved

    thread.join(timeout=2.0)
    assert result_container["approved"] is True


def test_interactive_gatekeeper_denial():
    store = OversightStore()
    gatekeeper = InteractiveGatekeeper(
        oversight_store=store,
        interactive_min_tier="T1",
        timeout_s=2.0,
        policy={"max_tier": "T3"},
    )

    def resolver() -> None:
        deadline = time.time() + 2.0
        approval_id = None
        while time.time() < deadline and approval_id is None:
            pending = store.list_pending_approvals()
            if pending:
                approval_id = pending[0].id
                decision = ApprovalDecision.build(
                    approval_id=approval_id,
                    approved=False,
                    reviewer="tester",
                    message="Rejected for review",
                )
                store.resolve_approval(approval_id, decision)
                return
            time.sleep(0.05)
        raise AssertionError("Approval request was not registered")

    thread = threading.Thread(target=resolver, daemon=True)
    thread.start()

    assert gatekeeper.review(
        "T3",
        tool="nuclear",
        context={"plan_id": "demo", "step_id": "2"},
    ) is False

    thread.join(timeout=2.0)
