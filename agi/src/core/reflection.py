from __future__ import annotations

"""Utilities for analysing working memory deliberation traces.

These helpers operate on :class:`WorkingMemory` snapshots emitted by the
orchestrator.  They surface lightweight summaries that higher level reflection
jobs can feed into continual learning loops, realising the reflective layer of
the Putnam-inspired architecture.
"""

from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

if TYPE_CHECKING:
    from .orchestrator import DeliberationAttempt, WorkingMemory


__all__ = [
    "normalise_insight_records",
    "summarise_reflection_insights",
    "summarise_working_memory",
]


def summarise_working_memory(working_memory: "WorkingMemory" | None) -> Dict[str, Any]:
    """Return a reflective summary for the provided working memory snapshot."""

    if working_memory is None:
        return {}

    attempts = list(working_memory.attempts)
    summary: Dict[str, Any] = {
        "run_id": working_memory.run_id,
        "goal": working_memory.goal,
        "attempt_count": len(attempts),
        "attempt_statuses": [
            {"index": attempt.index, "status": attempt.status} for attempt in attempts
        ],
    }
    if attempts:
        summary["final_status"] = attempts[-1].status

    plan_outcomes = _plan_outcome_summary(attempts)
    if any(plan_outcomes.values()):
        summary["plan_outcomes"] = plan_outcomes

    critique_tags = _collect_critique_tags(attempts)
    if critique_tags:
        summary["critique_tags"] = sorted(critique_tags)

    issues = _collect_issues(attempts)
    if issues:
        summary["issues"] = sorted(issues)

    hypothesis_ids = _collect_hypotheses(attempts)
    if hypothesis_ids:
        summary["hypotheses"] = sorted(hypothesis_ids)

    failures = _collect_execution_failures(attempts)
    if failures:
        summary["execution_failures"] = failures

    total_risk_events = sum(len(attempt.risk_assessments) for attempt in attempts)
    if total_risk_events:
        summary["risk_events"] = total_risk_events

    failure_motifs = _detect_failure_motifs(attempts)
    if failure_motifs:
        summary["failure_motifs"] = failure_motifs

    return summary


def _plan_outcome_summary(attempts: Sequence["DeliberationAttempt"]) -> Dict[str, List[str]]:
    buckets: Dict[str, Set[str]] = {
        "approved": set(),
        "rejected": set(),
        "succeeded": set(),
        "failed": set(),
    }
    for attempt in attempts:
        for plan in attempt.plans:
            if plan.approved:
                buckets["approved"].add(plan.plan_id)
            elif plan.approved is False:
                buckets["rejected"].add(plan.plan_id)
            if plan.execution_succeeded:
                buckets["succeeded"].add(plan.plan_id)
            elif plan.execution_succeeded is False:
                buckets["failed"].add(plan.plan_id)
    return {key: sorted(values) for key, values in buckets.items()}


def _collect_critique_tags(attempts: Sequence["DeliberationAttempt"]) -> Set[str]:
    tags: Set[str] = set()
    for attempt in attempts:
        for plan in attempt.plans:
            tags.update(plan.rationale_tags)
        for critique in attempt.critiques:
            for tag in critique.get("rationale_tags", []) or []:
                text = str(tag).strip()
                if text:
                    tags.add(text)
    return tags


def _collect_issues(attempts: Sequence["DeliberationAttempt"]) -> Set[str]:
    issues: Set[str] = set()
    for attempt in attempts:
        for critique in attempt.critiques:
            for issue in critique.get("issues", []) or []:
                text = str(issue).strip()
                if text:
                    issues.add(text)
    return issues


def _collect_execution_failures(attempts: Sequence["DeliberationAttempt"]) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    for attempt in attempts:
        for feedback in attempt.execution_feedback:
            failures.extend(
                _normalise_failure(failure, feedback.get("plan_id"))
                for failure in feedback.get("failures", []) or []
            )
    return failures


def _normalise_failure(data: Mapping[str, Any], plan_id: str | None) -> Dict[str, Any]:
    payload = {
        "plan_id": plan_id,
        "step_id": data.get("step_id"),
        "tool": data.get("tool"),
        "status": data.get("status"),
    }
    stdout = data.get("stdout")
    if stdout:
        payload["stdout"] = stdout
    return payload


def normalise_insight_records(raw: Any) -> List[Mapping[str, Any]]:
    """Return a sequence of mapping records containing ``insights`` payloads."""

    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    records: List[Mapping[str, Any]] = []
    for item in raw:
        if isinstance(item, Mapping) and item.get("insights"):
            records.append(item)
    return records


def summarise_reflection_insights(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate reflection insight records across runs."""

    tag_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    risk_events_total = 0
    attempt_total = 0
    hypothesis_counts: Counter[str] = Counter()
    motif_issue_counts: Counter[str] = Counter()
    branch_failure_counts: Counter[Tuple[str, str]] = Counter()
    recent_runs: List[Dict[str, Any]] = []

    for record in records:
        meta = record.get("insights")
        final_status = None
        if isinstance(meta, Mapping):
            tag_counts.update(str(tag) for tag in meta.get("critique_tags", []) or [])
            final_status = meta.get("final_status")
            if final_status:
                status_counts[str(final_status)] += 1
            risk_events_total += int(meta.get("risk_events", 0) or 0)
            attempt_total += int(meta.get("attempt_count", 0) or 0)
            hypotheses_meta = meta.get("hypotheses")
            if isinstance(hypotheses_meta, Sequence) and not isinstance(hypotheses_meta, (str, bytes)):
                for item in hypotheses_meta:
                    ident = None
                    if isinstance(item, Mapping):
                        ident = item.get("id") or item.get("claim_id") or item.get("value")
                    else:
                        ident = item
                    if ident:
                        hypothesis_counts[str(ident)] += 1
            motifs_meta = meta.get("failure_motifs", {})
            if isinstance(motifs_meta, Mapping):
                motif_issues = motifs_meta.get("issues")
                if isinstance(motif_issues, Mapping):
                    for name, count in motif_issues.items():
                        try:
                            motif_issue_counts[str(name)] += int(count)
                        except (TypeError, ValueError):
                            motif_issue_counts[str(name)] += 1
                motif_branches = motifs_meta.get("failure_branches")
                if isinstance(motif_branches, Sequence):
                    for entry in motif_branches:
                        if not isinstance(entry, Mapping):
                            continue
                        plan_id = entry.get("plan_id")
                        step_id = entry.get("step_id")
                        if not plan_id or not step_id:
                            continue
                        try:
                            count = int(entry.get("count", 1))
                        except (TypeError, ValueError):
                            count = 1
                        branch_failure_counts[(str(plan_id), str(step_id))] += count
            summary_text = str(meta.get("summary") or record.get("summary") or "")
        else:
            summary_text = str(record.get("summary") or "")
        recent_runs.append(
            {
                "run_id": record.get("run_id"),
                "time": record.get("time"),
                "summary": summary_text or None,
                "final_status": str(final_status) if final_status else None,
            }
        )

    total_records = len(recent_runs)
    dominant_tags = [tag for tag, _ in tag_counts.most_common(3)]
    caution_score = (
        tag_counts.get("safety", 0)
        + status_counts.get("needs_replan", 0)
        + status_counts.get("failed", 0)
    )
    motif_pressure = (
        sum(motif_issue_counts.values()) + sum(branch_failure_counts.values())
    )
    if motif_pressure:
        caution_score += motif_pressure
    average_attempts = attempt_total / total_records if total_records else 0.0

    summary: Dict[str, Any] = {
        "tag_counts": dict(tag_counts),
        "final_status_counts": dict(status_counts),
        "risk_events": risk_events_total,
        "total_runs": total_records,
        "average_attempts": round(average_attempts, 3),
        "dominant_tags": dominant_tags,
        "caution_score": caution_score,
        "recent_runs": recent_runs[-3:],
    }
    if hypothesis_counts:
        summary["hypothesis_focus"] = dict(hypothesis_counts.most_common(5))
    if motif_issue_counts:
        summary["motif_issue_counts"] = dict(motif_issue_counts.most_common(5))
    if branch_failure_counts:
        summary["failure_branches"] = [
            {"plan_id": plan_id, "step_id": step_id, "count": count}
            for (plan_id, step_id), count in branch_failure_counts.most_common(5)
        ]
    if motif_pressure:
        summary["motif_pressure"] = motif_pressure
    planner_bias: Dict[str, Any] = {}
    if branch_failure_counts:
        planner_bias["avoid_steps"] = [
            entry["step_id"] for entry in summary["failure_branches"]
        ]
    if motif_issue_counts:
        planner_bias["avoid_issues"] = list(summary["motif_issue_counts"].keys())
    if planner_bias:
        summary["planner_bias"] = planner_bias
    return summary


def _collect_hypotheses(attempts: Sequence["DeliberationAttempt"]) -> Set[str]:
    ids: Set[str] = set()
    for attempt in attempts:
        for hypothesis in getattr(attempt, "hypotheses", []) or []:
            if isinstance(hypothesis, Mapping):
                ident = hypothesis.get("id") or hypothesis.get("claim_id") or hypothesis.get("value")
            else:
                ident = hypothesis
            if isinstance(ident, (int, float)):
                ident = str(ident)
            if isinstance(ident, str):
                text = ident.strip()
                if text:
                    ids.add(text)
    return ids


def _detect_failure_motifs(attempts: Sequence["DeliberationAttempt"]) -> Dict[str, Any]:
    issue_counts: Counter[str] = Counter()
    branch_counts: Counter[Tuple[str, str]] = Counter()
    for attempt in attempts:
        for rationale in getattr(attempt, "critic_rationales", []) or []:
            if not isinstance(rationale, Mapping):
                continue
            for issue in rationale.get("issues", []) or []:
                text = str(issue).strip()
                if text:
                    issue_counts[text] += 1
        for branch in getattr(attempt, "branch_log", []) or []:
            if not isinstance(branch, Mapping):
                continue
            if not branch.get("taken"):
                continue
            metadata = branch.get("metadata") or {}
            condition = metadata.get("condition")
            kind = ""
            if isinstance(condition, Mapping):
                kind = str(condition.get("when") or condition.get("kind") or "").lower()
            elif isinstance(condition, str):
                kind = condition.lower()
            if "failure" not in kind:
                continue
            plan_id = branch.get("plan_id")
            step_id = branch.get("step_id")
            if not plan_id or not step_id:
                continue
            branch_counts[(str(plan_id), str(step_id))] += 1
    motifs: Dict[str, Any] = {}
    if issue_counts:
        motifs["issues"] = dict(issue_counts.most_common(5))
    if branch_counts:
        motifs["failure_branches"] = [
            {"plan_id": plan_id, "step_id": step_id, "count": count}
            for (plan_id, step_id), count in branch_counts.most_common(5)
        ]
    return motifs
