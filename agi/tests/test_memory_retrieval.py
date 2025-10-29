from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from agi.src.core.memory import MemoryStore
from agi.src.core.memory_retrieval import MemoryRetriever


@pytest.fixture
def memory_store(tmp_path: Path) -> MemoryStore:
    store = MemoryStore(tmp_path / "memory.jsonl")
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for idx, (record_type, tool, plan) in enumerate(
        [
            ("episode", "python_runner", "plan-1"),
            ("reflection", None, "plan-1"),
            ("episode", "calculator", "plan-2"),
        ]
    ):
        payload = {
            "type": record_type,
            "time": (base.replace(hour=idx)).isoformat(),
            "stdout": f"result-{idx}",
            "plan_id": plan,
        }
        if tool:
            payload["tool"] = tool
        store.append(payload)
    return store


def test_search_applies_tool_and_time_filters(memory_store: MemoryStore) -> None:
    retriever = MemoryRetriever(memory_store)
    slice_all = retriever.search("result", limit=5)
    assert len(slice_all.matches) == 3
    confidences = [match["confidence"] for match in slice_all.matches]
    assert all(0.0 <= value <= 1.0 for value in confidences)
    assert slice_all.confidence_summary["count"] == 3
    assert slice_all.confidence_summary["nonzero_fraction"] > 0

    filtered = retriever.search(
        "result",
        limit=5,
        tools=["calculator"],
        since="2024-01-01T02:00:00+00:00",
        until="2024-01-01T02:00:00+00:00",
    )
    assert len(filtered.matches) == 1
    assert filtered.matches[0]["tool"] == "calculator"
    assert filtered.window == {
        "since": "2024-01-01T02:00:00+00:00",
        "until": "2024-01-01T02:00:00+00:00",
    }
    assert filtered.confidence_summary["count"] == 1


def test_timeline_respects_limits(memory_store: MemoryStore) -> None:
    retriever = MemoryRetriever(memory_store)
    window = retriever.timeline(limit=2)
    assert len(window.records) == 2
    assert window.records[0]["stdout"] == "result-1"
    assert window.filters is None


def test_timeline_applies_tool_filters(memory_store: MemoryStore) -> None:
    retriever = MemoryRetriever(memory_store)
    window = retriever.timeline(limit=5, tools=["python_runner"])
    assert len(window.records) == 1
    assert window.records[0]["tool"] == "python_runner"
    assert window.filters == {"tools": ["python_runner"]}


def test_plan_context_groups_records(memory_store: MemoryStore) -> None:
    retriever = MemoryRetriever(memory_store)
    context = retriever.plan_context("plan-1", limit=5)
    assert context.plan_id == "plan-1"
    assert len(context.episodes) == 1
    assert len(context.reflections) == 1
