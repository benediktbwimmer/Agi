from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agi.src.core.memory import MemoryStore, _hash_source


@pytest.fixture
def memory_path(tmp_path: Path) -> Path:
    return tmp_path / "memory.jsonl"


def test_append_and_query(memory_path: Path) -> None:
    store = MemoryStore(memory_path)
    timestamp = datetime.now(timezone.utc).isoformat()
    record = {
        "type": "semantic",
        "claim": {"id": "c1"},
        "sources": [{"kind": "file", "ref": "artifact.txt"}],
        "time": timestamp,
    }
    store.append(record)

    assert store.query_by_claim("c1")

    digest = _hash_source(record["sources"][0])
    assert store.query_by_source_hash(digest)

    records = store.query_by_time(timestamp, timestamp)
    assert len(records) == 1


def test_crash_safe_write(memory_path: Path) -> None:
    store = MemoryStore(memory_path)
    record = {"type": "episode", "tool": "python_runner", "time": "2024-01-01T00:00:00+00:00"}
    store.append(record)

    with memory_path.open("r", encoding="utf-8") as f:
        data = f.read().strip().splitlines()
    assert len(data) == 1
    assert json.loads(data[0])["tool"] == "python_runner"


def test_tool_index(memory_path: Path) -> None:
    store = MemoryStore(memory_path)
    record = {
        "type": "episode",
        "tool": "python_runner",
        "time": "2024-01-01T00:00:00+00:00",
        "trace": [{"tool": "python_runner"}],
    }
    store.append(record)
    results = store.query_by_tool("python_runner")
    assert len(results) == 1


def test_plan_index(memory_path: Path) -> None:
    store = MemoryStore(memory_path)
    record = {
        "type": "episode",
        "tool": "python_runner",
        "time": "2024-01-01T00:00:00+00:00",
        "plan_id": "plan-42",
    }
    store.append(record)
    matches = store.query_by_plan("plan-42")
    assert len(matches) == 1
    assert matches[0]["plan_id"] == "plan-42"


def test_query_by_time_is_sorted(memory_path: Path) -> None:
    store = MemoryStore(memory_path)
    records = [
        {"type": "event", "time": "2024-01-01T00:02:00+00:00", "payload": 2},
        {"type": "event", "time": "2024-01-01T00:01:00+00:00", "payload": 1},
        {"type": "event", "time": "2024-01-01T00:03:00+00:00", "payload": 3},
    ]
    for record in records:
        store.append(record)

    results = store.query_by_time(
        "2024-01-01T00:00:00+00:00", "2024-01-01T00:04:00+00:00"
    )

    assert [entry["payload"] for entry in results] == [1, 2, 3]


def test_recent_returns_most_recent_records(memory_path: Path) -> None:
    store = MemoryStore(memory_path)
    records = [
        {"type": "event", "time": "2024-01-01T00:00:00+00:00", "payload": 1},
        {"type": "event", "time": "2024-01-01T00:01:00+00:00", "payload": 2},
        {"type": "event", "time": "2024-01-01T00:02:00+00:00", "payload": 3},
    ]
    for record in records:
        store.append(record)

    recent = store.recent(limit=2)
    assert [entry["payload"] for entry in recent] == [2, 3]

    recent_filtered = store.recent(limit=5, types=["event"])
    assert len(recent_filtered) == 3


def test_semantic_search_prioritises_recent_relevant_records(memory_path: Path) -> None:
    store = MemoryStore(memory_path)

    store.append(
        {
            "type": "episode",
            "tool": "calculator",
            "time": "2024-01-01T00:00:00+00:00",
            "stdout": "Computed optimal lunar transfer trajectory",
        }
    )
    store.append(
        {
            "type": "reflection",
            "summary": "Drafted research plan for lunar habitat",
            "time": "2024-01-01T01:00:00+00:00",
        }
    )
    store.append(
        {
            "type": "episode",
            "tool": "python_runner",
            "time": "2024-01-01T02:00:00+00:00",
            "stdout": "Analysed Martian soil sample chemistry",
        }
    )

    matches = store.semantic_search("lunar research plan", limit=2)
    assert len(matches) == 2
    assert matches[0]["summary"].startswith("Drafted research plan")
    assert matches[1]["stdout"].startswith("Computed optimal lunar")

    reflections_only = store.semantic_search("research plan", types=["reflection"])
    assert len(reflections_only) == 1
    assert reflections_only[0]["type"] == "reflection"
