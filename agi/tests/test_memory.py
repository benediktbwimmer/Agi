from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
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


def test_semantic_search_boosts_keyword_matches(memory_path: Path) -> None:
    store = MemoryStore(memory_path)
    store.append(
        {
            "type": "episode",
            "tool": "analysis",
            "time": "2024-01-01T00:00:00+00:00",
            "stdout": "General mission update",
        }
    )
    store.append(
        {
            "type": "episode",
            "tool": "spectrometer",
            "time": "2024-01-01T01:00:00+00:00",
            "stdout": "Oxygen yield rose sharply",
            "keywords": ["oxygen", "yield"],
        }
    )

    matches = store.semantic_search("oxygen yield", limit=1)

    assert matches
    assert matches[0]["tool"] == "spectrometer"
    assert "semantic_score" in matches[0]
    assert "lexical_hits" in matches[0]


def test_semantic_search_exposes_vector_similarity(memory_path: Path) -> None:
    pytest.importorskip("faiss")
    store = MemoryStore(memory_path)
    store.append(
        {
            "type": "reflection",
            "summary": "Analysed quantum anomaly in gravitational lensing experiment.",
            "time": "2024-01-01T00:00:00+00:00",
        }
    )

    matches = store.semantic_search("quantum anomaly", limit=1)
    assert matches
    record = matches[0]
    assert "semantic_score" in record
    assert "vector_similarity" in record
    assert record["vector_similarity"] >= 0.0


def test_temporal_window_respects_anchor_and_filters(memory_path: Path) -> None:
    store = MemoryStore(memory_path)
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for idx in range(5):
        store.append(
            {
                "type": "reflection" if idx % 2 else "episode",
                "time": (base + timedelta(minutes=idx)).isoformat(),
                "payload": idx,
            }
        )

    window = store.temporal_window(
        anchor=base + timedelta(minutes=2),
        before=timedelta(minutes=1),
        after=timedelta(minutes=1),
    )

    assert [entry["payload"] for entry in window] == [1, 2, 3]

    reflections_only = store.temporal_window(
        anchor=base + timedelta(minutes=2),
        before=120,
        after=120,
        types=["reflection"],
    )
    assert [entry["payload"] for entry in reflections_only] == [1, 3]

    limited = store.temporal_window(before=180, after=0, limit=1)
    assert len(limited) == 1

    with pytest.raises(ValueError):
        store.temporal_window()


def test_search_supports_semantic_and_temporal_filters(memory_path: Path) -> None:
    store = MemoryStore(memory_path)
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)

    store.append(
        {
            "type": "reflection",
            "summary": "Outlined lunar base construction plan",
            "time": (base + timedelta(minutes=0)).isoformat(),
        }
    )
    store.append(
        {
            "type": "reflection",
            "summary": "Drafted Martian research proposal",
            "time": (base + timedelta(minutes=5)).isoformat(),
        }
    )
    store.append(
        {
            "type": "episode",
            "tool": "python_runner",
            "stdout": "Processed lunar regolith sample",
            "time": (base + timedelta(minutes=10)).isoformat(),
        }
    )

    matches = store.search(
        query="lunar plan",
        start=base,
        end=base + timedelta(minutes=6),
        types=["reflection"],
    )

    assert len(matches) == 1
    assert matches[0]["summary"].startswith("Outlined lunar base")


def test_search_without_query_defaults_to_temporal(memory_path: Path) -> None:
    store = MemoryStore(memory_path)
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for idx in range(3):
        store.append(
            {
                "type": "episode" if idx < 2 else "reflection",
                "time": (base + timedelta(minutes=idx)).isoformat(),
                "payload": idx,
            }
        )

    results = store.search(
        start=base + timedelta(minutes=1),
        end=base + timedelta(minutes=2),
        types=["episode", "reflection"],
        limit=2,
    )

    assert [entry["payload"] for entry in results] == [1, 2]
