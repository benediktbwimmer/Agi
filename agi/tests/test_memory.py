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
