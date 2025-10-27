import asyncio
from datetime import datetime
from pathlib import Path

from agi.src.core.memory import MemoryStore
from agi.src.core.tools.retrieval import RetrievalTool
from agi.src.core.types import RunContext


def _run_ctx(tmp_path: Path) -> RunContext:
    return RunContext(
        working_dir=str(tmp_path),
        timeout_s=5,
        env_whitelist=[],
        network="off",
        record_provenance=True,
    )


def _append_episode(store: MemoryStore, **overrides) -> None:
    base = {
        "type": "episode",
        "plan_id": "plan-1",
        "tool": "python_runner",
        "call_id": overrides.get("call_id", "call"),
        "ok": True,
        "stdout": overrides.get("stdout", "computed result"),
        "time": overrides.get(
            "time",
            datetime(2024, 1, 1, 12, 0, 0).isoformat() + "+00:00",
        ),
        "provenance": overrides.get("provenance", []),
    }
    store.append({**base, **overrides})


def test_retrieval_returns_semantic_matches(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.jsonl")
    _append_episode(
        memory,
        call_id="a",
        stdout="Measured gravitational constant to high precision",
        time="2024-03-01T10:00:00+00:00",
    )
    _append_episode(
        memory,
        call_id="b",
        stdout="Performed unrelated calculation",
        time="2024-03-02T11:00:00+00:00",
    )
    tool = RetrievalTool(memory)
    ctx = _run_ctx(tmp_path)

    result = asyncio.run(tool.run({"query": "gravitational constant"}, ctx))

    assert result.ok
    records = result.data["records"]
    assert len(records) == 1
    assert "gravitational constant" in records[0]["stdout"]


def test_retrieval_filters_by_time_and_claim(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.jsonl")
    _append_episode(
        memory,
        call_id="old",
        stdout="Archived experiment",
        time="2023-01-01T00:00:00+00:00",
        claim={"id": "claim-1"},
    )
    _append_episode(
        memory,
        call_id="recent",
        stdout="Recent experiment",
        time="2024-05-01T00:00:00+00:00",
        claim={"id": "claim-1"},
    )
    _append_episode(
        memory,
        call_id="other",
        stdout="Other claim data",
        time="2024-05-02T00:00:00+00:00",
        claim={"id": "claim-2"},
    )
    tool = RetrievalTool(memory)
    ctx = _run_ctx(tmp_path)

    args = {
        "claim_id": "claim-1",
        "since": "2024-01-01T00:00:00+00:00",
        "limit": 2,
    }
    result = asyncio.run(tool.run(args, ctx))

    assert result.ok
    records = result.data["records"]
    assert len(records) == 1
    assert records[0]["call_id"] == "recent"
