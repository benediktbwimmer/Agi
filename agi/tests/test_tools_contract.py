from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agi.src.core.memory import MemoryStore
from agi.src.core.tools.python_runner import PythonRunner, PythonRunnerError
from agi.src.core.tools.retrieval import RetrievalTool
from agi.src.core.types import RunContext


def test_python_runner_executes_code(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    runner = PythonRunner(artifacts_root=artifacts)
    ctx = RunContext(
        working_dir=str(tmp_path),
        timeout_s=5,
        env_whitelist=[],
        network="off",
        record_provenance=True,
    )
    result = asyncio.run(runner.run({"code": "print('hello world')"}, ctx))
    assert result.ok
    assert "hello world" in (result.stdout or "")
    assert artifacts.exists()


def test_python_runner_blocks_escape(tmp_path: Path) -> None:
    runner = PythonRunner(artifacts_root=tmp_path / "artifacts")
    ctx = RunContext(
        working_dir=str(tmp_path),
        timeout_s=5,
        env_whitelist=[],
        network="off",
        record_provenance=True,
    )
    code = """
from pathlib import Path
Path('/tmp/outside.txt').write_text('nope')
"""
    with pytest.raises(PythonRunnerError):
        asyncio.run(runner.run({"code": code}, ctx))


def test_retrieval_tool_filters_memory(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.jsonl")
    store.append(
        {
            "type": "episode",
            "tool": "calculator",
            "call_id": "call-episodic-alpha",
            "stdout": "alpha found",
            "time": "2024-01-01T00:00:00+00:00",
        }
    )
    store.append(
        {
            "type": "episode",
            "tool": "calculator",
            "call_id": "call-episodic-beta",
            "stdout": "beta result",
            "time": "2024-01-01T00:01:00+00:00",
        }
    )
    ctx = RunContext(
        working_dir=str(tmp_path),
        timeout_s=5,
        env_whitelist=[],
        network="off",
        record_provenance=True,
        working_memory=[
            {
                "tool": "calculator",
                "call_id": "call-working-alpha",
                "stdout": "alpha working context",
                "time": "2024-01-01T00:02:00+00:00",
            }
        ],
        episodic_memory=store,
    )

    tool = RetrievalTool()
    result = asyncio.run(tool.run({"query": "alpha", "tool_hint": "calculator"}, ctx))

    assert result.ok
    assert "alpha" in (result.stdout or "")
    assert "beta" not in (result.stdout or "")


def test_retrieval_tool_respects_zero_limit(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.jsonl")
    store.append(
        {
            "type": "episode",
            "tool": "calculator",
            "call_id": "call-episodic-alpha",
            "stdout": "alpha found",
            "time": "2024-01-01T00:00:00+00:00",
        }
    )

    ctx = RunContext(
        working_dir=str(tmp_path),
        timeout_s=5,
        env_whitelist=[],
        network="off",
        record_provenance=True,
        working_memory=[],
        episodic_memory=store,
    )

    tool = RetrievalTool()
    result = asyncio.run(
        tool.run({"query": "alpha", "tool_hint": "calculator", "limit": 0}, ctx)
    )

    assert result.ok
    assert "no episodic memory matched query" in (result.stdout or "")
    assert any(source.ref == "retrieval" for source in result.provenance)
