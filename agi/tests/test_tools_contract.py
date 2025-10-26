from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agi.src.core.tools.python_runner import PythonRunner, PythonRunnerError
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
