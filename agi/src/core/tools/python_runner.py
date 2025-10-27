from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import textwrap
import time
import uuid
from pathlib import Path
from typing import Any, Dict

from . import ToolCapability, ToolParameter, ToolSpec
from ..types import RunContext, Source, ToolResult


class PythonRunnerError(RuntimeError):
    """Raised when the sandboxed execution fails."""


class PythonRunner:
    """Execute Python code safely inside a sandboxed temporary directory."""

    name = "python_runner"
    safety = "T0"

    def __init__(self, artifacts_root: Path | None = None) -> None:
        self._artifacts_root = artifacts_root or Path.cwd() / "artifacts"
        self._artifacts_root.mkdir(parents=True, exist_ok=True)

    async def run(self, args: Dict[str, Any], ctx: RunContext) -> ToolResult:
        if ctx.network != "off":
            raise PythonRunnerError("PythonRunner only supports network=off")

        code = args.get("code")
        if not isinstance(code, str):
            raise PythonRunnerError("'code' argument must be provided as a string")

        stdin_data = args.get("stdin")
        if stdin_data is not None and not isinstance(stdin_data, str):
            raise PythonRunnerError("'stdin' must be a string if provided")

        sandbox_timeout = ctx.timeout_s or 30
        run_id = uuid.uuid4().hex
        artifacts_dir = self._artifacts_root / f"run_{run_id}"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        env = {key: os.environ.get(key, "") for key in ctx.env_whitelist}
        env.update(
            {
                "PYTHONUNBUFFERED": "1",
                "PYTHON_RUNNER_SANDBOX": "",
            }
        )

        async with _Sandbox(code, artifacts_dir) as sandbox:
            env["PYTHON_RUNNER_SANDBOX"] = str(sandbox.root)
            env["PYTHON_RUNNER_ENTRY"] = str(sandbox.entrypoint)
            start_time = time.perf_counter()
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(sandbox.bootstrap),
                cwd=sandbox.root,
                env=env,
                stdin=asyncio.subprocess.PIPE if stdin_data is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                stdout_bytes, _ = await asyncio.wait_for(
                    proc.communicate(stdin_data.encode("utf-8") if stdin_data is not None else None),
                    timeout=sandbox_timeout,
                )
            except asyncio.TimeoutError as exc:
                proc.kill()
                await proc.communicate()
                raise PythonRunnerError("Python code execution timed out") from exc
            duration_ms = int((time.perf_counter() - start_time) * 1000)

        if proc.returncode != 0:
            raise PythonRunnerError(
                f"Python code exited with status {proc.returncode}:\n{stdout_bytes.decode()}"
            )

        stdout = stdout_bytes.decode()
        manifest = {
            "run_id": run_id,
            "tool": self.name,
            "stdout": stdout,
        }
        sandbox.write_manifest(manifest)

        provenance = [
            Source(kind="file", ref=_safe_relpath(sandbox.manifest_path)),
        ]

        return ToolResult(
            call_id=args.get("id", run_id),
            ok=True,
            stdout=stdout,
            wall_time_ms=duration_ms,
            provenance=provenance,
            data={"artifacts_dir": _safe_relpath(artifacts_dir)},
        )

    def describe(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description="Execute sandboxed Python code with filesystem isolation and no network access.",
            safety_tier=self.safety,
            metadata={"artifacts_root": str(self._artifacts_root)},
            input_schema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source code to execute."},
                    "stdin": {
                        "type": "string",
                        "description": "Optional standard input passed to the process.",
                    },
                },
                "required": ["code"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "stdout": {"type": "string"},
                    "data": {
                        "type": "object",
                        "properties": {
                            "artifacts_dir": {
                                "type": "string",
                                "description": "Relative path to the captured sandbox artifacts.",
                            }
                        },
                    },
                },
            },
            capabilities=(
                ToolCapability(
                    name="execute_python",
                    description="Run Python code in a hermetic sandbox with deterministic IO policies.",
                    safety_tier=self.safety,
                    parameters=(
                        ToolParameter(
                            name="code",
                            description="Python source code to execute inside the sandbox.",
                            required=True,
                            schema={"type": "string"},
                        ),
                        ToolParameter(
                            name="stdin",
                            description="Optional stdin passed as UTF-8 text.",
                            required=False,
                            schema={"type": "string"},
                        ),
                    ),
                    outputs=("stdout", "data.artifacts_dir"),
                ),
            ),
        )


class _Sandbox:
    """Context manager that prepares a sandboxed environment."""

    BOOTSTRAP_TEMPLATE = textwrap.dedent(
        """
        import builtins
        import importlib
        import io
        import os
        import runpy
        import socket
        import sys
        from pathlib import Path

        SANDBOX_ROOT = Path(os.environ["PYTHON_RUNNER_SANDBOX"]).resolve()
        ENTRYPOINT = Path(os.environ["PYTHON_RUNNER_ENTRY"]).resolve()

        _real_open = builtins.open
        _real_io_open = io.open

        def _normalise(path: str) -> Path:
            p = Path(path)
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            else:
                p = p.resolve()
            return p

        def _ensure_within_sandbox(path: Path) -> None:
            try:
                path.relative_to(SANDBOX_ROOT)
            except ValueError as exc:  # pragma: no cover - safety guard
                raise PermissionError(f"File access outside sandbox disallowed: {path}") from exc

        def sandbox_open(file, mode='r', *args, **kwargs):
            if any(flag in mode for flag in ('w', 'a', '+', 'x')):
                _ensure_within_sandbox(_normalise(file))
            return _real_open(file, mode, *args, **kwargs)

        builtins.open = sandbox_open
        io.open = sandbox_open

        def _blocked_socket(*_args, **_kwargs):
            raise PermissionError("network access disabled in sandbox")

        class _SocketWrapper(socket.socket):
            def __init__(self, *args, **kwargs):  # pragma: no cover - defensive
                raise PermissionError("network access disabled in sandbox")

        socket.socket = _SocketWrapper
        socket.create_connection = _blocked_socket
        socket.create_server = _blocked_socket
        socket.socketpair = _blocked_socket
        socket.fromfd = _blocked_socket

        os.chdir(SANDBOX_ROOT)
        if str(SANDBOX_ROOT) not in sys.path:
            sys.path.insert(0, str(SANDBOX_ROOT))

        runpy.run_path(str(ENTRYPOINT), run_name="__main__")
        """
    )

    def __init__(self, code: str, artifacts_dir: Path) -> None:
        self.code = code
        self.artifacts_dir = artifacts_dir
        self._tmp: tempfile.TemporaryDirectory[str] | None = None
        self.root: Path
        self.entrypoint: Path
        self.bootstrap: Path
        self.manifest_path: Path

    async def __aenter__(self) -> "_Sandbox":
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.entrypoint = self.root / "user_code.py"
        self.bootstrap = self.root / "bootstrap.py"
        self.manifest_path = self.artifacts_dir / "manifest.json"
        self.entrypoint.write_text(self.code, encoding="utf-8")
        self.bootstrap.write_text(self.BOOTSTRAP_TEMPLATE, encoding="utf-8")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._tmp is not None:
            shutil.copytree(self.root, self.artifacts_dir, dirs_exist_ok=True)
            self._tmp.cleanup()

    def write_manifest(self, manifest: Dict[str, Any]) -> None:
        import json

        with self.manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)


def _safe_relpath(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)

