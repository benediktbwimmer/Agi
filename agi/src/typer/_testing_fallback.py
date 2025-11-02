from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Any, Sequence

from . import Exit, Typer, secho


@dataclass
class Result:
    exit_code: int
    stdout: str
    stderr: str
    exception: Exception | None = None


class CliRunner:
    """Lightweight test runner compatible with the subset of Typer used in tests."""

    def invoke(self, app: Typer, args: Sequence[str] | None = None) -> Result:
        argv = list(args or [])
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        exit_code = 0
        exception: Exception | None = None
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            try:
                exit_code, _ = app._execute(argv)
            except Exit as exc:
                exit_code = exc.code
            except Exception as exc:  # pragma: no cover - surfaced in tests
                exit_code = 1
                exception = exc
                secho(f"Error: {exc}", err=True)
        return Result(
            exit_code=exit_code,
            stdout=stdout_buffer.getvalue(),
            stderr=stderr_buffer.getvalue(),
            exception=exception,
        )
