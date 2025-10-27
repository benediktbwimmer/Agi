from __future__ import annotations

import subprocess
from pathlib import Path


def _tracked_files() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "ls-files"], check=True, capture_output=True, text=True
    )
    return [repo_root / Path(line.strip()) for line in result.stdout.splitlines() if line]


def test_repository_contains_no_merge_conflicts() -> None:
    conflict_markers = ("<<<<<<<", "=======", ">>>>>>>")
    offenders: list[Path] = []
    for path in _tracked_files():
        # Binary files may raise decoding errors; ignore them quietly.
        try:
            contents = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(marker in contents for marker in conflict_markers):
            offenders.append(path)

    assert not offenders, f"merge conflict markers found in: {offenders}"
