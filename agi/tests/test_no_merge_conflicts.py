from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def _git_dir() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--git-dir"], check=True, capture_output=True, text=True
    )
    return Path(result.stdout.strip()).resolve()


def _tracked_files() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "ls-files"], check=True, capture_output=True, text=True
    )
    return [repo_root / Path(line.strip()) for line in result.stdout.splitlines() if line]


def test_repository_contains_no_merge_conflicts() -> None:
    start_marker = "<" * 7
    mid_marker = "=" * 7
    end_marker = ">" * 7
    diff3_marker = "|" * 7
    conflict_markers = (start_marker, mid_marker, end_marker, diff3_marker)
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


def test_worktree_has_no_unmerged_paths() -> None:
    """Ensure the git worktree is free from unmerged entries.

    This complements the marker scan by asserting that ``git status`` does not
    report merge conflict entries (lines beginning with ``U``).  It catches
    situations where a conflict has been staged without markers present in the
    file contents.
    """

    result = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    )
    conflict_lines = [
        line for line in result.stdout.splitlines() if line and line[0] == "U"
    ]
    assert not conflict_lines, f"unmerged paths detected: {conflict_lines}"


def test_repository_not_in_merge_flow() -> None:
    """Ensure ``.git`` does not contain sentinel merge/rebase state files."""

    git_dir = _git_dir()
    sentinels = [
        "MERGE_HEAD",
        "MERGE_MSG",
        "REBASE_HEAD",
        "CHERRY_PICK_HEAD",
        "REVERT_HEAD",
        "BISECT_LOG",
    ]
    offenders = [name for name in sentinels if (git_dir / name).exists()]
    assert not offenders, f"merge/rebase in-progress detected via {offenders}"


def _resolve_git_ref(name: str) -> str | None:
    """Return the full SHA for ``name`` if it exists, otherwise ``None``."""

    result = subprocess.run(
        ["git", "rev-parse", name], capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _merge_tree_has_conflicts(base: str, ours: str, theirs: str) -> bool:
    """Check ``git merge-tree`` output for conflict markers."""

    result = subprocess.run(
        ["git", "merge-tree", base, ours, theirs],
        capture_output=True,
        text=True,
        check=True,
    )
    markers = {"<" * 7, "=" * 7, ">" * 7, "|" * 7}
    return any(marker in result.stdout for marker in markers)


@pytest.mark.parametrize("candidate", ("main", "origin/main"))
def test_head_merges_cleanly_with_main(candidate: str) -> None:
    """If a ``main`` ref is present, ensure ``HEAD`` merges without conflicts."""

    target = _resolve_git_ref(candidate)
    if target is None:
        pytest.skip(f"{candidate} ref not available; skipping merge simulation")

    ours = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    merge_base = subprocess.run(
        ["git", "merge-base", ours, target],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    has_conflicts = _merge_tree_has_conflicts(merge_base, ours, target)
    assert not has_conflicts, (
        "simulated merge with main produced conflict markers; resolve conflicts"
    )
