# Merge and Rebase Guide

The repository ships with merge-guard tests (`agi/tests/test_no_merge_conflicts.py`)
that enforce conflict-free branches. The checklist below reflects the workflow
maintainers follow when refreshing a feature branch with the latest `main`.

## 1. Confirm a clean worktree

```bash
git status -sb
```

Commit or stash local edits before pulling new history. The merge guard rejects
branches containing conflict markers or unmerged paths.

## 2. Stash optional work in progress

```bash
git stash push -u -m "pre-merge backup"
```

Stashing keeps experimental edits safe while you refresh the branch. When there
are no local changes Git will report that nothing was stashed. Reapply later
with `git stash pop`.

## 3. Fetch the latest `main`

```bash
git fetch origin main
```

Ensure the `origin` remote is configured—`git remote add origin <url>` if you
are working from a fresh clone or fork. If HTTPS credentials fail (e.g. 403),
switch to an SSH remote instead.

## 4. Rebase or merge

Rebasing keeps history linear:

```bash
git rebase origin/main
```

If you prefer to retain branch commits intact, merge instead:

```bash
git merge origin/main
```

Resolve conflicts, rerun the relevant tests (`pytest` at minimum), and record
the merge with a descriptive commit message.

## 5. Reapply the stash (if used)

Bring back local edits with:

```bash
git stash pop
```

Fix any additional conflicts introduced by the stash replay before proceeding.

## 6. Verify the merge guard

```bash
pytest agi/tests/test_no_merge_conflicts.py
```

The guard checks for conflict markers, unmerged paths, and in-progress Git
operations (`MERGE_HEAD`, `REBASE_HEAD`, etc.). A clean run indicates the branch
is safe to push.

## 7. Push and review

```bash
git push --force-with-lease
```

Force-push is acceptable for rebased feature branches; the `--force-with-lease`
flag prevents overwriting remote changes you have not seen.

Following these steps keeps the automated merge simulations green and avoids
surprises when the PR is ready to land. The merge guard is intentionally
strict—if it fails, it highlights which merge artefacts still need attention.
