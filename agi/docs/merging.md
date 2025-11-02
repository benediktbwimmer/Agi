# Merge and Rebase Guide

The repository includes merge-guard tests (`agi/tests/test_no_merge_conflicts.py`)
that ensure feature branches stay in sync with `main`. The checklist below
captures the workflow the team uses when refreshing long-lived branches.

## 1. Confirm a clean worktree

```bash
git status -sb
```

Commit or stash local edits before pulling new history. The merge guard rejects
branches that contain conflict markers or unmerged paths.

## 2. Stash optional work in progress

When you have local patches that are not yet ready to commit:

```bash
git stash push -u -m "pre-merge backup"
```

Reapply the stash after the rebase/merge if needed with `git stash pop`.

## 3. Fetch the latest `main`

```bash
git fetch origin main
```

Ensure the `origin` remote is configured. If you work in a fork, add it once:

```bash
git remote add origin git@github.com:<user>/Agi.git
```

## 4. Rebase or merge

Rebasing keeps the history linear:

```bash
git rebase origin/main
```

If a merge is preferred (for example to retain branch commits intact):

```bash
git merge origin/main
```

Resolve conflicts, rerun the relevant tests (`pytest` at minimum), and record
the merge with a descriptive commit message.

## 5. Verify the merge guard

After updating the branch run:

```bash
pytest agi/tests/test_no_merge_conflicts.py
```

The test suite checks for conflict markers, unmerged paths, and unfinished
git operations (`MERGE_HEAD`, `REBASE_HEAD`, etc.). A clean run indicates the
branch is safe to push.

## 6. Push and review

```bash
git push --force-with-lease
```

Force-push is acceptable for rebased feature branches; the `--force-with-lease`
flag protects collaborators by refusing to overwrite remote updates you have
not seen.

---

Following these steps keeps the automated merge simulations green and avoids
surprises when the PR is ready to land. The merge guard test is intentionally
strict—when it fails, it highlights exactly which merge artefacts still need to
be resolved.
