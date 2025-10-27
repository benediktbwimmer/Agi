# Merge and rebase guide

Maintainers frequently need to refresh the feature branch with the latest
changes from `main`. The following checklist mirrors the workflow that repeatedly
came up while iterating on the memory subsystem and associated guard tests.

1. **Ensure the working tree is clean.**
   Run `git status -sb` and resolve or commit outstanding edits before continuing.
2. **Stash local changes.**
   Execute `git stash push -u` so you can safely reapply the edits after updating
   the branch. When there are no changes, Git will report that nothing was
   stashed.
3. **Fetch the latest `main`.**
   Use `git fetch origin main`. If your environment lacks network credentials,
   configure the `origin` remote first via `git remote add origin <url>`.
4. **Rebase or merge.**
   Update the feature branch with `git rebase origin/main` (preferred) or
   `git merge origin/main`. Resolve any conflicts, run the test suite, and keep
   commits focused on conflict resolution.
5. **Reapply the stash.**
   Once the branch tracks `main`, bring back local edits with
   `git stash pop`. Handle any additional conflicts introduced during this step.
6. **Run merge guards.**
   Execute `pytest agi/tests/test_no_merge_conflicts.py` to ensure the helper
   checks confirm there are no lingering conflict markers or in-progress merges.

If fetching from `origin` fails with a 403 response, verify that your credentials
are available in the environment or use an SSH remote instead of HTTPS. After
addressing connectivity, repeat the sequence above from step 3.
