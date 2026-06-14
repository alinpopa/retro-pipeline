---
name: pre-commit-secrets
description: >-
  Run the repository's secret scanner as a mandatory final gate before any git
  commit. Use whenever Codex is about to create a commit in this repository,
  including commits requested directly by the user or performed as part of a
  larger coding task.
---

# Pre-commit Secrets

Run the checked-in scanner immediately before committing. A successful scan is
required for the commit to proceed.

## Workflow

1. Finish all intended edits and staging changes.
2. Resolve the repository root:

   ```bash
   repo_root="$(git rev-parse --show-toplevel)"
   ```

3. Verify the scanner exists and is executable:

   ```bash
   test -x "$repo_root/.scripts/check-secrets"
   ```

4. From the repository root, scan the full worktree:

   ```bash
   cd "$repo_root"
   ./.scripts/check-secrets .
   ```

5. Run `git commit` only when the scanner exits successfully.

## Failure Handling

- If the scanner exits nonzero, stop and report the flagged paths and line
  numbers without exposing secret values.
- Remove the credential from the worktree and history as appropriate, then run
  the scanner again.
- Never bypass, weaken, or skip the scanner to make a commit succeed.
- If the scanner is missing or not executable, do not commit; report the
  repository setup problem.
