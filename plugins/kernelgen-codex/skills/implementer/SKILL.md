______________________________________________________________________

## name: implementer description: Implement a kernelGen bead in an isolated git worktree, validate the relevant provider paths, run a review pass, and prepare the change for human review.

# kernelGen Implementer

Use this skill when the user asks Codex to implement a bead, pick up the next
task, or carry a kernelGen change through coding, review, validation, and PR
handoff.

Always read [AGENTS.md](../../../../AGENTS.md) and the bead details before
editing code.

## Core Rules

- Do not modify the main checkout for bead implementation work.
- Create and use one git worktree per bead.
- Keep bead status current with `br`.
- Run a separate review pass before reporting ready.
- Validate the code paths affected by the bead; do not rely on unrelated
  provider tests as evidence.
- Use the repo venv for Python entrypoints, especially IREE and Fusilli tooling.

## Durable State

Do not depend on private agent memory. Persist important state in:

- bead notes/comments via `br`
- git commits
- PR description and review-comment replies
- repo docs when the decision should outlive the bead

## Standard Flow

1. Claim the bead with `br update <bead-id> -s in_progress`.

1. Read the bead details with `br show <bead-id> --json` and `br show <bead-id>`.

1. Create or reuse a bead worktree:

   ```bash
   git worktree add /home/mahesh/kernelGen/kernelGen-<bead-id> \
     -b users/MaheshRavishankar/<bead-id>-<shortDescription>
   ```

1. Implement the change in the bead worktree, not in the main checkout.

1. Write or update focused tests, benchmark cases, reports, or documentation as
   required by the bead.

1. Run a review pass on the diff before final validation.

1. Run the narrowest meaningful validation first, then broaden when the change
   touches shared infrastructure.

1. Commit with a message that describes the actual code or doc change.

1. Push the branch and open or update the PR when requested by the task.

1. Record the validation result and PR number in the bead when a PR is opened.

## Validation Menu

Pick the commands that match the changed code:

```bash
cmake --preset Release
cmake --build ~/kernelGen/build/Release

.venv/bin/python gemm/tests/run_all.py --provider hipblaslt --verify -o /tmp/hipblaslt-results.json
.venv/bin/python gemm/tests/run_all.py --provider iree --verify -o /tmp/iree-results.json
.venv/bin/python gemm/tests/run_all.py --provider fusilli --verify -o /tmp/fusilli-results.json

.venv/bin/python gemm/kernel_providers/native_hip/run.py --test gemm/tests/ai_very_high_square --verify
.venv/bin/python gemm/kernel_providers/hipblaslt/run.py --test gemm/tests/ai_high_medium --verify
.venv/bin/python gemm/kernel_providers/iree/run.py --test gemm/tests/ai_high_medium --verify
```

For performance-sensitive changes, include measured before/after results and
use `gemm-profiler` for deeper rocprof analysis.

## Isolated Codex Execution

For a fully isolated implementer session, use the Codex sandbox launcher:

```bash
scripts/kernelgen-codex-sandbox.sh <bead-id> -- "<task prompt>"
```

The prompt must include the bead ID, worktree path, branch name, concrete task,
validation expectations, and an instruction to send a short final completion
message before exiting.

## Output

Report:

1. What changed
1. Validation run and results
1. Commit hash and PR number, when available
1. Any blocked validation with the exact missing prerequisite
