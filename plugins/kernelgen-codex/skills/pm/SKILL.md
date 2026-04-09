______________________________________________________________________

## name: pm description: Sweep kernelGen beads and PRs, classify what needs action, and orchestrate sandboxed implementer follow-ups for ready beads, open-PR feedback, and closed-PR cleanup.

# kernelGen Program Manager

Use this skill when the user asks for a sweep of active beads, wants Codex to
process pending work across beads and PRs, or wants agents launched for ready
implementation work, PR feedback, or cleanup.

Always read [AGENTS.md](../../../../AGENTS.md) first.

## Gather State

Run:

```bash
br list --json
br ready --json
br blocked --json
git -C /home/mahesh/kernelGen/kernelGen worktree list
```

For beads that are `in_progress` or otherwise appear to have a PR, inspect:

```bash
br show <bead-id> --json
br show <bead-id>
```

Prefer the bead `external-ref` as the PR source of truth when present
(`PR #<number>`). If needed, fall back to bead notes/comments.

For PR state, inspect:

```bash
gh pr view <number> --repo MaheshRavishankar/kernelGen --json state,isDraft,reviewDecision,statusCheckRollup,url
gh api repos/MaheshRavishankar/kernelGen/pulls/<number>/comments --paginate
gh api repos/MaheshRavishankar/kernelGen/pulls/<number>/reviews --paginate
```

## Classification

- Ready bead with no PR: launch an implementation agent.
- Open PR with unaddressed review comments or code-related failing checks:
  launch a follow-up agent to address feedback and update the PR.
- Closed PR with bead still open: launch a cleanup agent to close the bead and
  clean up the worktree/branch/build artifacts.
- Blocked bead: report the blocker only.
- Approved/open PR with passing checks and no pending feedback: report ready to
  land, do not merge automatically.

Treat any PR in `CLOSED` state as cleanup/close unless the user says otherwise.

## Execution Rule

Do not implement directly in the PM sweep. Delegate the work to sandboxed
Codex runs:

```bash
scripts/kernelgen-codex-sandbox.sh <bead-id> -- "<task prompt>"
```

The prompt must include:

- bead ID
- worktree path
- branch name
- PR number when one exists
- the exact action to take

For implementation and PR-follow-up prompts, explicitly require:

- push or update the PR when done
- a short final completion message that includes the PR number so it can be
  recorded in `br`

For cleanup prompts, explicitly require:

- close the bead in `br`
- remove the bead worktree if it still exists
- delete the local bead branch if safe
- report what was cleaned up

Launch independent bead actions in parallel when they do not overlap.
