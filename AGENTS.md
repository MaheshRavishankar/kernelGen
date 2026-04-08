# Codex Instructions for kernelGen

Framework for benchmarking and comparing AMD GPU kernel providers for GEMM,
attention, and convolutions.

## Primary Repo Guidance

- Prefer additive Codex setup. Keep the existing Claude workflow working.
- Treat `CLAUDE.md` as Claude compatibility docs and `AGENTS.md` as the
  Codex source of truth.
- For bead-driven implementation work, use one git worktree per bead. Do not
  implement directly on the main checkout.
- Use camelCase for the descriptive branch suffix in worktree branches.
  Prefer `users/<author>/<bead-id>-<shortDescription>`.
- Record durable decisions in beads, git history, or repo docs. Do not rely on
  private session memory.

## Build

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cmake --preset Release
cmake --build ~/kernelGen/build/Release
```

## Running Tests

```bash
python gemm/tests/run_all.py --provider hipblaslt --verify -o results.json
python gemm/tests/run_all.py --provider fusilli --verify -o results.json
python gemm/tests/run_all.py --provider iree --verify -o results.json

python gemm/kernel_providers/hipblaslt/run.py --test gemm/tests/ai_high_medium --verify
python gemm/kernel_providers/fusilli/run.py --test gemm/tests/ai_high_medium --verify
python gemm/kernel_providers/iree/run.py --test gemm/tests/ai_high_medium --verify
python gemm/kernel_providers/native_hip/run.py --test gemm/tests/ai_very_high_square

python gemm/roofline.py results.json --arch gfx1100
python gemm/profiling/profile.py --provider hipblaslt --test gemm/tests/ai_high_small -o profile.json
python gemm/profiling/analyze.py profile.json --arch gfx1100
```

## Beads

Issues are managed with `br`. Never edit `.beads/issues.jsonl` by hand.

Common commands:

```bash
br list
br show <id>
br ready
br blocked
br create "Title" -d "Description" -t task -p 1
br close <id> -r "Reason"
br dep add <issue> <depends-on>
br dep tree <id>
br update <id> -s in_progress
br search "keyword"
```

Use `--json` when machine-readable output is useful.

## Worktree And Sandbox Flow

Use the same worktree-per-bead flow as the Claude setup:

```bash
cd /home/mahesh/kernelGen/kernelGen
git worktree add /home/mahesh/kernelGen/kernelGen-<bead-id> \
  -b users/MaheshRavishankar/<bead-id>-<shortDescription>
br update <bead-id> -s in_progress
```

Launch Codex inside the external bwrap sandbox:

```bash
scripts/kernelgen-codex-sandbox.sh <bead-id> --timeout 2h -- \
  -m alpine-alpha \
  "Your task prompt here. Bead: <bead-id>. Worktree: /home/mahesh/kernelGen/kernelGen-<bead-id>. Branch: users/MaheshRavishankar/<bead-id>-<shortDescription>. Commit validated changes on that branch before exiting; if blocked, summarize the blocker clearly. When complete, send a short final completion message and exit."
```

Resume the most recent Codex session for that worktree:

```bash
scripts/kernelgen-codex-sandbox.sh <bead-id> -- resume --last
```

The prompt should explicitly include:

- bead ID
- worktree path
- branch name
- the concrete task to perform
- an instruction to commit validated changes on the bead branch before exiting,
  or to report the blocker clearly if it cannot do so
- an instruction to send a short final completion message before exiting so the
  sandbox wrapper can terminate cleanly

For implementation tasks, prefer a deepthink model such as `alpine-alpha`
instead of the default model.

Use an explicit timeout for each Codex run. `scripts/kernelgen-codex-sandbox.sh`
defaults to `--timeout 2h`, supports overriding with `--timeout <duration>`,
and reports timeout with exit code `124` so the launcher can tell the difference
between timeout and task failure.

## Repo-Local Codex Workflows

Reusable Codex skills live in:

- `plugins/kernelgen-codex/`

Available skills:

- `gemm-kernel-writer`
- `gemm-profiler`

Use them when the task is clearly about native HIP GEMM kernel iteration or
rocprof-based bottleneck analysis.

## Key Conventions

- All BF16 GEMM tests use `compute_type: f32` for accumulation.
- Use `Release` for benchmarking unless debugging a correctness issue.
- Use the repo venv for Python entrypoints, especially IREE and Fusilli
  tooling.
- TheRock path defaults to `~/kernelGen/TheRock`, override with
  `-DTHEROCK_PATH=<path>`.
- GPU target defaults to `gfx1100`, override with `-DGPU_TARGETS=<target>`.
- IREE source defaults to `~/kernelGen/iree/iree/`, override with
  `-DIREE_SOURCE_DIR=<path>`.
- IREE/Fusilli `.vmfb` files are cached in
  `~/.cache/kernelgen/vmfb/<provider>/<gpu_target>/`.
- All providers use HIP events on a HIP stream for timing.
