# Codex Instructions for kernelGen

Framework for benchmarking and comparing AMD GPU kernel providers for GEMM,
attention, and convolutions.

## Primary Repo Guidance

- Prefer additive Codex setup. Keep the existing Claude workflow working.
- Treat `CLAUDE.md` as Claude compatibility docs and `AGENTS.md` as the
  Codex source of truth.
- For bead-driven implementation work, use one git worktree per bead. Do not
  implement directly on the main checkout.
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
python gemm/tests/run_all.py --provider iree --verify -o results.json

python gemm/kernel_providers/hipblaslt/run.py --test gemm/tests/ai_high_medium --verify
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
  -b users/MaheshRavishankar/<bead-id>-<short-description>
br update <bead-id> -s in_progress
```

Launch Codex inside the external bwrap sandbox:

```bash
scripts/kernelgen-codex-sandbox.sh <bead-id> -- \
  "Your task prompt here. Bead: <bead-id>. Worktree: /home/mahesh/kernelGen/kernelGen-<bead-id>. Branch: users/MaheshRavishankar/<bead-id>-<short-description>."
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
- TheRock path defaults to `~/kernelGen/TheRock`, override with
  `-DTHEROCK_PATH=<path>`.
- GPU target defaults to `gfx1100`, override with `-DGPU_TARGETS=<target>`.
- IREE source defaults to `~/kernelGen/iree/iree/`, override with
  `-DIREE_SOURCE_DIR=<path>`.
- IREE `.vmfb` files are cached in `~/.cache/kernelgen/vmfb/<gpu_target>/`.
- All providers use HIP events on a HIP stream for timing.
