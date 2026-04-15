______________________________________________________________________

## name: review description: Review kernelGen diffs and PRs for correctness, benchmark validity, provider API misuse, GPU timing errors, project convention violations, and missing validation.

# kernelGen Review

Use this skill when reviewing a diff, PR, or changed files in kernelGen.

Always read [AGENTS.md](../../../../AGENTS.md), the touched files, and the
relevant test or benchmark configuration before reviewing.

## Scope

- PR number: inspect `gh pr diff <number>` and PR metadata.
- File paths: read the files and their git diff.
- Default: review `git diff` and `git diff --cached`.

## Review Priorities

1. Correctness
   - GEMM shape, layout, dtype, stride, transposition, and accumulation handling
   - BF16 accumulation uses `compute_type: f32`
   - reference verification covers the modified path
   - generated artifacts match the source configs that claim to produce them
1. Provider API usage
   - HIP stream, event, allocation, and synchronization lifetimes
   - hipBLASLt descriptor, preference, algorithm, and workspace lifetimes
   - IREE module, VMFB cache, device, and invocation setup
   - Fusilli graph, compile, runtime handle, buffer, and stream handling
1. Benchmark integrity
   - timings use HIP events on the intended stream
   - warmup and measured iterations exercise the same code path
   - provider comparisons use the same shapes, dtypes, and verification mode
   - reported TFLOPS and roofline conclusions are supported by data
1. Memory safety and lifecycle
   - allocation/free pairing
   - host/device buffer sizes and element types
   - object ownership across C++ and Python boundaries
   - cleanup on error paths
1. Project conventions
   - code follows the existing provider and test structure
   - repo venv is used for Python entrypoints when relevant
   - build paths and cache paths respect `AGENTS.md`
   - bead work happens in the bead worktree, not the main checkout
1. Test integrity
   - relevant tests or benchmark smoke runs are reported
   - failures, skipped checks, or missing prerequisites are not hidden
   - performance claims include measured evidence

Only report issues you can support from the current diff and code.

## Hard Rules

- Lead with findings, ordered by severity, with `file:line` references.
- The change is not ready if relevant validation is failing.
- The change is not ready if there is no evidence that relevant validation ran,
  unless the review is explicitly scoped as code-only.
- Do not accept disabling, skipping, or narrowing validation to avoid fixing a
  regression.
- Do not claim a performance improvement without measured before/after data.

## Output

Report findings first. If no findings meet the threshold, state that clearly
and list any residual validation gaps.
