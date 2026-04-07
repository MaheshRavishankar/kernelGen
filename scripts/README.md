# Scripts

One-off helper scripts for generating test data, selecting shapes, and other tasks that are not part of the regular benchmark workflow.

## Sandboxes

- `kernelgen-sandbox.sh` — Bubblewrap launcher for Claude bead workflows. Uses
  the main checkout as read-only and grants read-write access to the bead
  worktree, build directories, `.beads/`, and the shared caches Claude needs.
- `kernelgen-codex-sandbox.sh` — Bubblewrap launcher for Codex bead workflows.
  Mirrors the Claude sandbox model while mounting Codex state and config
  directories instead of Claude's.
- `setup-bwrap-apparmor.sh` — One-time AppArmor setup needed on Ubuntu 24.04+
  so `bwrap` can create user namespaces.

## gemm/

- `select_gemm_shapes.py` — Parses a CSV of GEMM shapes (e.g., from nightly benchmark reports), computes arithmetic intensity, and selects a representative subset across AI bins.
- `create_benchmark_tests.py` — Creates test directories with `config.json` for the selected shapes. Generates `.npy` reference data for small shapes; large shapes get config-only (random init at runtime).
