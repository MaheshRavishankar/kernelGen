# kernelGen

A framework for benchmarking and comparing AMD GPU kernel providers
across operations (GEMM, attention, convolutions, etc.).

See [docs/InitialDesign.md](docs/InitialDesign.md) for motivation,
project layout, and usage instructions.

## Quick start

```bash
# Python setup
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Configure and build
cmake --preset Release
cmake --build ~/kernelGen/build/Release

# Generate test data and run
.venv/bin/python gemm/tests/generate_test.py gemm/tests/small_f16/config.json
.venv/bin/python gemm/kernel_providers/hipblaslt/run.py --test-dir gemm/tests
```

## Agent workflows

Both Claude and Codex use the same bead-driven, worktree-per-task flow.
Use `CLAUDE.md` for Claude-specific usage and `AGENTS.md` for Codex-specific
usage.

```bash
# One-time sandbox setup on Ubuntu 24.04+
sudo scripts/setup-bwrap-apparmor.sh

# Create a worktree for the bead
cd /home/mahesh/kernelGen/kernelGen
git worktree add /home/mahesh/kernelGen/kernelGen-<bead-id> \
  -b users/MaheshRavishankar/<bead-id>-<shortDescription>
br update <bead-id> -s in_progress

# Launch Claude in the sandbox
scripts/kernelgen-sandbox.sh <bead-id> -- \
  -p "Your task prompt here. Bead: <bead-id>. Worktree: /home/mahesh/kernelGen/kernelGen-<bead-id>. Branch: users/MaheshRavishankar/<bead-id>-<shortDescription>."

# Launch Codex in the sandbox
scripts/kernelgen-codex-sandbox.sh <bead-id> -- \
  "Your task prompt here. Bead: <bead-id>. Worktree: /home/mahesh/kernelGen/kernelGen-<bead-id>. Branch: users/MaheshRavishankar/<bead-id>-<shortDescription>."
```

## License

[MIT](LICENSE)
