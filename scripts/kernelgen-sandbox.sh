#!/bin/bash
# Sandboxed Claude Code runner for kernelGen implementer agents using bwrap.
#
# Restricts the agent to read-write access on a bead worktree and build
# directory while providing read-only access to the main checkout, SDKs,
# and system tools. Claude runs with --dangerously-skip-permissions inside
# the sandbox, so the filesystem restrictions ARE the permission model.
#
# SETUP (one-time, requires sudo):
#   sudo ./setup-bwrap-apparmor.sh
#
# Usage:
#   kernelgen-sandbox <bead-id> [-- claude-args...]
#
# Examples:
#   kernelgen-sandbox bd-ffh -- -p "Implement bd-ffh"
#   kernelgen-sandbox bd-k85 -- --resume
#
# The bead-id determines which worktree gets read-write access:
#   /home/mahesh/kernelGen/kernelGen-<bead-id>/
#
# Mount policy:
#   Read-write:  worktree, build dirs, .beads/, .git/worktrees/, ~/.claude*,
#                ~/.config/gh, ~/.cache
#   Read-only:   main checkout (rest), TheRock, IREE, ~/.gitconfig, ~/.ssh,
#                system tools
#   Devices:     /dev/kfd, /dev/dri (AMD GPU)

set -euo pipefail

# --- Parse arguments ---

BEAD_ID=""
CLAUDE_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --)
      shift
      CLAUDE_ARGS=("$@")
      break
      ;;
    *)
      if [[ -z "$BEAD_ID" ]]; then
        BEAD_ID="$1"
      else
        echo "Error: unexpected argument '$1'" >&2
        echo "Usage: kernelgen-sandbox <bead-id> [-- claude-args...]" >&2
        exit 1
      fi
      shift
      ;;
  esac
done

if [[ -z "$BEAD_ID" ]]; then
  echo "Error: bead-id is required." >&2
  echo "Usage: kernelgen-sandbox <bead-id> [-- claude-args...]" >&2
  exit 1
fi

# --- Paths ---

MAIN_CHECKOUT="/home/mahesh/kernelGen/kernelGen"
WORKTREE="/home/mahesh/kernelGen/kernelGen-${BEAD_ID}"
BUILD_BASE="/home/mahesh/kernelGen/build"
VENV="${MAIN_CHECKOUT}/.venv"

# SDK paths — fall back to defaults if env vars are unset.
THEROCK="${THEROCK_PATH:-/home/mahesh/kernelGen/TheRock}"
IREE_SRC="${IREE_SOURCE_DIR:-/home/mahesh/kernelGen/iree/iree}"

# Claude location (via nvm).
CLAUDE_BIN="$(command -v claude 2>/dev/null || echo "$HOME/.nvm/versions/node/v25.0.0/bin/claude")"
NVM_NODE_DIR="$HOME/.nvm/versions/node/v25.0.0"

# --- Preflight checks ---

if ! /usr/bin/bwrap --ro-bind / / -- true 2>/dev/null; then
  echo "Error: bwrap cannot create user namespaces." >&2
  echo "Run the one-time AppArmor setup:" >&2
  echo "  sudo ${MAIN_CHECKOUT}/scripts/setup-bwrap-apparmor.sh" >&2
  exit 1
fi

if [[ ! -d "$MAIN_CHECKOUT" ]]; then
  echo "Error: main checkout not found at ${MAIN_CHECKOUT}" >&2
  exit 1
fi

# --- Build conditional mount args ---

EXTRA_BINDS=()

# Worktree (read-write). Pre-create the directory so that git worktree add
# inside the sandbox writes to the real filesystem, not a tmpfs overlay.
mkdir -p "$WORKTREE"
EXTRA_BINDS+=(--bind "$WORKTREE" "$WORKTREE")

# Ensure .git/worktrees exists (created on first `git worktree add`).
mkdir -p "$MAIN_CHECKOUT/.git/worktrees"

# Build directory (read-write, create if needed).
mkdir -p "$BUILD_BASE"
EXTRA_BINDS+=(--bind "$BUILD_BASE" "$BUILD_BASE")

# TheRock SDK (read-only, if it exists).
if [[ -d "$THEROCK" ]]; then
  EXTRA_BINDS+=(--ro-bind "$THEROCK" "$THEROCK")
fi

# IREE source (read-only, if it exists).
if [[ -d "$IREE_SRC" ]]; then
  EXTRA_BINDS+=(--ro-bind "$IREE_SRC" "$IREE_SRC")
fi

# IREE build tools (read-only, if they exist).
if [[ -d "/home/mahesh/kernelGen/iree/build" ]]; then
  EXTRA_BINDS+=(--ro-bind "/home/mahesh/kernelGen/iree/build" "/home/mahesh/kernelGen/iree/build")
fi

# Python venv (read-only — agents should not pip install).
if [[ -d "$VENV" ]]; then
  EXTRA_BINDS+=(--ro-bind "$VENV" "$VENV")
fi

# Claude state dir (read-write, if it exists).
if [[ -d "$HOME/.local/state/claude" ]]; then
  EXTRA_BINDS+=(--bind "$HOME/.local/state/claude" "$HOME/.local/state/claude")
fi

echo "Sandboxed claude for bead ${BEAD_ID}"
echo "  Main checkout: ${MAIN_CHECKOUT} (read-only)"
echo "  Worktree:      ${WORKTREE} (read-write)"
echo "  Build dir:     ${BUILD_BASE} (read-write)"

# --- Build bwrap command ---
#
# On Ubuntu, /bin, /lib, /lib64 are symlinks into /usr, so binding /usr
# and recreating the symlinks is sufficient for all system tools.

exec /usr/bin/bwrap \
  --ro-bind /usr /usr \
  --symlink usr/bin /bin \
  --symlink usr/lib /lib \
  --symlink usr/lib64 /lib64 \
  --ro-bind /etc /etc \
  --ro-bind /opt /opt \
  --proc /proc \
  --ro-bind /sys /sys \
  --dev /dev \
  --dev-bind-try /dev/dri /dev/dri \
  --dev-bind-try /dev/kfd /dev/kfd \
  --bind /tmp /tmp \
  --ro-bind /run/systemd/resolve /run/systemd/resolve \
  \
  `# --- Home: tmpfs base with selective mounts ---` \
  --tmpfs "$HOME" \
  --ro-bind "$HOME/.bashrc" "$HOME/.bashrc" \
  --ro-bind "$HOME/.gitconfig" "$HOME/.gitconfig" \
  --ro-bind "$HOME/.ssh" "$HOME/.ssh" \
  --ro-bind "$HOME/.local" "$HOME/.local" \
  --ro-bind "$HOME/.nvm" "$HOME/.nvm" \
  \
  `# --- Claude state (read-write) ---` \
  --bind "$HOME/.claude" "$HOME/.claude" \
  --bind "$HOME/.claude.json" "$HOME/.claude.json" \
  --bind "$HOME/.config/gh" "$HOME/.config/gh" \
  --bind "$HOME/.cache" "$HOME/.cache" \
  \
  `# --- Main checkout: read-only with surgical read-write overlays ---` \
  --ro-bind "$MAIN_CHECKOUT" "$MAIN_CHECKOUT" \
  --bind "$MAIN_CHECKOUT/.beads" "$MAIN_CHECKOUT/.beads" \
  --bind "$MAIN_CHECKOUT/.git" "$MAIN_CHECKOUT/.git" \
  \
  `# --- Conditional mounts (worktree, build, SDKs) ---` \
  "${EXTRA_BINDS[@]}" \
  \
  `# --- Environment ---` \
  --setenv HOME "$HOME" \
  --setenv THEROCK_PATH "$THEROCK" \
  --setenv PATH "${VENV}/bin:$HOME/.local/bin:${NVM_NODE_DIR}/bin:/usr/local/bin:/usr/bin:/bin" \
  --setenv ANTHROPIC_API_KEY "${ANTHROPIC_API_KEY:-}" \
  --unsetenv VSCODE_GIT_ASKPASS_MAIN \
  --unsetenv VSCODE_GIT_ASKPASS_NODE \
  \
  --chdir "$MAIN_CHECKOUT" \
  --die-with-parent \
  -- \
  "$CLAUDE_BIN" --dangerously-skip-permissions "${CLAUDE_ARGS[@]}"
