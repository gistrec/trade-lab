#!/usr/bin/env bash
# Install the trading runtime into a venv, pinned to constraints.txt.
#
# This is THE versioned install procedure. Every path that puts code on a
# host must go through it — the SSH wrapper (deploy.sh, owner-only and
# gitignored because it holds host details), an ansible role, or a human
# rebuilding the box by hand.
#
# Why it is a tracked file rather than three lines inside deploy.sh: the
# whole point of constraints.txt is that the versions the suite ran
# against are the versions that place orders. A pin applied only by an
# untracked helper on one laptop is not applied at all — a rebuilt host,
# or a deploy from any other machine, silently resolves ccxt>=4.0 to
# whatever shipped that morning. That is exactly the drift this is meant
# to prevent (measured 2026-08-30: three ccxt versions in play at once).
#
# Usage:
#   scripts/install_runtime.sh [venv_path] [extras]
#
#   scripts/install_runtime.sh                    # runtime deps only (server)
#   scripts/install_runtime.sh .venv dev,research # with extras (dev box)
set -euo pipefail

VENV="${1:-.venv}"
EXTRAS="${2:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f constraints.txt ]]; then
    echo "FATAL: constraints.txt not found in $REPO_ROOT." >&2
    echo "Refusing an unconstrained install into a real-money runtime." >&2
    exit 2
fi

if [[ ! -x "$VENV/bin/pip" ]]; then
    echo "FATAL: no venv at $VENV (expected $VENV/bin/pip)." >&2
    echo "Create it first: python3 -m venv $VENV" >&2
    exit 2
fi

TARGET="."
[[ -n "$EXTRAS" ]] && TARGET=".[$EXTRAS]"

echo "[install] $VENV <- $TARGET  (constrained)"
"$VENV/bin/pip" install -e "$TARGET" -c constraints.txt -q

# Verify the venv is internally consistent BEFORE reporting success. A
# constrained install of the runtime alone will happily upgrade a shared
# dependency underneath a package that came from an extra: on 2026-08-31
# this box ended up with pyarrow 25.0.1 under streamlit 1.61.1, which
# declares pyarrow<25. It rendered anyway, so nothing failed — the
# mismatch was found by hand afterwards. That is the failure mode this
# pin exists to remove, so the installer has to say it out loud.
if ! "$VENV/bin/pip" check; then
    echo "[install] FATAL: the venv has unsatisfied requirements (above)." >&2
    echo "[install] Most likely an extra is installed but was not passed to" >&2
    echo "[install] this script — e.g. a host running the dashboard needs" >&2
    echo "[install] 'monitoring'. Re-run with the right extras." >&2
    exit 3
fi

# Echo what actually landed. The point of the pin is that this line matches
# what CI tested; printing it makes a mismatch visible in the deploy log
# instead of only in an incident.
"$VENV/bin/python" - <<'PY'
import importlib.metadata as md
for pkg in ("ccxt", "pandas", "numpy"):
    try:
        print(f"[install] {pkg}=={md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"[install] {pkg}: not installed")
PY
