#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required." >&2
  exit 1
fi

if command -v pipx >/dev/null 2>&1; then
  pipx install -e "$ROOT_DIR" --force
  echo "Installed Ulam Prover with pipx."
  exit 0
fi

set +e
PIP_ARGS=(--user -e "$ROOT_DIR")
if [[ "${ULAM_BREAK_SYSTEM_PACKAGES:-}" != "" ]]; then
  PIP_ARGS+=(--break-system-packages)
fi
python3 -m pip install "${PIP_ARGS[@]}" 2> /tmp/ulam_pip_err.txt
PIP_STATUS=$?
set -e

if [[ $PIP_STATUS -ne 0 ]]; then
  if grep -qiE "externally managed|externally-managed-environment" /tmp/ulam_pip_err.txt 2>/dev/null; then
    echo "Detected externally managed Python. Falling back to a user venv."
  else
    echo "pip install --user failed. Falling back to a user venv."
  fi
  VENV_DIR="${ULAM_VENV_DIR:-$HOME/.local/ulam-venv}"
  python3 -m venv "$VENV_DIR"
  "$VENV_DIR/bin/python" -m pip install -U pip setuptools wheel
  "$VENV_DIR/bin/python" -m pip install -e "$ROOT_DIR"
  BIN_DIR="$HOME/.local/bin"
  mkdir -p "$BIN_DIR"
  ln -sf "$VENV_DIR/bin/ulam" "$BIN_DIR/ulam"
else
  BIN_DIR=$(python3 - <<'PY'
import os
import site
print(os.path.join(site.USER_BASE, "bin"))
PY
)
fi

if [[ ":$PATH:" != *":${BIN_DIR}:"* ]]; then
  echo "Added to PATH? Not detected. Add this to your shell config:"
  echo "export PATH=\"${BIN_DIR}:$PATH\""
fi

echo "Installed Ulam Prover. You can now run: ulam --help"
