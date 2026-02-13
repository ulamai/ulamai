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

python3 -m pip install --user -e "$ROOT_DIR"

BIN_DIR=$(python3 - <<'PY'
import os
import site
print(os.path.join(site.USER_BASE, "bin"))
PY
)

if [[ ":$PATH:" != *":${BIN_DIR}:"* ]]; then
  echo "Added to PATH? Not detected. Add this to your shell config:"
  echo "export PATH=\"${BIN_DIR}:$PATH\""
fi

echo "Installed Ulam Prover. You can now run: ulam --help"
