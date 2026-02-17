#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ ! -f "${ROOT_DIR}/pyproject.toml" && ! -f "${ROOT_DIR}/setup.py" ]]; then
  SUBDIR=$(find "$ROOT_DIR" -maxdepth 2 -type f -name pyproject.toml | head -n 1 || true)
  if [[ -n "${SUBDIR}" ]]; then
    ROOT_DIR=$(cd "$(dirname "${SUBDIR}")" && pwd)
  fi
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required." >&2
  exit 1
fi

IN_VENV=$(python3 - <<'PY'
import os, sys
in_venv = (hasattr(sys, "base_prefix") and sys.prefix != sys.base_prefix) or bool(os.environ.get("VIRTUAL_ENV"))
print("1" if in_venv else "0")
PY
)

set +e
PIP_ARGS=(-e "$ROOT_DIR")
if [[ "${ULAM_BREAK_SYSTEM_PACKAGES:-}" != "" ]]; then
  PIP_ARGS+=(--break-system-packages)
fi

if [[ "$IN_VENV" == "1" ]]; then
  python3 -m pip install -U pip setuptools wheel
  python3 -m pip install "${PIP_ARGS[@]}" 2> /tmp/ulam_pip_err.txt
  PIP_STATUS=$?
else
  python3 -m pip install "${PIP_ARGS[@]}" 2> /tmp/ulam_pip_err.txt
  PIP_STATUS=$?
fi
set -e

if [[ $PIP_STATUS -ne 0 ]]; then
  if grep -qiE "externally managed|externally-managed-environment" /tmp/ulam_pip_err.txt 2>/dev/null; then
    echo "Detected externally managed Python."
    echo "Re-run with: ULAM_BREAK_SYSTEM_PACKAGES=1 ./install.sh"
    exit 1
  fi
  if grep -qiE "permission denied|not permitted" /tmp/ulam_pip_err.txt 2>/dev/null; then
    echo "Permission error installing globally."
    echo "Try: sudo ./install.sh"
    exit 1
  fi
  echo "pip install failed. See /tmp/ulam_pip_err.txt for details."
  exit 1
fi

BIN_DIR=$(python3 - <<'PY'
import sysconfig
print(sysconfig.get_path("scripts"))
PY
)

if [[ ":$PATH:" != *":${BIN_DIR}:"* ]]; then
  echo "Added to PATH? Not detected. Add this to your shell config:"
  echo "export PATH=\"${BIN_DIR}:$PATH\""
fi

echo "Installed Ulam Prover. You can now run: ulam --help"
