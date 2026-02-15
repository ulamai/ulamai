from __future__ import annotations

import os
import subprocess
from pathlib import Path


def lean_cli_check(
    file_path: Path,
    project_path: Path | None = None,
    timeout_s: float = 60.0,
) -> str | None:
    file_path = file_path.resolve()
    cmd = _lean_check_cmd(file_path)
    if not cmd:
        return "Lean CLI not found. Install Lean (elan) or ensure `lake` is on PATH."
    env = os.environ.copy()
    cwd = str(project_path) if project_path else None
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return f"Lean check timed out after {timeout_s:.0f}s"
    if proc.returncode == 0:
        return None
    stderr = (proc.stderr or "").strip()
    stdout = (proc.stdout or "").strip()
    if stderr and stdout:
        return stderr + "\n" + stdout
    return stderr or stdout or "Lean check failed"


def _lean_check_cmd(file_path: Path) -> list[str] | None:
    from shutil import which

    if which("lake") is not None:
        return ["lake", "env", "lean", str(file_path)]
    if which("lean") is not None:
        return ["lean", str(file_path)]
    return None
