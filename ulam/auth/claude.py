from __future__ import annotations

import re
import subprocess
from typing import Optional


def run_claude_setup_token() -> Optional[str]:
    try:
        proc = subprocess.run(
            ["claude", "setup-token"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("claude CLI not found on PATH") from exc

    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    token = _extract_token(output)
    if token:
        return token
    return None


def run_claude_login() -> None:
    try:
        subprocess.run(
            ["claude", "auth", "login"],
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("claude CLI not found on PATH") from exc


def _extract_token(text: str) -> Optional[str]:
    matches = re.findall(r"\bsk-[A-Za-z0-9_-]{20,}\b", text)
    if matches:
        return matches[-1]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        last = lines[-1]
        if len(last) >= 20 and " " not in last:
            return last
    return None
