from __future__ import annotations

import re
import subprocess


_DATA_COLLECTION_DISABLED_RE = re.compile(
    r"^\s*data\s+collection\s+is\s+disabled\.?\s*$",
    re.IGNORECASE,
)


def run_gemini_login() -> None:
    verify_gemini_login()


def verify_gemini_login(timeout_s: float = 300.0) -> None:
    try:
        proc = subprocess.run(
            ["gemini", "-p", "Reply with exactly: OK"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("gemini CLI not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"gemini verification timed out after {int(timeout_s)}s"
        ) from exc
    if proc.returncode == 0:
        return
    out = _strip_known_noise(proc.stdout or "")
    err = _strip_known_noise(proc.stderr or "")
    msg = (err or out).strip()
    raise RuntimeError(msg or "gemini verification failed")


def _strip_known_noise(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        if _DATA_COLLECTION_DISABLED_RE.match(line):
            continue
        lines.append(line)
    return "\n".join(lines).strip()
