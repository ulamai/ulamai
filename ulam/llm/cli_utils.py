from __future__ import annotations

import re
import subprocess
import tempfile
import time
from pathlib import Path

_GEMINI_NOISE_LINE_PATTERNS = (
    re.compile(r"^\s*data\s+collection\s+is\s+disabled\.?\s*$", re.IGNORECASE),
    re.compile(
        r"^\s*accessing\s+resource\s+attributes\s+before\s+async\s+attributes\s+settled\.?\s*$",
        re.IGNORECASE,
    ),
)
_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def codex_exec(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    timeout_s: float | None = None,
    heartbeat_s: float | None = None,
) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "codex_out.txt"
        cmd = [
            "codex",
            "exec",
            "-",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--output-last-message",
            str(out_path),
        ]
        if model:
            cmd.extend(["--model", model])
        prompt = system_prompt.strip() + "\n\n" + user_prompt.strip()
        proc = subprocess.Popen(
            cmd,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        start = time.time()
        last_beat = start
        input_data = prompt
        while True:
            try:
                if input_data is not None:
                    stdout, stderr = proc.communicate(input=input_data, timeout=1)
                    input_data = None
                else:
                    stdout, stderr = proc.communicate(timeout=1)
                break
            except subprocess.TimeoutExpired:
                input_data = None
                now = time.time()
                if heartbeat_s and heartbeat_s > 0 and now - last_beat >= heartbeat_s:
                    elapsed = int(now - start)
                    print(f"[llm] still running ({elapsed}s)...")
                    last_beat = now
                if timeout_s and timeout_s > 0 and now - start >= timeout_s:
                    proc.kill()
                    raise RuntimeError(f"codex exec timed out after {timeout_s:.0f}s")
        if proc.returncode != 0:
            raise RuntimeError((stderr or "").strip() or "codex exec failed")
        if not out_path.exists():
            raise RuntimeError("codex exec did not produce output")
        return out_path.read_text(encoding="utf-8")


def claude_print(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    timeout_s: float | None = None,
    heartbeat_s: float | None = None,
) -> str:
    cmd = [
        "claude",
        "-p",
        "--output-format",
        "text",
        "--tools",
        "",
        "--system-prompt",
        system_prompt,
    ]
    if model:
        cmd.extend(["--model", model])
    return _claude_print_impl(
        cmd,
        user_prompt,
        timeout_s=timeout_s,
        heartbeat_s=heartbeat_s,
        retry_login=True,
    )


def gemini_exec(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    timeout_s: float | None = None,
    heartbeat_s: float | None = None,
) -> str:
    prompt = system_prompt.strip() + "\n\n" + user_prompt.strip()
    cmd = ["gemini", "-p", prompt]
    if model:
        cmd.extend(["-m", model])
    return _gemini_exec_impl(
        cmd,
        timeout_s=timeout_s,
        heartbeat_s=heartbeat_s,
    )


def _gemini_exec_impl(
    cmd: list[str],
    timeout_s: float | None,
    heartbeat_s: float | None,
) -> str:
    proc = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    start = time.time()
    last_beat = start
    while True:
        try:
            stdout, stderr = proc.communicate(timeout=1)
            break
        except subprocess.TimeoutExpired:
            now = time.time()
            if heartbeat_s and heartbeat_s > 0 and now - last_beat >= heartbeat_s:
                elapsed = int(now - start)
                print(f"[llm] still running ({elapsed}s)...")
                last_beat = now
            if timeout_s and timeout_s > 0 and now - start >= timeout_s:
                proc.kill()
                raise RuntimeError(f"gemini exec timed out after {timeout_s:.0f}s")
    if proc.returncode != 0:
        err = _strip_gemini_startup_noise((stderr or "").strip())
        out = _strip_gemini_startup_noise((stdout or "").strip())
        msg = err or out or f"gemini exec failed (exit {proc.returncode})"
        if _looks_like_gemini_auth_error(msg):
            raise RuntimeError(
                "Gemini CLI is not authenticated. Run `ulam auth gemini` or use "
                "Settings -> Configure LLM -> Gemini and complete OAuth login."
            )
        raise RuntimeError(msg)
    return _strip_gemini_startup_noise(stdout)


def _strip_gemini_startup_noise(text: str) -> str:
    if not text:
        return ""
    lines: list[str] = []
    for line in text.splitlines():
        normalized = _ANSI_ESCAPE_RE.sub("", line).strip()
        if any(pattern.match(normalized) for pattern in _GEMINI_NOISE_LINE_PATTERNS):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _looks_like_gemini_auth_error(text: str) -> bool:
    lower = text.lower()
    markers = (
        "not logged in",
        "login required",
        "oauth",
        "authorization required",
        "authenticate",
    )
    return any(marker in lower for marker in markers)


def _claude_print_impl(
    cmd: list[str],
    user_prompt: str,
    timeout_s: float | None,
    heartbeat_s: float | None,
    retry_login: bool,
) -> str:
    proc = subprocess.Popen(
        cmd,
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    start = time.time()
    last_beat = start
    input_data = user_prompt
    while True:
        try:
            if input_data is not None:
                stdout, stderr = proc.communicate(input=input_data, timeout=1)
                input_data = None
            else:
                stdout, stderr = proc.communicate(timeout=1)
            break
        except subprocess.TimeoutExpired:
            input_data = None
            now = time.time()
            if heartbeat_s and heartbeat_s > 0 and now - last_beat >= heartbeat_s:
                elapsed = int(now - start)
                print(f"[llm] still running ({elapsed}s)...")
                last_beat = now
            if timeout_s and timeout_s > 0 and now - start >= timeout_s:
                proc.kill()
                raise RuntimeError(f"claude print timed out after {timeout_s:.0f}s")
    if proc.returncode != 0:
        err = (stderr or "").strip()
        out = (stdout or "").strip()
        msg = err or out or f"claude print failed (exit {proc.returncode})"
        if retry_login and "Not logged in" in msg:
            _claude_auth_login()
            return _claude_print_impl(
                cmd,
                user_prompt,
                timeout_s=timeout_s,
                heartbeat_s=heartbeat_s,
                retry_login=False,
            )
        raise RuntimeError(msg)
    return stdout


def _claude_auth_login() -> None:
    try:
        subprocess.run(["claude", "auth", "login"], check=False)
    except Exception:
        return
