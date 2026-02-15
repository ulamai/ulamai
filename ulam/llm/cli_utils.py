from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path


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
        raise RuntimeError((stderr or "").strip() or "claude print failed")
    return stdout
