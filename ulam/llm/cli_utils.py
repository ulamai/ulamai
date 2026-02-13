from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


def codex_exec(system_prompt: str, user_prompt: str, model: str | None = None) -> str:
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
        proc = subprocess.run(cmd, input=prompt, text=True, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "codex exec failed")
        if not out_path.exists():
            raise RuntimeError("codex exec did not produce output")
        return out_path.read_text(encoding="utf-8")


def claude_print(system_prompt: str, user_prompt: str, model: str | None = None) -> str:
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
    proc = subprocess.run(cmd, input=user_prompt, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "claude print failed")
    return proc.stdout
