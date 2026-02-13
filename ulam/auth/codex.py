from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Optional


def codex_auth_path() -> Path:
    base = os.environ.get("CODEX_HOME", "~/.codex")
    return Path(base).expanduser() / "auth.json"


def run_codex_login() -> None:
    try:
        proc = subprocess.run(["codex", "login"], check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("codex CLI not found on PATH") from exc
    if proc.returncode != 0:
        raise RuntimeError("codex login failed")


def load_codex_api_key(path: Optional[Path] = None) -> Optional[str]:
    auth_path = path or codex_auth_path()
    if not auth_path.exists():
        return None
    try:
        data = json.loads(auth_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return _extract_api_key(data)


def load_codex_tokens(path: Optional[Path] = None) -> Optional[dict]:
    auth_path = path or codex_auth_path()
    if not auth_path.exists():
        return None
    try:
        data = json.loads(auth_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    tokens = data.get("tokens") if isinstance(data, dict) else None
    if not isinstance(tokens, dict):
        return None
    if "access_token" in tokens:
        return tokens
    return None


def _extract_api_key(data: Any) -> Optional[str]:
    if isinstance(data, dict):
        openai_key = data.get("OPENAI_API_KEY")
        if isinstance(openai_key, str) and _looks_like_openai_key(openai_key):
            return openai_key
        for key in ("api_key", "apiKey", "openai_api_key", "openaiApiKey", "key"):
            value = data.get(key)
            if isinstance(value, str) and _looks_like_openai_key(value):
                return value
        for value in data.values():
            found = _extract_api_key(value)
            if found:
                return found
    elif isinstance(data, list):
        for value in data:
            found = _extract_api_key(value)
            if found:
                return found
    elif isinstance(data, str):
        if _looks_like_openai_key(data):
            return data
    return None


def _looks_like_openai_key(text: str) -> bool:
    return re.search(r"\bsk-[A-Za-z0-9_-]{20,}\b", text) is not None
