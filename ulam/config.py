from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Any] = {
    "llm_provider": "openai",
    "llm": {
        "timeout_s": 0,
        "heartbeat_s": 60,
    },
    "openai": {
        "api_key": "",
        "base_url": "https://api.openai.com",
        "model": "gpt-4.1",
        "codex_model": "",
    },
    "ollama": {
        "base_url": "http://localhost:11434",
        "model": "llama3.1",
    },
    "anthropic": {
        "api_key": "",
        "setup_token": "",
        "base_url": "https://api.anthropic.com",
        "model": "claude-3-5-sonnet-20240620",
        "claude_model": "",
    },
    "embed": {
        "api_key": "",
        "base_url": "https://api.openai.com",
        "model": "text-embedding-3-small",
        "cache": ".ulam/embeddings.json",
    },
    "prove": {
        "allow_axioms": True,
    },
    "formalize": {
        "proof_backend": "inherit",
        "lean_backend": "dojo",
        "max_rounds": 5,
        "max_proof_rounds": 1,
        "proof_repair": 2,
    },
    "lean": {
        "project": "",
        "imports": [],
        "dojo_timeout_s": 180,
    },
}


def config_path() -> Path:
    if "ULAM_CONFIG" in os.environ:
        return Path(os.environ["ULAM_CONFIG"]).expanduser()
    config_dir = Path(os.environ.get("ULAM_CONFIG_DIR", ".ulam")).expanduser()
    if not config_dir.is_absolute():
        config_dir = Path.cwd() / config_dir
    return config_dir / "config.json"


def load_config() -> Dict[str, Any]:
    path = config_path()
    data = _deep_copy(DEFAULT_CONFIG)
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                _deep_update(data, raw)
        except Exception:
            pass
    return data


def save_config(data: Dict[str, Any]) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass


def _deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _deep_copy(data: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(data))
