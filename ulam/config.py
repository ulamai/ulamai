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
    "gemini": {
        "api_key": "",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-3.1-pro-preview",
        "cli_model": "",
    },
    "embed": {
        "api_key": "",
        "base_url": "https://api.openai.com",
        "model": "text-embedding-3-small",
        "cache": ".ulam/embeddings.json",
    },
    "prove": {
        "mode": "tactic",
        "output_format": "lean",
        "solver": "script",
        "autop": True,
        "k": 1,
        "tex_out_dir": "proofs",
        "tex_rounds": 3,
        "tex_judge_repairs": 2,
        "tex_worker_drafts": 2,
        "tex_concurrency": False,
        "tex_replan_passes": 2,
        "tex_action_steps": 10,
        "tex_planner_model": "",
        "tex_worker_model": "",
        "tex_artifacts_dir": "runs/prove_tex",
        "llm_rounds": 4,
        "llm_cycle_patience": 2,
        "llm_allow_helper_lemmas": True,
        "llm_edit_scope": "full",
        "llm_typecheck_backend": "cli",
        "search_lean_backend": "dojo",
        "lemma_max": 60,
        "lemma_depth": 60,
        "allow_axioms": True,
        "typecheck_timeout_s": 60,
        "retriever_k": 8,
        "retriever_source": "local",
        "retriever_build": "auto",
        "retriever_index": "",
    },
    "formalize": {
        "proof_backend": "inherit",
        "lean_backend": "dojo",
        "max_rounds": 5,
        "max_repairs": 5,
        "max_equivalence_repairs": 2,
        "max_proof_rounds": 1,
        "proof_repair": 2,
        "typecheck_timeout_s": 60,
        "llm_check": True,
        "llm_check_timing": "end",
        "llm_check_repairs": 2,
    },
    "segmentation": {
        "chunk_words": 1000,
    },
    "lean": {
        "project": "",
        "imports": [],
        "dojo_timeout_s": 180,
    },
    "policy": {
        "proof_profile": "balanced",
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
