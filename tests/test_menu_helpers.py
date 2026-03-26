from __future__ import annotations

import json
from pathlib import Path

from ulam.config import DEFAULT_CONFIG
from ulam.menu import (
    _build_args_from_config,
    _default_tex_launch_overrides,
)


def _config_copy() -> dict:
    return json.loads(json.dumps(DEFAULT_CONFIG))


def test_default_tex_launch_overrides_use_one_worker_defaults() -> None:
    overrides = _default_tex_launch_overrides(
        {
            "tex_worker_drafts": 4,
            "tex_concurrency": True,
            "tex_planner_model": "planner-pro",
            "tex_worker_model": "worker-fast",
        }
    )
    assert overrides == {
        "tex_worker_drafts": 1,
        "tex_concurrency": False,
        "tex_planner_model": "",
        "tex_worker_model": "",
    }


def test_build_args_from_config_applies_tex_overrides() -> None:
    config = _config_copy()
    config["llm_provider"] = "openai"
    config["openai"]["model"] = "gpt-5.4"
    config["prove"]["tex_action_steps"] = 10
    config["prove"]["tex_planner_model"] = "planner-pro"
    config["prove"]["tex_worker_model"] = "worker-fast"
    args = _build_args_from_config(
        config,
        file_path=None,
        theorem="demo_theorem",
        instruction="prove it",
        context_files=[],
        statement="For all n, P(n).",
        prove_overrides={
            "tex_worker_drafts": 1,
            "tex_concurrency": False,
            "tex_action_steps": 7,
            "tex_planner_model": "",
            "tex_worker_model": "",
            "tex_artifacts_dir": "runs/custom_tex",
        },
    )
    assert args.tex_worker_drafts == 1
    assert args.tex_concurrency is False
    assert args.tex_action_steps == 7
    assert args.tex_planner_model == ""
    assert args.tex_worker_model == ""
    assert args.tex_artifacts_dir == Path("runs/custom_tex")
