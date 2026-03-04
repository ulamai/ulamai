from __future__ import annotations

from argparse import Namespace

from ulam.cli import _inference_signature, _resolve_inference_runtime
from ulam.search.best_first import _rank_and_limit_suggestions


def test_resolve_inference_runtime_balanced_defaults() -> None:
    args = Namespace(
        k=1,
        inference_profile="balanced",
        gen_k=0,
        exec_k=0,
        verify_level="auto",
    )
    profile, gen_k, exec_k, verify_level = _resolve_inference_runtime(args)

    assert profile == "balanced"
    assert gen_k == 6
    assert exec_k == 3
    assert verify_level == "strict"


def test_resolve_inference_runtime_overrides_are_applied() -> None:
    args = Namespace(
        k=2,
        inference_profile="explore",
        gen_k=4,
        exec_k=2,
        verify_level="none",
    )
    profile, gen_k, exec_k, verify_level = _resolve_inference_runtime(args)

    assert profile == "explore"
    assert gen_k == 4
    assert exec_k == 2
    assert verify_level == "none"


def test_inference_signature_formats_exec_all() -> None:
    meta = {
        "inference_profile": "default",
        "generation_budget_per_state": 1,
        "execution_budget_per_state": 0,
        "verification_level": "light",
    }
    assert (
        _inference_signature(meta)
        == "profile=default, gen=1, exec=all, verify=light"
    )


def test_rank_and_limit_suggestions_strict_filters_and_prefers_cached() -> None:
    ranked = _rank_and_limit_suggestions(
        [
            "aesop",
            "rw [h]; simp",
            "simpa",
            "exact h",
            "repeat' simp",
            "nlinarith",
        ],
        cached_tactics=["aesop"],
        execution_budget_per_state=3,
        verification_level="strict",
    )

    assert ranked == ["aesop", "simpa", "exact h"]
