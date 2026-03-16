from __future__ import annotations

from argparse import Namespace

from ulam.cli import (
    _apply_proof_profile_to_args,
    _normalize_proof_profile,
    _resolve_formalize_max_repairs,
    _resolve_formalize_max_rounds,
    _tex_claim_pass_gate,
)


def test_normal_alias_maps_to_balanced() -> None:
    assert _normalize_proof_profile("normal") == "balanced"
    assert _normalize_proof_profile("balanced") == "balanced"
    assert _normalize_proof_profile("fast") == "fast"
    assert _normalize_proof_profile("strict") == "strict"


def test_apply_fast_profile_sets_unset_budgets() -> None:
    args = Namespace(
        proof_profile=None,
        tex_rounds=None,
        tex_judge_repairs=None,
        tex_worker_drafts=None,
        tex_replan_passes=None,
        tex_verifier_policy=None,
        tex_compose_policy=None,
        max_rounds=None,
        max_repairs=None,
        max_proof_rounds=None,
        proof_repair=None,
        llm_check=None,
        llm_check_timing=None,
        llm_check_repairs=None,
    )
    _apply_proof_profile_to_args(args, "fast")
    assert args.proof_profile == "fast"
    assert args.tex_rounds == 2
    assert args.tex_worker_drafts == 1
    assert args.tex_verifier_policy == "final_only"
    assert args.tex_compose_policy == "on_complete"
    assert args.max_rounds == 3
    assert args.max_repairs == 3
    assert args.llm_check_timing == "end"


def test_apply_strict_profile_forces_guardrails() -> None:
    args = Namespace(
        proof_profile=None,
        allow_axioms=True,
        llm_allow_helper_lemmas=True,
        llm_edit_scope="full",
        tex_rounds=None,
        tex_judge_repairs=None,
        tex_worker_drafts=None,
        tex_replan_passes=None,
        tex_verifier_policy=None,
        tex_compose_policy=None,
        max_rounds=None,
        max_repairs=None,
        max_proof_rounds=None,
        proof_repair=None,
        llm_check=None,
        llm_check_timing=None,
        llm_check_repairs=None,
    )
    _apply_proof_profile_to_args(args, "strict")
    assert args.proof_profile == "strict"
    assert args.allow_axioms is False
    assert args.llm_allow_helper_lemmas is False
    assert args.llm_edit_scope == "errors_only"
    assert args.tex_verifier_policy == "worker"
    assert args.tex_compose_policy == "on_complete"
    assert args.llm_check_timing == "mid+end"


def test_resolve_formalize_max_repairs_defaults_to_rounds_when_unset() -> None:
    args = Namespace(max_rounds=None, max_repairs=None)
    cfg = {"formalize": {"max_rounds": 7}}
    rounds = _resolve_formalize_max_rounds(args, cfg)
    assert rounds == 7
    assert _resolve_formalize_max_repairs(args, rounds, cfg) == 7


def test_tex_pass_gate_can_skip_verifier_for_fast_path() -> None:
    judge = {"verdict": "pass"}
    verifier = {"verdict": "revise"}
    checker = {"status": "ok"}
    assert _tex_claim_pass_gate(judge, verifier, checker, [], require_verifier=False) is True
    assert _tex_claim_pass_gate(judge, verifier, checker, [], require_verifier=True) is False
