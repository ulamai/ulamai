from __future__ import annotations

from argparse import Namespace

from ulam.cli import _evaluate_bench_parity_gate


def _gate_args(**overrides: object) -> Namespace:
    base = {
        "gate": True,
        "max_solved_drop": 0.0,
        "max_success_rate_drop": 0.0,
        "max_semantic_pass_rate_drop": 0.0,
        "max_regression_rejection_rate_increase": 0.0,
        "max_median_time_increase_pct": 25.0,
        "max_planner_replan_triggers_increase": 0.0,
        "max_planner_cached_tactic_tries_drop": 0.0,
        "allow_profile_mismatch": False,
        "allow_suite_mismatch": False,
    }
    base.update(overrides)
    return Namespace(**base)


def test_gate_rejects_profile_mismatch_by_default() -> None:
    metrics = {
        "solved": 1.0,
        "success_rate_percent": 100.0,
        "semantic_pass_rate_percent": 0.0,
        "regression_rejection_rate_percent": 0.0,
        "median_duration_s": 1.0,
        "planner_replan_triggers_total": 0.0,
        "planner_cached_tactic_tries_total": 0.0,
    }
    gate = _evaluate_bench_parity_gate(
        metrics,
        metrics,
        _gate_args(),
        comparable_suite_sha=True,
        same_suite_sha=True,
        comparable_inference=True,
        same_inference=False,
    )
    assert gate["passed"] is False
    assert any("inference profile/budgets mismatch" in reason for reason in gate["reasons"])


def test_gate_rejects_planner_regression_by_default() -> None:
    metrics_a = {
        "solved": 1.0,
        "success_rate_percent": 100.0,
        "semantic_pass_rate_percent": 0.0,
        "regression_rejection_rate_percent": 0.0,
        "median_duration_s": 1.0,
        "planner_replan_triggers_total": 1.0,
        "planner_cached_tactic_tries_total": 5.0,
    }
    metrics_b = dict(metrics_a)
    metrics_b["planner_replan_triggers_total"] = 2.0
    metrics_b["planner_cached_tactic_tries_total"] = 4.0
    gate = _evaluate_bench_parity_gate(
        metrics_a,
        metrics_b,
        _gate_args(),
        comparable_suite_sha=True,
        same_suite_sha=True,
        comparable_inference=True,
        same_inference=True,
    )
    assert gate["passed"] is False
    assert any("planner-replan-trigger increase" in reason for reason in gate["reasons"])
    assert any("planner-cached-tactic-tries drop" in reason for reason in gate["reasons"])


def test_gate_allows_comparability_mismatch_with_opt_out_flags() -> None:
    metrics = {
        "solved": 1.0,
        "success_rate_percent": 100.0,
        "semantic_pass_rate_percent": 0.0,
        "regression_rejection_rate_percent": 0.0,
        "median_duration_s": 1.0,
        "planner_replan_triggers_total": 0.0,
        "planner_cached_tactic_tries_total": 0.0,
    }
    gate = _evaluate_bench_parity_gate(
        metrics,
        metrics,
        _gate_args(
            allow_profile_mismatch=True,
            allow_suite_mismatch=True,
        ),
        comparable_suite_sha=False,
        same_suite_sha=False,
        comparable_inference=False,
        same_inference=False,
    )
    assert gate["passed"] is True
