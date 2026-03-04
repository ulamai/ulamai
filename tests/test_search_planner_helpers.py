from __future__ import annotations

from ulam.search.best_first import (
    SearchResult,
    _cached_plan_tactics,
    _remember_plan_tactic,
    _with_planner_stats,
)


def test_remember_plan_tactic_adds_once_and_is_stable() -> None:
    cache: dict[str, list[str]] = {}
    state = "x : Nat\n⊢ x = x"
    next_state = "⊢ True"
    added = _remember_plan_tactic(cache, state, "simp", next_state)
    assert added is True

    rows = _cached_plan_tactics(cache, state)
    assert rows == ["simp"]

    added_again = _remember_plan_tactic(cache, state, "simp", next_state)
    assert added_again is False
    assert _cached_plan_tactics(cache, state) == ["simp"]


def test_remember_plan_tactic_rejects_strongly_worse_state() -> None:
    cache: dict[str, list[str]] = {}
    state = "x : Nat\n⊢ x = x"
    # Much larger state should be rejected by complexity threshold.
    next_state = "\n".join(
        [f"h{i} : Nat" for i in range(40)] + ["⊢ ∃ n : Nat, n = n"]
    )
    added = _remember_plan_tactic(cache, state, "linarith", next_state)
    assert added is False
    assert _cached_plan_tactics(cache, state) == []


def test_with_planner_stats_includes_cache_entry_count() -> None:
    base = SearchResult(solved=False, proof=[], steps=3, error="exhausted")
    stats = {
        "planner_replan_triggers": 2,
        "planner_cached_tactic_tries": 4,
    }
    cache = {"a": ["simp"], "b": ["linarith"]}
    merged = _with_planner_stats(base, stats, cache)

    assert merged.solved is False
    assert merged.steps == 3
    assert merged.stats["planner_replan_triggers"] == 2
    assert merged.stats["planner_cached_tactic_tries"] == 4
    assert merged.stats["planner_cache_entries"] == 2
