from __future__ import annotations

from ulam.cli import _llm_error_cluster, _llm_strategy_pivot_error


def test_llm_error_cluster_normalizes_known_cases() -> None:
    assert _llm_error_cluster("1:2: unknown identifier 'foo'") == "unknown_identifier"
    assert _llm_error_cluster("type mismatch at line 42") == "type_mismatch"
    assert _llm_error_cluster("Declaration `T` still contains sorry/admit.") == "placeholder"


def test_llm_strategy_pivot_error_activates_on_repeat() -> None:
    counts: dict[str, int] = {}
    first = _llm_strategy_pivot_error(reason="type mismatch", cluster_counts=counts)
    second = _llm_strategy_pivot_error(reason="type mismatch", cluster_counts=counts)

    assert first == "type mismatch"
    assert "STRATEGY PIVOT REQUIRED" in second
    assert counts.get("type_mismatch", 0) == 2
