from __future__ import annotations

from ulam.lean.lsp_runner import (
    _locate_tactic_declaration,
    _render_candidate,
    _script_state_key,
)


def test_locate_tactic_declaration_extracts_goal_and_indent() -> None:
    text = """theorem demo_goal (n : Nat) : n = n := by
  sorry

theorem next_goal : True := by
  trivial
"""
    decl = _locate_tactic_declaration(text, "demo_goal")
    assert decl is not None
    assert decl.theorem == "demo_goal"
    assert decl.goal_hint.startswith("⊢")
    assert "n = n" in decl.goal_hint
    assert decl.indent == "  "


def test_render_candidate_adds_guard_and_probe_line() -> None:
    text = """theorem demo_goal : True := by
  sorry
"""
    decl = _locate_tactic_declaration(text, "demo_goal")
    assert decl is not None

    rendered, probe_line, probe_col = _render_candidate(
        decl=decl,
        script=["intro h", "exact True.intro"],
        with_sorry_guard=True,
    )
    assert "all_goals sorry" in rendered
    assert probe_col == 2
    assert probe_line >= 2


def test_script_state_key_is_stable_and_sensitive() -> None:
    a = _script_state_key(["simp", "exact h"])
    b = _script_state_key(["simp", "exact h"])
    c = _script_state_key(["simp", "exact h1"])
    assert a == b
    assert a != c
