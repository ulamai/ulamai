from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from ulam.cli import (
    _normalize_tex_proof,
    _resolve_prove_output_format,
    _resolve_tex_output_path,
)
from ulam.formalize.llm import _parse_tex_judge, _parse_tex_plan


def test_resolve_prove_output_format_prefers_explicit_arg() -> None:
    args = Namespace(output_format="tex")
    cfg = {"prove": {"output_format": "lean"}}
    assert _resolve_prove_output_format(args, cfg) == "tex"


def test_normalize_tex_proof_wraps_plain_text() -> None:
    rendered = _normalize_tex_proof(
        "Assume h. Therefore contradiction.",
        theorem="my_theorem",
        theorem_statement="P -> False",
    )
    assert "\\begin{theorem}[my_theorem]" in rendered
    assert "\\begin{proof}" in rendered
    assert "\\end{proof}" in rendered


def test_resolve_tex_output_path_uses_configured_dir(tmp_path: Path) -> None:
    args = Namespace(tex_out=None)
    cfg = {"prove": {"tex_out_dir": "proofs"}}
    file_path = tmp_path / "Example.lean"
    out = _resolve_tex_output_path(args, cfg, "theorem one/two", file_path=file_path)
    assert out.parent == (tmp_path / "proofs")
    assert out.name == "theorem_one_two.tex"


def test_parse_tex_plan_json() -> None:
    parsed = _parse_tex_plan(
        """
        {"strategy":"induction",
         "outline":["base case","step case"],
         "key_lemmas":["lemma A"],
         "checks":["no missing hypotheses"]}
        """
    )
    assert parsed["strategy"] == "induction"
    assert parsed["outline"] == ["base case", "step case"]
    assert parsed["key_lemmas"] == ["lemma A"]


def test_parse_tex_judge_fallback_on_plain_text() -> None:
    parsed = _parse_tex_judge("Needs revision: one implication is missing.")
    assert parsed["verdict"] == "revise"
    assert parsed["score"] == 20
    assert parsed["required_changes"]
