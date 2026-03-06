from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from ulam.cli import (
    _build_tex_replan_instruction,
    _normalize_tex_proof,
    _resolve_prove_output_format,
    _resolve_tex_artifacts_root,
    _resolve_tex_claim_graph,
    _resolve_tex_output_path,
    _resolve_tex_replan_passes,
    _resolve_tex_resume_snapshot,
    _tex_static_claim_issues,
)
from ulam.formalize.llm import (
    _parse_tex_claim_checker,
    _parse_tex_claim_draft,
    _parse_tex_claim_judge,
    _parse_tex_claim_verifier,
    _parse_tex_judge,
    _parse_tex_plan,
)


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


def test_parse_tex_plan_claim_graph_included() -> None:
    parsed = _parse_tex_plan(
        """
        {"strategy":"split into cases",
         "outline":["set up","derive contradiction"],
         "claims":[
           {"id":"A 1","goal":"show P","depends_on":[]},
           {"id":"A 2","goal":"show Q","depends_on":["A_1","A 1","missing"]}
         ]}
        """
    )
    claims = parsed.get("claims", [])
    assert isinstance(claims, list)
    assert len(claims) == 2
    assert claims[0]["id"] == "A_1"
    assert claims[1]["depends_on"] == ["A_1"]


def test_resolve_tex_claim_graph_removes_forward_dependencies() -> None:
    plan = {
        "claims": [
            {"id": "C1", "goal": "g1", "depends_on": ["C2"]},
            {"id": "C2", "goal": "g2", "depends_on": []},
        ]
    }
    claims = _resolve_tex_claim_graph(plan, "goal")
    assert claims[0]["depends_on"] == []
    assert claims[1]["depends_on"] == []


def test_parse_tex_claim_json_helpers() -> None:
    draft = _parse_tex_claim_draft(
        '{"claim_id":"C1","proof_tex":"By contradiction.","assumptions_used":["h"],"depends_on_used":["C0"],"cited_facts":["Euclid"],"confidence":88}',
        fallback_claim_id="C1",
    )
    judge = _parse_tex_claim_judge(
        '{"verdict":"pass","score":91,"summary":"ok","required_changes":[],"missing_assumptions":[],"citation_issues":[],"polished_proof_tex":"Refined."}'
    )
    verifier = _parse_tex_claim_verifier(
        '{"verdict":"pass","score":85,"summary":"no material flaw","critical_issues":[],"counterexample_attempt":"","suggested_repairs":[]}'
    )
    checker = _parse_tex_claim_checker(
        '{"status":"ok","score":90,"issues":[],"warnings":[],"sanity_checks":["checked quantifiers"]}'
    )
    assert draft["confidence"] == 88
    assert judge["verdict"] == "pass"
    assert verifier["verdict"] == "pass"
    assert checker["status"] == "ok"


def test_build_tex_replan_instruction_mentions_previous_strategy() -> None:
    text = _build_tex_replan_instruction(
        "Use direct counting.",
        pass_idx=2,
        pass_history=[
            {
                "strategy": "direct contradiction",
                "unresolved_claims": ["C3", "C4"],
                "feedback": ["missing bound", "uses unstated assumption"],
            }
        ],
    )
    lowered = text.lower()
    assert "replan pass 2" in lowered
    assert "direct contradiction" in text
    assert "c3" in lowered
    assert "missing bound" in lowered


def test_resolve_tex_replan_passes_prefers_explicit() -> None:
    args = Namespace(tex_replan_passes=5)
    cfg = {"prove": {"tex_replan_passes": 2}}
    assert _resolve_tex_replan_passes(args, cfg) == 5


def test_resolve_tex_resume_snapshot_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run1"
    run_dir.mkdir(parents=True)
    snapshot = run_dir / "state.json"
    snapshot.write_text("{}", encoding="utf-8")
    args = Namespace(tex_resume=run_dir)
    resolved = _resolve_tex_resume_snapshot(args)
    assert resolved == snapshot.resolve()


def test_resolve_tex_artifacts_root_relative_to_file(tmp_path: Path) -> None:
    lean_file = tmp_path / "Math" / "Demo.lean"
    lean_file.parent.mkdir(parents=True)
    lean_file.write_text("theorem demo : True := by trivial\n", encoding="utf-8")
    args = Namespace(tex_artifacts_dir=None)
    cfg = {"prove": {"tex_artifacts_dir": "runs/prove_tex"}}
    root = _resolve_tex_artifacts_root(args, cfg, lean_file)
    assert root == (lean_file.parent / "runs/prove_tex").resolve()


def test_tex_static_claim_issues_flags_missing_dependencies_and_facts() -> None:
    claim = {
        "id": "C2",
        "goal": "goal",
        "depends_on": ["C1"],
        "assumptions": [],
        "required_facts": ["Euclid lemma"],
    }
    candidate = {
        "claim_id": "C2",
        "proof_tex": "This is TODO.",
        "depends_on_used": [],
        "assumptions_used": [],
        "cited_facts": [],
    }
    accepted = {"C1": {"assumptions_used": []}}
    issues = _tex_static_claim_issues(claim, candidate, accepted)
    joined = " | ".join(issues).lower()
    assert "missing dependency citation" in joined
    assert "required fact not cited" in joined
    assert "placeholder text found" in joined
