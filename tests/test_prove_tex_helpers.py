from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import time

from ulam.cli import (
    _build_tex_memory_context,
    _build_tex_replan_instruction,
    _evaluate_tex_claim_workers,
    _normalize_tex_proof,
    _resolve_prove_output_format,
    _resolve_tex_artifacts_root,
    _resolve_tex_claim_graph,
    _resolve_tex_concurrency,
    _resolve_tex_output_path,
    _resolve_tex_replan_passes,
    _resolve_tex_resume_snapshot,
    _sync_tex_memory_state,
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


def test_resolve_tex_concurrency_defaults_off() -> None:
    args = Namespace(tex_concurrency=None)
    cfg = {"prove": {"tex_concurrency": False}}
    assert _resolve_tex_concurrency(args, cfg) is False


def test_resolve_tex_concurrency_prefers_explicit() -> None:
    args = Namespace(tex_concurrency=True)
    cfg = {"prove": {"tex_concurrency": False}}
    assert _resolve_tex_concurrency(args, cfg) is True


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


def test_sync_tex_memory_state_creates_whiteboard_and_repo_items() -> None:
    run_state = {
        "status": "running",
        "current_pass": 1,
        "current_round": 2,
        "current_claim_index": 0,
        "plan": {
            "strategy": "contradiction",
            "outline": ["assume the negation", "derive divisibility contradiction"],
            "key_lemmas": ["Euclid lemma"],
            "checks": ["no hidden assumptions"],
        },
        "claims": [
            {
                "id": "C1",
                "goal": "derive a contradiction",
                "depends_on": [],
                "assumptions": [],
                "required_facts": [],
                "acceptance_checks": [],
            }
        ],
        "accepted_claims": {},
        "best_claim_candidates": {
            "C1": {
                "draft": {"proof_tex": "Assume the contrary and continue."},
                "judge": {"required_changes": ["justify the divisibility step"]},
                "verifier": {"critical_issues": ["one implication is implicit"]},
                "checker": {"issues": ["symbol mismatch"]},
                "static_issues": ["missing citation"],
                "score": 42.0,
            }
        },
        "claim_feedback": {"C1": "- justify the divisibility step"},
        "pass_history": [
            {
                "pass": 1,
                "strategy": "contradiction",
                "accepted_count": 0,
                "total_claims": 1,
                "unresolved_claims": ["C1"],
                "feedback": ["justify the divisibility step"],
                "status": "round_limit",
            }
        ],
        "best_pass": None,
        "final": {},
    }
    _sync_tex_memory_state(run_state, "demo_theorem", "For all n, P(n).")
    whiteboard = str(run_state.get("whiteboard", "") or "")
    repo_items = run_state.get("repo_items", {})
    assert "# TeX Whiteboard" in whiteboard
    assert "contradiction" in whiteboard
    assert isinstance(repo_items, dict)
    assert "theorem" in repo_items
    assert "pass_1_plan" in repo_items
    assert "pass_1_summary" in repo_items
    assert "claim_c1" in repo_items
    assert "open_claims" in repo_items


def test_build_tex_memory_context_includes_whiteboard_and_repo_materials() -> None:
    run_state = {
        "status": "running",
        "current_pass": 1,
        "current_round": 1,
        "current_claim_index": 0,
        "plan": {"strategy": "cases"},
        "claims": [{"id": "C1", "goal": "show P", "depends_on": [], "assumptions": [], "required_facts": [], "acceptance_checks": []}],
        "accepted_claims": {},
        "best_claim_candidates": {},
        "claim_feedback": {},
        "pass_history": [],
        "best_pass": None,
        "final": {},
    }
    _sync_tex_memory_state(run_state, "demo_theorem", "show P")
    text = _build_tex_memory_context(
        "base context",
        run_state,
        claim={"id": "C1", "depends_on": []},
        pass_idx=1,
        max_items=4,
    )
    assert "base context" in text
    assert "[persistent whiteboard]" in text
    assert "[repo index]" in text
    assert "[repo item: theorem]" in text


class _DummyTexLLM:
    def tex_claim_draft(
        self,
        *,
        theorem_name: str,
        theorem_statement: str,
        instruction: str,
        plan: dict,
        claim: dict,
        accepted_claims: list[dict],
        ledger: dict,
        prior_draft: str,
        prior_feedback: str,
        context: str,
        round_idx: int,
        worker_id: int,
    ) -> dict:
        if worker_id == 1:
            time.sleep(0.05)
        else:
            time.sleep(0.01)
        return {
            "claim_id": str(claim.get("id", "C1")),
            "proof_tex": f"worker {worker_id} proof",
            "assumptions_used": [],
            "depends_on_used": [],
            "cited_facts": [],
            "confidence": 80 + worker_id,
        }

    def tex_claim_judge(
        self,
        *,
        theorem_name: str,
        theorem_statement: str,
        instruction: str,
        plan: dict,
        claim: dict,
        candidate: dict,
        accepted_claims: list[dict],
        ledger: dict,
        context: str,
    ) -> dict:
        return {
            "verdict": "pass",
            "score": 90,
            "summary": "ok",
            "required_changes": [],
            "missing_assumptions": [],
            "citation_issues": [],
            "polished_proof_tex": "",
        }

    def tex_claim_verifier(
        self,
        *,
        theorem_name: str,
        theorem_statement: str,
        instruction: str,
        plan: dict,
        claim: dict,
        candidate: dict,
        accepted_claims: list[dict],
        ledger: dict,
        context: str,
    ) -> dict:
        return {
            "verdict": "pass",
            "score": 88,
            "summary": "ok",
            "critical_issues": [],
            "counterexample_attempt": "",
            "suggested_repairs": [],
        }

    def tex_claim_domain_check(
        self,
        *,
        theorem_name: str,
        theorem_statement: str,
        plan: dict,
        claim: dict,
        candidate: dict,
        context: str,
    ) -> dict:
        return {
            "status": "ok",
            "score": 87,
            "issues": [],
            "warnings": [],
            "sanity_checks": [],
        }


def test_evaluate_tex_claim_workers_returns_sorted_results() -> None:
    results = _evaluate_tex_claim_workers(
        llm=_DummyTexLLM(),  # type: ignore[arg-type]
        theorem="demo_theorem",
        theorem_statement="show P",
        instruction="",
        plan={"strategy": "direct"},
        claim={
            "id": "C1",
            "goal": "show P",
            "depends_on": [],
            "assumptions": [],
            "required_facts": [],
            "acceptance_checks": [],
        },
        accepted_claims_context=[],
        ledger={},
        prior_draft="",
        prior_feedback="",
        prompt_context="",
        round_idx=1,
        worker_drafts=2,
        concurrent=True,
        verifier_policy="final_only",
    )
    assert [row["worker"] for row in results] == [1, 2]
    assert all(row["status"] == "ok" for row in results)
    assert all(isinstance(row.get("candidate"), dict) for row in results)


def test_evaluate_tex_claim_workers_can_run_serially() -> None:
    results = _evaluate_tex_claim_workers(
        llm=_DummyTexLLM(),  # type: ignore[arg-type]
        theorem="demo_theorem",
        theorem_statement="show P",
        instruction="",
        plan={"strategy": "direct"},
        claim={
            "id": "C1",
            "goal": "show P",
            "depends_on": [],
            "assumptions": [],
            "required_facts": [],
            "acceptance_checks": [],
        },
        accepted_claims_context=[],
        ledger={},
        prior_draft="",
        prior_feedback="",
        prompt_context="",
        round_idx=1,
        worker_drafts=2,
        concurrent=False,
        verifier_policy="final_only",
    )
    assert [row["worker"] for row in results] == [1, 2]
