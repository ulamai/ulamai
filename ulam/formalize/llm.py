from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Iterable

from ..llm.runtime import run_with_runtime_controls


class FormalizationLLM:
    def __init__(self, provider: str, config: dict) -> None:
        self._provider = provider
        self._config = config

    def statement(self, text: str, context: str) -> str:
        prompt = _build_statement_prompt(text, context)
        return self._call(prompt)

    def draft(self, tex: str, context: str, hints: list[str]) -> str:
        prompt = _build_draft_prompt(tex, context, hints)
        return self._call(prompt)

    def repair(self, lean_code: str, error: str, context: str) -> str:
        prompt = _build_repair_prompt(lean_code, error, context)
        return self._call(prompt)

    def improve(self, lean_code: str, failures: list[str], context: str) -> str:
        prompt = _build_improve_prompt(lean_code, failures, context)
        return self._call(prompt)

    def prove(
        self,
        lean_code: str,
        name: str,
        instruction: str,
        tex_snippet: str,
        context: str,
        error: str | None = None,
        allow_helper_lemmas: bool = True,
        edit_scope: str = "full",
    ) -> str:
        prompt = _build_proof_prompt(
            lean_code,
            name,
            instruction,
            tex_snippet,
            context,
            error,
            allow_helper_lemmas=allow_helper_lemmas,
            edit_scope=edit_scope,
        )
        return self._call(prompt)

    def equivalence_check(self, tex_statement: str, lean_statement: str) -> dict:
        prompt = _build_equivalence_prompt(tex_statement, lean_statement)
        raw = self._call(prompt)
        return _parse_equivalence(raw)

    def repair_statement(self, lean_code: str, name: str, tex_statement: str, context: str) -> str:
        prompt = _build_statement_repair_prompt(lean_code, name, tex_statement, context)
        return self._call(prompt)

    def plan_lemmas(
        self,
        theorem_name: str,
        theorem_statement: str,
        original_statement: str,
        context: str,
    ) -> str:
        prompt = _build_lemma_plan_prompt(
            theorem_name, theorem_statement, original_statement, context
        )
        return self._call(prompt)

    def expand_lemmas(
        self,
        lemma_name: str,
        lemma_statement: str,
        last_goal: str,
        failures: list[str],
        successes: list[str],
        context: str,
    ) -> str:
        prompt = _build_lemma_expand_prompt(
            lemma_name, lemma_statement, last_goal, failures, successes, context
        )
        return self._call(prompt)

    def summarize_attempt(
        self,
        theorem_name: str,
        theorem_statement: str,
        last_goal: str,
        failures: list[str],
        successes: list[str],
        context: str,
    ) -> str:
        prompt = _build_summary_prompt(
            theorem_name, theorem_statement, last_goal, failures, successes, context
        )
        return self._call(prompt)

    def semantic_check(
        self,
        tex: str,
        lean_code: str,
        deterministic_issues: list[dict],
        context: str,
        stage: str = "end",
    ) -> dict:
        prompt = _build_semantic_check_prompt(
            tex, lean_code, deterministic_issues, context, stage=stage
        )
        raw = self._call(prompt)
        return _parse_semantic_check(raw)

    def semantic_repair(
        self,
        lean_code: str,
        tex: str,
        deterministic_issues: list[dict],
        audit: dict,
        context: str,
    ) -> str:
        prompt = _build_semantic_repair_prompt(
            lean_code,
            tex,
            deterministic_issues,
            audit,
            context,
        )
        return self._call(prompt)

    def tex_plan(
        self,
        theorem_name: str,
        theorem_statement: str,
        instruction: str,
        context: str,
    ) -> dict:
        prompt = _build_tex_plan_prompt(
            theorem_name=theorem_name,
            theorem_statement=theorem_statement,
            instruction=instruction,
            context=context,
        )
        raw = self._call(prompt)
        return _parse_tex_plan(raw)

    def tex_claim_draft(
        self,
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
        prompt = _build_tex_claim_worker_prompt(
            theorem_name=theorem_name,
            theorem_statement=theorem_statement,
            instruction=instruction,
            plan=plan,
            claim=claim,
            accepted_claims=accepted_claims,
            ledger=ledger,
            prior_draft=prior_draft,
            prior_feedback=prior_feedback,
            context=context,
            round_idx=round_idx,
            worker_id=worker_id,
        )
        raw = self._call(prompt)
        return _parse_tex_claim_draft(raw, fallback_claim_id=str(claim.get("id", "C1")))

    def tex_claim_judge(
        self,
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
        prompt = _build_tex_claim_judge_prompt(
            theorem_name=theorem_name,
            theorem_statement=theorem_statement,
            instruction=instruction,
            plan=plan,
            claim=claim,
            candidate=candidate,
            accepted_claims=accepted_claims,
            ledger=ledger,
            context=context,
        )
        raw = self._call(prompt)
        return _parse_tex_claim_judge(raw)

    def tex_claim_verifier(
        self,
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
        prompt = _build_tex_claim_verifier_prompt(
            theorem_name=theorem_name,
            theorem_statement=theorem_statement,
            instruction=instruction,
            plan=plan,
            claim=claim,
            candidate=candidate,
            accepted_claims=accepted_claims,
            ledger=ledger,
            context=context,
        )
        raw = self._call(prompt)
        return _parse_tex_claim_verifier(raw)

    def tex_claim_domain_check(
        self,
        theorem_name: str,
        theorem_statement: str,
        plan: dict,
        claim: dict,
        candidate: dict,
        context: str,
    ) -> dict:
        prompt = _build_tex_claim_checker_prompt(
            theorem_name=theorem_name,
            theorem_statement=theorem_statement,
            plan=plan,
            claim=claim,
            candidate=candidate,
            context=context,
        )
        raw = self._call(prompt)
        return _parse_tex_claim_checker(raw)

    def tex_compose(
        self,
        theorem_name: str,
        theorem_statement: str,
        instruction: str,
        plan: dict,
        accepted_claims: list[dict],
        ledger: dict,
        context: str,
    ) -> str:
        prompt = _build_tex_compose_prompt(
            theorem_name=theorem_name,
            theorem_statement=theorem_statement,
            instruction=instruction,
            plan=plan,
            accepted_claims=accepted_claims,
            ledger=ledger,
            context=context,
        )
        raw = self._call(prompt)
        return _extract_tex_document(raw)

    def tex_worker_draft(
        self,
        theorem_name: str,
        theorem_statement: str,
        instruction: str,
        plan: dict,
        prior_draft: str,
        judge_feedback: str,
        context: str,
        worker_id: int,
    ) -> str:
        prompt = _build_tex_worker_prompt(
            theorem_name=theorem_name,
            theorem_statement=theorem_statement,
            instruction=instruction,
            plan=plan,
            prior_draft=prior_draft,
            judge_feedback=judge_feedback,
            context=context,
            worker_id=worker_id,
        )
        raw = self._call(prompt)
        return _extract_tex_document(raw)

    def tex_judge(
        self,
        theorem_name: str,
        theorem_statement: str,
        instruction: str,
        plan: dict,
        draft_tex: str,
        context: str,
    ) -> dict:
        prompt = _build_tex_judge_prompt(
            theorem_name=theorem_name,
            theorem_statement=theorem_statement,
            instruction=instruction,
            plan=plan,
            draft_tex=draft_tex,
            context=context,
        )
        raw = self._call(prompt)
        return _parse_tex_judge(raw)

    def _call(self, prompt: str) -> str:
        if self._provider == "openai":
            return _call_openai(self._config, prompt)
        if self._provider == "ollama":
            return _call_ollama(self._config, prompt)
        if self._provider == "anthropic":
            return _call_anthropic(self._config, prompt)
        if self._provider == "codex_cli":
            return _call_codex_cli(self._config, prompt)
        if self._provider == "claude_cli":
            return _call_claude_cli(self._config, prompt)
        if self._provider == "gemini":
            return _call_gemini(self._config, prompt)
        if self._provider == "gemini_cli":
            return _call_gemini_cli(self._config, prompt)
        return ""


def _build_draft_prompt(tex: str, context: str, hints: list[str]) -> str:
    hint_block = "\n\n".join(hints)
    prompt = (
        "You are a Lean 4 formalization assistant. Output Lean code only.\n"
        "Task: formalize the following LaTeX into Lean 4.\n"
        "- Add missing definitions.\n"
        "- Use `sorry` for incomplete proofs.\n"
        "- Keep theorem names stable and human-readable.\n\n"
        "LaTeX:\n"
        f"{tex}\n\n"
    )
    if hint_block:
        prompt += "Hints from theorem/proof text:\n" + hint_block + "\n\n"
    if context:
        prompt += "Context files:\n" + context + "\n\n"
    prompt += "Return ONLY Lean code."
    return prompt


def _build_statement_prompt(text: str, context: str) -> str:
    prompt = (
        "You are a Lean 4 formalization assistant.\n"
        "Convert the following informal statement into a Lean 4 proposition.\n"
        "- Return ONLY the Lean proposition (type), without `theorem`/`lemma` keywords.\n"
        "- Do not include any proofs or code fences.\n"
        "- If the statement is already Lean, return it unchanged.\n\n"
        "Statement:\n"
        f"{text}\n\n"
    )
    if context:
        prompt += "Context files:\n" + context + "\n\n"
    prompt += "Return ONLY the Lean proposition."
    return prompt


def _build_repair_prompt(lean_code: str, error: str, context: str) -> str:
    prompt = (
        "You are a Lean 4 formalization assistant. Output Lean code only.\n"
        "The following Lean file fails to typecheck. Fix it.\n"
        "- Preserve existing structure where possible.\n"
        "- If needed, add imports or small helper lemmas.\n\n"
        "Lean file:\n"
        f"{lean_code}\n\n"
        "Lean error:\n"
        f"{error}\n\n"
    )
    if context:
        prompt += "Context files:\n" + context + "\n\n"
    prompt += "Return ONLY corrected Lean code."
    return prompt


def _build_improve_prompt(lean_code: str, failures: list[str], context: str) -> str:
    failure_block = "\n".join(f"- {item}" for item in failures)
    prompt = (
        "You are a Lean 4 formalization assistant. Output Lean code only.\n"
        "Some theorems remain unsolved. Improve the Lean file by:\n"
        "- Adding missing intermediate lemmas.\n"
        "- Tweaking statements to match intended informal meaning.\n"
        "- Preserving already solved proofs.\n\n"
        "Lean file:\n"
        f"{lean_code}\n\n"
        "Unsolved items:\n"
        f"{failure_block}\n\n"
    )
    if context:
        prompt += "Context files:\n" + context + "\n\n"
    prompt += "Return ONLY updated Lean code."
    return prompt


def _build_proof_prompt(
    lean_code: str,
    name: str,
    instruction: str,
    tex_snippet: str,
    context: str,
    error: str | None,
    allow_helper_lemmas: bool = True,
    edit_scope: str = "full",
) -> str:
    prompt = (
        "You are a Lean 4 assistant. Output Lean code only.\n"
        f"Task: complete the proof of `{name}` in the Lean file below.\n"
        "- Preserve other declarations and imports.\n"
        "- Do NOT use `sorry` or `admit` in the proof of the target.\n\n"
        "Lean file:\n"
        f"{lean_code}\n\n"
    )
    if allow_helper_lemmas:
        prompt += "- You may add helper lemmas before the target declaration if needed.\n"
    else:
        prompt += (
            "- Do NOT add/remove/rename declarations.\n"
            f"- Only edit the proof body of `{name}`.\n"
        )
    if edit_scope == "errors_only":
        prompt += (
            "- Edit scope is errors-only: declarations without `sorry`/`admit` must remain unchanged.\n"
        )
    if instruction:
        prompt += "User guidance:\n" + instruction.strip() + "\n\n"
    if tex_snippet:
        snippet = tex_snippet.strip()
        if len(snippet) > 1600:
            snippet = snippet[:1600] + "\n-- (truncated)"
        prompt += f"Informal proof snippet for {name}:\n{snippet}\n\n"
    if context:
        prompt += "Context files:\n" + context + "\n\n"
    if error:
        trimmed = error.strip()
        if len(trimmed) > 2000:
            trimmed = trimmed[:2000] + "\n-- (truncated)"
        prompt += "Lean error:\n" + trimmed + "\n\n"
    prompt += "Return ONLY the updated Lean file."
    return prompt


def _build_equivalence_prompt(tex_statement: str, lean_statement: str) -> str:
    prompt = (
        "You compare informal math statements to Lean statements.\n"
        "Return a JSON object with fields: match (yes|no|unknown), reason.\n"
        "Only return JSON.\n\n"
        "Informal statement:\n"
        f"{tex_statement}\n\n"
        "Lean statement:\n"
        f"{lean_statement}\n"
    )
    return prompt


def _build_statement_repair_prompt(lean_code: str, name: str, tex_statement: str, context: str) -> str:
    prompt = (
        "You are a Lean 4 formalization assistant. Output Lean code only.\n"
        "Update the statement of the declaration below to match the informal statement.\n"
        "- Edit ONLY the named declaration.\n"
        "- Do NOT change any unrelated declaration (statement or proof).\n"
        "- If the target proof no longer matches, `by sorry` is allowed only for the target declaration.\n"
        "- Keep theorem/lemma names unchanged.\n\n"
        f"Declaration name: {name}\n\n"
        "Informal statement:\n"
        f"{tex_statement}\n\n"
        "Current Lean file:\n"
        f"{lean_code}\n\n"
    )
    if context:
        prompt += "Context files:\n" + context + "\n\n"
    prompt += "Return ONLY the updated Lean file."
    return prompt


def _build_lemma_plan_prompt(
    theorem_name: str,
    theorem_statement: str,
    original_statement: str,
    context: str,
) -> str:
    prompt = (
        "You are a Lean 4 assistant. Produce a lemma-first proof plan as Lean code.\n"
        "- Output Lean code only.\n"
        "- Introduce helper definitions/lemmas if needed.\n"
        "- Use `by sorry` for proofs.\n"
        f"- The final theorem MUST be named `{theorem_name}`.\n"
        f"- The final theorem statement MUST be: {theorem_statement}\n\n"
        "Original informal statement:\n"
        f"{original_statement}\n\n"
    )
    if context:
        prompt += "Context files:\n" + context + "\n\n"
    prompt += "Return ONLY Lean code."
    return prompt


def _build_lemma_expand_prompt(
    lemma_name: str,
    lemma_statement: str,
    last_goal: str,
    failures: list[str],
    successes: list[str],
    context: str,
) -> str:
    failure_block = "\n".join(f"- {item}" for item in failures[:12])
    success_block = "\n".join(f"- {item}" for item in successes[:12])
    prompt = (
        "You are a Lean 4 assistant. Suggest new helper lemmas/defs to prove a stuck lemma.\n"
        "- Output Lean code only.\n"
        "- Do NOT restate the target lemma.\n"
        "- Use `by sorry` for proofs.\n\n"
        f"Stuck lemma name: {lemma_name}\n"
        f"Stuck lemma statement:\n{lemma_statement}\n\n"
        f"Last goal:\n{last_goal}\n\n"
        "Recent failed tactics/errors:\n"
        f"{failure_block}\n\n"
        "Recent successful tactics:\n"
        f"{success_block}\n\n"
    )
    if context:
        prompt += "Context files:\n" + context + "\n\n"
    prompt += "Return ONLY Lean code for new helper declarations."
    return prompt


def _build_summary_prompt(
    theorem_name: str,
    theorem_statement: str,
    last_goal: str,
    failures: list[str],
    successes: list[str],
    context: str,
) -> str:
    failure_block = "\n".join(f"- {item}" for item in failures[:12])
    success_block = "\n".join(f"- {item}" for item in successes[:12])
    prompt = (
        "You are a Lean 4 assistant. Summarize a failed proof attempt and propose next steps.\n"
        "- Keep it short and actionable.\n"
        "- Prefer Lean tactics/lemmas likely to work.\n"
        "- Do not output code fences.\n\n"
        f"Theorem name: {theorem_name}\n"
        f"Theorem statement:\n{theorem_statement}\n\n"
        f"Last goal:\n{last_goal}\n\n"
        "Recent failed tactics/errors:\n"
        f"{failure_block}\n\n"
        "Recent successful tactics:\n"
        f"{success_block}\n\n"
    )
    if context:
        prompt += "Context files:\n" + context + "\n\n"
    prompt += "Return a brief summary and 3-7 next-step suggestions."
    return prompt


def _truncate_block(text: str, max_chars: int) -> str:
    content = (text or "").strip()
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + "\n-- (truncated)"


def _build_semantic_check_prompt(
    tex: str,
    lean_code: str,
    deterministic_issues: list[dict],
    context: str,
    stage: str = "end",
) -> str:
    issues = deterministic_issues[:32]
    issue_json = json.dumps(issues, ensure_ascii=True, indent=2)
    prompt = (
        "You are auditing semantic integrity of a Lean formalization.\n"
        "Goal: detect whether the Lean file preserves the intended theorem(s) from LaTeX,\n"
        "or if it was made trivially true by weakening definitions/statements.\n"
        "Focus on cheats such as:\n"
        "- introducing `axiom`/`constant` (including `private axiom` / `private constant`) to bypass proofs,\n"
        "- redefining hard concepts to `True`/`False` or constants,\n"
        "- replacing counting/probability objects by trivial identities/constants,\n"
        "- changing theorem statements away from intended meaning.\n\n"
        f"Audit stage: {stage}\n\n"
        "Deterministic findings (pre-check):\n"
        f"{issue_json}\n\n"
        "Return ONLY JSON with schema:\n"
        "{\n"
        '  "verdict": "pass|fail|unknown",\n'
        '  "summary": "short text",\n'
        '  "issues": [{"kind":"...", "severity":"low|medium|high", "evidence":"..."}],\n'
        '  "should_repair": true|false\n'
        "}\n\n"
        "LaTeX source:\n"
        f"{_truncate_block(tex, 18000)}\n\n"
        "Lean file:\n"
        f"{_truncate_block(lean_code, 26000)}\n\n"
    )
    if context:
        prompt += "Context files:\n" + _truncate_block(context, 6000) + "\n\n"
    return prompt


def _build_semantic_repair_prompt(
    lean_code: str,
    tex: str,
    deterministic_issues: list[dict],
    audit: dict,
    context: str,
) -> str:
    audit_json = json.dumps(audit, ensure_ascii=True, indent=2)
    deterministic_json = json.dumps(deterministic_issues[:32], ensure_ascii=True, indent=2)
    prompt = (
        "You are a Lean 4 formalization assistant. Output Lean code only.\n"
        "Repair semantic integrity issues in this Lean file.\n"
        "Constraints:\n"
        "- Preserve intended meaning of the original LaTeX theorem(s).\n"
        "- Do NOT trivialize definitions (e.g., Prop := True/False, constant stubs).\n"
        "- Keep theorem names stable.\n"
        "- Avoid introducing `axiom` or `constant` unless mathematically unavoidable.\n"
        "- Keep already-correct parts when possible.\n\n"
        "Semantic audit result:\n"
        f"{audit_json}\n\n"
        "Deterministic findings:\n"
        f"{deterministic_json}\n\n"
        "LaTeX source:\n"
        f"{_truncate_block(tex, 14000)}\n\n"
        "Current Lean file:\n"
        f"{_truncate_block(lean_code, 26000)}\n\n"
    )
    if context:
        prompt += "Context files:\n" + _truncate_block(context, 6000) + "\n\n"
    prompt += "Return ONLY the corrected Lean file."
    return prompt


def _build_tex_plan_prompt(
    theorem_name: str,
    theorem_statement: str,
    instruction: str,
    context: str,
) -> str:
    prompt = (
        "You are a mathematical proof planner.\n"
        "Create a compact proof plan for writing an informal LaTeX proof.\n"
        "Return ONLY JSON with schema:\n"
        "{\n"
        '  "strategy": "short summary",\n'
        '  "outline": ["step 1", "step 2", "..."],\n'
        '  "key_lemmas": ["lemma/fact 1", "..."],\n'
        '  "checks": ["what must be validated", "..."],\n'
        '  "claims": [\n'
        "    {\n"
        '      "id": "C1",\n'
        '      "goal": "subclaim statement",\n'
        '      "depends_on": ["Ck"],\n'
        '      "assumptions": ["explicit assumption 1"],\n'
        '      "required_facts": ["fact/lemma to cite"],\n'
        '      "acceptance_checks": ["what makes this subclaim complete"]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Use at most 8 claims. Keep dependencies acyclic.\n"
        "Every claim goal must be precise enough for downstream verification.\n\n"
        f"Theorem name: {theorem_name}\n"
        f"Theorem statement:\n{theorem_statement}\n\n"
    )
    if instruction.strip():
        prompt += "User guidance:\n" + instruction.strip() + "\n\n"
    if context:
        prompt += "Context files:\n" + _truncate_block(context, 8000) + "\n\n"
    return prompt


def _build_tex_claim_worker_prompt(
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
) -> str:
    plan_json = json.dumps(plan, ensure_ascii=True, indent=2)
    claim_json = json.dumps(claim, ensure_ascii=True, indent=2)
    accepted_json = json.dumps(accepted_claims, ensure_ascii=True, indent=2)
    ledger_json = json.dumps(ledger, ensure_ascii=True, indent=2)
    prompt = (
        "You are writing one subclaim proof for an informal LaTeX theorem.\n"
        "Return ONLY JSON with schema:\n"
        "{\n"
        '  "claim_id": "C1",\n'
        '  "proof_tex": "proof text for this claim only (plain LaTeX, no markdown fences)",\n'
        '  "assumptions_used": ["explicit assumptions used in this subproof"],\n'
        '  "depends_on_used": ["claim ids used from accepted claims"],\n'
        '  "cited_facts": ["facts/lemmas cited in this subproof"],\n'
        '  "confidence": 0,\n'
        '  "notes": "short notes"\n'
        "}\n\n"
        "Constraints:\n"
        "- Focus only on the provided claim goal.\n"
        "- Be explicit about implications and avoid hidden assumptions.\n"
        "- If a dependency claim is used, list it in depends_on_used.\n"
        "- If a required fact is used, list it in cited_facts.\n"
        "- Do not include Lean code.\n\n"
        f"Round: {round_idx}\n"
        f"Worker id: {worker_id}\n"
        f"Theorem name: {theorem_name}\n"
        f"Theorem statement:\n{theorem_statement}\n\n"
        "Global plan:\n"
        f"{plan_json}\n\n"
        "Target claim:\n"
        f"{claim_json}\n\n"
        "Accepted claims so far:\n"
        f"{_truncate_block(accepted_json, 12000)}\n\n"
        "Assumption/dependency ledger:\n"
        f"{_truncate_block(ledger_json, 12000)}\n\n"
    )
    if instruction.strip():
        prompt += "User guidance:\n" + instruction.strip() + "\n\n"
    if prior_draft.strip():
        prompt += "Previous draft for this claim:\n" + _truncate_block(prior_draft, 12000) + "\n\n"
    if prior_feedback.strip():
        prompt += "Feedback to address:\n" + _truncate_block(prior_feedback, 7000) + "\n\n"
    if context:
        prompt += "Context files:\n" + _truncate_block(context, 8000) + "\n\n"
    return prompt


def _build_tex_claim_judge_prompt(
    theorem_name: str,
    theorem_statement: str,
    instruction: str,
    plan: dict,
    claim: dict,
    candidate: dict,
    accepted_claims: list[dict],
    ledger: dict,
    context: str,
) -> str:
    plan_json = json.dumps(plan, ensure_ascii=True, indent=2)
    claim_json = json.dumps(claim, ensure_ascii=True, indent=2)
    candidate_json = json.dumps(candidate, ensure_ascii=True, indent=2)
    accepted_json = json.dumps(accepted_claims, ensure_ascii=True, indent=2)
    ledger_json = json.dumps(ledger, ensure_ascii=True, indent=2)
    prompt = (
        "You are a strict mathematical judge for one subclaim.\n"
        "Evaluate correctness and formalization-readiness.\n"
        "Return ONLY JSON with schema:\n"
        "{\n"
        '  "verdict": "pass|revise|fail",\n'
        '  "score": 0,\n'
        '  "summary": "short diagnosis",\n'
        '  "required_changes": ["must-fix issue"],\n'
        '  "missing_assumptions": ["assumption that was used but not stated"],\n'
        '  "citation_issues": ["missing or incorrect citation"],\n'
        '  "polished_proof_tex": "optional corrected claim proof text or empty"\n'
        "}\n\n"
        "Rules:\n"
        "- `pass` only if this claim is mathematically coherent and dependency-safe.\n"
        "- Reject hidden assumptions and circular dependencies.\n"
        "- Keep feedback concrete.\n\n"
        f"Theorem name: {theorem_name}\n"
        f"Theorem statement:\n{theorem_statement}\n\n"
        "Global plan:\n"
        f"{plan_json}\n\n"
        "Target claim:\n"
        f"{claim_json}\n\n"
        "Candidate claim proof:\n"
        f"{_truncate_block(candidate_json, 18000)}\n\n"
        "Accepted claims:\n"
        f"{_truncate_block(accepted_json, 12000)}\n\n"
        "Assumption/dependency ledger:\n"
        f"{_truncate_block(ledger_json, 12000)}\n\n"
    )
    if instruction.strip():
        prompt += "User guidance:\n" + instruction.strip() + "\n\n"
    if context:
        prompt += "Context files:\n" + _truncate_block(context, 8000) + "\n\n"
    return prompt


def _build_tex_claim_verifier_prompt(
    theorem_name: str,
    theorem_statement: str,
    instruction: str,
    plan: dict,
    claim: dict,
    candidate: dict,
    accepted_claims: list[dict],
    ledger: dict,
    context: str,
) -> str:
    plan_json = json.dumps(plan, ensure_ascii=True, indent=2)
    claim_json = json.dumps(claim, ensure_ascii=True, indent=2)
    candidate_json = json.dumps(candidate, ensure_ascii=True, indent=2)
    accepted_json = json.dumps(accepted_claims, ensure_ascii=True, indent=2)
    ledger_json = json.dumps(ledger, ensure_ascii=True, indent=2)
    prompt = (
        "You are an adversarial mathematical verifier.\n"
        "Try to break the candidate claim proof.\n"
        "Return ONLY JSON with schema:\n"
        "{\n"
        '  "verdict": "pass|revise|fail",\n'
        '  "score": 0,\n'
        '  "summary": "short verifier summary",\n'
        '  "critical_issues": ["logical gap or contradiction"],\n'
        '  "counterexample_attempt": "short attempt or empty",\n'
        '  "suggested_repairs": ["repair direction"]\n'
        "}\n\n"
        "Rules:\n"
        "- Assume the proof is wrong until convinced otherwise.\n"
        "- Search for missing quantifier conditions, hidden case splits, circularity.\n"
        "- `pass` only when you cannot identify a material flaw.\n\n"
        f"Theorem name: {theorem_name}\n"
        f"Theorem statement:\n{theorem_statement}\n\n"
        "Global plan:\n"
        f"{plan_json}\n\n"
        "Target claim:\n"
        f"{claim_json}\n\n"
        "Candidate claim proof:\n"
        f"{_truncate_block(candidate_json, 18000)}\n\n"
        "Accepted claims:\n"
        f"{_truncate_block(accepted_json, 12000)}\n\n"
        "Assumption/dependency ledger:\n"
        f"{_truncate_block(ledger_json, 12000)}\n\n"
    )
    if instruction.strip():
        prompt += "User guidance:\n" + instruction.strip() + "\n\n"
    if context:
        prompt += "Context files:\n" + _truncate_block(context, 8000) + "\n\n"
    return prompt


def _build_tex_claim_checker_prompt(
    theorem_name: str,
    theorem_statement: str,
    plan: dict,
    claim: dict,
    candidate: dict,
    context: str,
) -> str:
    plan_json = json.dumps(plan, ensure_ascii=True, indent=2)
    claim_json = json.dumps(claim, ensure_ascii=True, indent=2)
    candidate_json = json.dumps(candidate, ensure_ascii=True, indent=2)
    prompt = (
        "You are a symbolic/domain consistency checker for informal math proofs.\n"
        "Do not rewrite the proof. Identify objective consistency issues.\n"
        "Return ONLY JSON with schema:\n"
        "{\n"
        '  "status": "ok|issues",\n'
        '  "score": 0,\n'
        '  "issues": ["concrete consistency issue"],\n'
        '  "warnings": ["non-fatal concern"],\n'
        '  "sanity_checks": ["checks attempted"]\n'
        "}\n\n"
        "Focus on:\n"
        "- contradiction between stated assumptions and conclusions,\n"
        "- invalid algebraic transformations,\n"
        "- unjustified leaps in equalities/inequalities,\n"
        "- quantifier/domain mismatch.\n\n"
        f"Theorem name: {theorem_name}\n"
        f"Theorem statement:\n{theorem_statement}\n\n"
        "Global plan:\n"
        f"{plan_json}\n\n"
        "Target claim:\n"
        f"{claim_json}\n\n"
        "Candidate claim proof:\n"
        f"{_truncate_block(candidate_json, 18000)}\n\n"
    )
    if context:
        prompt += "Context files:\n" + _truncate_block(context, 8000) + "\n\n"
    return prompt


def _build_tex_compose_prompt(
    theorem_name: str,
    theorem_statement: str,
    instruction: str,
    plan: dict,
    accepted_claims: list[dict],
    ledger: dict,
    context: str,
) -> str:
    plan_json = json.dumps(plan, ensure_ascii=True, indent=2)
    claims_json = json.dumps(accepted_claims, ensure_ascii=True, indent=2)
    ledger_json = json.dumps(ledger, ensure_ascii=True, indent=2)
    prompt = (
        "You are composing a final informal LaTeX theorem proof from verified subclaims.\n"
        "Return ONLY LaTeX text (no markdown fences).\n"
        "Output shape:\n"
        "\\begin{theorem}[<name>] ... \\end{theorem}\n"
        "\\begin{proof} ... \\end{proof}\n\n"
        "Rules:\n"
        "- Use accepted subclaims consistently and respect dependency order.\n"
        "- Keep assumptions explicit.\n"
        "- Do not introduce new major lemmas that are not in the accepted claim set.\n"
        "- Ensure the final proof is coherent as a single narrative.\n\n"
        f"Theorem name: {theorem_name}\n"
        f"Theorem statement:\n{theorem_statement}\n\n"
        "Global plan:\n"
        f"{plan_json}\n\n"
        "Accepted claim proofs:\n"
        f"{_truncate_block(claims_json, 22000)}\n\n"
        "Assumption/dependency ledger:\n"
        f"{_truncate_block(ledger_json, 12000)}\n\n"
    )
    if instruction.strip():
        prompt += "User guidance:\n" + instruction.strip() + "\n\n"
    if context:
        prompt += "Context files:\n" + _truncate_block(context, 8000) + "\n\n"
    prompt += "Return ONLY LaTeX."
    return prompt


def _build_tex_worker_prompt(
    theorem_name: str,
    theorem_statement: str,
    instruction: str,
    plan: dict,
    prior_draft: str,
    judge_feedback: str,
    context: str,
    worker_id: int,
) -> str:
    plan_json = json.dumps(plan, ensure_ascii=True, indent=2)
    prompt = (
        "You are writing an informal mathematical proof in LaTeX.\n"
        "Return ONLY LaTeX text (no markdown fences).\n"
        "Output a theorem/proof block using this shape:\n"
        "\\begin{theorem}[<name>] ... \\end{theorem}\n"
        "\\begin{proof} ... \\end{proof}\n\n"
        "Constraints:\n"
        "- Be mathematically coherent and explicit about key implications.\n"
        "- Avoid hand-wavy jumps.\n"
        "- Keep notation consistent.\n"
        "- Do not include Lean code.\n\n"
        f"Worker id: {worker_id}\n"
        f"Theorem name: {theorem_name}\n"
        f"Theorem statement:\n{theorem_statement}\n\n"
        "Proof plan:\n"
        f"{plan_json}\n\n"
    )
    if instruction.strip():
        prompt += "User guidance:\n" + instruction.strip() + "\n\n"
    if prior_draft.strip():
        prompt += "Previous best draft:\n" + _truncate_block(prior_draft, 14000) + "\n\n"
    if judge_feedback.strip():
        prompt += "Judge feedback to address:\n" + _truncate_block(judge_feedback, 6000) + "\n\n"
    if context:
        prompt += "Context files:\n" + _truncate_block(context, 8000) + "\n\n"
    prompt += "Return ONLY LaTeX."
    return prompt


def _build_tex_judge_prompt(
    theorem_name: str,
    theorem_statement: str,
    instruction: str,
    plan: dict,
    draft_tex: str,
    context: str,
) -> str:
    plan_json = json.dumps(plan, ensure_ascii=True, indent=2)
    prompt = (
        "You are a strict mathematical proof judge.\n"
        "Evaluate the candidate informal LaTeX proof for correctness and clarity.\n"
        "Return ONLY JSON with schema:\n"
        "{\n"
        '  "verdict": "pass|revise|fail",\n'
        '  "score": 0,\n'
        '  "summary": "short diagnosis",\n'
        '  "required_changes": ["must-fix issue", "..."],\n'
        '  "style_notes": ["optional style note", "..."],\n'
        '  "polished_tex": "optional improved full LaTeX proof text or empty"\n'
        "}\n\n"
        "Rules:\n"
        "- `pass` only if the proof is mathematically valid and complete enough to formalize.\n"
        "- `revise` when there are fixable logical gaps or unclear steps.\n"
        "- `fail` when the approach is fundamentally incorrect.\n\n"
        f"Theorem name: {theorem_name}\n"
        f"Theorem statement:\n{theorem_statement}\n\n"
        "Proof plan:\n"
        f"{plan_json}\n\n"
    )
    if instruction.strip():
        prompt += "User guidance:\n" + instruction.strip() + "\n\n"
    prompt += "Candidate LaTeX proof:\n" + _truncate_block(draft_tex, 18000) + "\n\n"
    if context:
        prompt += "Context files:\n" + _truncate_block(context, 8000) + "\n\n"
    return prompt


def _call_openai(config: dict, prompt: str) -> str:
    openai = config.get("openai", {})
    api_key = openai.get("api_key", "") or os.environ.get("ULAM_OPENAI_API_KEY", "")
    if not api_key:
        return ""
    base_url = openai.get("base_url", "https://api.openai.com").rstrip("/")
    model = openai.get("model", "gpt-4.1")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a Lean 4 formalization assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 3000,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    timeout_s, heartbeat_s = _llm_runtime_settings(config)
    raw = run_with_runtime_controls(
        lambda: _urlopen_read(req, timeout_s),
        timeout_s=timeout_s,
        heartbeat_s=heartbeat_s,
    )
    return _extract_openai(raw)


def _call_ollama(config: dict, prompt: str) -> str:
    ollama = config.get("ollama", {})
    base_url = ollama.get("base_url", "http://localhost:11434").rstrip("/")
    model = ollama.get("model", "llama3.1")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a Lean 4 formalization assistant."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    endpoints: list[str] = []
    if base_url.endswith("/api"):
        base_url = base_url[: -len("/api")]
    if base_url.endswith("/v1"):
        base_url = base_url[: -len("/v1")]
        endpoints.append(f"{base_url}/v1/chat/completions")
    endpoints.append(f"{base_url}/api/chat")
    endpoints.append(f"{base_url}/v1/chat/completions")
    seen = set()
    endpoints = [url for url in endpoints if not (url in seen or seen.add(url))]
    last_error: Exception | None = None
    timeout_s, heartbeat_s = _llm_runtime_settings(config)
    for url in endpoints:
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            raw = run_with_runtime_controls(
                lambda req=req: _urlopen_read(req, timeout_s),
                timeout_s=timeout_s,
                heartbeat_s=heartbeat_s,
            )
            return _extract_ollama(raw)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code in (404, 405):
                continue
            raise
    if last_error:
        raise last_error
    return ""


def _call_anthropic(config: dict, prompt: str) -> str:
    anthropic = config.get("anthropic", {})
    api_key = anthropic.get("api_key", "") or anthropic.get("setup_token", "")
    api_key = api_key or os.environ.get("ULAM_ANTHROPIC_API_KEY", "")
    if not api_key:
        return ""
    base_url = anthropic.get("base_url", "https://api.anthropic.com").rstrip("/")
    model = anthropic.get("model", "claude-3-5-sonnet-20240620")
    payload = {
        "model": model,
        "max_tokens": 3000,
        "temperature": 0.2,
        "messages": [
            {"role": "user", "content": prompt},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/messages",
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    timeout_s, heartbeat_s = _llm_runtime_settings(config)
    raw = run_with_runtime_controls(
        lambda: _urlopen_read(req, timeout_s),
        timeout_s=timeout_s,
        heartbeat_s=heartbeat_s,
    )
    return _extract_anthropic(raw)


def _call_gemini(config: dict, prompt: str) -> str:
    gemini = config.get("gemini", {})
    api_key = (
        gemini.get("api_key", "")
        or os.environ.get("ULAM_GEMINI_API_KEY", "")
        or os.environ.get("GEMINI_API_KEY", "")
    )
    if not api_key:
        return ""
    base_url = gemini.get("base_url", "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
    model = gemini.get("model", "gemini-3.1-pro-preview")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a Lean 4 formalization assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 3000,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _gemini_chat_endpoint(base_url),
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    timeout_s, heartbeat_s = _llm_runtime_settings(config)
    raw = run_with_runtime_controls(
        lambda: _urlopen_read(req, timeout_s),
        timeout_s=timeout_s,
        heartbeat_s=heartbeat_s,
    )
    return _extract_openai(raw)


def _llm_runtime_settings(config: dict) -> tuple[float | None, float | None]:
    llm_cfg = config.get("llm", {})
    if not isinstance(llm_cfg, dict):
        return None, 60.0
    try:
        timeout = float(llm_cfg.get("timeout_s", 0))
    except Exception:
        timeout = 0.0
    try:
        heartbeat = float(llm_cfg.get("heartbeat_s", 60))
    except Exception:
        heartbeat = 60.0
    timeout_s: float | None = timeout if timeout > 0 else None
    heartbeat_s: float | None = heartbeat if heartbeat > 0 else None
    return timeout_s, heartbeat_s


def _urlopen_read(req: urllib.request.Request, timeout_s: float | None) -> str:
    timeout = timeout_s if timeout_s and timeout_s > 0 else None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _extract_openai(raw: str) -> str:
    data = json.loads(raw)
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = choices[0]
    if "message" in msg and "content" in msg["message"]:
        return msg["message"]["content"]
    if "text" in msg:
        return msg["text"]
    return ""


def _extract_ollama(raw: str) -> str:
    data = json.loads(raw)
    message = data.get("message")
    if isinstance(message, dict) and "content" in message:
        return message["content"]
    choices = data.get("choices") or []
    if choices:
        choice = choices[0]
        if "message" in choice and "content" in choice["message"]:
            return choice["message"]["content"]
        if "text" in choice:
            return choice["text"]
    if "response" in data and isinstance(data["response"], str):
        return data["response"]
    return ""


def _extract_anthropic(raw: str) -> str:
    data = json.loads(raw)
    content = data.get("content")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts)
    return data.get("text", "") if isinstance(data.get("text", ""), str) else ""


def _call_codex_cli(config: dict, prompt: str) -> str:
    from ..llm.cli_utils import codex_exec

    openai = config.get("openai", {})
    llm_cfg = config.get("llm", {})
    model = openai.get("codex_model") or openai.get("model") or None
    system = "You are a Lean 4 formalization assistant. Output Lean code only."
    timeout_s = float(llm_cfg.get("timeout_s", 0))
    heartbeat_s = float(llm_cfg.get("heartbeat_s", 60))
    return codex_exec(
        system,
        prompt,
        model=model,
        timeout_s=timeout_s,
        heartbeat_s=heartbeat_s,
    )


def _call_claude_cli(config: dict, prompt: str) -> str:
    from ..llm.cli_utils import claude_print

    anthropic = config.get("anthropic", {})
    llm_cfg = config.get("llm", {})
    model = anthropic.get("claude_model") or anthropic.get("model") or None
    system = "You are a Lean 4 formalization assistant. Output Lean code only."
    timeout_s = float(llm_cfg.get("timeout_s", 0))
    heartbeat_s = float(llm_cfg.get("heartbeat_s", 60))
    return claude_print(
        system,
        prompt,
        model=model,
        timeout_s=timeout_s,
        heartbeat_s=heartbeat_s,
    )


def _call_gemini_cli(config: dict, prompt: str) -> str:
    from ..llm.cli_utils import gemini_exec

    gemini = config.get("gemini", {})
    llm_cfg = config.get("llm", {})
    model = gemini.get("cli_model") or gemini.get("model") or None
    system = "You are a Lean 4 formalization assistant. Output Lean code only."
    timeout_s = float(llm_cfg.get("timeout_s", 0))
    heartbeat_s = float(llm_cfg.get("heartbeat_s", 60))
    return gemini_exec(
        system,
        prompt,
        model=model,
        timeout_s=timeout_s,
        heartbeat_s=heartbeat_s,
    )


def _gemini_chat_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/openai") or base.endswith("/openai/v1") or base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _parse_equivalence(raw: str) -> dict:
    if not raw.strip():
        return {"match": "unknown", "reason": "empty response"}
    text = raw.strip()
    try:
        return json.loads(_extract_json(text))
    except Exception:
        match = "unknown"
        reason = "unparsed response"
        lowered = text.lower()
        if "match" in lowered and "yes" in lowered:
            match = "yes"
        elif "match" in lowered and "no" in lowered:
            match = "no"
        return {"match": match, "reason": reason, "raw": text[:400]}


def _parse_semantic_check(raw: str) -> dict:
    if not raw.strip():
        return {
            "verdict": "unknown",
            "summary": "empty response",
            "issues": [],
            "should_repair": True,
        }
    text = raw.strip()
    try:
        payload = json.loads(_extract_json(text))
    except Exception:
        lowered = text.lower()
        verdict = "unknown"
        if "verdict" in lowered and "pass" in lowered:
            verdict = "pass"
        elif "verdict" in lowered and "fail" in lowered:
            verdict = "fail"
        return {
            "verdict": verdict,
            "summary": "unparsed response",
            "issues": [],
            "should_repair": verdict != "pass",
            "raw": text[:600],
        }
    if not isinstance(payload, dict):
        return {
            "verdict": "unknown",
            "summary": "invalid payload",
            "issues": [],
            "should_repair": True,
        }
    verdict = str(payload.get("verdict", "unknown")).strip().lower()
    if verdict not in {"pass", "fail", "unknown"}:
        verdict = "unknown"
    issues = payload.get("issues", [])
    if not isinstance(issues, list):
        issues = []
    summary = str(payload.get("summary", "")).strip()
    should_repair = payload.get("should_repair", verdict != "pass")
    if isinstance(should_repair, str):
        should_repair = should_repair.strip().lower() in {"1", "true", "yes", "y"}
    else:
        should_repair = bool(should_repair)
    return {
        "verdict": verdict,
        "summary": summary or ("ok" if verdict == "pass" else "issues found"),
        "issues": issues[:32],
        "should_repair": should_repair,
    }


def _extract_tex_document(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse_tex_plan(raw: str) -> dict:
    if not raw.strip():
        return {
            "strategy": "direct proof",
            "outline": [],
            "key_lemmas": [],
            "checks": [],
            "claims": [
                {
                    "id": "C1",
                    "goal": "Prove the theorem statement directly.",
                    "depends_on": [],
                    "assumptions": [],
                    "required_facts": [],
                    "acceptance_checks": [],
                }
            ],
        }
    text = raw.strip()
    payload: dict | None = None
    try:
        parsed = json.loads(_extract_json(text))
        if isinstance(parsed, dict):
            payload = parsed
    except Exception:
        payload = None
    if payload is None:
        lines = [line.strip("- ").strip() for line in text.splitlines() if line.strip()]
        return {
            "strategy": lines[0] if lines else "direct proof",
            "outline": lines[1:7],
            "key_lemmas": [],
            "checks": [],
            "claims": [
                {
                    "id": "C1",
                    "goal": lines[0] if lines else "Prove the theorem statement directly.",
                    "depends_on": [],
                    "assumptions": [],
                    "required_facts": [],
                    "acceptance_checks": [],
                }
            ],
        }

    strategy = str(payload.get("strategy", "direct proof")).strip() or "direct proof"
    outline = payload.get("outline", [])
    key_lemmas = payload.get("key_lemmas", [])
    checks = payload.get("checks", [])
    claims = payload.get("claims", [])
    if not isinstance(outline, list):
        outline = []
    if not isinstance(key_lemmas, list):
        key_lemmas = []
    if not isinstance(checks, list):
        checks = []
    normalized_claims = _normalize_tex_claims(claims, outline)
    return {
        "strategy": strategy[:400],
        "outline": [str(item).strip()[:300] for item in outline[:12] if str(item).strip()],
        "key_lemmas": [str(item).strip()[:200] for item in key_lemmas[:12] if str(item).strip()],
        "checks": [str(item).strip()[:200] for item in checks[:12] if str(item).strip()],
        "claims": normalized_claims,
    }


def _parse_tex_judge(raw: str) -> dict:
    if not raw.strip():
        return {
            "verdict": "revise",
            "score": 0,
            "summary": "empty judge response",
            "required_changes": ["Judge returned empty output."],
            "style_notes": [],
            "polished_tex": "",
        }
    text = raw.strip()
    payload: dict | None = None
    try:
        parsed = json.loads(_extract_json(text))
        if isinstance(parsed, dict):
            payload = parsed
    except Exception:
        payload = None
    if payload is None:
        return {
            "verdict": "revise",
            "score": 20,
            "summary": "unparsed judge response",
            "required_changes": [text[:240]],
            "style_notes": [],
            "polished_tex": "",
            "raw": text[:1200],
        }

    verdict = str(payload.get("verdict", "revise")).strip().lower()
    if verdict not in {"pass", "revise", "fail"}:
        verdict = "revise"
    score_raw = payload.get("score", 0)
    try:
        score = int(float(score_raw))
    except Exception:
        score = 0
    score = max(0, min(100, score))
    required_changes = payload.get("required_changes", [])
    if not isinstance(required_changes, list):
        required_changes = []
    style_notes = payload.get("style_notes", [])
    if not isinstance(style_notes, list):
        style_notes = []
    polished_tex = _extract_tex_document(str(payload.get("polished_tex", "")))
    return {
        "verdict": verdict,
        "score": score,
        "summary": str(payload.get("summary", "")).strip()[:1200],
        "required_changes": [
            str(item).strip()[:500] for item in required_changes[:16] if str(item).strip()
        ],
        "style_notes": [
            str(item).strip()[:300] for item in style_notes[:16] if str(item).strip()
        ],
        "polished_tex": polished_tex,
    }


def _normalize_tex_claims(raw_claims, outline: list) -> list[dict]:
    fallback_goal = " ; ".join(str(item).strip() for item in outline[:4] if str(item).strip())
    if not fallback_goal:
        fallback_goal = "Prove the theorem statement directly."
    if not isinstance(raw_claims, list):
        raw_claims = []
    claims: list[dict] = []
    seen: set[str] = set()
    for idx, item in enumerate(raw_claims, start=1):
        if not isinstance(item, dict):
            continue
        raw_id = str(item.get("id", "")).strip() or f"C{idx}"
        claim_id = raw_id.replace(" ", "_")[:24]
        if not claim_id:
            claim_id = f"C{idx}"
        if claim_id in seen:
            claim_id = f"C{idx}"
        seen.add(claim_id)
        goal = str(
            item.get("goal", "")
            or item.get("statement", "")
            or item.get("claim", "")
        ).strip()
        if not goal:
            goal = fallback_goal
        depends_on = item.get("depends_on", [])
        assumptions = item.get("assumptions", [])
        required_facts = item.get("required_facts", [])
        acceptance_checks = item.get("acceptance_checks", [])
        if not isinstance(depends_on, list):
            depends_on = []
        if not isinstance(assumptions, list):
            assumptions = []
        if not isinstance(required_facts, list):
            required_facts = []
        if not isinstance(acceptance_checks, list):
            acceptance_checks = []
        claims.append(
            {
                "id": claim_id,
                "goal": goal[:800],
                "depends_on": [str(dep).strip()[:24] for dep in depends_on[:8] if str(dep).strip()],
                "assumptions": [str(v).strip()[:240] for v in assumptions[:16] if str(v).strip()],
                "required_facts": [str(v).strip()[:240] for v in required_facts[:16] if str(v).strip()],
                "acceptance_checks": [
                    str(v).strip()[:240] for v in acceptance_checks[:16] if str(v).strip()
                ],
            }
        )
    if not claims:
        claims = [
            {
                "id": "C1",
                "goal": fallback_goal,
                "depends_on": [],
                "assumptions": [],
                "required_facts": [],
                "acceptance_checks": [],
            }
        ]

    claim_ids = {c["id"] for c in claims}
    for claim in claims:
        claim["depends_on"] = [
            dep for dep in claim.get("depends_on", []) if dep in claim_ids and dep != claim["id"]
        ]
    return claims[:8]


def _parse_tex_claim_draft(raw: str, fallback_claim_id: str) -> dict:
    if not raw.strip():
        return {
            "claim_id": fallback_claim_id,
            "proof_tex": "",
            "assumptions_used": [],
            "depends_on_used": [],
            "cited_facts": [],
            "confidence": 0,
            "notes": "empty draft response",
        }
    text = raw.strip()
    payload: dict | None = None
    try:
        parsed = json.loads(_extract_json(text))
        if isinstance(parsed, dict):
            payload = parsed
    except Exception:
        payload = None
    if payload is None:
        return {
            "claim_id": fallback_claim_id,
            "proof_tex": _extract_tex_document(text),
            "assumptions_used": [],
            "depends_on_used": [],
            "cited_facts": [],
            "confidence": 25,
            "notes": "unparsed draft response",
        }
    claim_id = str(payload.get("claim_id", "")).strip() or fallback_claim_id
    proof_tex = _extract_tex_document(str(payload.get("proof_tex", "")))
    assumptions = payload.get("assumptions_used", [])
    depends_on = payload.get("depends_on_used", [])
    cited_facts = payload.get("cited_facts", [])
    if not isinstance(assumptions, list):
        assumptions = []
    if not isinstance(depends_on, list):
        depends_on = []
    if not isinstance(cited_facts, list):
        cited_facts = []
    confidence_raw = payload.get("confidence", 0)
    try:
        confidence = int(float(confidence_raw))
    except Exception:
        confidence = 0
    confidence = max(0, min(100, confidence))
    return {
        "claim_id": claim_id[:24],
        "proof_tex": proof_tex,
        "assumptions_used": [str(v).strip()[:240] for v in assumptions[:20] if str(v).strip()],
        "depends_on_used": [str(v).strip()[:24] for v in depends_on[:12] if str(v).strip()],
        "cited_facts": [str(v).strip()[:240] for v in cited_facts[:20] if str(v).strip()],
        "confidence": confidence,
        "notes": str(payload.get("notes", "")).strip()[:800],
    }


def _parse_tex_claim_judge(raw: str) -> dict:
    if not raw.strip():
        return {
            "verdict": "revise",
            "score": 0,
            "summary": "empty judge response",
            "required_changes": ["Judge returned empty output."],
            "missing_assumptions": [],
            "citation_issues": [],
            "polished_proof_tex": "",
        }
    text = raw.strip()
    payload: dict | None = None
    try:
        parsed = json.loads(_extract_json(text))
        if isinstance(parsed, dict):
            payload = parsed
    except Exception:
        payload = None
    if payload is None:
        return {
            "verdict": "revise",
            "score": 20,
            "summary": "unparsed judge response",
            "required_changes": [text[:300]],
            "missing_assumptions": [],
            "citation_issues": [],
            "polished_proof_tex": "",
        }
    verdict = str(payload.get("verdict", "revise")).strip().lower()
    if verdict not in {"pass", "revise", "fail"}:
        verdict = "revise"
    score_raw = payload.get("score", 0)
    try:
        score = int(float(score_raw))
    except Exception:
        score = 0
    score = max(0, min(100, score))
    required_changes = payload.get("required_changes", [])
    missing_assumptions = payload.get("missing_assumptions", [])
    citation_issues = payload.get("citation_issues", [])
    if not isinstance(required_changes, list):
        required_changes = []
    if not isinstance(missing_assumptions, list):
        missing_assumptions = []
    if not isinstance(citation_issues, list):
        citation_issues = []
    polished = _extract_tex_document(str(payload.get("polished_proof_tex", "")))
    return {
        "verdict": verdict,
        "score": score,
        "summary": str(payload.get("summary", "")).strip()[:1200],
        "required_changes": [str(v).strip()[:500] for v in required_changes[:20] if str(v).strip()],
        "missing_assumptions": [
            str(v).strip()[:500] for v in missing_assumptions[:20] if str(v).strip()
        ],
        "citation_issues": [str(v).strip()[:500] for v in citation_issues[:20] if str(v).strip()],
        "polished_proof_tex": polished,
    }


def _parse_tex_claim_verifier(raw: str) -> dict:
    if not raw.strip():
        return {
            "verdict": "revise",
            "score": 0,
            "summary": "empty verifier response",
            "critical_issues": ["Verifier returned empty output."],
            "counterexample_attempt": "",
            "suggested_repairs": [],
        }
    text = raw.strip()
    payload: dict | None = None
    try:
        parsed = json.loads(_extract_json(text))
        if isinstance(parsed, dict):
            payload = parsed
    except Exception:
        payload = None
    if payload is None:
        return {
            "verdict": "revise",
            "score": 20,
            "summary": "unparsed verifier response",
            "critical_issues": [text[:300]],
            "counterexample_attempt": "",
            "suggested_repairs": [],
        }
    verdict = str(payload.get("verdict", "revise")).strip().lower()
    if verdict not in {"pass", "revise", "fail"}:
        verdict = "revise"
    score_raw = payload.get("score", 0)
    try:
        score = int(float(score_raw))
    except Exception:
        score = 0
    score = max(0, min(100, score))
    issues = payload.get("critical_issues", [])
    repairs = payload.get("suggested_repairs", [])
    if not isinstance(issues, list):
        issues = []
    if not isinstance(repairs, list):
        repairs = []
    return {
        "verdict": verdict,
        "score": score,
        "summary": str(payload.get("summary", "")).strip()[:1200],
        "critical_issues": [str(v).strip()[:500] for v in issues[:20] if str(v).strip()],
        "counterexample_attempt": str(payload.get("counterexample_attempt", "")).strip()[:1000],
        "suggested_repairs": [str(v).strip()[:500] for v in repairs[:20] if str(v).strip()],
    }


def _parse_tex_claim_checker(raw: str) -> dict:
    if not raw.strip():
        return {
            "status": "issues",
            "score": 0,
            "issues": ["empty checker response"],
            "warnings": [],
            "sanity_checks": [],
        }
    text = raw.strip()
    payload: dict | None = None
    try:
        parsed = json.loads(_extract_json(text))
        if isinstance(parsed, dict):
            payload = parsed
    except Exception:
        payload = None
    if payload is None:
        return {
            "status": "issues",
            "score": 20,
            "issues": [text[:300]],
            "warnings": [],
            "sanity_checks": [],
        }
    status = str(payload.get("status", "issues")).strip().lower()
    if status not in {"ok", "issues"}:
        status = "issues"
    score_raw = payload.get("score", 0)
    try:
        score = int(float(score_raw))
    except Exception:
        score = 0
    score = max(0, min(100, score))
    issues = payload.get("issues", [])
    warnings = payload.get("warnings", [])
    checks = payload.get("sanity_checks", [])
    if not isinstance(issues, list):
        issues = []
    if not isinstance(warnings, list):
        warnings = []
    if not isinstance(checks, list):
        checks = []
    return {
        "status": status,
        "score": score,
        "issues": [str(v).strip()[:500] for v in issues[:20] if str(v).strip()],
        "warnings": [str(v).strip()[:400] for v in warnings[:20] if str(v).strip()],
        "sanity_checks": [str(v).strip()[:240] for v in checks[:20] if str(v).strip()],
    }


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text
