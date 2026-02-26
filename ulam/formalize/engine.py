from __future__ import annotations

import ast
import datetime as dt
import difflib
import json
import os
import re
from pathlib import Path
from typing import Optional

from .llm import FormalizationLLM
from .segment import segment_tex, collect_segment_hints, attach_proofs
from .types import FormalizationConfig, FormalizationResult
from ..search import best_first_search, scripted_search
from ..trace import TraceLogger
from ..types import RunConfig


class FormalizationEngine:
    def __init__(self, config: FormalizationConfig, llm: FormalizationLLM) -> None:
        self._config = config
        self._llm = llm

    def run(self) -> FormalizationResult:
        tex = self._config.tex_path.read_text(encoding="utf-8", errors="ignore")
        context = _read_context(self._config.context_files)
        segments = attach_proofs(segment_tex(tex))
        hints = collect_segment_hints(segments, "theorem") + collect_segment_hints(segments, "lemma")
        artifact_dir = _prepare_artifacts(self._config, tex, context, segments)

        if self._config.resume_path and self._config.resume_path.exists():
            lean_code = self._config.resume_path.read_text(encoding="utf-8", errors="ignore")
            _log(self._config, f"[resume] using {self._config.resume_path}")
        else:
            lean_code = self._llm.draft(tex, context, hints)
            if not lean_code.strip():
                lean_code = _fallback_lean(tex)
        lean_code = _normalize_lean_output(lean_code)
        lean_code = _inject_tex_snippets(lean_code, segments)
        initial_sorries = _count_sorries(lean_code)
        baseline_decls = _decl_statement_map(lean_code)
        resume_locked_decls = (
            _locked_declaration_map(lean_code)
            if self._config.resume_path and self._config.resume_path.exists()
            else {}
        )
        context_decl_map = _context_decl_statement_map(self._config.context_files)
        best_proven_decls = _count_proven_declarations(lean_code)
        rejection_memory = _load_rejection_memory(artifact_dir / "rejection_memory.json")
        if resume_locked_decls:
            _log(
                self._config,
                f"[resume] locked {len(resume_locked_decls)} previously proven declaration(s)",
            )
        if rejection_memory:
            _log(
                self._config,
                f"[equivalence] loaded rejection memory for {len(rejection_memory)} declaration(s)",
            )

        rounds = 0
        repairs = 0
        equiv_repairs = 0
        semantic_repairs = 0
        solved = 0
        typecheck_ok = False
        last_typecheck_error: str | None = None
        error_counts: dict[str, int] = {}
        round_offset = _max_existing_round_index(artifact_dir)
        while rounds < self._config.max_rounds:
            rounds += 1
            _log(self._config, f"[formalize] round {rounds}/{self._config.max_rounds}")
            _log_progress(self._config, initial_sorries, _count_sorries(lean_code))
            _log_decl_progress(self._config, lean_code)
            round_idx = round_offset + rounds
            round_dir = _ensure_round_dir(artifact_dir, round_idx)
            _persist_rejection_memory(artifact_dir, round_dir, rejection_memory)
            _write_artifact(round_dir / "start.lean", lean_code)
            self._write_output(lean_code)
            check_error = _typecheck(lean_code, self._config)
            if check_error:
                typecheck_ok = False
                last_typecheck_error = check_error
                fingerprint = _error_fingerprint(check_error)
                repeat_count = error_counts.get(fingerprint, 0) + 1
                error_counts[fingerprint] = repeat_count
                _log(self._config, f"[typecheck] error: {_error_preview(check_error)}")
                _write_artifact(round_dir / "typecheck_error.txt", check_error)
                repairs += 1
                if repeat_count >= 3:
                    _log(self._config, "[stagnation] repeated typecheck error; stopping early")
                    break
                if repairs > self._config.max_repairs:
                    return FormalizationResult(
                        output_path=self._config.output_path,
                        rounds=rounds,
                        typecheck_ok=False,
                        solved=0,
                        remaining_sorries=_count_sorries(lean_code),
                        error=f"max repairs reached ({self._config.max_repairs})",
                        artifact_dir=artifact_dir,
                    )
                if rounds >= self._config.max_rounds:
                    return FormalizationResult(
                        output_path=self._config.output_path,
                        rounds=rounds,
                        typecheck_ok=False,
                        solved=0,
                        remaining_sorries=_count_sorries(lean_code),
                        error=check_error,
                        artifact_dir=artifact_dir,
                    )
                _log(self._config, "[repair] requesting LLM repair")
                repair_error = check_error
                if repeat_count >= 2:
                    repair_error = (
                        check_error
                        + "\n\nStagnation note: a similar typecheck error has repeated in prior rounds. "
                        "Make a materially different repair strategy."
                    )
                repaired = self._llm.repair(
                    lean_code,
                    repair_error,
                    _progress_guard_context(
                        context, resume_locked_decls, context_decl_map, best_proven_decls
                    ),
                )
                if repaired.strip() and repaired != lean_code:
                    repaired = _normalize_lean_output(repaired)
                    repaired = _inject_tex_snippets(repaired, segments)
                    guard_error = _progress_guard_error(
                        repaired,
                        min_proven=best_proven_decls,
                        locked_decls=resume_locked_decls,
                        context_decls=context_decl_map,
                    )
                    if guard_error:
                        _log(self._config, f"[progress-guard] {guard_error}")
                        _write_artifact(round_dir / "repair_rejected.lean", repaired)
                        _write_artifact(round_dir / "repair_rejected.txt", guard_error)
                    else:
                        _write_artifact(round_dir / "repair.lean", repaired)
                        _write_diff(round_dir / "repair.diff", lean_code, repaired)
                        lean_code = repaired
                        best_proven_decls = max(
                            best_proven_decls, _count_proven_declarations(lean_code)
                        )
                else:
                    if not repaired.strip():
                        _log(self._config, "[repair] empty response; stopping early")
                    else:
                        _log(self._config, "[repair] no effective changes; stopping early")
                    break
                if not lean_code.strip():
                    break
                continue

            typecheck_ok = True
            best_proven_decls = max(best_proven_decls, _count_proven_declarations(lean_code))
            if self._config.equivalence_checks:
                eq_results, mismatches = _equivalence_check(
                    segments, lean_code, self._llm, context, self._config
                )
                _write_artifact(round_dir / "equivalence.json", json.dumps(eq_results, indent=2))
                if mismatches and equiv_repairs < self._config.max_equivalence_repairs:
                    actionable_mismatches: list[dict] = []
                    skipped_locked = 0
                    statement_repair_attempts = 2
                    for mismatch in mismatches:
                        name = str(mismatch.get("name", "")).strip()
                        if not name:
                            continue
                        if name in resume_locked_decls:
                            skipped_locked += 1
                            continue
                        actionable_mismatches.append(mismatch)
                    _log(
                        self._config,
                        f"[equivalence] {len(mismatches)} mismatches"
                        + (
                            f" ({len(actionable_mismatches)} actionable, {skipped_locked} locked skipped)"
                            if skipped_locked
                            else ""
                        ),
                    )
                    did_repair = False
                    for idx, mismatch in enumerate(actionable_mismatches, start=1):
                        if equiv_repairs >= self._config.max_equivalence_repairs:
                            break
                        target_name = str(mismatch.get("name", "")).strip()
                        target_tex = str(mismatch.get("tex", ""))
                        if not target_name:
                            continue
                        rejection_reasons: list[str] = list(rejection_memory.get(target_name, []))
                        if rejection_reasons:
                            _log(
                                self._config,
                                f"[equivalence] using {len(rejection_reasons)} prior rejection hint(s) for {target_name}",
                            )
                        base_repair_context = _progress_guard_context(
                            context, resume_locked_decls, context_decl_map, best_proven_decls
                        )
                        for attempt in range(1, statement_repair_attempts + 1):
                            if attempt == 1:
                                _log(self._config, f"[equivalence] repair {idx}: {target_name}")
                            else:
                                _log(
                                    self._config,
                                    f"[equivalence] retry {idx}.{attempt - 1}: {target_name}",
                                )
                            repaired = self._llm.repair_statement(
                                lean_code,
                                target_name,
                                target_tex,
                                _context_with_rejection_feedback(
                                    base_repair_context, target_name, rejection_reasons
                                ),
                            )
                            if not repaired.strip() or repaired == lean_code:
                                reason = (
                                    f"No effective update was produced for `{target_name}`; "
                                    "change only the target declaration and keep locked declarations untouched."
                                )
                                _record_rejection_reason(rejection_memory, target_name, reason)
                                _persist_rejection_memory(artifact_dir, round_dir, rejection_memory)
                                if attempt < statement_repair_attempts:
                                    rejection_reasons.append(reason)
                                continue
                            repaired = _normalize_lean_output(repaired)
                            repaired = _inject_tex_snippets(repaired, segments)
                            guard_error = _progress_guard_error(
                                repaired,
                                min_proven=best_proven_decls,
                                locked_decls=resume_locked_decls,
                                context_decls=context_decl_map,
                            )
                            if guard_error:
                                _log(self._config, f"[progress-guard] {guard_error}")
                                rejection_reasons.append(guard_error)
                                _record_rejection_reason(rejection_memory, target_name, guard_error)
                                _persist_rejection_memory(artifact_dir, round_dir, rejection_memory)
                                _write_artifact(
                                    round_dir / f"equivalence_repair_{idx}_attempt_{attempt}_rejected.lean",
                                    repaired,
                                )
                                _write_artifact(
                                    round_dir / f"equivalence_repair_{idx}_attempt_{attempt}_rejected.txt",
                                    guard_error,
                                )
                                continue
                            suffix = "" if attempt == 1 else f"_attempt_{attempt}"
                            _write_artifact(round_dir / f"equivalence_repair_{idx}{suffix}.lean", repaired)
                            _write_diff(
                                round_dir / f"equivalence_repair_{idx}{suffix}.diff",
                                lean_code,
                                repaired,
                            )
                            lean_code = repaired
                            best_proven_decls = max(
                                best_proven_decls, _count_proven_declarations(lean_code)
                            )
                            equiv_repairs += 1
                            did_repair = True
                            break
                    if did_repair:
                        continue

            if self._config.llm_check and _llm_check_stage_enabled(self._config.llm_check_timing, "mid"):
                mid_check = _semantic_integrity_check(
                    tex=tex,
                    lean_code=lean_code,
                    baseline_decls=baseline_decls,
                    llm=self._llm,
                    context=context,
                    stage="mid",
                )
                _write_artifact(round_dir / "llm_check_mid.json", json.dumps(mid_check, indent=2))
                if not mid_check.get("ok", False):
                    message = _semantic_failure_message(mid_check, stage="mid")
                    if semantic_repairs >= self._config.llm_check_repairs:
                        return FormalizationResult(
                            output_path=self._config.output_path,
                            rounds=rounds,
                            typecheck_ok=False,
                            solved=0,
                            remaining_sorries=_count_sorries(lean_code),
                            error=message,
                            artifact_dir=artifact_dir,
                        )
                    _log(self._config, f"[llm-check] {message}")
                    _log(self._config, "[llm-check] requesting semantic repair")
                    repaired = self._llm.semantic_repair(
                        lean_code=lean_code,
                        tex=tex,
                        deterministic_issues=mid_check.get("deterministic_issues", []),
                        audit=mid_check.get("audit", {}),
                        context=_progress_guard_context(
                            context, resume_locked_decls, context_decl_map, best_proven_decls
                        ),
                    )
                    if repaired.strip() and repaired != lean_code:
                        repaired = _normalize_lean_output(repaired)
                        repaired = _inject_tex_snippets(repaired, segments)
                        guard_error = _progress_guard_error(
                            repaired,
                            min_proven=best_proven_decls,
                            locked_decls=resume_locked_decls,
                            context_decls=context_decl_map,
                        )
                        if guard_error:
                            _log(self._config, f"[progress-guard] {guard_error}")
                            _write_artifact(round_dir / "llm_check_mid_repair_rejected.lean", repaired)
                            _write_artifact(round_dir / "llm_check_mid_repair_rejected.txt", guard_error)
                        else:
                            semantic_repairs += 1
                            _write_artifact(
                                round_dir / f"llm_check_mid_repair_{semantic_repairs}.lean", repaired
                            )
                            _write_diff(
                                round_dir / f"llm_check_mid_repair_{semantic_repairs}.diff",
                                lean_code,
                                repaired,
                            )
                            lean_code = repaired
                            best_proven_decls = max(
                                best_proven_decls, _count_proven_declarations(lean_code)
                            )
                            continue
                    return FormalizationResult(
                        output_path=self._config.output_path,
                        rounds=rounds,
                        typecheck_ok=False,
                        solved=0,
                        remaining_sorries=_count_sorries(lean_code),
                        error=message,
                        artifact_dir=artifact_dir,
                    )

            before_proof_code = lean_code
            solved, failures, updated_lean_code = _attempt_proofs(
                lean_code,
                self._config,
                _progress_guard_context(context, resume_locked_decls, context_decl_map, best_proven_decls),
                segments,
                self._llm,
                locked_decls=resume_locked_decls,
                context_decls=context_decl_map,
                min_proven=best_proven_decls,
            )
            proof_guard_error = _progress_guard_error(
                updated_lean_code,
                min_proven=best_proven_decls,
                locked_decls=resume_locked_decls,
                context_decls=context_decl_map,
            )
            if proof_guard_error:
                _log(self._config, f"[progress-guard] {proof_guard_error}")
                _write_artifact(round_dir / "proof_update_rejected.lean", updated_lean_code)
                _write_artifact(round_dir / "proof_update_rejected.txt", proof_guard_error)
                lean_code = before_proof_code
                solved = 0
                failures = [name for name in _extract_decl_names(lean_code) if _decl_has_placeholder(lean_code, name)]
                self._write_output(lean_code)
            else:
                lean_code = updated_lean_code
                best_proven_decls = max(best_proven_decls, _count_proven_declarations(lean_code))
            if failures:
                _write_artifact(round_dir / "proof_failures.json", json.dumps(failures, indent=2))
            if not failures:
                if typecheck_ok and _count_sorries(lean_code) == 0:
                    axiom_error = _axiom_guardrail_error(lean_code, self._config.allow_axioms)
                    if axiom_error:
                        _log(self._config, f"[axiom] {axiom_error}")
                        repaired = self._llm.repair(
                            lean_code,
                            axiom_error,
                            _progress_guard_context(
                                context, resume_locked_decls, context_decl_map, best_proven_decls
                            ),
                        )
                        if repaired.strip() and repaired != lean_code:
                            repaired = _normalize_lean_output(repaired)
                            repaired = _inject_tex_snippets(repaired, segments)
                            guard_error = _progress_guard_error(
                                repaired,
                                min_proven=best_proven_decls,
                                locked_decls=resume_locked_decls,
                                context_decls=context_decl_map,
                            )
                            if guard_error:
                                _log(self._config, f"[progress-guard] {guard_error}")
                                _write_artifact(round_dir / "axiom_repair_rejected.lean", repaired)
                                _write_artifact(round_dir / "axiom_repair_rejected.txt", guard_error)
                            else:
                                _write_artifact(round_dir / "axiom_repair.lean", repaired)
                                _write_diff(round_dir / "axiom_repair.diff", lean_code, repaired)
                                lean_code = repaired
                                best_proven_decls = max(
                                    best_proven_decls, _count_proven_declarations(lean_code)
                                )
                                continue
                    if self._config.llm_check and _llm_check_stage_enabled(
                        self._config.llm_check_timing, "end"
                    ):
                        end_check = _semantic_integrity_check(
                            tex=tex,
                            lean_code=lean_code,
                            baseline_decls=baseline_decls,
                            llm=self._llm,
                            context=context,
                            stage="end",
                        )
                        _write_artifact(
                            round_dir / "llm_check_end.json", json.dumps(end_check, indent=2)
                        )
                        if not end_check.get("ok", False):
                            message = _semantic_failure_message(end_check, stage="end")
                            if semantic_repairs >= self._config.llm_check_repairs:
                                return FormalizationResult(
                                    output_path=self._config.output_path,
                                    rounds=rounds,
                                    typecheck_ok=False,
                                    solved=0,
                                    remaining_sorries=_count_sorries(lean_code),
                                    error=message,
                                    artifact_dir=artifact_dir,
                                )
                            _log(self._config, f"[llm-check] {message}")
                            _log(self._config, "[llm-check] requesting semantic repair")
                            repaired = self._llm.semantic_repair(
                                lean_code=lean_code,
                                tex=tex,
                                deterministic_issues=end_check.get("deterministic_issues", []),
                                audit=end_check.get("audit", {}),
                                context=_progress_guard_context(
                                    context, resume_locked_decls, context_decl_map, best_proven_decls
                                ),
                            )
                            if repaired.strip() and repaired != lean_code:
                                repaired = _normalize_lean_output(repaired)
                                repaired = _inject_tex_snippets(repaired, segments)
                                guard_error = _progress_guard_error(
                                    repaired,
                                    min_proven=best_proven_decls,
                                    locked_decls=resume_locked_decls,
                                    context_decls=context_decl_map,
                                )
                                if guard_error:
                                    _log(self._config, f"[progress-guard] {guard_error}")
                                    _write_artifact(
                                        round_dir / "llm_check_end_repair_rejected.lean", repaired
                                    )
                                    _write_artifact(
                                        round_dir / "llm_check_end_repair_rejected.txt", guard_error
                                    )
                                else:
                                    semantic_repairs += 1
                                    _write_artifact(
                                        round_dir / f"llm_check_end_repair_{semantic_repairs}.lean",
                                        repaired,
                                    )
                                    _write_diff(
                                        round_dir / f"llm_check_end_repair_{semantic_repairs}.diff",
                                        lean_code,
                                        repaired,
                                    )
                                    lean_code = repaired
                                    best_proven_decls = max(
                                        best_proven_decls, _count_proven_declarations(lean_code)
                                    )
                                    continue
                            return FormalizationResult(
                                output_path=self._config.output_path,
                                rounds=rounds,
                                typecheck_ok=False,
                                solved=0,
                                remaining_sorries=_count_sorries(lean_code),
                                error=message,
                                artifact_dir=artifact_dir,
                            )
                break
            if rounds >= self._config.max_rounds:
                break
            _log(self._config, "[improve] requesting LLM improvements")
            improved = self._llm.improve(
                lean_code,
                failures,
                _progress_guard_context(context, resume_locked_decls, context_decl_map, best_proven_decls),
            )
            if improved.strip() and improved != lean_code:
                improved = _normalize_lean_output(improved)
                improved = _inject_tex_snippets(improved, segments)
                guard_error = _progress_guard_error(
                    improved,
                    min_proven=best_proven_decls,
                    locked_decls=resume_locked_decls,
                    context_decls=context_decl_map,
                )
                if guard_error:
                    _log(self._config, f"[progress-guard] {guard_error}")
                    _write_artifact(round_dir / "improve_rejected.lean", improved)
                    _write_artifact(round_dir / "improve_rejected.txt", guard_error)
                else:
                    _write_artifact(round_dir / "improve.lean", improved)
                    _write_diff(round_dir / "improve.diff", lean_code, improved)
                    lean_code = improved
                    best_proven_decls = max(best_proven_decls, _count_proven_declarations(lean_code))
            else:
                if not improved.strip():
                    _log(self._config, "[improve] empty response; stopping early")
                else:
                    _log(self._config, "[improve] no effective changes; stopping early")
                break
            if not lean_code.strip():
                break

        if (
            typecheck_ok
            and self._config.llm_check
            and _llm_check_stage_enabled(self._config.llm_check_timing, "end")
        ):
            final_check = _semantic_integrity_check(
                tex=tex,
                lean_code=lean_code,
                baseline_decls=baseline_decls,
                llm=self._llm,
                context=context,
                stage="end-final",
            )
            _write_artifact(artifact_dir / "llm_check_final.json", json.dumps(final_check, indent=2))
            if not final_check.get("ok", False):
                message = _semantic_failure_message(final_check, stage="end-final")
                self._write_output(lean_code)
                _log_progress(self._config, initial_sorries, _count_sorries(lean_code))
                _log_decl_progress(self._config, lean_code)
                return FormalizationResult(
                    output_path=self._config.output_path,
                    rounds=rounds,
                    typecheck_ok=False,
                    solved=0,
                    remaining_sorries=_count_sorries(lean_code),
                    error=message,
                    artifact_dir=artifact_dir,
                )

        self._write_output(lean_code)
        _log_progress(self._config, initial_sorries, _count_sorries(lean_code))
        _log_decl_progress(self._config, lean_code)
        return FormalizationResult(
            output_path=self._config.output_path,
            rounds=rounds,
            typecheck_ok=typecheck_ok,
            solved=solved,
            remaining_sorries=_count_sorries(lean_code),
            error=None if typecheck_ok else (last_typecheck_error or "typecheck failed"),
            artifact_dir=artifact_dir,
        )

    def _write_output(self, lean_code: str) -> None:
        self._config.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._config.output_path.write_text(_normalize_lean_output(lean_code), encoding="utf-8")


def _attempt_proofs(
    lean_code: str,
    config: FormalizationConfig,
    context: str,
    segments,
    llm: FormalizationLLM | None = None,
    *,
    locked_decls: dict[str, dict[str, str]] | None = None,
    context_decls: dict[str, str] | None = None,
    min_proven: int | None = None,
) -> tuple[int, list[str], str]:
    if config.max_proof_rounds <= 0:
        _log(config, "[prove] skipping proof search (disabled)")
        return 0, [], lean_code
    mode = config.proof_backend or "tactic"
    if mode == "dojo":
        mode = "tactic"
    if mode == "llm":
        if llm is None:
            _log(config, "[prove] skipping proof search (no LLM configured)")
            return 0, [], lean_code
        return _attempt_proofs_llm(
            lean_code,
            config,
            context,
            segments,
            llm,
            locked_decls=locked_decls or {},
            context_decls=context_decls or {},
            min_proven=min_proven,
        )
    if mode == "lemma":
        if llm is None:
            _log(config, "[prove] skipping lemma mode (no LLM configured)")
            return 0, [], lean_code
        return _attempt_proofs_lemma(lean_code, config, context, segments, llm)
    if not config.lean_project:
        _log(config, "[prove] skipping proof search (no Lean project configured)")
        return 0, [], lean_code

    return _attempt_proofs_tactic(lean_code, config, context, segments, llm)


def _attempt_proofs_tactic(
    lean_code: str,
    config: FormalizationConfig,
    context: str,
    segments,
    llm: FormalizationLLM | None = None,
) -> tuple[int, list[str], str]:
    names = _extract_decl_names(lean_code)
    failures: list[str] = []
    solved = 0
    rounds = 0
    pending = [name for name in names if _decl_has_sorry(lean_code, name)]
    tex_snippets = _build_tex_snippet_map(segments, lean_code)

    while pending and rounds < config.max_proof_rounds:
        rounds += 1
        _log(config, f"[prove] round {rounds}/{config.max_proof_rounds}")
        next_pending: list[str] = []
        for name in pending:
            _log(config, f"[prove] attempting {name}")
            snippet = tex_snippets.get(name, "")
            result, proof_lines, _, error = _run_prover(name, config, context, snippet)
            if result:
                lean_code = _replace_sorry(lean_code, name, proof_lines)
                _write_output_inline(config, lean_code)
                solved += 1
                _log(config, f"[prove] solved {name}")
            else:
                if error:
                    _log(config, f"[prove] prover error for {name}: {error}")
                    if llm is not None:
                        repaired = llm.repair(lean_code, error, context)
                        if repaired.strip() and repaired != lean_code:
                            repaired = _normalize_lean_output(repaired)
                            repaired = _inject_tex_snippets(repaired, segments)
                            _write_output_inline(config, repaired)
                            lean_code = repaired
                next_pending.append(name)
                _log(config, f"[prove] failed {name}")
        pending = next_pending

    failures.extend(pending)
    return solved, failures, lean_code


def _attempt_proofs_lemma(
    lean_code: str,
    config: FormalizationConfig,
    context: str,
    segments,
    llm: FormalizationLLM,
) -> tuple[int, list[str], str]:
    names = _extract_decl_names(lean_code)
    pending = [name for name in names if _decl_has_sorry(lean_code, name)]
    if not pending:
        return 0, [], lean_code
    tex_snippets = _build_tex_snippet_map(segments, lean_code)

    for name in list(pending):
        marker = _lemma_plan_marker(name)
        if marker in lean_code:
            continue
        theorem_stmt, original_stmt = _extract_theorem_statement_from_text(lean_code, name)
        if not theorem_stmt:
            continue
        original = tex_snippets.get(name, "") or original_stmt or theorem_stmt
        plan_code = llm.plan_lemmas(
            theorem_name=name,
            theorem_statement=theorem_stmt,
            original_statement=original,
            context=context,
        )
        if not plan_code.strip():
            continue
        snippet = _strip_imports_from_snippet(plan_code)
        blocks = _extract_decl_blocks(snippet)
        lean_code, new_names = _insert_lemmas_before_in_text(lean_code, name, blocks)
        if new_names:
            lean_code = _insert_lemma_plan_marker(lean_code, name)
            _write_output_inline(config, lean_code)

    # Recompute pending after inserting lemma plans.
    pending = [name for name in _extract_decl_names(lean_code) if _decl_has_sorry(lean_code, name)]
    if not pending:
        return 0, [], lean_code

    queue: list[tuple[str, int]] = [(name, 0) for name in pending]
    solved: set[str] = set()
    failures: list[str] = []
    total_decl_count = len(_extract_decl_names(lean_code))
    completed = total_decl_count - len(queue)

    while queue:
        _log(config, f"[lemma-first] Progress {completed}/{total_decl_count} solved")
        name, depth = queue.pop(0)
        if name in solved:
            continue
        _log(config, f"[lemma-first] Proving {name}...")
        trace_path = _lemma_trace_path(name)
        result, proof_lines, trace_path, error = _run_prover(
            name, config, context, tex_snippets.get(name, ""), trace_path=trace_path
        )
        if result:
            lean_code = _replace_sorry(lean_code, name, proof_lines)
            _write_output_inline(config, lean_code)
            solved.add(name)
            completed += 1
            continue

        if error:
            _log(config, f"[lemma-first] prover error for {name}: {error}")
            repaired = llm.repair(lean_code, error, context)
            if repaired.strip() and repaired != lean_code:
                repaired = _normalize_lean_output(repaired)
                repaired = _inject_tex_snippets(repaired, segments)
                _write_output_inline(config, repaired)
                lean_code = repaired
                queue.append((name, depth))
            else:
                failures.append(name)
            continue

        failures.append(name)
        total_decl_count = len(_extract_decl_names(lean_code))
        if total_decl_count >= config.lemma_max or depth >= config.lemma_depth:
            _log(config, "[lemma-first] limits reached; stopping.")
            break

        expanded = _expand_lemmas_for_failure(
            lean_code,
            name,
            trace_path,
            context,
            llm,
        )
        if not expanded:
            _log(config, f"[lemma-first] failed to expand lemma {name}")
            break
        lean_code, new_names = expanded
        if new_names:
            _write_output_inline(config, lean_code)
            for new_name in reversed(new_names):
                queue.insert(0, (new_name, depth + 1))
        queue.insert(0, (name, depth + 1))

    pending = [name for name in _extract_decl_names(lean_code) if _decl_has_sorry(lean_code, name)]
    failures.extend(name for name in pending if name not in solved)
    return len(solved), failures, lean_code


def _attempt_proofs_llm(
    lean_code: str,
    config: FormalizationConfig,
    context: str,
    segments,
    llm: FormalizationLLM,
    *,
    locked_decls: dict[str, dict[str, str]],
    context_decls: dict[str, str],
    min_proven: int | None,
) -> tuple[int, list[str], str]:
    names = _extract_decl_names(lean_code)
    failures: list[str] = []
    solved = 0
    rounds = 0
    pending = [name for name in names if _decl_has_placeholder(lean_code, name)]
    tex_snippets = _build_tex_snippet_map(segments, lean_code)
    min_proven_local = int(min_proven) if min_proven is not None else _count_proven_declarations(lean_code)
    if min_proven_local < 0:
        min_proven_local = 0

    if not pending:
        return 0, [], lean_code

    while pending and rounds < config.max_proof_rounds:
        rounds += 1
        _log(config, f"[prove] round {rounds}/{config.max_proof_rounds}")
        next_pending: list[str] = []
        for name in pending:
            _log(config, f"[prove] attempting {name}")
            snippet = tex_snippets.get(name, "")
            error: str | None = None
            error_counts: dict[str, int] = {}
            success = False
            attempts = max(1, int(config.proof_repair) + 1)
            for attempt in range(1, attempts + 1):
                _log(config, "[llm] requesting proof update")
                try:
                    updated = llm.prove(
                        lean_code=lean_code,
                        name=name,
                        instruction=(
                            f"Edit only declaration `{name}`. "
                            "Do not modify any other declaration."
                        ),
                        tex_snippet=snippet,
                        context=context,
                        error=error,
                    )
                except Exception as exc:
                    _log(config, f"[llm] error: {exc}")
                    break
                if not updated.strip():
                    break
                candidate = _normalize_lean_output(updated)
                candidate = _inject_tex_snippets(candidate, segments)
                if _decl_has_placeholder(candidate, name):
                    error = f"Declaration `{name}` still contains sorry/admit."
                    repeat = _record_error_count(error_counts, error)
                    if repeat >= 3:
                        _log(config, f"[stagnation] repeated placeholder error for {name}")
                        break
                    continue
                check_error = _typecheck(candidate, config)
                if check_error:
                    error = check_error
                    repeat = _record_error_count(error_counts, check_error)
                    if repeat >= 3:
                        _log(config, f"[stagnation] repeated typecheck error for {name}")
                        break
                    continue
                guard_error = _progress_guard_error(
                    candidate,
                    min_proven=min_proven_local,
                    locked_decls=locked_decls,
                    context_decls=context_decls,
                )
                if guard_error:
                    _log(config, f"[progress-guard] {guard_error}")
                    error = (
                        f"Progress guard rejected the update: {guard_error}\n"
                        f"Only change declaration `{name}` and keep all other declarations unchanged."
                    )
                    repeat = _record_error_count(error_counts, error)
                    if repeat >= 3:
                        _log(config, f"[stagnation] repeated guard rejection for {name}")
                        break
                    continue
                lean_code = candidate
                _write_output_inline(config, lean_code)
                min_proven_local = max(min_proven_local, _count_proven_declarations(lean_code))
                solved += 1
                success = True
                break
            if success:
                _log(config, f"[prove] solved {name}")
            else:
                next_pending.append(name)
                _log(config, f"[prove] failed {name}")
        pending = next_pending

    failures.extend(pending)
    return solved, failures, lean_code


def _run_prover(
    name: str,
    config: FormalizationConfig,
    context: str,
    tex_snippet: str,
    trace_path: Path | None = None,
) -> tuple[bool, list[str], Path | None, str | None]:
    from ..lean.dojo import LeanDojoRunner

    trace = TraceLogger(trace_path)
    runner = None
    error: str | None = None
    result = None
    try:
        llm_client = _make_llm_client(config)
        retriever = _proof_retriever(config.output_path, name)
        runner = LeanDojoRunner(
            project_path=config.lean_project,
            imports=config.lean_imports,
            timeout_s=config.dojo_timeout_s,
        )
        instruction = _format_proof_instruction(name, tex_snippet, context)
        run_config = RunConfig(
            file_path=config.output_path,
            theorem=name,
            max_steps=config.proof_max_steps,
            beam_width=config.proof_beam,
            suggestions_per_state=config.proof_k,
            timeout_s=config.proof_timeout_s,
            repair_attempts=config.proof_repair,
            seed=0,
            trace_path=None,
            retriever_k=8,
            autop=True,
            instruction=instruction,
            context=None,
            verbose=config.verbose,
        )
        result = scripted_search(runner, llm_client, retriever, trace, run_config)
    except Exception as exc:
        error = _augment_lean_error(
            str(exc) or repr(exc),
            output_path=config.output_path,
            project_path=config.lean_project,
        )
    finally:
        trace.close()
        if runner is not None:
            try:
                runner.close()
            except Exception:
                pass

    if error:
        return False, [], trace_path, error
    if result and result.solved:
        return True, list(result.proof), trace_path, None
    return False, [], trace_path, None


def _make_llm_client(config: FormalizationConfig):
    from ..llm import (
        OpenAICompatClient,
        OllamaClient,
        AnthropicClient,
        CodexCLIClient,
        ClaudeCLIClient,
        GeminiClient,
        GeminiCLIClient,
    )

    cfg = _load_global_config()
    provider = cfg.get("llm_provider", "openai")
    if provider == "openai":
        return OpenAICompatClient(
            api_key=cfg.get("openai", {}).get("api_key", "") or "",
            base_url=cfg.get("openai", {}).get("base_url", "https://api.openai.com"),
            model=cfg.get("openai", {}).get("model", "gpt-4.1"),
        )
    if provider == "codex_cli":
        openai_cfg = cfg.get("openai", {})
        model = openai_cfg.get("codex_model") or openai_cfg.get("model") or "gpt-5.2-codex"
        return CodexCLIClient(model=model)
    if provider == "ollama":
        return OllamaClient(
            base_url=cfg.get("ollama", {}).get("base_url", "http://localhost:11434"),
            model=cfg.get("ollama", {}).get("model", "llama3.1"),
        )
    if provider == "anthropic":
        token = cfg.get("anthropic", {}).get("api_key", "") or cfg.get("anthropic", {}).get("setup_token", "")
        return AnthropicClient(
            api_key=token,
            base_url=cfg.get("anthropic", {}).get("base_url", "https://api.anthropic.com"),
            model=cfg.get("anthropic", {}).get("model", "claude-3-5-sonnet-20240620"),
        )
    if provider == "claude_cli":
        anthropic_cfg = cfg.get("anthropic", {})
        model = anthropic_cfg.get("claude_model") or anthropic_cfg.get("model") or "claude-3-5-sonnet-20240620"
        return ClaudeCLIClient(model=model)
    if provider == "gemini":
        gemini_cfg = cfg.get("gemini", {})
        api_key = gemini_cfg.get("api_key", "") or os.environ.get("ULAM_GEMINI_API_KEY", "") or os.environ.get(
            "GEMINI_API_KEY", ""
        )
        return GeminiClient(
            api_key=api_key,
            base_url=gemini_cfg.get("base_url", "https://generativelanguage.googleapis.com/v1beta/openai"),
            model=gemini_cfg.get("model", "gemini-3.1-pro-preview"),
        )
    if provider == "gemini_cli":
        gemini_cfg = cfg.get("gemini", {})
        model = gemini_cfg.get("cli_model") or gemini_cfg.get("model") or "gemini-3.1-pro-preview"
        return GeminiCLIClient(model=model)
    raise RuntimeError("No LLM provider configured")


def _typecheck(lean_code: str, config: FormalizationConfig) -> Optional[str]:
    if config.lean_backend == "cli":
        from ..lean.cli_check import lean_cli_check

        try:
            project_path = config.lean_project or _find_lean_project(config.output_path)
            return lean_cli_check(config.output_path, project_path=project_path, timeout_s=60.0)
        except Exception as exc:
            return str(exc) or repr(exc)

    if not config.lean_project:
        return None
    try:
        from ..lean.dojo import _load_pantograph_server  # type: ignore

        Server = _load_pantograph_server()
    except RuntimeError as exc:
        return str(exc)
    except Exception as exc:
        return str(exc) or repr(exc)

    from ..lean.dojo import _create_server, _find_project_root  # type: ignore

    server = None
    project_path = config.lean_project or _find_project_root(config.output_path)
    try:
        parsed_imports, body = _split_imports_for_typecheck(lean_code)
        imports = _merge_imports(config.lean_imports, parsed_imports)
        server = _create_server(Server, project_path, imports, timeout_s=config.dojo_timeout_s)
        server.load_sorry(body)
        return None
    except Exception as exc:  # pragma: no cover
        return _augment_lean_error(
            str(exc) or repr(exc),
            output_path=config.output_path,
            project_path=project_path,
        )
    finally:
        if server is not None:
            try:
                server.close()
            except Exception:
                pass


def _augment_lean_error(error: str, output_path: Path, project_path: Path | None) -> str:
    base = _normalize_lean_error_message(error)
    detail = _lean_cli_diagnostic(output_path, project_path)
    if not detail:
        return base
    normalized = detail.strip()
    if normalized and normalized in base:
        return base
    return f"{base}\n\nLean CLI diagnostic:\n{detail}"


def _lean_cli_diagnostic(output_path: Path, project_path: Path | None) -> str | None:
    try:
        from ..lean.cli_check import lean_cli_check

        return lean_cli_check(output_path, project_path=project_path, timeout_s=60.0)
    except Exception:
        return None


def _normalize_lean_error_message(error: str) -> str:
    text = (error or "").strip()
    if not text:
        return text
    if text.startswith("{") and text.endswith("}"):
        try:
            payload = ast.literal_eval(text)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            desc = payload.get("desc")
            code = payload.get("error")
            if isinstance(desc, str) and desc.strip():
                if isinstance(code, str) and code.strip():
                    return f"{desc}\n(error: {code})"
                return desc
    return text


def _error_preview(error: str, max_lines: int = 10, max_chars: int = 900) -> str:
    text = (error or "").strip()
    if not text:
        return text
    truncated = False
    lines = text.splitlines()
    clipped = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        truncated = True
    if len(clipped) > max_chars:
        clipped = clipped[:max_chars].rstrip()
        truncated = True
    if truncated:
        clipped = clipped.rstrip() + "\n[truncated]"
    return clipped


def _error_fingerprint(error: str) -> str:
    text = (error or "").strip()
    if not text:
        return "<empty>"
    first_line = text.splitlines()[0].strip().lower()
    first_line = re.sub(r":[0-9]+:[0-9]+", ":#:#", first_line)
    first_line = re.sub(r"\b[0-9]{2,}\b", "#", first_line)
    return first_line[:220]


def _record_error_count(counter: dict[str, int], error: str) -> int:
    fingerprint = _error_fingerprint(error)
    counter[fingerprint] = counter.get(fingerprint, 0) + 1
    return counter[fingerprint]


def _extract_decl_names(text: str) -> list[str]:
    pattern = re.compile(r"\b(theorem|lemma|example|proposition|corollary)\s+([A-Za-z0-9_']+)")
    return [match.group(2) for match in pattern.finditer(text)]


def _decl_has_sorry(text: str, name: str) -> bool:
    match = re.search(rf"\b(theorem|lemma|example|proposition|corollary)\s+{re.escape(name)}\b", text)
    if not match:
        return False
    rest = text[match.end():]
    next_decl = re.search(r"\b(theorem|lemma|example|proposition|corollary)\b", rest)
    scope = rest if not next_decl else rest[: next_decl.start()]
    return "sorry" in scope


def _decl_has_placeholder(text: str, name: str) -> bool:
    match = re.search(rf"\b(theorem|lemma|example|proposition|corollary)\s+{re.escape(name)}\b", text)
    if not match:
        return False
    rest = text[match.end():]
    next_decl = re.search(r"\b(theorem|lemma|example|proposition|corollary)\b", rest)
    scope = rest if not next_decl else rest[: next_decl.start()]
    return bool(re.search(r"\b(sorry|admit)\b", scope))


def _replace_sorry(text: str, name: str, proof_lines: list[str]) -> str:
    match = re.search(rf"\b(theorem|lemma|example|proposition|corollary)\s+{re.escape(name)}\b", text)
    if not match:
        return text
    rest = text[match.end():]
    next_decl = re.search(r"\b(theorem|lemma|example|proposition|corollary)\b", rest)
    end_idx = len(text) if not next_decl else match.end() + next_decl.start()
    block = text[match.end():end_idx]
    sorry_match = re.search(r"\bsorry\b", block)
    if not sorry_match:
        return text
    start = match.end() + sorry_match.start()
    end = match.end() + sorry_match.end()
    line_start = text.rfind("\n", 0, start) + 1
    line_prefix = text[line_start:start]
    indent = re.match(r"\s*", line_prefix).group(0) if line_prefix is not None else ""
    proof_block = "\n".join(f"{indent}{line}" for line in proof_lines)
    return text[:line_start] + proof_block + text[end:]


def _normalize_lean_output(text: str) -> str:
    lines = []
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    cleaned = _normalize_imports(cleaned)
    return cleaned.strip() + "\n"


def _normalize_imports(text: str) -> str:
    if not text.strip():
        return "import Mathlib\n"
    lines = text.splitlines()
    import_lines: list[str] = []
    body_lines: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("import "):
            import_lines.append(stripped)
        else:
            body_lines.append(line)

    modules: list[str] = []
    for line in import_lines:
        remainder = line[len("import ") :].strip()
        if remainder:
            modules.extend(part.strip() for part in remainder.split() if part.strip())

    if not modules:
        modules = ["Mathlib"]
    elif "Mathlib" not in modules:
        modules.insert(0, "Mathlib")

    deduped: list[str] = []
    seen: set[str] = set()
    for mod in modules:
        if mod in seen:
            continue
        seen.add(mod)
        deduped.append(mod)

    import_block = "import " + " ".join(deduped)

    header: list[str] = []
    rest: list[str] = []
    in_block_comment = False
    for line in body_lines:
        stripped = line.strip()
        if in_block_comment:
            header.append(line)
            if "-/" in stripped:
                in_block_comment = False
            continue
        if stripped.startswith("/-"):
            in_block_comment = True
            header.append(line)
            continue
        if stripped.startswith("--") or stripped == "":
            header.append(line)
            continue
        rest = body_lines[len(header) :]
        break
    else:
        rest = []

    output_lines = [import_block, ""]
    if header:
        output_lines.extend(header)
        if output_lines and output_lines[-1].strip() != "":
            output_lines.append("")
    output_lines.extend(rest)
    return "\n".join(output_lines).strip()


def _axiom_guardrail_error(text: str, allow_axioms: bool) -> Optional[str]:
    if allow_axioms:
        return None

    cleaned = _strip_comments(text)
    if re.search(r"\b(axiom|constant)\b", cleaned):
        return "Axioms/constants are disabled (--no-allow-axioms)."
    return None


def _strip_comments(text: str) -> str:
    no_block = re.sub(r"/-.*?-/", "", text, flags=re.S)
    no_line = re.sub(r"--.*", "", no_block)
    return no_line


def _build_tex_snippet_map(segments, lean_code: str, max_chars: int = 1800) -> dict[str, str]:
    tex_segments = [
        seg
        for seg in segments
        if seg.kind in {"theorem", "lemma", "proposition", "corollary", "example"}
    ]
    decls = _extract_decl_blocks(lean_code)
    pairs = _map_segments_to_decls(tex_segments, decls)
    mapping: dict[str, str] = {}
    for seg, decl in pairs:
        body = seg.body.strip()
        if not body:
            continue
        snippet = body
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars] + "\n-- (truncated)"
        mapping[decl["name"]] = snippet
    return mapping


def _inject_tex_snippets(lean_code: str, segments) -> str:
    mapping = _build_tex_snippet_map(segments, lean_code)
    text = lean_code
    for name, snippet in mapping.items():
        marker = f"ULAMAI_TEX_SNIPPET: {name}"
        if marker in text:
            continue
        safe_snippet = snippet.replace("-/", "- /")
        block = f"/- {marker}\n{safe_snippet}\n-/\n"
        pattern = re.compile(
            rf"^\s*(theorem|lemma|example|proposition|corollary)\s+{re.escape(name)}\b",
            re.M,
        )
        match = pattern.search(text)
        if not match:
            continue
        text = text[: match.start()] + block + text[match.start() :]
    return text


def _merge_imports(config_imports: list[str], parsed_imports: list[str]) -> list[str]:
    merged: list[str] = []
    for item in config_imports + parsed_imports:
        if item and item not in merged:
            merged.append(item)
    if "Mathlib" not in merged:
        merged.insert(0, "Mathlib")
    return merged


def _split_imports_for_typecheck(text: str) -> tuple[list[str], str]:
    imports: list[str] = []
    body: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("import "):
            remainder = stripped[len("import ") :].strip()
            if remainder:
                imports.extend(part.strip() for part in remainder.split() if part.strip())
            continue
        body.append(line)
    return imports, "\n".join(body).lstrip("\n")


def _find_lean_project(path: Path) -> Path | None:
    root = path if path.is_dir() else path.parent
    for parent in [root, *root.parents]:
        if (
            (parent / "lakefile.lean").exists()
            or (parent / "lakefile.toml").exists()
            or (parent / "lean-toolchain").exists()
        ):
            return parent
    return None


def _fallback_lean(tex: str) -> str:
    lines = ["/-", "Original .tex:", tex, "-/", "", "-- TODO: formalize above."]
    return "\n".join(lines)


def _count_sorries(text: str) -> int:
    return len(re.findall(r"\bsorry\b", text))


def _decl_block_has_placeholder(block: str) -> bool:
    return bool(re.search(r"\b(sorry|admit)\b", block))


def _declaration_progress(text: str) -> tuple[int, int]:
    decls = _extract_decl_blocks(text)
    total = len(decls)
    proven = 0
    for decl in decls:
        block = str(decl.get("block", ""))
        if not _decl_block_has_placeholder(block):
            proven += 1
    return proven, total


def _count_proven_declarations(text: str) -> int:
    proven, _ = _declaration_progress(text)
    return proven


def _write_output_inline(config: FormalizationConfig, lean_code: str) -> None:
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(_normalize_lean_output(lean_code), encoding="utf-8")


def _log_progress(config: FormalizationConfig, total: int, remaining: int) -> None:
    if total <= 0:
        return
    solved = max(0, total - remaining)
    pct = int(round((solved / total) * 100))
    _log(config, f"[progress] solved {solved}/{total} ({pct}%), remaining {remaining}")


def _log_decl_progress(config: FormalizationConfig, text: str) -> None:
    proven, total = _declaration_progress(text)
    if total <= 0:
        return
    pct = int(round((proven / total) * 100))
    _log(config, f"[progress] proven declarations {proven}/{total} ({pct}%)")


def _format_proof_instruction(name: str, tex_snippet: str, context: str, max_chars: int = 2500) -> str | None:
    parts = []
    if tex_snippet:
        snippet = tex_snippet.strip()
        if len(snippet) > 1400:
            snippet = snippet[:1400] + "\n-- (truncated)"
        parts.append(f"Informal proof snippet for {name}:\n{snippet}")
    if context:
        ctx = context.strip()
        if len(ctx) > max_chars:
            ctx = ctx[:max_chars] + "\n-- (truncated)"
        parts.append(f"Additional context:\n{ctx}")
    if not parts:
        return None
    return "\n\n".join(parts)


def _read_context(paths: list[Path], max_chars: int = 8000) -> str:
    chunks = []
    for path in paths:
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        if len(content) > max_chars:
            content = content[:max_chars] + "\n-- (truncated)"
        chunks.append(f"[file: {path}]\n{content}")
    return "\n\n".join(chunks)


def _log(config: FormalizationConfig, message: str) -> None:
    if config.verbose:
        print(message)


def _null_retriever():
    from ..retrieve import NullRetriever

    return NullRetriever()


def _proof_retriever(file_path: Path, theorem_name: str):
    # Optional local retrieval: collect nearby declarations from the current
    # Lean file and feed them to script search as lightweight premises.
    from ..retrieve import SimpleRetriever

    enabled = os.environ.get("ULAM_FORMALIZE_LOCAL_RETRIEVER", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return _null_retriever()
    premises = _local_decl_premises(file_path, theorem_name)
    if not premises:
        return _null_retriever()
    return SimpleRetriever(premises)


def _local_decl_premises(
    file_path: Path,
    theorem_name: str,
    *,
    max_items: int = 256,
    max_statement_chars: int = 320,
) -> list[str]:
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    decls = _extract_decl_blocks(text)
    premises: list[str] = []
    for decl in decls:
        name = str(decl.get("name", "")).strip()
        if not name or name == theorem_name:
            continue
        statement = " ".join(str(decl.get("statement", "")).split())
        if not statement:
            continue
        if len(statement) > max_statement_chars:
            statement = statement[:max_statement_chars] + " ..."
        premises.append(f"{name}: {statement}")
        if len(premises) >= max_items:
            break
    return premises


def _load_global_config() -> dict:
    from ..config import load_config

    return load_config()


def _prepare_artifacts(
    config: FormalizationConfig, tex: str, context: str, segments
) -> Path:
    if config.artifact_dir:
        artifact_dir = config.artifact_dir
    else:
        stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        artifact_dir = Path("runs") / f"formalize_{stamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_artifact(artifact_dir / "input.tex", tex)
    _write_artifact(artifact_dir / "context.txt", context or "")
    _write_artifact(artifact_dir / "segments.json", json.dumps([seg.__dict__ for seg in segments], indent=2))
    _write_artifact(artifact_dir / "config.json", json.dumps(_config_snapshot(config), indent=2))
    return artifact_dir


def _ensure_round_dir(artifact_dir: Path, round_idx: int) -> Path:
    round_dir = artifact_dir / f"round_{round_idx:02d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    return round_dir


def _max_existing_round_index(artifact_dir: Path) -> int:
    max_idx = 0
    for path in artifact_dir.glob("round_*"):
        if not path.is_dir():
            continue
        match = re.fullmatch(r"round_(\d+)", path.name)
        if not match:
            continue
        try:
            idx = int(match.group(1))
        except Exception:
            continue
        if idx > max_idx:
            max_idx = idx
    return max_idx


def _write_artifact(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_diff(path: Path, before: str, after: str) -> None:
    diff = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile="before.lean",
        tofile="after.lean",
        lineterm="",
    )
    text = "\n".join(diff)
    _write_artifact(path, text)


def _config_snapshot(config: FormalizationConfig) -> dict:
    return {
        "tex_path": str(config.tex_path),
        "output_path": str(config.output_path),
        "context_files": [str(p) for p in config.context_files],
        "max_rounds": config.max_rounds,
        "max_repairs": config.max_repairs,
        "max_equivalence_repairs": config.max_equivalence_repairs,
        "max_proof_rounds": config.max_proof_rounds,
        "proof_max_steps": config.proof_max_steps,
        "proof_beam": config.proof_beam,
        "proof_k": config.proof_k,
        "proof_timeout_s": config.proof_timeout_s,
        "proof_repair": config.proof_repair,
        "dojo_timeout_s": config.dojo_timeout_s,
        "lemma_max": config.lemma_max,
        "lemma_depth": config.lemma_depth,
        "allow_axioms": config.allow_axioms,
        "proof_backend": config.proof_backend,
        "lean_backend": config.lean_backend,
        "lean_project": str(config.lean_project) if config.lean_project else None,
        "lean_imports": list(config.lean_imports),
        "resume_path": str(config.resume_path) if config.resume_path else None,
        "verbose": config.verbose,
        "artifact_dir": str(config.artifact_dir) if config.artifact_dir else None,
        "equivalence_checks": config.equivalence_checks,
        "llm_check": config.llm_check,
        "llm_check_timing": config.llm_check_timing,
        "llm_check_repairs": config.llm_check_repairs,
    }


def _llm_check_stage_enabled(timing: str, stage: str) -> bool:
    mode = str(timing or "end").strip().lower()
    if mode == "mid+end":
        return stage in {"mid", "end"}
    return stage == "end"


def _semantic_integrity_check(
    *,
    tex: str,
    lean_code: str,
    baseline_decls: dict[str, str],
    llm: FormalizationLLM,
    context: str,
    stage: str,
) -> dict:
    deterministic_issues = _deterministic_semantic_issues(lean_code, baseline_decls)
    blocking_issues = [issue for issue in deterministic_issues if issue.get("severity") == "high"]
    try:
        audit = llm.semantic_check(
            tex=tex,
            lean_code=lean_code,
            deterministic_issues=deterministic_issues,
            context=context,
            stage=stage,
        )
    except Exception as exc:
        audit = {
            "verdict": "unknown",
            "summary": f"semantic check failed: {exc}",
            "issues": [],
            "should_repair": True,
        }
    verdict = str(audit.get("verdict", "unknown")).strip().lower()
    if verdict not in {"pass", "fail", "unknown"}:
        verdict = "unknown"
    should_repair = audit.get("should_repair", verdict == "fail")
    if isinstance(should_repair, str):
        should_repair = should_repair.strip().lower() in {"1", "true", "yes", "y"}
    else:
        should_repair = bool(should_repair)
    audit_issues = audit.get("issues", [])
    if not isinstance(audit_issues, list):
        audit_issues = []
    has_flagged_audit_issues = False
    for issue in audit_issues:
        if not isinstance(issue, dict):
            continue
        severity = str(issue.get("severity", "")).strip().lower()
        if severity in {"medium", "high"}:
            has_flagged_audit_issues = True
            break
    audit_summary = str(audit.get("summary", "")).strip()
    llm_blocks = verdict == "fail" or (verdict == "unknown" and should_repair and has_flagged_audit_issues)
    ok = not blocking_issues and not llm_blocks
    summary_parts: list[str] = []
    if blocking_issues:
        summary_parts.append(f"{len(blocking_issues)} high-risk deterministic issue(s)")
    if llm_blocks:
        summary_parts.append(f"LLM verdict: {verdict}")
    elif verdict == "unknown":
        summary_parts.append("LLM verdict: unknown (treated as advisory)")
    if audit_summary:
        summary_parts.append(audit_summary)
    summary = "; ".join(summary_parts) if summary_parts else "semantic integrity check passed"
    return {
        "ok": ok,
        "stage": stage,
        "summary": summary,
        "verdict": verdict,
        "deterministic_issues": deterministic_issues,
        "blocking_issues": blocking_issues,
        "should_repair": should_repair,
        "audit": audit,
    }


def _semantic_failure_message(report: dict, stage: str) -> str:
    summary = str(report.get("summary", "")).strip()
    if summary:
        return f"LLM check ({stage}) failed: {summary}"
    return f"LLM check ({stage}) failed."


def _decl_statement_map(lean_code: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for decl in _extract_decl_blocks(lean_code):
        name = str(decl.get("name", "")).strip()
        statement = _normalize_statement_text(str(decl.get("statement", "")))
        if name and statement:
            mapping[name] = statement
    return mapping


def _normalize_statement_text(text: str) -> str:
    return " ".join(text.split())


def _normalize_decl_block_text(block: str) -> str:
    return " ".join(_strip_comments(block).split())


def _locked_declaration_map(lean_code: str) -> dict[str, dict[str, str]]:
    locked: dict[str, dict[str, str]] = {}
    for decl in _extract_decl_blocks(lean_code):
        name = str(decl.get("name", "")).strip()
        block = str(decl.get("block", ""))
        statement = _normalize_statement_text(str(decl.get("statement", "")))
        if not name or _decl_block_has_placeholder(block):
            continue
        locked[name] = {
            "statement": statement,
            "block": _normalize_decl_block_text(block),
        }
    return locked


def _context_decl_statement_map(paths: list[Path]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in paths:
        try:
            if path.suffix != ".lean" or not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for name, statement in _decl_statement_map(text).items():
            if name and statement and name not in mapping:
                mapping[name] = statement
    return mapping


def _locked_declaration_error(lean_code: str, locked_decls: dict[str, dict[str, str]]) -> str | None:
    if not locked_decls:
        return None
    current = {str(item.get("name", "")).strip(): item for item in _extract_decl_blocks(lean_code)}
    for name, locked in locked_decls.items():
        decl = current.get(name)
        if not decl:
            return f"Regression: previously proven declaration `{name}` was removed."
        current_statement = _normalize_statement_text(str(decl.get("statement", "")))
        locked_statement = str(locked.get("statement", "")).strip()
        if locked_statement and current_statement and current_statement != locked_statement:
            return f"Regression: previously proven declaration `{name}` changed statement."
        current_block = _normalize_decl_block_text(str(decl.get("block", "")))
        locked_block = str(locked.get("block", "")).strip()
        if locked_block and current_block != locked_block:
            return f"Regression: previously proven declaration `{name}` was modified."
    return None


def _context_declaration_conflict_error(lean_code: str, context_decls: dict[str, str]) -> str | None:
    if not context_decls:
        return None
    current = _decl_statement_map(lean_code)
    for name, current_statement in current.items():
        ctx_statement = context_decls.get(name)
        if not ctx_statement:
            continue
        if current_statement and current_statement != ctx_statement:
            return (
                f"Context conflict: declaration `{name}` differs from the same declaration "
                "in a provided .lean context file."
            )
    return None


def _progress_guard_error(
    lean_code: str,
    *,
    min_proven: int,
    locked_decls: dict[str, dict[str, str]],
    context_decls: dict[str, str],
) -> str | None:
    locked_error = _locked_declaration_error(lean_code, locked_decls)
    if locked_error:
        return locked_error
    context_error = _context_declaration_conflict_error(lean_code, context_decls)
    if context_error:
        return context_error
    proven = _count_proven_declarations(lean_code)
    if proven < min_proven:
        return f"Progress regression: proven declarations dropped from {min_proven} to {proven}."
    return None


def _progress_guard_context(
    base_context: str,
    locked_decls: dict[str, dict[str, str]],
    context_decls: dict[str, str],
    min_proven: int,
) -> str:
    notes: list[str] = []
    if locked_decls:
        names = sorted(locked_decls.keys())[:24]
        lines = "\n".join(f"- {name}" for name in names)
        suffix = "" if len(locked_decls) <= 24 else "\n- ..."
        notes.append(
            "Hard constraints for this update:\n"
            "- Keep these previously proven declarations unchanged:\n"
            f"{lines}{suffix}"
        )
    if context_decls:
        notes.append(
            "Hard constraints for this update:\n"
            "- If you define a declaration with a name present in provided .lean context files, "
            "its statement must match exactly."
        )
    notes.append(
        "Hard constraints for this update:\n"
        f"- Do not reduce the number of fully proven declarations below {min_proven}."
    )
    extra = "\n\n".join(notes)
    if not base_context.strip():
        return extra
    return base_context.rstrip() + "\n\n" + extra


def _normalize_rejection_reason(reason: str, max_chars: int = 300) -> str:
    text = " ".join(str(reason).split())
    if not text:
        return ""
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + " ..."
    return text


def _load_rejection_memory(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    memory: dict[str, list[str]] = {}
    for raw_name, raw_reasons in payload.items():
        name = str(raw_name).strip()
        if not name or not isinstance(raw_reasons, list):
            continue
        items: list[str] = []
        for raw_reason in raw_reasons:
            reason = _normalize_rejection_reason(str(raw_reason))
            if not reason:
                continue
            items.append(reason)
        if items:
            memory[name] = items[-5:]
    return memory


def _record_rejection_reason(
    memory: dict[str, list[str]],
    name: str,
    reason: str,
    *,
    max_reasons: int = 5,
    max_decls: int = 50,
) -> None:
    decl = str(name).strip()
    if not decl:
        return
    normalized = _normalize_rejection_reason(reason)
    if not normalized:
        return
    if decl not in memory and len(memory) >= max_decls:
        oldest = next(iter(memory), None)
        if oldest is not None:
            memory.pop(oldest, None)
    reasons = memory.setdefault(decl, [])
    if normalized in reasons:
        reasons.remove(normalized)
    reasons.append(normalized)
    if len(reasons) > max_reasons:
        del reasons[: len(reasons) - max_reasons]


def _persist_rejection_memory(
    artifact_dir: Path,
    round_dir: Path,
    memory: dict[str, list[str]],
) -> None:
    if not memory:
        return
    content = json.dumps(memory, indent=2, ensure_ascii=True)
    _write_artifact(artifact_dir / "rejection_memory.json", content)
    _write_artifact(round_dir / "rejection_memory.json", content)


def _context_with_rejection_feedback(base_context: str, target_name: str, reasons: list[str]) -> str:
    if not reasons:
        return base_context
    trimmed: list[str] = []
    for item in reasons[-4:]:
        text = str(item).strip()
        if not text:
            continue
        if len(text) > 300:
            text = text[:300].rstrip() + " ..."
        trimmed.append(text)
    if not trimmed:
        return base_context
    bullet_block = "\n".join(f"- {reason}" for reason in trimmed)
    feedback = (
        f"Validator feedback for `{target_name}`:\n"
        "Your previous candidate was rejected.\n"
        "Fix these issues and regenerate while keeping all locked declarations unchanged:\n"
        f"{bullet_block}"
    )
    if not base_context.strip():
        return feedback
    return base_context.rstrip() + "\n\n" + feedback


def _deterministic_semantic_issues(lean_code: str, baseline_decls: dict[str, str]) -> list[dict]:
    issues: list[dict] = []
    cleaned = _strip_comments(lean_code)

    axiom_decl_pattern = re.compile(
        r"^\s*(?:(?:private|protected|local|unsafe|noncomputable|scoped)\s+)*"
        r"(axiom|constant)\s+([A-Za-z_][A-Za-z0-9_']*)",
        re.M,
    )
    for match in axiom_decl_pattern.finditer(cleaned):
        qualifier = " ".join(match.group(0).strip().split()[:-2])
        issues.append(
            {
                "kind": "axiom_or_constant",
                "severity": "high",
                "evidence": match.group(0).strip(),
                "reason": (
                    "Introduces unproven assumptions/constants in generated Lean."
                    if not qualifier
                    else f"Introduces unproven assumption/constant via `{qualifier}` declaration."
                ),
            }
        )

    taut_pattern = re.compile(
        r"^\s*(def|abbrev)\s+([A-Za-z_][A-Za-z0-9_']*)[^\n]*:\s*Prop\s*:=\s*(True|False)\b",
        re.M,
    )
    for match in taut_pattern.finditer(cleaned):
        issues.append(
            {
                "kind": "tautological_prop_definition",
                "severity": "high",
                "evidence": match.group(0).strip(),
                "reason": "Prop definition collapses to True/False and may trivialize obligations.",
            }
        )

    issues.extend(_simple_def_integrity_issues(cleaned))

    current_decls = _decl_statement_map(lean_code)
    for name, baseline_statement in baseline_decls.items():
        current = current_decls.get(name)
        if current is None:
            issues.append(
                {
                    "kind": "declaration_removed",
                    "severity": "high",
                    "evidence": name,
                    "reason": "A declaration present in the initial draft was removed.",
                }
            )
            continue
        if baseline_statement and current and baseline_statement != current:
            issues.append(
                {
                    "kind": "statement_changed",
                    "severity": "medium",
                    "evidence": name,
                    "reason": "Declaration statement differs from initial draft; verify meaning was preserved.",
                }
            )

    return issues[:80]


def _simple_def_integrity_issues(cleaned: str) -> list[dict]:
    issues: list[dict] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line or not line.startswith(("def ", "abbrev ")):
            continue
        if ":=" not in line:
            continue
        m = re.match(r"(def|abbrev)\s+([A-Za-z_][A-Za-z0-9_']*)\s*(.*)", line)
        if not m:
            continue
        name = m.group(2)
        tail = m.group(3)
        has_params = any(ch in tail for ch in ("(", "{", "["))
        if not has_params:
            continue
        body = line.split(":=", 1)[1].strip()
        if not re.fullmatch(r"(True|False|[0-9]+(?:\.[0-9]+)?|[A-Za-z_][A-Za-z0-9_']*)", body):
            continue
        lower_name = name.lower()
        severity = "medium"
        if any(tok in lower_name for tok in ("prob", "measure", "independent")):
            severity = "high"
        if lower_name in {"ra", "r_a", "count", "cardinality"}:
            severity = "high"
        issues.append(
            {
                "kind": "suspicious_simple_definition",
                "severity": severity,
                "evidence": line,
                "reason": "Parameterized definition with a trivial body; may bypass mathematical content.",
            }
        )
    return issues


def _equivalence_check(segments, lean_code: str, llm: FormalizationLLM, context: str, config: FormalizationConfig):
    tex_segments = [seg for seg in segments if seg.kind in {"theorem", "lemma", "proposition", "corollary", "example"}]
    decls = _extract_decl_blocks(lean_code)
    pairs = _map_segments_to_decls(tex_segments, decls)
    results = []
    mismatches = []
    for tex_seg, decl in pairs:
        statement = decl.get("statement", "")
        tex_stmt = tex_seg.body.strip()
        if not statement or not tex_stmt:
            result = {"name": decl.get("name", ""), "match": "unknown", "reason": "missing statement"}
        else:
            result = llm.equivalence_check(tex_stmt, statement)
            result["name"] = decl.get("name", "")
        result["tex"] = tex_stmt[:500]
        result["lean"] = statement[:500]
        results.append(result)
        if result.get("match") in {"no", "unknown"}:
            mismatches.append({"name": decl.get("name", ""), "tex": tex_stmt})
    return results, mismatches


def _extract_decl_blocks(text: str) -> list[dict]:
    pattern = re.compile(r"\b(theorem|lemma|example|proposition|corollary)\s+([A-Za-z0-9_']+)")
    matches = list(pattern.finditer(text))
    blocks = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[start:end]
        name = match.group(2)
        blocks.append(
            {
                "name": name,
                "block": block,
                "statement": _decl_statement(block),
            }
        )
    return blocks


def _lemma_plan_marker(name: str) -> str:
    return f"ULAMAI_LEMMA_PLAN:{name}"


def _insert_lemma_plan_marker(text: str, name: str) -> str:
    marker = _lemma_plan_marker(name)
    if marker in text:
        return text
    block = f"/- {marker} -/\n"
    pattern = re.compile(
        rf"^\s*(theorem|lemma|example|proposition|corollary)\s+{re.escape(name)}\b",
        re.M,
    )
    match = pattern.search(text)
    if not match:
        return text
    return text[: match.start()] + block + text[match.start() :]


def _extract_theorem_statement_from_text(text: str, theorem: str) -> tuple[str, str]:
    original = _extract_original_statement(text)
    decl_match = re.search(
        rf"\b(theorem|lemma|example|proposition|corollary)\s+{re.escape(theorem)}\b",
        text,
    )
    if not decl_match:
        return "", original
    proof_match = re.search(r":=\s*by\b", text[decl_match.end():], re.S)
    if not proof_match:
        return "", original
    start = decl_match.end()
    end = decl_match.end() + proof_match.start()
    header = text[start:end]
    depth_paren = 0
    depth_brace = 0
    depth_bracket = 0
    colon_idx = None
    i = 0
    while i < len(header):
        ch = header[i]
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren = max(0, depth_paren - 1)
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace = max(0, depth_brace - 1)
        elif ch == "[":
            depth_bracket += 1
        elif ch == "]":
            depth_bracket = max(0, depth_bracket - 1)
        elif ch == ":" and i + 1 < len(header) and header[i + 1] == "=":
            i += 1
        elif ch == ":" and depth_paren == 0 and depth_brace == 0 and depth_bracket == 0:
            colon_idx = i
            break
        i += 1
    if colon_idx is None:
        return "", original
    stmt_raw = header[colon_idx + 1 :]
    stmt = " ".join(stmt_raw.split())
    return stmt, original


def _extract_original_statement(text: str) -> str:
    match = re.search(r"/-\s*ULAMAI_ORIGINAL_STATEMENT\s*(.*?)\s*-/", text, re.S)
    if not match:
        return ""
    return match.group(1).strip()


def _strip_imports_from_snippet(snippet: str) -> str:
    lines = []
    for line in snippet.splitlines():
        if line.strip().startswith("import "):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _insert_lemmas_before_in_text(
    text: str, target_name: str, blocks: list[dict]
) -> tuple[str, list[str]]:
    existing = set(_extract_decl_names(text))
    kept_blocks: list[str] = []
    new_names: list[str] = []
    for block in blocks:
        name = block.get("name") or ""
        if not name or name == target_name:
            continue
        if name in existing:
            continue
        block_text = block.get("block", "").rstrip()
        if not block_text:
            continue
        kept_blocks.append(block_text + "\n")
        new_names.append(name)

    if not kept_blocks:
        return text, []

    match = re.search(
        rf"\b(theorem|lemma|example|proposition|corollary)\s+{re.escape(target_name)}\b",
        text,
    )
    if match is None:
        return text, []

    insert_at = match.start()
    insert_block = "\n".join(kept_blocks).rstrip() + "\n\n"
    new_text = text[:insert_at] + insert_block + text[insert_at:]
    return new_text, new_names


def _lemma_trace_path(name: str) -> Path:
    stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    return Path("runs") / f"trace_{safe}_{stamp}.jsonl"


def _read_trace_steps(path: Path, max_lines: int = 200) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if max_lines and len(lines) > max_lines:
        lines = lines[-max_lines:]
    steps: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            steps.append(json.loads(line))
        except Exception:
            continue
    return steps


def _summarize_trace_steps(steps: list[dict]) -> tuple[str, list[str], list[str]]:
    last_goal = ""
    failures: list[str] = []
    successes: list[str] = []
    for step in steps:
        tactic = step.get("tactic", "")
        ok = step.get("ok", False)
        if step.get("state_pretty"):
            last_goal = step["state_pretty"]
        if ok:
            if tactic:
                successes.append(tactic)
            continue
        err = step.get("error") or "error"
        if tactic:
            failures.append(f"{tactic}: {err}")
        else:
            failures.append(str(err))
    return last_goal, failures, successes


def _expand_lemmas_for_failure(
    lean_code: str,
    lemma_name: str,
    trace_path: Path | None,
    context: str,
    llm: FormalizationLLM,
) -> tuple[str, list[str]] | None:
    lemma_stmt, original_stmt = _extract_theorem_statement_from_text(lean_code, lemma_name)
    if not lemma_stmt:
        return None
    steps = _read_trace_steps(trace_path) if trace_path else []
    last_goal, failures, successes = _summarize_trace_steps(steps)
    snippet = llm.expand_lemmas(
        lemma_name=lemma_name,
        lemma_statement=lemma_stmt,
        last_goal=last_goal,
        failures=failures,
        successes=successes,
        context=context,
    )
    if not snippet.strip():
        return None
    snippet = _strip_imports_from_snippet(snippet)
    blocks = _extract_decl_blocks(snippet)
    updated, new_names = _insert_lemmas_before_in_text(lean_code, lemma_name, blocks)
    if not new_names:
        return None
    return updated, new_names


def _decl_statement(block: str) -> str:
    split = re.search(r"\bwhere\b|:=", block)
    if split:
        return block[: split.start()].strip()
    return block.strip()


def _map_segments_to_decls(segments, decls: list[dict]) -> list[tuple]:
    pairs = []
    used = set()
    for seg in segments:
        if seg.title:
            for decl in decls:
                if decl["name"] == seg.title or seg.title in decl["name"]:
                    pairs.append((seg, decl))
                    used.add(decl["name"])
                    break
    for seg in segments:
        if any(seg is pair[0] for pair in pairs):
            continue
        for decl in decls:
            if decl["name"] in used:
                continue
            pairs.append((seg, decl))
            used.add(decl["name"])
            break
    return pairs
