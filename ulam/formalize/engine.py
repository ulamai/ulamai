from __future__ import annotations

import datetime as dt
import difflib
import json
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

        rounds = 0
        repairs = 0
        equiv_repairs = 0
        solved = 0
        typecheck_ok = False
        while rounds < self._config.max_rounds:
            rounds += 1
            _log(self._config, f"[formalize] round {rounds}/{self._config.max_rounds}")
            _log_progress(self._config, initial_sorries, _count_sorries(lean_code))
            round_dir = _ensure_round_dir(artifact_dir, rounds)
            _write_artifact(round_dir / "start.lean", lean_code)
            self._write_output(lean_code)
            check_error = _typecheck(lean_code, self._config)
            if check_error:
                _log(self._config, f"[typecheck] error: {check_error[:200]}")
                _write_artifact(round_dir / "typecheck_error.txt", check_error)
                repairs += 1
                if repairs > self._config.max_repairs:
                    return FormalizationResult(
                        output_path=self._config.output_path,
                        rounds=rounds,
                        typecheck_ok=False,
                        solved=0,
                        remaining_sorries=_count_sorries(lean_code),
                        error="max repairs reached",
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
                repaired = self._llm.repair(lean_code, check_error, context)
                if repaired.strip() and repaired != lean_code:
                    repaired = _normalize_lean_output(repaired)
                    repaired = _inject_tex_snippets(repaired, segments)
                    _write_artifact(round_dir / "repair.lean", repaired)
                    _write_diff(round_dir / "repair.diff", lean_code, repaired)
                    lean_code = repaired
                else:
                    break
                if not lean_code.strip():
                    break
                continue

            typecheck_ok = True
            if self._config.equivalence_checks:
                eq_results, mismatches = _equivalence_check(
                    segments, lean_code, self._llm, context, self._config
                )
                _write_artifact(round_dir / "equivalence.json", json.dumps(eq_results, indent=2))
                if mismatches and equiv_repairs < self._config.max_equivalence_repairs:
                    _log(self._config, f"[equivalence] {len(mismatches)} mismatches")
                    did_repair = False
                    for idx, mismatch in enumerate(mismatches, start=1):
                        if equiv_repairs >= self._config.max_equivalence_repairs:
                            break
                        _log(self._config, f"[equivalence] repair {idx}: {mismatch['name']}")
                        repaired = self._llm.repair_statement(
                            lean_code, mismatch["name"], mismatch["tex"], context
                        )
                        if repaired.strip() and repaired != lean_code:
                            _write_artifact(round_dir / f"equivalence_repair_{idx}.lean", repaired)
                            _write_diff(
                                round_dir / f"equivalence_repair_{idx}.diff", lean_code, repaired
                            )
                            lean_code = _normalize_lean_output(repaired)
                            lean_code = _inject_tex_snippets(lean_code, segments)
                            equiv_repairs += 1
                            did_repair = True
                    if did_repair:
                        continue

            solved, failures, lean_code = _attempt_proofs(
                lean_code, self._config, context, segments, self._llm
            )
            if failures:
                _write_artifact(round_dir / "proof_failures.json", json.dumps(failures, indent=2))
            if not failures:
                if typecheck_ok and _count_sorries(lean_code) == 0:
                    axiom_error = _axiom_guardrail_error(lean_code, self._config.allow_axioms)
                    if axiom_error:
                        _log(self._config, f"[axiom] {axiom_error}")
                        repaired = self._llm.repair(lean_code, axiom_error, context)
                        if repaired.strip() and repaired != lean_code:
                            repaired = _normalize_lean_output(repaired)
                            repaired = _inject_tex_snippets(repaired, segments)
                            _write_artifact(round_dir / "axiom_repair.lean", repaired)
                            _write_diff(round_dir / "axiom_repair.diff", lean_code, repaired)
                            lean_code = repaired
                            continue
                break
            if rounds >= self._config.max_rounds:
                break
            _log(self._config, "[improve] requesting LLM improvements")
            improved = self._llm.improve(lean_code, failures, context)
            if improved.strip() and improved != lean_code:
                improved = _normalize_lean_output(improved)
                improved = _inject_tex_snippets(improved, segments)
                _write_artifact(round_dir / "improve.lean", improved)
                _write_diff(round_dir / "improve.diff", lean_code, improved)
                lean_code = improved
            else:
                break
            if not lean_code.strip():
                break

        self._write_output(lean_code)
        _log_progress(self._config, initial_sorries, _count_sorries(lean_code))
        return FormalizationResult(
            output_path=self._config.output_path,
            rounds=rounds,
            typecheck_ok=typecheck_ok,
            solved=solved,
            remaining_sorries=_count_sorries(lean_code),
            error=None if typecheck_ok else "typecheck failed",
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
        return _attempt_proofs_llm(lean_code, config, context, segments, llm)
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
) -> tuple[int, list[str], str]:
    names = _extract_decl_names(lean_code)
    failures: list[str] = []
    solved = 0
    rounds = 0
    pending = [name for name in names if _decl_has_placeholder(lean_code, name)]
    tex_snippets = _build_tex_snippet_map(segments, lean_code)

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
            success = False
            attempts = max(1, int(config.proof_repair) + 1)
            for attempt in range(1, attempts + 1):
                _log(config, "[llm] requesting proof update")
                try:
                    updated = llm.prove(
                        lean_code=lean_code,
                        name=name,
                        instruction="",
                        tex_snippet=snippet,
                        context=context,
                        error=error,
                    )
                except Exception as exc:
                    _log(config, f"[llm] error: {exc}")
                    break
                if not updated.strip():
                    break
                updated = _normalize_lean_output(updated)
                updated = _inject_tex_snippets(updated, segments)
                config.output_path.write_text(updated, encoding="utf-8")
                lean_code = updated
                if _decl_has_placeholder(updated, name):
                    error = f"Declaration `{name}` still contains sorry/admit."
                    continue
                check_error = _typecheck(updated, config)
                if check_error:
                    error = check_error
                    continue
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

    llm_client = _make_llm_client(config)
    runner = LeanDojoRunner(
        project_path=config.lean_project,
        imports=config.lean_imports,
        timeout_s=config.dojo_timeout_s,
    )
    trace = TraceLogger(trace_path)
    error: str | None = None
    result = None
    try:
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
            autop=True,
            instruction=instruction,
            context=None,
            verbose=config.verbose,
        )
        result = scripted_search(runner, llm_client, _null_retriever(), trace, run_config)
    except Exception as exc:
        error = str(exc)
    finally:
        trace.close()
        runner.close()

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
    raise RuntimeError("No LLM provider configured")


def _typecheck(lean_code: str, config: FormalizationConfig) -> Optional[str]:
    if config.lean_backend == "cli":
        from ..lean.cli_check import lean_cli_check

        project_path = config.lean_project or _find_lean_project(config.output_path)
        return lean_cli_check(config.output_path, project_path=project_path, timeout_s=60.0)

    if not config.lean_project:
        return None
    try:
        from pantograph import Server  # type: ignore
    except ImportError:
        return "Pantograph not installed."

    from ..lean.dojo import _create_server, _find_project_root  # type: ignore

    project_path = config.lean_project or _find_project_root(config.output_path)
    parsed_imports, body = _split_imports_for_typecheck(lean_code)
    imports = _merge_imports(config.lean_imports, parsed_imports)
    server = _create_server(Server, project_path, imports, timeout_s=config.dojo_timeout_s)
    try:
        server.load_sorry(body)
        return None
    except Exception as exc:  # pragma: no cover
        return str(exc)
    finally:
        try:
            server.close()
        except Exception:
            pass


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


def _write_output_inline(config: FormalizationConfig, lean_code: str) -> None:
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(_normalize_lean_output(lean_code), encoding="utf-8")


def _log_progress(config: FormalizationConfig, total: int, remaining: int) -> None:
    if total <= 0:
        return
    solved = max(0, total - remaining)
    pct = int(round((solved / total) * 100))
    _log(config, f"[progress] solved {solved}/{total} ({pct}%), remaining {remaining}")


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
    }


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
