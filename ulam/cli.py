from __future__ import annotations

import argparse
import statistics
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from .lean.base import LeanRunner
from .lean.dojo import LeanDojoRunner
from .lean.mock import MockLeanRunner
from .llm import (
    AnthropicClient,
    ClaudeCLIClient,
    CodexCLIClient,
    GeminiCLIClient,
    GeminiClient,
    MockLLMClient,
    OllamaClient,
    OpenAICompatClient,
)
from .formalize.engine import FormalizationEngine
from .formalize.llm import FormalizationLLM
from .formalize.types import FormalizationConfig
from .config import load_config, save_config
from .auth import (
    codex_auth_path,
    load_codex_api_key,
    load_codex_tokens,
    run_codex_login,
    run_claude_setup_token,
    run_claude_login,
    run_gemini_login,
)
from .retrieve import (
    EmbeddingRetriever,
    NullRetriever,
    OpenAIEmbeddingClient,
    SimpleRetriever,
)
from .search import best_first_search, scripted_search
from .trace import TraceLogger
from .types import RunConfig


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        from .menu import run_menu

        run_menu()
        return
    if _has_lean_setup_flag(argv):
        run_lean_setup(_parse_lean_setup_args(_strip_lean_setup_flags(argv)))
        return
    parser = argparse.ArgumentParser(prog="ulam", description="Ulam Prover CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    prove = sub.add_parser("prove", help="attempt to prove a Lean theorem")
    prove.add_argument("file", type=Path, help="path to a Lean file")
    prove.add_argument("--theorem", required=True, help="theorem name to prove")
    prove.add_argument(
        "--llm",
        choices=[
            "mock",
            "openai",
            "ollama",
            "anthropic",
            "codex_cli",
            "claude_cli",
            "gemini",
            "gemini_cli",
        ],
        default="mock",
    )
    prove.add_argument("--lean", choices=["mock", "dojo", "cli"], default="mock")
    prove.add_argument(
        "--lean-project",
        type=Path,
        default=Path(os.environ["ULAM_LEAN_PROJECT"]) if "ULAM_LEAN_PROJECT" in os.environ else None,
        help="Lean project root (defaults to nearest lakefile.lean/lean-toolchain)",
    )
    prove.add_argument(
        "--lean-import",
        action="append",
        default=[],
        help="additional Lean modules to import (repeatable)",
    )
    prove.add_argument("--premises", type=Path, default=None, help="optional premises file")
    prove.add_argument("--instruction", default="", help="natural language guidance for the prover")
    prove.add_argument(
        "--context",
        action="append",
        default=[],
        help="additional context files (.lean/.tex), repeatable",
    )
    prove.add_argument(
        "--retriever",
        choices=["none", "simple", "embedding"],
        default="simple",
        help="retriever type (requires --premises for simple/embedding)",
    )
    prove.add_argument(
        "--retriever-k",
        type=int,
        default=8,
        help="number of retrieved premises per state",
    )
    prove.add_argument(
        "--embed-api-key",
        default=os.environ.get("ULAM_EMBED_API_KEY", os.environ.get("ULAM_OPENAI_API_KEY", "")),
    )
    prove.add_argument(
        "--embed-base-url",
        default=os.environ.get("ULAM_EMBED_BASE_URL", os.environ.get("ULAM_OPENAI_BASE_URL", "https://api.openai.com")),
    )
    prove.add_argument(
        "--embed-model",
        default=os.environ.get("ULAM_EMBED_MODEL", "text-embedding-3-small"),
    )
    prove.add_argument("--embed-cache", type=Path, default=None)
    prove.add_argument("--embed-batch-size", type=int, default=16)
    prove.add_argument("--max-steps", type=int, default=64)
    prove.add_argument("--beam", type=int, default=4)
    prove.add_argument("--k", type=int, default=1, help="suggestions per state")
    prove.add_argument("--llm-rounds", type=int, default=4, help="LLM-only max rounds")
    prove.add_argument("--timeout", type=float, default=5.0, help="tactic timeout (seconds)")
    prove.add_argument("--repair", type=int, default=2, help="repair attempts per failure")
    prove.add_argument(
        "--allow-axioms",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="allow axioms/constants anywhere (default: enabled)",
    )
    prove.add_argument("--no-autop", action="store_true", help="disable autop fallback tactics")
    prove.add_argument("--seed", type=int, default=0)
    prove.add_argument("--trace", type=Path, default=Path("run.jsonl"))
    prove.add_argument("--verbose", action="store_true", help="print search/LLM logs")
    prove.add_argument(
        "--prove-mode",
        choices=["tactic", "lemma", "llm"],
        default="tactic",
        help="proof search mode",
    )
    prove.add_argument(
        "--solver",
        choices=["auto", "search", "script", "portfolio"],
        default="script",
        help="tactic solver (auto=script for lemma-first, search otherwise)",
    )
    prove.add_argument(
        "--lemma-max",
        type=int,
        default=60,
        help="maximum number of lemmas (lemma-first mode)",
    )
    prove.add_argument(
        "--lemma-depth",
        type=int,
        default=60,
        help="maximum lemma expansion depth (lemma-first mode)",
    )

    prove.add_argument("--openai-key", default=os.environ.get("ULAM_OPENAI_API_KEY", ""))
    prove.add_argument(
        "--openai-base-url",
        default=os.environ.get("ULAM_OPENAI_BASE_URL", "https://api.openai.com"),
    )
    prove.add_argument("--openai-model", default=os.environ.get("ULAM_OPENAI_MODEL", "gpt-4.1"))

    prove.add_argument(
        "--ollama-base-url",
        default=os.environ.get("ULAM_OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    prove.add_argument("--ollama-model", default=os.environ.get("ULAM_OLLAMA_MODEL", "llama3.1"))
    prove.add_argument("--anthropic-key", default=os.environ.get("ULAM_ANTHROPIC_API_KEY", ""))
    prove.add_argument("--anthropic-setup-token", default=os.environ.get("ULAM_ANTHROPIC_SETUP_TOKEN", ""))
    prove.add_argument(
        "--anthropic-base-url",
        default=os.environ.get("ULAM_ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
    )
    prove.add_argument("--anthropic-model", default=os.environ.get("ULAM_ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620"))
    prove.add_argument(
        "--gemini-api-key",
        default=os.environ.get("ULAM_GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", "")),
    )
    prove.add_argument(
        "--gemini-base-url",
        default=os.environ.get(
            "ULAM_GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai",
        ),
    )
    prove.add_argument(
        "--gemini-model",
        default=os.environ.get("ULAM_GEMINI_MODEL", "gemini-3-pro-preview"),
    )

    replay = sub.add_parser("replay", help="replay or summarize a run trace")
    replay.add_argument("trace", type=Path, help="trace jsonl path")

    auth = sub.add_parser("auth", help="authenticate with Codex, Claude, or Gemini CLI")
    auth.add_argument("provider", choices=["codex", "claude", "gemini"])

    formalize = sub.add_parser("formalize", help="formalize a .tex document to Lean")
    formalize.add_argument("tex", type=Path, help="path to .tex file")
    formalize.add_argument("--out", type=Path, default=None, help="output .lean path")
    formalize.add_argument("--context", action="append", default=[], help="context files (.lean/.tex)")
    formalize.add_argument("--max-rounds", type=int, default=5)
    formalize.add_argument(
        "--max-repairs",
        type=int,
        default=None,
        help="max typecheck repairs (default: same as --max-rounds)",
    )
    formalize.add_argument("--max-equivalence-repairs", type=int, default=2)
    formalize.add_argument("--max-proof-rounds", type=int, default=1)
    formalize.add_argument("--proof-max-steps", type=int, default=64)
    formalize.add_argument("--proof-beam", type=int, default=4)
    formalize.add_argument("--proof-k", type=int, default=1)
    formalize.add_argument("--proof-timeout", type=float, default=5.0)
    formalize.add_argument("--proof-repair", type=int, default=2)
    formalize.add_argument(
        "--allow-axioms",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="allow axioms/constants anywhere (default: enabled)",
    )
    formalize.add_argument(
        "--segment",
        action="store_true",
        help="segment long TeX and formalize piece-wise",
    )
    formalize.add_argument(
        "--segment-words",
        type=int,
        default=1000,
        help="word threshold/chunk size for segmentation",
    )
    formalize.add_argument("--lean-project", type=Path, default=None)
    formalize.add_argument("--lean-import", action="append", default=[])
    formalize.add_argument(
        "--proof-backend",
        choices=["tactic", "lemma", "llm", "dojo"],
        default="tactic",
        help="proof backend (tactic/lemma use LeanDojo, llm uses Lean CLI)",
    )
    formalize.add_argument(
        "--lean-backend",
        choices=["dojo", "cli"],
        default="dojo",
        help="typecheck backend (dojo uses Pantograph, cli uses lake/lean)",
    )
    formalize.add_argument("--no-equivalence", action="store_true", help="skip equivalence checks")
    formalize.add_argument("--artifacts-dir", type=Path, default=None)
    formalize.add_argument("--verbose", action="store_true")

    bench = sub.add_parser("bench", help="run a regression suite")
    bench.add_argument("--suite", type=Path, required=True, help="jsonl suite path")
    bench.add_argument(
        "--llm",
        choices=[
            "mock",
            "openai",
            "ollama",
            "anthropic",
            "codex_cli",
            "claude_cli",
            "gemini",
            "gemini_cli",
        ],
        default="mock",
    )
    bench.add_argument("--lean", choices=["mock", "dojo"], default="mock")
    bench.add_argument(
        "--lean-project",
        type=Path,
        default=Path(os.environ["ULAM_LEAN_PROJECT"]) if "ULAM_LEAN_PROJECT" in os.environ else None,
    )
    bench.add_argument("--lean-import", action="append", default=[])
    bench.add_argument("--premises", type=Path, default=None)
    bench.add_argument("--instruction", default="", help="natural language guidance for the prover")
    bench.add_argument(
        "--context",
        action="append",
        default=[],
        help="additional context files (.lean/.tex), repeatable",
    )
    bench.add_argument(
        "--retriever",
        choices=["none", "simple", "embedding"],
        default="simple",
    )
    bench.add_argument("--retriever-k", type=int, default=8)
    bench.add_argument(
        "--embed-api-key",
        default=os.environ.get("ULAM_EMBED_API_KEY", os.environ.get("ULAM_OPENAI_API_KEY", "")),
    )
    bench.add_argument(
        "--embed-base-url",
        default=os.environ.get("ULAM_EMBED_BASE_URL", os.environ.get("ULAM_OPENAI_BASE_URL", "https://api.openai.com")),
    )
    bench.add_argument(
        "--embed-model",
        default=os.environ.get("ULAM_EMBED_MODEL", "text-embedding-3-small"),
    )
    bench.add_argument("--embed-cache", type=Path, default=None)
    bench.add_argument("--embed-batch-size", type=int, default=16)
    bench.add_argument("--max-steps", type=int, default=64)
    bench.add_argument("--beam", type=int, default=4)
    bench.add_argument("--k", type=int, default=1, help="suggestions per state")
    bench.add_argument("--timeout", type=float, default=5.0)
    bench.add_argument("--repair", type=int, default=2)
    bench.add_argument(
        "--solver",
        choices=["search", "script", "portfolio"],
        default="search",
    )
    bench.add_argument("--no-autop", action="store_true", help="disable autop fallback tactics")
    bench.add_argument("--seed", type=int, default=0)
    bench.add_argument("--trace-dir", type=Path, default=Path("bench_traces"))
    bench.add_argument("--verbose", action="store_true", help="print search/LLM logs")

    bench.add_argument("--openai-key", default=os.environ.get("ULAM_OPENAI_API_KEY", ""))
    bench.add_argument(
        "--openai-base-url",
        default=os.environ.get("ULAM_OPENAI_BASE_URL", "https://api.openai.com"),
    )
    bench.add_argument("--openai-model", default=os.environ.get("ULAM_OPENAI_MODEL", "gpt-4.1"))

    bench.add_argument(
        "--ollama-base-url",
        default=os.environ.get("ULAM_OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    bench.add_argument("--ollama-model", default=os.environ.get("ULAM_OLLAMA_MODEL", "llama3.1"))
    bench.add_argument("--anthropic-key", default=os.environ.get("ULAM_ANTHROPIC_API_KEY", ""))
    bench.add_argument("--anthropic-setup-token", default=os.environ.get("ULAM_ANTHROPIC_SETUP_TOKEN", ""))
    bench.add_argument(
        "--anthropic-base-url",
        default=os.environ.get("ULAM_ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
    )
    bench.add_argument(
        "--anthropic-model",
        default=os.environ.get("ULAM_ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620"),
    )
    bench.add_argument(
        "--gemini-api-key",
        default=os.environ.get("ULAM_GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", "")),
    )
    bench.add_argument(
        "--gemini-base-url",
        default=os.environ.get(
            "ULAM_GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai",
        ),
    )
    bench.add_argument(
        "--gemini-model",
        default=os.environ.get("ULAM_GEMINI_MODEL", "gemini-3-pro-preview"),
    )

    lean_setup = sub.add_parser(
        "lean-setup", help="install Lean + LeanDojo and create a Lean project"
    )
    _add_lean_setup_args(lean_setup)

    args = parser.parse_args(argv)

    if args.command == "prove":
        run_prove(args)
        return
    if args.command == "replay":
        run_replay(args)
        return
    if args.command == "auth":
        run_auth(args)
        return
    if args.command == "formalize":
        run_formalize(args)
        return
    if args.command == "bench":
        run_bench(args)
        return
    if args.command == "lean-setup":
        run_lean_setup(args)
        return


def run_prove(args: argparse.Namespace) -> None:
    allow_axioms = _resolve_allow_axioms(args)
    context = _read_context_files(args.context)
    config = RunConfig(
        file_path=args.file,
        theorem=args.theorem,
        max_steps=args.max_steps,
        beam_width=args.beam,
        suggestions_per_state=args.k,
        timeout_s=args.timeout,
        repair_attempts=args.repair,
        seed=args.seed,
        trace_path=args.trace,
        retriever_k=max(1, int(getattr(args, "retriever_k", 8))),
        autop=_autop_enabled(args),
        instruction=args.instruction.strip() if args.instruction else None,
        context=context,
        verbose=bool(args.verbose),
    )
    _preflight_lean_alignment(args)

    def _run_once() -> SearchResult:
        solver = _resolve_solver(args)
        if config.verbose:
            model = ""
            if args.llm in {"openai", "codex_cli"}:
                model = args.openai_model
            elif args.llm in {"anthropic", "claude_cli"}:
                model = args.anthropic_model
            elif args.llm in {"gemini", "gemini_cli"}:
                model = args.gemini_model
            elif args.llm == "ollama":
                model = args.ollama_model
            label = args.llm
            if model:
                label = f"{label} ({model})"
            print(f"[run] llm={label}")
            print(f"[run] lean={args.lean} file={args.file} theorem={args.theorem}")
            if args.instruction:
                lines = len(args.instruction.splitlines())
                print(f"[run] instruction_lines={lines}")
            if args.context:
                print(f"[run] context_files={len(args.context)}")
            print(f"[run] trace={config.trace_path}")
            print(f"[run] solver={solver}")
            print(f"[run] autop={'on' if config.autop else 'off'}")

        runner = _make_runner(args)
        llm = _make_llm(args)
        retriever = _make_retriever(args)
        trace = TraceLogger(config.trace_path)
        try:
            return _run_with_solver(solver, runner, llm, retriever, trace, config)
        finally:
            trace.close()
            runner.close()

    try:
        if args.prove_mode == "llm":
            run_prove_llm(args)
            return
        if args.prove_mode == "lemma":
            run_prove_lemma_first(args)
            return
        result = _run_once()
    except Exception as exc:
        if args.lean == "dojo" and _is_parse_error(exc):
            if _attempt_statement_repair(args, exc):
                print("Retrying prover after statement repair...")
                result = _run_once()
            else:
                raise
        elif args.lean == "dojo" and _is_lean_mismatch_error(exc):
            print("Lean toolchain mismatch detected. Attempting to repair...")
            if _repair_lean_toolchain(args):
                print("Retrying prover after toolchain repair...")
                result = _run_once()
            else:
                raise
        else:
            raise

    if result.solved:
        print("Solved.")
        print("Proof:")
        print("by")
        for line in result.proof:
            print(f"  {line}")
        if _write_proof_to_file(args.file, args.theorem, result.proof):
            print(f"Wrote proof to: {args.file}")
        try:
            updated = args.file.read_text(encoding="utf-8")
        except Exception:
            updated = ""
        if updated:
            axiom_error = _axiom_guardrail_error(updated, allow_axioms)
            if axiom_error:
                print(f"[axiom] {axiom_error}")
        return

    if _resolve_solver(args) == "portfolio" and args.prove_mode == "tactic":
        print("[portfolio] search stages failed; trying LLM-only fallback.")
        if run_prove_llm(args):
            return

    print("Failed to solve.")
    print(result.error or "unknown error")
    _summarize_failed_run(args)


def run_prove_llm(args: argparse.Namespace) -> bool:
    allow_axioms = _resolve_allow_axioms(args)
    try:
        text = args.file.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"Failed to read Lean file: {exc}")
        return False
    if not _file_has_decl(text, args.theorem):
        print(f"Could not find `{args.theorem}` in {args.file}")
        _suggest_proof_targets(args.file, args.theorem)
        return False

    config = _llm_config_from_args(args)
    llm = FormalizationLLM(args.llm, config)
    instruction = args.instruction.strip() if args.instruction else ""
    context = "\n\n".join(_read_context_files(args.context))
    max_rounds = max(1, int(getattr(args, "llm_rounds", 4)))
    project = args.lean_project or _find_lean_project_for_file(args.file)
    if project:
        print(f"[llm] using Lean project: {project}")
    else:
        print("[llm] no Lean project detected; attempting Lean CLI directly.")

    from .lean.cli_check import lean_cli_check

    error: str | None = None
    error_counts: dict[str, int] = {}
    for round_idx in range(1, max_rounds + 1):
        print(f"[llm] round {round_idx}/{max_rounds}")
        tex_snippet = _extract_tex_snippet(text, args.theorem)
        print("[llm] requesting proof update...")
        try:
            updated = llm.prove(
                lean_code=text,
                name=args.theorem,
                instruction=instruction,
                tex_snippet=tex_snippet,
                context=context,
                error=error,
            )
        except Exception as exc:
            print(f"[llm] error: {exc}")
            break
        if not updated.strip():
            print("LLM returned empty output.")
            break
        updated = _normalize_llm_output(updated)
        if updated.strip() == text.strip():
            print("[stagnation] LLM returned no effective code changes.")
            break
        args.file.write_text(updated, encoding="utf-8")
        text = updated
        if _decl_has_placeholder(updated, args.theorem):
            error = f"Declaration `{args.theorem}` still contains sorry/admit."
            print(f"[typecheck] {error}")
            count = _record_error_count(error_counts, error)
            if count >= 3:
                print("[stagnation] same error repeated multiple rounds; stopping.")
                break
            continue
        check_error = lean_cli_check(
            args.file,
            project_path=project,
            timeout_s=max(30.0, float(args.timeout)),
        )
        if check_error:
            error = check_error
            print(f"[typecheck] error: {check_error[:200]}")
            count = _record_error_count(error_counts, check_error)
            if count >= 3:
                print("[stagnation] same typecheck error repeated multiple rounds; stopping.")
                break
            continue
        axiom_error = _axiom_guardrail_error(updated, allow_axioms)
        if axiom_error:
            error = axiom_error
            print(f"[axiom] {axiom_error}")
            count = _record_error_count(error_counts, axiom_error)
            if count >= 3:
                print("[stagnation] same axiom guardrail error repeated multiple rounds; stopping.")
                break
            continue
        print("Solved.")
        print(f"Wrote proof to: {args.file}")
        return True

    print("Failed to solve with LLM-only mode.")
    return False


def run_prove_lemma_first(args: argparse.Namespace) -> None:
    print("Lemma-first mode: generating lemma plan...")
    _preflight_lean_alignment(args)
    file_text = ""
    try:
        file_text = args.file.read_text(encoding="utf-8")
    except Exception:
        pass
    has_decl = _file_has_decl(file_text, args.theorem) if file_text else False
    if not _has_lemma_plan(args.file):
        if has_decl:
            print("Using existing declarations; skipping lemma plan generation.")
            _update_lemma_list_block(args.file)
        else:
            stmt, _ = _extract_theorem_statement(args.file, args.theorem)
            if not stmt:
                _suggest_proof_targets(args.file, args.theorem)
                return
            try:
                plan = _generate_lemma_plan(args)
            except Exception as exc:
                print(f"Failed to generate lemma plan: {exc}")
                return
            if not plan.declarations:
                print("Lemma plan produced no declarations. Aborting.")
                return
            if not _write_lemma_plan(args, plan):
                print("Failed to write lemma plan to file.")
                return
    if not file_text:
        try:
            file_text = args.file.read_text(encoding="utf-8")
        except Exception:
            print("Failed to read lemma plan file.")
            return

    decls = _extract_decl_names(file_text)
    if not decls:
        print("No lemma/theorem declarations found in plan.")
        return

    max_lemmas = max(1, int(getattr(args, "lemma_max", 60)))
    max_depth = max(1, int(getattr(args, "lemma_depth", 60)))
    queue: list[tuple[str, int]] = [
        (name, 0) for name in decls if _decl_needs_proof(file_text, name)
    ]
    solved: set[str] = set()
    attempts: dict[str, int] = {}
    if not queue:
        print("All lemmas already solved.")
        return
    total_decl_count = _count_decl_names(args.file)
    completed = total_decl_count - len(queue)

    while queue:
        print(f"[lemma-first] Progress {completed}/{total_decl_count} solved")
        name, depth = queue.pop(0)
        if name in solved:
            continue
        attempts[name] = attempts.get(name, 0) + 1
        print(f"[lemma-first] Proving {name}...")
        result = _run_search_for_theorem(args, name)
        if result is None:
            print("Aborted.")
            return
        if result.solved:
            _write_proof_to_file(args.file, name, result.proof)
            solved.add(name)
            completed += 1
            continue

        print(f"Failed to solve lemma {name}.")
        total_lemmas = _count_decl_names(args.file)
        if total_lemmas >= max_lemmas or depth >= max_depth:
            print("Lemma-first limits reached; stopping.")
            _summarize_failed_run_for(
                args,
                name,
                _trace_path_for_theorem(args, name),
                note=f"lemma-first limits reached (lemmas={total_lemmas}, depth={depth})",
            )
            _cleanup_failed_lemmas(args.file)
            return

        expanded = _expand_lemmas_for_failure(args, name)
        if not expanded:
            _summarize_failed_run_for(
                args, name, _trace_path_for_theorem(args, name)
            )
            _cleanup_failed_lemmas(args.file)
            return
        for new_name in reversed(expanded):
            queue.insert(0, (new_name, depth + 1))
        queue.insert(0, (name, depth))

    print("Lemma-first mode complete.")


@dataclass(frozen=True)
class LemmaPlan:
    lean_code: str
    declarations: list[str]


LEMMA_PLAN_MARKER = "ULAMAI_LEMMA_PLAN"


def _generate_lemma_plan(args: argparse.Namespace) -> LemmaPlan:
    theorem_stmt, original_stmt = _extract_theorem_statement(args.file, args.theorem)
    if not theorem_stmt:
        raise RuntimeError("Could not read theorem statement.")
    config = _formalization_config_from_args(args)
    llm = FormalizationLLM(args.llm, config)
    context = "\n\n".join(_read_context_files(args.context))
    plan_code = llm.plan_lemmas(
        theorem_name=args.theorem,
        theorem_statement=theorem_stmt,
        original_statement=original_stmt or theorem_stmt,
        context=context,
    )
    normalized = _normalize_plan_code(plan_code, args.theorem, theorem_stmt, original_stmt)
    decls = _extract_decl_names(normalized)
    return LemmaPlan(lean_code=normalized, declarations=decls)


def _normalize_plan_code(code: str, theorem: str, statement: str, original: str | None) -> str:
    text = (code or "").strip()
    if not text:
        text = f"import Mathlib\n\n" f"theorem {theorem} : {statement} := by\n  sorry\n"
    if "import" not in text.splitlines()[0:5]:
        text = "import Mathlib\n\n" + text
    if not re.search(rf"\b(theorem|lemma|example)\s+{re.escape(theorem)}\b", text):
        text = (
            text
            + "\n\n"
            + f"theorem {theorem} : {statement} := by\n"
            + "  sorry\n"
        )
    if original:
        header = (
            "/- ULAMAI_ORIGINAL_STATEMENT\n"
            + original.strip()
            + "\n-/\n\n"
            + f"/- {LEMMA_PLAN_MARKER} -/\n\n"
        )
        if "ULAMAI_ORIGINAL_STATEMENT" not in text:
            text = header + text
    text = _ensure_lemma_list_block(text)
    return text


def _extract_decl_names(text: str) -> list[str]:
    names: list[str] = []
    for match in re.finditer(r"\b(?:theorem|lemma|example)\s+([A-Za-z0-9_']+)\b", text):
        names.append(match.group(1))
    return names


def _decl_needs_proof(text: str, name: str) -> bool:
    block = _decl_block(text, name)
    if not block:
        return False
    return re.search(r"\bsorry\b", block) is not None


def _decl_has_placeholder(text: str, name: str) -> bool:
    block = _decl_block(text, name)
    if not block:
        return False
    return re.search(r"\b(sorry|admit)\b", block) is not None


def _decl_block(text: str, name: str) -> str:
    pattern = re.compile(rf"^\s*(theorem|lemma|example)\s+{re.escape(name)}\b", re.M)
    match = pattern.search(text)
    if not match:
        return ""
    start = match.start()
    next_match = pattern.search(text, match.end())
    end = next_match.start() if next_match else len(text)
    return text[start:end]


def _count_decl_names(file_path: Path) -> int:
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception:
        return 0
    return len(_extract_decl_names(text))


def _write_lemma_plan(args: argparse.Namespace, plan: LemmaPlan) -> bool:
    try:
        args.file.write_text(plan.lean_code, encoding="utf-8")
    except Exception:
        return False
    _update_lemma_list_block(args.file)
    return True


def _has_lemma_plan(file_path: Path) -> bool:
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception:
        return False
    return LEMMA_PLAN_MARKER in text


def _expand_lemmas_for_failure(args: argparse.Namespace, lemma_name: str) -> list[str]:
    trace_path = _trace_path_for_theorem(args, lemma_name)
    steps = _read_trace_steps(trace_path, max_lines=200) if trace_path else []
    last_goal = steps[-1].get("state_pretty", "") if steps else ""
    failures = []
    successes = []
    for step in steps:
        tactic = step.get("tactic", "")
        ok = step.get("ok", False)
        if ok:
            if tactic:
                successes.append(tactic)
            continue
        err = step.get("error") or "error"
        if tactic:
            failures.append(f"{tactic}: {err}")
        else:
            failures.append(str(err))

    lemma_statement, original_statement = _extract_theorem_statement(args.file, lemma_name)
    if not lemma_statement:
        return []

    config = _formalization_config_from_args(args)
    llm = FormalizationLLM(args.llm, config)
    context = _read_context_files(args.context)
    print(f"[lemma-first] Expanding lemma {lemma_name}...")
    snippet = llm.expand_lemmas(
        lemma_name=lemma_name,
        lemma_statement=lemma_statement,
        last_goal=last_goal,
        failures=_dedupe_items(failures),
        successes=_dedupe_items(successes),
        context=context,
    )
    if not snippet.strip():
        return []
    return _insert_lemmas_before(args.file, lemma_name, snippet)


def _insert_lemmas_before(file_path: Path, target_name: str, snippet: str) -> list[str]:
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception:
        return []
    clean_snippet = _strip_imports_from_snippet(snippet)
    clean_snippet = _remove_decl_by_name(clean_snippet, target_name)
    blocks = _split_decl_blocks(clean_snippet)
    if not blocks:
        return []

    existing = set(_extract_decl_names(text))
    kept_blocks: list[str] = []
    new_names: list[str] = []
    for name, block in blocks:
        if name in existing:
            continue
        kept_blocks.append(block.rstrip() + "\n")
        new_names.append(name)

    if not kept_blocks:
        return []

    match = re.search(rf"\b(theorem|lemma|example)\s+{re.escape(target_name)}\b", text)
    if match is None:
        return []
    insert_at = match.start()
    insert_block = "\n".join(kept_blocks).rstrip() + "\n\n"
    new_text = text[:insert_at] + insert_block + text[insert_at:]
    try:
        file_path.write_text(new_text, encoding="utf-8")
    except Exception:
        return []
    _update_lemma_list_block(file_path)
    return new_names


def _strip_imports_from_snippet(snippet: str) -> str:
    lines = []
    for line in snippet.splitlines():
        if line.strip().startswith("import "):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _split_decl_blocks(text: str) -> list[tuple[str, str]]:
    pattern = re.compile(r"^\s*(theorem|lemma|example|def|abbrev|structure)\s+([A-Za-z0-9_']+)\b", re.M)
    matches = list(pattern.finditer(text))
    if not matches:
        return []
    blocks: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        name = match.group(2)
        block = text[start:end].rstrip()
        blocks.append((name, block))
    return blocks


def _remove_decl_by_name(text: str, name: str) -> str:
    pattern = re.compile(rf"^\s*(theorem|lemma|example|def|abbrev|structure)\s+{re.escape(name)}\b", re.M)
    match = pattern.search(text)
    if not match:
        return text
    start = match.start()
    next_match = pattern.search(text, match.end())
    end = next_match.start() if next_match else len(text)
    return (text[:start] + text[end:]).strip()


def _ensure_lemma_list_block(text: str) -> str:
    if "ULAMAI_LEMMA_LIST" in text:
        return text
    decls = _extract_decl_names(text)
    statuses = _lemma_status_map(text)
    block_lines = []
    for name in decls:
        status = statuses.get(name, "unknown")
        block_lines.append(f"- {name}  [{status}]")
    block = "/- ULAMAI_LEMMA_LIST\n" + "\n".join(block_lines) + "\n-/\n\n"
    return block + text


def _update_lemma_list_block(file_path: Path) -> None:
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception:
        return
    decls = _extract_decl_names(text)
    statuses = _lemma_status_map(text)
    block_lines = []
    for name in decls:
        status = statuses.get(name, "unknown")
        block_lines.append(f"- {name}  [{status}]")
    block = "/- ULAMAI_LEMMA_LIST\n" + "\n".join(block_lines) + "\n-/"
    if "ULAMAI_LEMMA_LIST" in text:
        new_text = re.sub(
            r"/-\s*ULAMAI_LEMMA_LIST.*?-/",
            block,
            text,
            flags=re.S,
        )
    else:
        new_text = block + "\n\n" + text
    if new_text != text:
        try:
            file_path.write_text(new_text, encoding="utf-8")
        except Exception:
            return


def _lemma_status_map(text: str) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for name in _extract_decl_names(text):
        block = _decl_block(text, name)
        if not block:
            continue
        if re.search(r"\bsorry\b", block):
            statuses[name] = "sorry"
        else:
            statuses[name] = "solved"
    return statuses


def _cleanup_failed_lemmas(file_path: Path) -> None:
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception:
        return
    cleaned = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    try:
        file_path.write_text(cleaned, encoding="utf-8")
    except Exception:
        return
    _update_lemma_list_block(file_path)


def _file_has_decl(text: str, name: str) -> bool:
    if not text or not name:
        return False
    return re.search(rf"\b(theorem|lemma|example)\s+{re.escape(name)}\b", text) is not None


def _suggest_proof_targets(file_path: Path, theorem: str, max_items: int = 20) -> None:
    cwd = Path.cwd()
    print("Could not read theorem statement.")
    if not file_path.exists():
        print(f"File not found: {file_path}")
        _print_available_lean_files(cwd, max_items=max_items)
        return
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"Could not read file: {exc}")
        _print_available_lean_files(file_path.parent, max_items=max_items)
        return
    decls = _extract_decl_names(text)
    if not decls:
        print("No theorem/lemma/example declarations found in the file.")
    else:
        print("Declarations found in file:")
        for name in decls[:max_items]:
            print(f"  - {name}")
        if theorem and theorem not in decls:
            print(f"Note: '{theorem}' is not present in this file.")
    print("Tip: choose a name from the list above or edit the file to add a theorem.")


def _print_available_lean_files(root: Path, max_items: int = 20) -> None:
    try:
        candidates = sorted(root.glob("*.lean"))
    except Exception:
        candidates = []
    if not candidates:
        return
    print(f"Lean files in {root}:")
    for path in candidates[:max_items]:
        print(f"  - {path}")


def _run_search_for_theorem(args: argparse.Namespace, theorem: str) -> SearchResult | None:
    context = _read_context_files(args.context)
    trace_path = _trace_path_for_theorem(args, theorem)
    solver = _resolve_solver(args)
    config = RunConfig(
        file_path=args.file,
        theorem=theorem,
        max_steps=args.max_steps,
        beam_width=args.beam,
        suggestions_per_state=args.k,
        timeout_s=args.timeout,
        repair_attempts=args.repair,
        seed=args.seed,
        trace_path=trace_path,
        retriever_k=max(1, int(getattr(args, "retriever_k", 8))),
        autop=_autop_enabled(args),
        instruction=args.instruction.strip() if args.instruction else None,
        context=context,
        verbose=bool(args.verbose),
    )

    def _run_once() -> SearchResult:
        if config.verbose:
            model = ""
            if args.llm in {"openai", "codex_cli"}:
                model = args.openai_model
            elif args.llm in {"anthropic", "claude_cli"}:
                model = args.anthropic_model
            elif args.llm in {"gemini", "gemini_cli"}:
                model = args.gemini_model
            elif args.llm == "ollama":
                model = args.ollama_model
            label = args.llm
            if model:
                label = f"{label} ({model})"
            print(f"[run] llm={label}")
            print(f"[run] lean={args.lean} file={args.file} theorem={theorem}")
            if args.instruction:
                lines = len(args.instruction.splitlines())
                print(f"[run] instruction_lines={lines}")
            if args.context:
                print(f"[run] context_files={len(args.context)}")
            print(f"[run] trace={config.trace_path}")
            print(f"[run] solver={solver}")

        runner = _make_runner(args)
        llm = _make_llm(args)
        retriever = _make_retriever(args)
        trace = TraceLogger(config.trace_path)
        try:
            return _run_with_solver(solver, runner, llm, retriever, trace, config)
        finally:
            trace.close()
            runner.close()

    try:
        return _run_once()
    except Exception as exc:
        if args.lean == "dojo" and _is_parse_error(exc):
            if _attempt_statement_repair(args, exc):
                return _run_once()
            return None
        if args.lean == "dojo" and _is_lean_mismatch_error(exc):
            print("Lean toolchain mismatch detected. Attempting to repair...")
            if _repair_lean_toolchain(args):
                print("Retrying prover after toolchain repair...")
                return _run_once()
        print(f"Prover error: {exc}")
        return None


def _trace_path_for_theorem(args: argparse.Namespace, theorem: str) -> Path | None:
    trace_path = args.trace
    if trace_path is not None and trace_path.name == "run.jsonl":
        return trace_path.with_name(f"run_{theorem}.jsonl")
    return trace_path


def _resolve_solver(args: argparse.Namespace) -> str:
    solver = getattr(args, "solver", "auto") or "auto"
    if solver not in {"auto", "search", "script", "portfolio"}:
        solver = "auto"
    if solver == "auto":
        return "script" if getattr(args, "prove_mode", "tactic") == "lemma" else "search"
    return solver


def _run_with_solver(
    solver: str,
    runner,
    llm,
    retriever,
    trace: TraceLogger,
    config: RunConfig,
) -> SearchResult:
    if solver == "script":
        return scripted_search(runner, llm, retriever, trace, config)
    if solver == "portfolio":
        return _portfolio_search(runner, llm, retriever, trace, config)
    return best_first_search(runner, llm, retriever, trace, config)


def _portfolio_search(
    runner,
    llm,
    retriever,
    trace: TraceLogger,
    config: RunConfig,
) -> SearchResult:
    total_steps = max(1, int(config.max_steps))
    stage1_steps = min(total_steps, max(1, int(total_steps * 0.4)))
    first_cfg = replace(config, max_steps=stage1_steps)
    first = scripted_search(runner, llm, retriever, trace, first_cfg)
    if first.solved:
        return first
    remaining = total_steps - first.steps
    if remaining <= 0:
        return first
    second_cfg = replace(config, max_steps=remaining)
    second = best_first_search(runner, llm, retriever, trace, second_cfg)
    if second.solved:
        return SearchResult(True, second.proof, first.steps + second.steps, None)
    error = second.error or first.error or "portfolio exhausted"
    return SearchResult(False, [], first.steps + second.steps, error)


def _bench_error_kind(error: str | None) -> str | None:
    if not error:
        return None
    lowered = error.lower()
    if "timeout" in lowered:
        return "timeout"
    if "unknown identifier" in lowered:
        return "unknown_identifier"
    if "type mismatch" in lowered:
        return "type_mismatch"
    if "unsolved goals" in lowered:
        return "unsolved_goals"
    if "unexpected token" in lowered or "parse error" in lowered:
        return "parse_error"
    return "other"


def _record_error_count(counter: dict[str, int], error: str) -> int:
    fingerprint = _error_fingerprint(error)
    counter[fingerprint] = counter.get(fingerprint, 0) + 1
    return counter[fingerprint]


def _error_fingerprint(error: str) -> str:
    text = (error or "").strip()
    if not text:
        return "<empty>"
    first_line = text.splitlines()[0].strip().lower()
    first_line = re.sub(r":[0-9]+:[0-9]+", ":#:#", first_line)
    first_line = re.sub(r"\b[0-9]{2,}\b", "#", first_line)
    return first_line[:220]


def _autop_enabled(args: argparse.Namespace) -> bool:
    if hasattr(args, "autop"):
        return bool(getattr(args, "autop"))
    if hasattr(args, "no_autop"):
        return not bool(getattr(args, "no_autop"))
    return True


def _resolve_allow_axioms(args: argparse.Namespace, config: dict | None = None) -> bool:
    explicit = getattr(args, "allow_axioms", None)
    if explicit is not None:
        return bool(explicit)
    cfg = config if config is not None else load_config()
    return bool(cfg.get("prove", {}).get("allow_axioms", True))


def run_replay(args: argparse.Namespace) -> None:
    if not args.trace.exists():
        print(f"Trace not found: {args.trace}")
        sys.exit(1)

    steps = 0
    solved = False
    tactics: list[str] = []
    with args.trace.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            steps += 1
            tactics.append(payload.get("tactic", ""))
            if payload.get("solved"):
                solved = True
                break

    print(f"Steps: {steps}")
    print(f"Solved: {solved}")
    if tactics:
        print("Tactics:")
        for tactic in tactics:
            print(f"- {tactic}")


def run_auth(args: argparse.Namespace) -> None:
    config = load_config()
    if args.provider == "codex":
        print("Launching Codex CLI login...")
        try:
            run_codex_login()
        except Exception as exc:
            print(f"Codex login failed: {exc}")
            return
        auth_path = codex_auth_path()
        api_key = load_codex_api_key(auth_path)
        if api_key:
            config.setdefault("openai", {})["api_key"] = api_key
            config["llm_provider"] = "openai"
            save_config(config)
            print("Codex login imported API key successfully.")
            return
        tokens = load_codex_tokens(auth_path)
        if tokens:
            config["llm_provider"] = "codex_cli"
            save_config(config)
            print("Codex login successful (subscription token detected).")
            return
        print(f"Could not read credentials from {auth_path}.")
        print("If your Codex CLI uses a different auth file, set CODEX_HOME.")
        return
    if args.provider == "claude":
        print("Claude auth options:")
        print("1. Claude Code CLI login (claude auth login)")
        print("2. Claude subscription (setup-token)")
        choice = input("Auth method [1]: ").strip() or "1"
        if choice == "1":
            print("Launching Claude Code login...")
            try:
                run_claude_login()
            except Exception as exc:
                print(f"Claude login failed: {exc}")
                return
            config["llm_provider"] = "claude_cli"
            save_config(config)
            print("Claude CLI login completed.")
            return
        print("Launching Claude Code setup-token...")
        try:
            token = run_claude_setup_token()
        except Exception as exc:
            print(f"Claude setup-token failed: {exc}")
            return
        if not token:
            print("Could not capture setup-token output. Run `claude setup-token` and paste it into config.")
            return
        config.setdefault("anthropic", {})["setup_token"] = token
        config["llm_provider"] = "anthropic"
        save_config(config)
        print("Claude setup-token saved.")
        return
    if args.provider == "gemini":
        print("Gemini auth options:")
        print("1. Gemini CLI OAuth login (browser callback + manual fallback)")
        print("2. Use API key")
        choice = input("Auth method [1]: ").strip() or "1"
        if choice == "1":
            print("Starting Gemini OAuth login...")
            try:
                run_gemini_login()
            except Exception as exc:
                print(f"Gemini login failed: {exc}")
                return
            config["llm_provider"] = "gemini_cli"
            save_config(config)
            print("Gemini OAuth login completed.")
            return
        api_key = input("Gemini API key: ").strip()
        if not api_key:
            print("No API key entered.")
            return
        section = config.setdefault("gemini", {})
        section["api_key"] = api_key
        config["llm_provider"] = "gemini"
        save_config(config)
        print("Gemini API key saved.")
        return


def _has_lean_setup_flag(argv: list[str]) -> bool:
    return any(flag in argv for flag in ("-lean", "--lean", "--lean-setup"))


def _strip_lean_setup_flags(argv: list[str]) -> list[str]:
    return [arg for arg in argv if arg not in ("-lean", "--lean", "--lean-setup")]


def _parse_lean_setup_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ulam -lean",
        description="Install Lean + LeanDojo and create a Mathlib project",
    )
    _add_lean_setup_args(parser)
    return parser.parse_args(argv)


def _add_lean_setup_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dir",
        default="",
        help="Lean project directory (default: ./ulam-lean or current Lean project)",
    )
    parser.add_argument(
        "--toolchain",
        default="",
        help="Lean toolchain to pin (default: Pantograph toolchain if available, else leanprover/lean4:stable)",
    )
    parser.add_argument(
        "--use-mathlib-toolchain",
        action="store_true",
        help="use the toolchain from mathlib4 template instead of --toolchain",
    )
    parser.add_argument("--yes", action="store_true", help="run non-interactively with defaults")
    parser.add_argument("--skip-elan", action="store_true", help="skip installing elan (Lean)")
    parser.add_argument("--no-build", action="store_true", help="skip `lake build`")
    parser.add_argument("--no-dojo", action="store_true", help="skip LeanDojo + Pantograph install")
    parser.add_argument("--no-config", action="store_true", help="do not write .ulam/config.json")
    parser.add_argument("--pip-timeout", type=int, default=120, help="pip download timeout (seconds)")
    parser.add_argument("--pip-retries", type=int, default=5, help="pip retry count")


def run_lean_setup(args: argparse.Namespace) -> None:
    print("UlamAI Lean setup")
    if not args.yes:
        print("This will install Lean (elan), create a Mathlib project, and install LeanDojo.")
        choice = input("Continue? (Y/n): ").strip().lower()
        if choice in {"n", "no"}:
            print("Aborted.")
            return

    project_dir = _resolve_lean_project_dir(args)
    if project_dir is None:
        return

    env = os.environ.copy()
    env = _ensure_elan(env, args)
    if env is None:
        return

    if not _looks_like_lean_project(project_dir):
        created = _create_mathlib_project(project_dir, env, args)
        if not created:
            return
        _pin_toolchain(project_dir, args)

    if not args.no_dojo:
        if not _ensure_lean_dojo(env, args):
            return
        pantograph_toolchain = _pantograph_toolchain()
        if pantograph_toolchain and not args.use_mathlib_toolchain:
            if not _align_mathlib_to_toolchain(project_dir, env, pantograph_toolchain):
                print(
                    "Unable to align Mathlib to Pantograph toolchain automatically; "
                    "continuing with Mathlib's default toolchain."
                )

    if not args.no_build:
        if not _run_cmd(["lake", "build"], cwd=project_dir, env=env):
            if _maybe_fallback_toolchain(project_dir, env, args):
                if not _run_cmd(["lake", "build"], cwd=project_dir, env=env):
                    return
            else:
                return

    # If dojo install was skipped earlier, still allow it after build.
    if args.no_dojo:
        pass
    else:
        # Already handled above.
        pass

    if not args.no_config:
        config = load_config()
        config.setdefault("lean", {})["project"] = str(project_dir)
        save_config(config)
        print(f"Saved Lean project path to config: {project_dir}")

    print("Lean setup complete.")


def _resolve_lean_project_dir(args: argparse.Namespace) -> Path | None:
    if args.dir:
        project_dir = Path(args.dir).expanduser()
    else:
        cwd = Path.cwd()
        if _looks_like_lean_project(cwd):
            project_dir = cwd
        elif args.yes:
            project_dir = cwd / "ulam-lean"
        else:
            value = input("Lean project directory [./ulam-lean]: ").strip()
            project_dir = Path(value).expanduser() if value else (cwd / "ulam-lean")

    while project_dir.exists() and not _looks_like_lean_project(project_dir):
        if args.yes:
            print(f"Directory exists and is not a Lean project: {project_dir}")
            return None
        print(f"Directory exists and is not a Lean project: {project_dir}")
        value = input("Choose a different path (blank to abort): ").strip()
        if not value:
            print("Aborted.")
            return None
        project_dir = Path(value).expanduser()

    return project_dir


def _ensure_elan(env: dict, args: argparse.Namespace) -> dict | None:
    env = _extend_path(env, Path("~/.elan/bin").expanduser())
    if _which("elan", env):
        return env
    if args.skip_elan:
        print("Elan not found and --skip-elan set. Aborting.")
        return None
    if not args.yes:
        choice = input("Elan not found. Install now? (Y/n): ").strip().lower()
        if choice in {"n", "no"}:
            print("Aborted.")
            return None
    if not _run_shell("curl https://elan.lean-lang.org/elan-init.sh -sSf | sh", env=env):
        return None
    env = _extend_path(env, Path("~/.elan/bin").expanduser())
    if not _which("elan", env):
        print("Elan was installed but not found on PATH. Open a new shell or source ~/.elan/env.")
        return None
    return env


def _create_mathlib_project(project_dir: Path, env: dict, args: argparse.Namespace) -> bool:
    if not _which("lake", env):
        print("Lake not found. Ensure elan is installed and ~/.elan/bin is on PATH.")
        return False
    parent = project_dir.parent
    name = project_dir.name
    if project_dir.exists() and not _looks_like_lean_project(project_dir):
        print(f"Directory exists and is not a Lean project: {project_dir}")
        return False
    parent.mkdir(parents=True, exist_ok=True)
    if not _run_cmd(
        ["lake", "+leanprover-community/mathlib4:lean-toolchain", "new", name, "math"],
        cwd=parent,
        env=env,
    ):
        return False
    return True


def _align_mathlib_to_toolchain(project_dir: Path, env: dict, toolchain: str) -> bool:
    toolchain = _normalize_toolchain(toolchain)
    if not toolchain:
        return False
    toolchain_file = project_dir / "lean-toolchain"
    lean_path = project_dir / "lakefile.lean"
    toml_path = project_dir / "lakefile.toml"

    old_toolchain = toolchain_file.read_text(encoding="utf-8") if toolchain_file.exists() else ""
    old_lean = lean_path.read_text(encoding="utf-8") if lean_path.exists() else None
    old_toml = toml_path.read_text(encoding="utf-8") if toml_path.exists() else None

    success = False
    try:
        toolchain_file.write_text(toolchain + "\n", encoding="utf-8")
        mathlib_rev = _mathlib_rev_for_toolchain(toolchain)
        if mathlib_rev:
            if toml_path.exists():
                if not _update_mathlib_rev_toml(toml_path, mathlib_rev):
                    return False
            elif lean_path.exists():
                if not _pin_mathlib_rev_in_lakefile(project_dir, mathlib_rev):
                    return False
            else:
                return False

        if not _ensure_toolchain_installed(toolchain, env):
            return False

        shutil.rmtree(project_dir / ".lake" / "build", ignore_errors=True)
        if not _run_cmd(["lake", "update"], cwd=project_dir, env=env):
            return False
        success = True
        return True
    finally:
        if not success:
            if old_toolchain:
                try:
                    toolchain_file.write_text(old_toolchain, encoding="utf-8")
                except Exception:
                    pass
            if old_lean is not None:
                try:
                    lean_path.write_text(old_lean, encoding="utf-8")
                except Exception:
                    pass
            if old_toml is not None:
                try:
                    toml_path.write_text(old_toml, encoding="utf-8")
                except Exception:
                    pass


def _mathlib_rev_for_toolchain(toolchain: str) -> str | None:
    if not toolchain:
        return None
    version = toolchain.split(":", 1)[-1].strip()
    if not version:
        return None
    if version in {"stable", "nightly"}:
        return None
    if not version.startswith("v"):
        version = f"v{version}"
    return version


def _pin_mathlib_rev_in_lakefile(project_dir: Path, rev: str) -> bool:
    lakefile = project_dir / "lakefile.lean"
    if not lakefile.exists():
        return False
    text = lakefile.read_text(encoding="utf-8")
    if "require mathlib" not in text or "mathlib4" not in text:
        return False
    url_pat = r"https?://github\.com/leanprover-community/mathlib4(?:\.git)?"
    replace_pat = rf'({url_pat}"\s*@\s*")[^"]+(")'
    if re.search(replace_pat, text):
        new_text = re.sub(replace_pat, rf"\1{rev}\2", text)
        if new_text != text:
            lakefile.write_text(new_text, encoding="utf-8")
        return True
    insert_pat = rf'({url_pat}(?:\.git)?")'
    if re.search(insert_pat, text):
        new_text = re.sub(insert_pat, rf'\1 @ "{rev}"', text, count=1)
        if new_text != text:
            lakefile.write_text(new_text, encoding="utf-8")
            return True
    return False


def _update_mathlib_rev_toml(path: Path, rev: str) -> bool:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    in_require = False
    is_mathlib = False
    rev_written = False
    seen_mathlib = False

    def flush_rev() -> None:
        nonlocal rev_written
        if in_require and is_mathlib and not rev_written:
            out.append(f'rev = "{rev}"')
            rev_written = True

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped != "[[require]]":
            flush_rev()
            in_require = False
            is_mathlib = False
            rev_written = False
            out.append(line)
            continue
        if stripped.startswith("[[require]]"):
            flush_rev()
            in_require = True
            is_mathlib = False
            rev_written = False
            out.append(line)
            continue
        if in_require:
            if stripped.startswith("name"):
                is_mathlib = "mathlib" in stripped
                if is_mathlib:
                    seen_mathlib = True
            if stripped.startswith("rev") and is_mathlib:
                out.append(f'rev = "{rev}"')
                rev_written = True
                continue
        out.append(line)

    flush_rev()
    if not seen_mathlib:
        return False
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return True


def _ensure_lean_dojo(env: dict, args: argparse.Namespace) -> bool:
    if _pantograph_available():
        print("LeanDojo already installed.")
        return True
    print("Installing LeanDojo + Pantograph...")
    if not _pip_install(env, args, ["lean-dojo-v2"]):
        return False
    return _pip_install(env, args, ["git+https://github.com/stanford-centaur/PyPantograph"])


def _pantograph_available() -> bool:
    try:
        import pantograph  # type: ignore
    except Exception:
        return False
    return True


def _looks_like_lean_project(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if (path / "lakefile.lean").exists():
        return True
    if (path / "lakefile.toml").exists():
        return True
    if (path / "lean-toolchain").exists():
        return True
    return False


def _extend_path(env: dict, extra: Path) -> dict:
    if not extra.exists():
        return env
    path = env.get("PATH", "")
    extra_str = str(extra)
    parts = path.split(os.pathsep) if path else []
    if extra_str not in parts:
        env = env.copy()
        env["PATH"] = extra_str + os.pathsep + path if path else extra_str
    return env


def _which(cmd: str, env: dict | None = None) -> str | None:
    path = env.get("PATH") if env else None
    return shutil.which(cmd, path=path)


def _run_shell(command: str, env: dict | None = None) -> bool:
    try:
        subprocess.run(command, shell=True, check=True, env=env)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Command failed ({exc.returncode}): {command}")
        return False


def _run_cmd(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> bool:
    try:
        subprocess.run(cmd, check=True, cwd=cwd, env=env)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Command failed ({exc.returncode}): {' '.join(cmd)}")
        return False


def _ensure_toolchain_installed(toolchain: str, env: dict) -> bool:
    proc = subprocess.run(
        ["elan", "toolchain", "install", toolchain],
        env=env,
        text=True,
        capture_output=True,
    )
    if proc.returncode == 0:
        return True
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if "already installed" in output or "is already installed" in output:
        return True
    print(f"Command failed ({proc.returncode}): elan toolchain install {toolchain}")
    if output.strip():
        print(output.strip())
    return False


def _pip_install(env: dict, args: argparse.Namespace, packages: list[str]) -> bool:
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--timeout",
        str(args.pip_timeout),
        "--retries",
        str(args.pip_retries),
        *packages,
    ]
    return _run_cmd(cmd, env=env)


def _is_lean_mismatch_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "incompatible header" in text
        or "lean version mismatch" in text
        or "server failed to emit ready signal" in text
    )


def _is_parse_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "unexpected token" in text
        or "function expected at" in text
        or "parse error" in text
        or "invalid 'import' command" in text
    )


def _extract_original_statement(text: str) -> str:
    match = re.search(r"/-\s*ULAMAI_ORIGINAL_STATEMENT\s*(.*?)\s*-/", text, re.S)
    if not match:
        return ""
    return match.group(1).strip()


def _formalization_config_from_args(args: argparse.Namespace) -> dict:
    return {
        "llm_provider": args.llm,
        "openai": {
            "api_key": args.openai_key,
            "base_url": args.openai_base_url,
            "model": args.openai_model,
            "codex_model": args.openai_model,
        },
        "anthropic": {
            "api_key": args.anthropic_key,
            "setup_token": args.anthropic_setup_token,
            "base_url": args.anthropic_base_url,
            "model": args.anthropic_model,
            "claude_model": args.anthropic_model,
        },
        "gemini": {
            "api_key": args.gemini_api_key,
            "base_url": args.gemini_base_url,
            "model": args.gemini_model,
            "cli_model": args.gemini_model,
        },
        "ollama": {
            "base_url": args.ollama_base_url,
            "model": args.ollama_model,
        },
    }


def _attempt_statement_repair(args: argparse.Namespace, exc: Exception) -> bool:
    file_path = args.file
    try:
        lean_code = file_path.read_text(encoding="utf-8")
    except Exception:
        return False
    original = _extract_original_statement(lean_code)
    if not original:
        return False
    config = _formalization_config_from_args(args)
    llm = FormalizationLLM(args.llm, config)
    print("Statement parse failed; asking LLM to repair the theorem statement...")
    repaired = llm.repair_statement(lean_code, args.theorem, original, context="")
    if not repaired.strip():
        return False
    try:
        file_path.write_text(repaired, encoding="utf-8")
    except Exception:
        return False
    return True


def _write_proof_to_file(file_path: Path, theorem: str, proof: list[str]) -> bool:
    if not proof:
        return False
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception:
        return False
    match = re.search(rf"\b(theorem|lemma|example)\s+{re.escape(theorem)}\b", text)
    if match is None:
        return False
    sorry_match = re.search(r"\bsorry\b", text[match.start() :])
    if sorry_match is None:
        return False
    start = match.start() + sorry_match.start()
    end = match.start() + sorry_match.end()
    line_start = text.rfind("\n", 0, start) + 1
    indent = text[line_start:start]
    prefix = text[:line_start]
    has_by = re.search(r":=\s*by\s*$", prefix) is not None or re.search(r"\bby\s*$", prefix) is not None
    if has_by:
        proof_block = "\n".join(f"{indent}{line}" for line in proof)
    else:
        proof_block = "by\n" + "\n".join(f"{indent}{line}" for line in proof)
    new_text = prefix + proof_block + text[end:]
    try:
        file_path.write_text(new_text, encoding="utf-8")
    except Exception:
        return False
    return True


def _summarize_failed_run(args: argparse.Namespace) -> None:
    _summarize_failed_run_for(args, args.theorem, args.trace)


def _summarize_failed_run_for(
    args: argparse.Namespace, theorem: str, trace_path: Path | None, note: str | None = None
) -> None:
    if trace_path is None or str(trace_path) == "-":
        return
    try:
        steps = _read_trace_steps(trace_path, max_lines=200)
    except Exception:
        return
    if not steps:
        return

    last_goal = steps[-1].get("state_pretty", "")
    failures = []
    successes = []
    for step in steps:
        tactic = step.get("tactic", "")
        ok = step.get("ok", False)
        if ok:
            if tactic:
                successes.append(tactic)
            continue
        err = step.get("error") or "error"
        if tactic:
            failures.append(f"{tactic}: {err}")
        else:
            failures.append(str(err))

    if note:
        failures.insert(0, note)

    theorem_statement, original_statement = _extract_theorem_statement(args.file, theorem)
    statement = original_statement or theorem_statement
    if not statement:
        statement = "(unknown statement)"

    config = _formalization_config_from_args(args)
    llm = FormalizationLLM(args.llm, config)
    summary = llm.summarize_attempt(
        theorem_name=theorem,
        theorem_statement=statement,
        last_goal=last_goal,
        failures=_dedupe_items(failures),
        successes=_dedupe_items(successes),
        context="",
    )
    if not summary.strip():
        return
    _append_run_summary(args.file, summary.strip())
    print(f"Wrote run summary to: {args.file}")


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


def _extract_theorem_statement(file_path: Path, theorem: str) -> tuple[str, str]:
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception:
        return "", ""
    original = _extract_original_statement(text)
    decl_match = re.search(
        rf"\b(theorem|lemma|example)\s+{re.escape(theorem)}\b",
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


def _dedupe_items(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _append_run_summary(file_path: Path, summary: str) -> None:
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception:
        return
    block = (
        "\n\n/- ULAMAI_RUN_SUMMARY\n"
        + summary
        + "\n-/"
    )
    try:
        file_path.write_text(text + block, encoding="utf-8")
    except Exception:
        return


def _preflight_lean_alignment(args: argparse.Namespace) -> None:
    if args.lean != "dojo":
        return
    project = args.lean_project or _find_lean_project_for_file(args.file)
    if project is None:
        return
    args.lean_project = project
    toolchain_file = project / "lean-toolchain"
    current = toolchain_file.read_text(encoding="utf-8").strip() if toolchain_file.exists() else ""
    pantograph = _pantograph_toolchain()
    if not pantograph or current == pantograph:
        return
    env = _extend_path(os.environ.copy(), Path("~/.elan/bin").expanduser())
    if not _which("elan", env):
        return
    print(f"Aligning project toolchain to Pantograph ({pantograph}) before starting prover...")
    if not _align_mathlib_to_toolchain(project, env, pantograph):
        print("Warning: failed to align project toolchain automatically.")


def _repair_lean_toolchain(args: argparse.Namespace) -> bool:
    project = args.lean_project or _find_lean_project_for_file(args.file)
    if project is None:
        print("Unable to locate Lean project to repair.")
        return False
    toolchain_file = project / "lean-toolchain"
    if not toolchain_file.exists():
        print("Lean toolchain file not found; cannot repair automatically.")
        return False
    original_toolchain_text = toolchain_file.read_text(encoding="utf-8")
    original_toolchain = original_toolchain_text.strip()
    if not original_toolchain:
        print("Lean toolchain file is empty; cannot repair automatically.")
        return False
    lakefile_lean = project / "lakefile.lean"
    lakefile_toml = project / "lakefile.toml"
    original_lakefile_lean = lakefile_lean.read_text(encoding="utf-8") if lakefile_lean.exists() else None
    original_lakefile_toml = lakefile_toml.read_text(encoding="utf-8") if lakefile_toml.exists() else None
    env = _extend_path(os.environ.copy(), Path("~/.elan/bin").expanduser())
    if not _which("elan", env):
        print("Elan not found on PATH; cannot repair automatically.")
        return False
    candidates = _repair_toolchain_candidates(original_toolchain)
    for candidate in candidates:
        if candidate != original_toolchain:
            print(f"Trying toolchain: {candidate}")
        try:
            toolchain_file.write_text(original_toolchain_text, encoding="utf-8")
        except Exception:
            pass
        if original_lakefile_lean is not None:
            try:
                lakefile_lean.write_text(original_lakefile_lean, encoding="utf-8")
            except Exception:
                pass
        if original_lakefile_toml is not None:
            try:
                lakefile_toml.write_text(original_lakefile_toml, encoding="utf-8")
            except Exception:
                pass

        if not _align_mathlib_to_toolchain(project, env, candidate):
            continue
        if _run_cmd(["lake", "build"], cwd=project, env=env):
            if args.lean_project is None:
                args.lean_project = project
            return True
    try:
        toolchain_file.write_text(original_toolchain_text, encoding="utf-8")
    except Exception:
        pass
    if original_lakefile_lean is not None:
        try:
            lakefile_lean.write_text(original_lakefile_lean, encoding="utf-8")
        except Exception:
            pass
    if original_lakefile_toml is not None:
        try:
            lakefile_toml.write_text(original_lakefile_toml, encoding="utf-8")
        except Exception:
            pass
    print("Toolchain repair failed after trying alternatives.")
    return False


def _find_lean_project_for_file(file_path: Path) -> Path | None:
    root = file_path.parent
    for parent in [root, *root.parents]:
        if (
            (parent / "lakefile.lean").exists()
            or (parent / "lakefile.toml").exists()
            or (parent / "lean-toolchain").exists()
        ):
            return parent
    return None


def _pin_toolchain(project_dir: Path, args: argparse.Namespace) -> None:
    if args.use_mathlib_toolchain or not args.toolchain:
        return
    toolchain = _normalize_toolchain(args.toolchain)
    if not toolchain:
        return
    toolchain_file = project_dir / "lean-toolchain"
    if toolchain_file.exists():
        current = toolchain_file.read_text(encoding="utf-8").strip()
        if current == toolchain:
            return
    toolchain_file.write_text(toolchain + "\n", encoding="utf-8")


def _maybe_fallback_toolchain(project_dir: Path, env: dict, args: argparse.Namespace) -> bool:
    if args.use_mathlib_toolchain or args.no_build:
        return False
    toolchain_file = project_dir / "lean-toolchain"
    if not toolchain_file.exists():
        return False
    current = toolchain_file.read_text(encoding="utf-8").strip()
    if not current:
        return False
    pantograph = _pantograph_toolchain()
    fallback = pantograph or "leanprover/lean4:stable"
    if current == fallback:
        return False
    print(f"Build failed with current toolchain; retrying with {fallback}.")
    toolchain_file.write_text(fallback + "\n", encoding="utf-8")
    _ensure_toolchain_installed(fallback, env)
    return True


def _normalize_toolchain(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    if text in {"stable", "nightly"}:
        return f"leanprover/lean4:{text}"
    return text


def _is_rc_toolchain(value: str) -> bool:
    return "-rc" in value or "rc" in value


def _select_toolchain(args: argparse.Namespace) -> str:
    if args.toolchain:
        return _normalize_toolchain(args.toolchain)
    return ""


def _repair_toolchain_candidates(original: str) -> list[str]:
    candidates: list[str] = []
    pantograph = _pantograph_toolchain()
    if pantograph:
        candidates.append(pantograph)
    if _is_rc_toolchain(original):
        candidates.append("leanprover/lean4:stable")
    candidates.append(original)
    # Deduplicate while preserving order.
    seen: set[str] = set()
    ordered: list[str] = []
    for item in candidates:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _try_align_with_pantograph(project_dir: Path, env: dict, args: argparse.Namespace) -> bool:
    pantograph = _pantograph_toolchain()
    if not pantograph:
        return True
    if args.use_mathlib_toolchain:
        print("Pantograph toolchain differs from Mathlib toolchain; keeping Mathlib toolchain.")
        return True
    toolchain_file = project_dir / "lean-toolchain"
    current = toolchain_file.read_text(encoding="utf-8").strip() if toolchain_file.exists() else ""
    if current == pantograph:
        return True
    print(f"Pantograph toolchain detected ({pantograph}); attempting to align project toolchain.")
    original = current
    toolchain_file.write_text(pantograph + "\n", encoding="utf-8")
    _run_cmd(["elan", "toolchain", "install", pantograph], env=env)
    if not _run_cmd(["lake", "update"], cwd=project_dir, env=env):
        print("Failed to update dependencies for Pantograph toolchain; restoring original toolchain.")
        if original:
            toolchain_file.write_text(original + "\n", encoding="utf-8")
        return True
    if not args.no_build:
        if not _run_cmd(["lake", "build"], cwd=project_dir, env=env):
            print("Build failed after aligning to Pantograph toolchain; restoring original toolchain.")
            if original:
                toolchain_file.write_text(original + "\n", encoding="utf-8")
                _run_cmd(["lake", "build"], cwd=project_dir, env=env)
            return True
    return True


def _pantograph_toolchain() -> str | None:
    try:
        import importlib.util
        spec = importlib.util.find_spec("pantograph")
        if spec is None or spec.origin is None:
            return None
        base = Path(spec.origin).resolve().parent
    except Exception:
        return None
    candidates = [
        base / "lean-toolchain",
        base / "src" / "lean-toolchain",
        base.parent / "src" / "lean-toolchain",
        base.parent / "pantograph" / "lean-toolchain",
    ]
    for path in candidates:
        toolchain = _read_toolchain_file(path)
        if toolchain:
            return toolchain
    # As a last resort, look a couple levels up for a toolchain file.
    toolchain = _scan_for_toolchain(base.parent, max_depth=3)
    return toolchain


def _scan_for_toolchain(root: Path, max_depth: int = 3) -> str | None:
    queue: list[tuple[Path, int]] = [(root, 0)]
    while queue:
        path, depth = queue.pop(0)
        toolchain = _read_toolchain_file(path / "lean-toolchain")
        if toolchain:
            return toolchain
        if depth >= max_depth:
            continue
        try:
            for child in path.iterdir():
                if child.is_dir():
                    name = child.name
                    if name.startswith(".") or name.endswith(".dist-info") or name == "__pycache__":
                        continue
                    queue.append((child, depth + 1))
        except Exception:
            continue
    return None


def _read_toolchain_file(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        value = path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    return value or None


def run_formalize(args: argparse.Namespace) -> None:
    config = load_config()
    tex_path = args.tex
    if not tex_path.exists():
        print(f"Tex file not found: {tex_path}")
        sys.exit(1)
    output_path = args.out if args.out else tex_path.with_suffix(".lean")
    context_files = [Path(p) for p in args.context]

    proof_backend = args.proof_backend
    if proof_backend == "dojo":
        proof_backend = "tactic"
    lean_backend = args.lean_backend
    if proof_backend == "llm":
        lean_backend = "cli"
    max_rounds = max(1, int(args.max_rounds))
    max_repairs = max_rounds if args.max_repairs is None else max(0, int(args.max_repairs))
    dojo_timeout_s = float(config.get("lean", {}).get("dojo_timeout_s", 180))
    allow_axioms = _resolve_allow_axioms(args, config)
    cfg = FormalizationConfig(
        tex_path=tex_path,
        output_path=output_path,
        context_files=context_files,
        max_rounds=max_rounds,
        max_repairs=max_repairs,
        max_equivalence_repairs=args.max_equivalence_repairs,
        max_proof_rounds=args.max_proof_rounds,
        proof_max_steps=args.proof_max_steps,
        proof_beam=args.proof_beam,
        proof_k=args.proof_k,
        proof_timeout_s=args.proof_timeout,
        proof_repair=args.proof_repair,
        dojo_timeout_s=dojo_timeout_s,
        lemma_max=60,
        lemma_depth=60,
        allow_axioms=allow_axioms,
        lean_project=args.lean_project,
        lean_imports=args.lean_import,
        verbose=bool(args.verbose),
        proof_backend=proof_backend,
        lean_backend=lean_backend,
        resume_path=None,
        artifact_dir=args.artifacts_dir,
        equivalence_checks=not args.no_equivalence,
    )
    llm = FormalizationLLM(config.get("llm_provider", "openai"), config)
    if args.segment:
        from .formalize.segmentation import run_segmented_formalize

        max_words = max(200, int(args.segment_words))
        out_path = run_segmented_formalize(cfg, llm, max_words=max_words)
        print(f"Wrote: {out_path}")
        return

    engine = FormalizationEngine(cfg, llm)
    result = engine.run()
    print(f"Wrote: {result.output_path}")
    print(f"Typecheck: {'ok' if result.typecheck_ok else 'failed'}")
    print(f"Proof-search solved: {result.solved}, Remaining sorries: {result.remaining_sorries}")
    if result.error:
        print(f"Failure reason: {str(result.error).splitlines()[0]}")
    if not result.typecheck_ok and result.remaining_sorries == 0:
        print("Note: no sorries remain, but Lean typecheck errors remain.")
    if result.artifact_dir:
        print(f"Artifacts: {result.artifact_dir}")


def run_bench(args: argparse.Namespace) -> None:
    if not args.suite.exists():
        print(f"Suite not found: {args.suite}")
        sys.exit(1)

    cases = []
    with args.suite.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            file_path = Path(payload["file"])
            theorem = payload["theorem"]
            premises = Path(payload["premises"]) if payload.get("premises") else None
            cases.append((file_path, theorem, premises))

    llm = _make_llm(args)
    solved = 0
    results: list[dict[str, object]] = []
    error_kinds: Counter[str] = Counter()
    step_counts: list[int] = []
    durations_s: list[float] = []
    for idx, (file_path, theorem, premises) in enumerate(cases, start=1):
        case_args = argparse.Namespace(**vars(args))
        case_args.premises = premises if premises is not None else args.premises
        context = _read_context_files(case_args.context)
        solver = _resolve_solver(case_args)
        trace_path = None
        if args.trace_dir:
            args.trace_dir.mkdir(parents=True, exist_ok=True)
            trace_name = _sanitize_case_name(idx, theorem)
            trace_path = args.trace_dir / f"{trace_name}.jsonl"
        config = RunConfig(
            file_path=file_path,
            theorem=theorem,
            max_steps=args.max_steps,
            beam_width=args.beam,
            suggestions_per_state=args.k,
            timeout_s=args.timeout,
            repair_attempts=args.repair,
            seed=args.seed,
            trace_path=trace_path,
            retriever_k=max(1, int(getattr(args, "retriever_k", 8))),
            autop=_autop_enabled(case_args),
            instruction=args.instruction.strip() if args.instruction else None,
            context=context,
            verbose=bool(args.verbose),
        )
        runner = _make_runner(case_args)
        retriever = _make_retriever(case_args)
        trace = TraceLogger(trace_path)
        start = time.perf_counter()
        try:
            result = _run_with_solver(solver, runner, llm, retriever, trace, config)
        finally:
            trace.close()
            runner.close()
        duration_s = time.perf_counter() - start
        durations_s.append(duration_s)
        step_counts.append(result.steps)
        kind = _bench_error_kind(result.error)
        if kind:
            error_kinds[kind] += 1
        results.append(
            {
                "theorem": theorem,
                "solved": result.solved,
                "steps": result.steps,
                "duration_s": duration_s,
                "error_kind": kind,
            }
        )
        if result.solved:
            solved += 1
        status = "solved" if result.solved else "failed"
        print(
            f"[{idx}/{len(cases)}] {theorem}: {status} "
            f"(steps={result.steps}, time={duration_s:.2f}s)"
        )

    print(f"Total: {len(cases)}")
    print(f"Solved: {solved}")
    if results:
        success_rate = (100.0 * solved) / len(results)
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Median steps: {statistics.median(step_counts):.1f}")
        print(f"Median time: {statistics.median(durations_s):.2f}s")
    if error_kinds:
        summary = ", ".join(f"{name}={count}" for name, count in error_kinds.most_common(5))
        print(f"Top failure kinds: {summary}")


def _read_context_files(paths: list[Path], max_chars: int = 8000) -> list[str]:
    context = []
    for path in paths:
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        if len(content) > max_chars:
            content = content[:max_chars] + "\n-- (truncated)"
        context.append(f"[file: {path}]\n{content}")
    return context


def _sanitize_case_name(index: int, theorem: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in theorem)
    return f"{index:03d}_{safe}"


def _make_runner(args: argparse.Namespace) -> LeanRunner:
    if args.lean == "mock":
        return MockLeanRunner()
    if args.lean == "dojo":
        imports = args.lean_import if args.lean_import else None
        return LeanDojoRunner(project_path=args.lean_project, imports=imports)
    raise RuntimeError(f"unknown Lean backend: {args.lean}")


def _make_llm(args: argparse.Namespace):
    if args.llm == "mock":
        return MockLLMClient()
    if args.llm == "openai":
        if not args.openai_key:
            raise RuntimeError("OpenAI API key missing. Set ULAM_OPENAI_API_KEY or --openai-key.")
        return OpenAICompatClient(
            api_key=args.openai_key,
            base_url=args.openai_base_url,
            model=args.openai_model,
        )
    if args.llm == "ollama":
        return OllamaClient(base_url=args.ollama_base_url, model=args.ollama_model)
    if args.llm == "anthropic":
        token = args.anthropic_key or args.anthropic_setup_token
        return AnthropicClient(
            api_key=token,
            base_url=args.anthropic_base_url,
            model=args.anthropic_model,
        )
    if args.llm == "gemini":
        if not args.gemini_api_key:
            raise RuntimeError("Gemini API key missing. Set ULAM_GEMINI_API_KEY or --gemini-api-key.")
        return GeminiClient(
            api_key=args.gemini_api_key,
            base_url=args.gemini_base_url,
            model=args.gemini_model,
        )
    if args.llm == "codex_cli":
        model = args.openai_model or None
        return CodexCLIClient(model=model)
    if args.llm == "claude_cli":
        return ClaudeCLIClient(model=args.anthropic_model or None)
    if args.llm == "gemini_cli":
        return GeminiCLIClient(model=args.gemini_model or None)
    raise RuntimeError(f"unknown LLM backend: {args.llm}")


def _llm_config_from_args(args: argparse.Namespace) -> dict:
    cfg = load_config()
    cfg["llm_provider"] = args.llm
    openai = cfg.setdefault("openai", {})
    anthropic = cfg.setdefault("anthropic", {})
    ollama = cfg.setdefault("ollama", {})
    gemini = cfg.setdefault("gemini", {})
    if args.llm in {"openai", "codex_cli"}:
        if args.openai_key:
            openai["api_key"] = args.openai_key
        if args.openai_base_url:
            openai["base_url"] = args.openai_base_url
        if args.openai_model:
            openai["model"] = args.openai_model
            openai["codex_model"] = args.openai_model
    if args.llm in {"anthropic", "claude_cli"}:
        if args.anthropic_key:
            anthropic["api_key"] = args.anthropic_key
        if args.anthropic_setup_token:
            anthropic["setup_token"] = args.anthropic_setup_token
        if args.anthropic_base_url:
            anthropic["base_url"] = args.anthropic_base_url
        if args.anthropic_model:
            anthropic["model"] = args.anthropic_model
            anthropic["claude_model"] = args.anthropic_model
    if args.llm == "ollama":
        if args.ollama_base_url:
            ollama["base_url"] = args.ollama_base_url
        if args.ollama_model:
            ollama["model"] = args.ollama_model
    if args.llm in {"gemini", "gemini_cli"}:
        if args.gemini_api_key:
            gemini["api_key"] = args.gemini_api_key
        if args.gemini_base_url:
            gemini["base_url"] = args.gemini_base_url
        if args.gemini_model:
            gemini["model"] = args.gemini_model
            gemini["cli_model"] = args.gemini_model
    return cfg


def _normalize_llm_output(text: str) -> str:
    lines = []
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            lines.append(line)
        else:
            lines.append(line)
    cleaned = "\n".join(lines).strip()
    return cleaned + "\n" if cleaned else ""


def _axiom_guardrail_error(text: str, allow_axioms: bool) -> str | None:
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


def _extract_tex_snippet(text: str, name: str) -> str:
    marker = f"ULAMAI_TEX_SNIPPET: {name}"
    pattern = re.compile(rf"/-\\s*{re.escape(marker)}\\s*(.*?)\\s*-/", re.S)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    original = re.search(r"/-\\s*ULAMAI_ORIGINAL_STATEMENT\\s*(.*?)\\s*-/", text, re.S)
    if original:
        return original.group(1).strip()
    return ""


def _make_retriever(args: argparse.Namespace):
    if args.retriever == "none" or args.premises is None:
        return NullRetriever()
    if not args.premises.exists():
        raise RuntimeError(f"Premises file not found: {args.premises}")
    with args.premises.open("r", encoding="utf-8") as fh:
        premises = [line.rstrip("\n") for line in fh]
    if args.retriever == "simple":
        return SimpleRetriever(premises)
    if args.retriever == "embedding":
        if not args.embed_api_key:
            raise RuntimeError(
                "Embedding API key missing. Set ULAM_EMBED_API_KEY or --embed-api-key."
            )
        embedder = OpenAIEmbeddingClient(
            api_key=args.embed_api_key,
            base_url=args.embed_base_url,
            model=args.embed_model,
        )
        return EmbeddingRetriever(
            premises=premises,
            embedder=embedder,
            cache_path=args.embed_cache,
            batch_size=args.embed_batch_size,
        )
    raise RuntimeError(f"unknown retriever: {args.retriever}")
