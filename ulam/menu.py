from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path

from . import __version__
from .cli import run_prove
from .auth import (
    codex_auth_path,
    load_codex_api_key,
    load_codex_tokens,
    run_codex_login,
    run_claude_setup_token,
    run_claude_login,
)
from .formalize.engine import FormalizationEngine
from .formalize.llm import FormalizationLLM
from .formalize.types import FormalizationConfig
from .config import load_config, save_config


def run_menu() -> None:
    config = load_config()
    while True:
        _print_banner()
        print(f"Version: {__version__}")
        print(f"Provider: {_provider_label(config)}")
        print()
        print("1. Configure LLM (Codex/OpenAI, Claude, Ollama)")
        print("2. Prove with natural language guidance")
        print("3. Formalize .tex to Lean")
        print("4. Resume last formalization")
        print("5. Settings")
        print("6. Exit")
        print()
        choice = _prompt("Select option", default="6")
        if choice == "1":
            _configure_llm(config)
            save_config(config)
            continue
        if choice == "2":
            _menu_prove(config)
            continue
        if choice == "3":
            _menu_formalize(config)
            continue
        if choice == "4":
            _menu_formalize_resume(config)
            continue
        if choice == "5":
            _configure_prover(config)
            save_config(config)
            continue
        if choice == "6":
            print("Goodbye.")
            return
        print("Invalid choice.\n")


def _configure_llm(config: dict) -> None:
    print("\nChoose provider:")
    print("1. Codex/OpenAI (subscription or API key)")
    print("2. Claude (Anthropic)")
    print("3. Ollama")
    choice = _prompt("Provider", default="1")
    if choice == "1":
        config["llm_provider"] = "openai"
        _configure_openai(config)
    elif choice == "2":
        config["llm_provider"] = "anthropic"
        _configure_anthropic(config)
    elif choice == "3":
        config["llm_provider"] = "ollama"
        _configure_ollama(config)
    else:
        print("Unknown provider.")
    print("\nSaved configuration.\n")


def _configure_openai(config: dict) -> None:
    section = config.setdefault("openai", {})
    print("\nOpenAI/Codex auth:")
    print("1. Sign in with ChatGPT (Codex CLI)")
    print("2. Use API key")
    choice = _prompt("Auth method", default="1")
    if choice == "1":
        _login_codex(section)
        config["llm_provider"] = "codex_cli"
        default_model = _default_codex_model(section)
        suggestions = _codex_model_suggestions(section, default_model)
        section["codex_model"] = _prompt_model_choice("Codex model", default_model, suggestions)
        return
    section["api_key"] = _prompt("API key", default=section.get("api_key", ""))
    config["llm_provider"] = "openai"
    section["base_url"] = _prompt("Base URL", default=section.get("base_url", "https://api.openai.com"))
    section["model"] = _prompt("Model", default=section.get("model", "gpt-4.1"))


def _configure_anthropic(config: dict) -> None:
    section = config.setdefault("anthropic", {})
    print("\nClaude auth:")
    print("1. Claude Code CLI login (claude auth login)")
    print("2. Claude subscription (setup-token)")
    print("3. Use API key")
    choice = _prompt("Auth method", default="1")
    if choice == "1":
        _login_claude_cli()
        config["llm_provider"] = "claude_cli"
        default_model = _default_claude_model(section)
        suggestions = _claude_model_suggestions(section, default_model)
        section["claude_model"] = _prompt_model_choice("Claude model", default_model, suggestions)
        return
    if choice == "2":
        _login_claude_setup_token(section)
        config["llm_provider"] = "anthropic"
        section["model"] = _prompt(
            "Model", default=section.get("model", "claude-3-5-sonnet-20240620")
        )
        section["base_url"] = _prompt(
            "Base URL", default=section.get("base_url", "https://api.anthropic.com")
        )
        return
    section["api_key"] = _prompt("API key", default=section.get("api_key", ""))
    config["llm_provider"] = "anthropic"
    section["model"] = _prompt(
        "Model", default=section.get("model", "claude-3-5-sonnet-20240620")
    )
    section["base_url"] = _prompt(
        "Base URL", default=section.get("base_url", "https://api.anthropic.com")
    )


def _configure_ollama(config: dict) -> None:
    section = config.setdefault("ollama", {})
    section["base_url"] = _prompt(
        "Base URL", default=section.get("base_url", "http://localhost:11434")
    )
    section["model"] = _prompt("Model", default=section.get("model", "llama3.1"))


def _configure_prover(config: dict) -> None:
    prove = config.setdefault("prove", {})
    mode = _prompt("Default proof mode (tactic|lemma|llm)", default=prove.get("mode", "tactic")).strip().lower()
    if mode not in {"tactic", "lemma", "llm"}:
        mode = "tactic"
    prove["mode"] = mode
    solver = _prompt(
        "Default solver (auto|search|script)",
        default=prove.get("solver", "script"),
    ).strip().lower()
    if solver not in {"auto", "search", "script"}:
        solver = "auto"
    prove["solver"] = solver
    autop_default = "y" if prove.get("autop", True) else "n"
    autop_choice = _prompt("Enable autop tactics (aesop/simp/linarith/ring) (Y/n)", default=autop_default).strip().lower()
    prove["autop"] = autop_choice not in {"n", "no", "false", "0"}
    suggestion_default = str(prove.get("k", 1))
    suggestion_raw = _prompt("Number of LLM suggestions per state", default=suggestion_default).strip()
    try:
        prove["k"] = max(1, int(suggestion_raw))
    except Exception:
        prove["k"] = 1
    llm_rounds_default = str(prove.get("llm_rounds", 4))
    llm_rounds_raw = _prompt("LLM-only max rounds", default=llm_rounds_default).strip()
    try:
        prove["llm_rounds"] = max(1, int(llm_rounds_raw))
    except Exception:
        prove["llm_rounds"] = 4
    llm_section = config.setdefault("llm", {})
    timeout_default = str(llm_section.get("timeout_s", 0))
    timeout_raw = _prompt("LLM request timeout (seconds, 0 = no timeout)", default=timeout_default).strip()
    try:
        llm_section["timeout_s"] = max(0, int(float(timeout_raw)))
    except Exception:
        llm_section["timeout_s"] = 0
    heartbeat_default = str(llm_section.get("heartbeat_s", 60))
    heartbeat_raw = _prompt("LLM heartbeat interval (seconds, 0 = off)", default=heartbeat_default).strip()
    try:
        llm_section["heartbeat_s"] = max(0, int(float(heartbeat_raw)))
    except Exception:
        llm_section["heartbeat_s"] = 60
    lemma_max = _prompt("Lemma max count", default=str(prove.get("lemma_max", 60))).strip()
    lemma_depth = _prompt("Lemma max depth", default=str(prove.get("lemma_depth", 60))).strip()
    try:
        prove["lemma_max"] = max(1, int(lemma_max))
    except Exception:
        prove["lemma_max"] = 60
    try:
        prove["lemma_depth"] = max(1, int(lemma_depth))
    except Exception:
        prove["lemma_depth"] = 60
    allow_axioms_default = "y" if prove.get("allow_axioms", False) else "n"
    allow_axioms_raw = _prompt("Allow axioms (Y/n)", default=allow_axioms_default).strip().lower()
    allow_axioms = allow_axioms_raw in {"y", "yes", "true", "1"}
    prove["allow_axioms"] = allow_axioms
    lean_section = config.setdefault("lean", {})
    dojo_timeout_default = str(lean_section.get("dojo_timeout_s", 180))
    dojo_timeout_raw = _prompt(
        "LeanDojo server startup timeout (seconds)",
        default=dojo_timeout_default,
    ).strip()
    try:
        lean_section["dojo_timeout_s"] = max(30, int(float(dojo_timeout_raw)))
    except Exception:
        lean_section["dojo_timeout_s"] = 180

    formalize = config.setdefault("formalize", {})
    formalize_mode_default = formalize.get("proof_backend", "inherit")
    formalize_mode = _prompt(
        "Formalize proof mode (inherit|tactic|lemma|llm)",
        default=formalize_mode_default,
    ).strip().lower()
    if formalize_mode not in {"inherit", "tactic", "lemma", "llm"}:
        formalize_mode = "inherit"
    formalize["proof_backend"] = formalize_mode
    effective_mode = prove.get("mode", "tactic") if formalize_mode == "inherit" else formalize_mode
    lean_backend_default = formalize.get(
        "lean_backend", "cli" if effective_mode == "llm" else "dojo"
    )
    lean_backend = _prompt(
        "Formalize typecheck backend (dojo|cli)",
        default=lean_backend_default,
    ).strip().lower()
    if lean_backend not in {"dojo", "cli"}:
        lean_backend = "dojo"
    formalize["lean_backend"] = lean_backend
    print("\nSaved prover settings.\n")


def _menu_prove(config: dict) -> None:
    instruction = _prompt_multiline("Enter guidance for the prover")
    print("Optional: provide a Lean file path to run immediately, or leave blank to auto-generate from text.")
    _print_lean_file_suggestions()
    file_path = _prompt("Lean file path (optional)", default="").strip()
    theorem = ""
    extra_paths = _prompt("Additional context files (.lean/.tex), comma-separated", default="")
    context_files = _parse_paths(extra_paths)
    prove_defaults = config.get("prove", {})
    prove_mode = _prompt(
        "Proof mode (tactic|lemma|llm)", default=prove_defaults.get("mode", "tactic")
    ).strip().lower()
    if prove_mode not in {"tactic", "lemma", "llm"}:
        prove_mode = "tactic"
    config.setdefault("prove", {})["mode"] = prove_mode
    save_config(config)

    if file_path:
        theorem = _prompt("Theorem name", default="").strip()
        if not theorem:
            print("Theorem name is required to run the prover.")
            return
        theorem = _sanitize_lean_name(theorem)
    else:
        statement = _prompt_multiline("Theorem statement (informal or Lean)")
        if not statement:
            _save_task(
                kind="nl_prove",
                payload={
                    "instruction": instruction,
                    "context_files": [str(p) for p in context_files],
                    "statement": "",
                },
            )
            print("Saved task only. Provide a statement to auto-generate Lean.")
            return
        theorem = _prompt("Theorem name", default="ulam_theorem").strip() or "ulam_theorem"
        theorem = _sanitize_lean_name(theorem)
        if not _ensure_llm_ready(config, allow_placeholder=False):
            _save_task(
                kind="nl_prove",
                payload={
                    "instruction": instruction,
                    "context_files": [str(p) for p in context_files],
                    "statement": statement,
                    "theorem": theorem,
                },
            )
            print("LLM not configured. Saved task only; configure an LLM to run.")
            return
        try:
            file_path = str(_generate_lean_stub(config, theorem, statement, context_files))
        except Exception as exc:
            _save_task(
                kind="nl_prove",
                payload={
                    "instruction": instruction,
                    "context_files": [str(p) for p in context_files],
                    "statement": statement,
                    "theorem": theorem,
                },
            )
            print(f"Failed to generate Lean stub: {exc}")
            return
        print(f"Generated Lean stub: {file_path}")

    if not _ensure_llm_ready(config, allow_placeholder=False):
        _save_task(
            kind="nl_prove",
            payload={
                "instruction": instruction,
                "context_files": [str(p) for p in context_files],
                "file_path": file_path,
                "theorem": theorem,
            },
        )
        print("LLM not configured. Saved task only; configure an LLM to run.")
        return

    proceed, lean_project = _ensure_lean_backend(
        config, Path(file_path), require_dojo=(prove_mode != "llm")
    )
    if not proceed:
        return

    args = _build_args_from_config(config, file_path, theorem, instruction, context_files)
    args.prove_mode = prove_mode
    if prove_mode == "llm":
        args.lean = "cli"
        args.lean_project = lean_project
    elif lean_project is None:
        args.lean = "mock"
        args.lean_project = None
        print("Running with mock Lean backend (no Lean project configured).")
    else:
        args.lean = "dojo"
        args.lean_project = lean_project
    try:
        print("Starting prover...")
        run_prove(args)
    except Exception as exc:
        print(f"Error: {exc}")


def _menu_formalize(config: dict) -> None:
    _ensure_llm_ready(config, allow_placeholder=True)
    tex_path = _prompt(".tex path", default="").strip()
    if not tex_path:
        print(".tex path is required.")
        return
    extra_paths = _prompt("Additional context files (.lean/.tex), comma-separated", default="")
    context_files = _parse_paths(extra_paths)
    output_path = _prompt("Output .lean path", default=_default_lean_output(tex_path))

    tex_file = Path(tex_path)
    if not tex_file.exists():
        print(f"File not found: {tex_file}")
        return

    payload = {
        "tex_path": tex_path,
        "context_files": [str(p) for p in context_files],
        "output_path": output_path,
    }
    _save_task(kind="formalize_tex", payload=payload)

    lean_project_raw = config.get("lean", {}).get("project", "")
    lean_project = Path(lean_project_raw) if lean_project_raw else None
    formalize_cfg = config.get("formalize", {})
    proof_backend = formalize_cfg.get("proof_backend", "inherit")
    if proof_backend == "inherit":
        proof_backend = config.get("prove", {}).get("mode", "tactic")
    if proof_backend == "dojo":
        proof_backend = "tactic"
    if proof_backend not in {"tactic", "lemma", "llm"}:
        proof_backend = "tactic"
    lean_backend = formalize_cfg.get("lean_backend", "cli" if proof_backend == "llm" else "dojo")
    dojo_timeout_s = float(config.get("lean", {}).get("dojo_timeout_s", 180))
    cfg = FormalizationConfig(
        tex_path=tex_file,
        output_path=Path(output_path),
        context_files=context_files,
        max_rounds=5,
        max_repairs=2,
        max_equivalence_repairs=2,
        max_proof_rounds=1,
        proof_max_steps=64,
        proof_beam=4,
        proof_k=int(config.get("prove", {}).get("k", 1)),
        proof_timeout_s=5.0,
        proof_repair=2,
        dojo_timeout_s=dojo_timeout_s,
        lemma_max=int(config.get("prove", {}).get("lemma_max", 60)),
        lemma_depth=int(config.get("prove", {}).get("lemma_depth", 60)),
        allow_axioms=bool(config.get("prove", {}).get("allow_axioms", False)),
        lean_project=lean_project,
        lean_imports=config.get("lean", {}).get("imports", []),
        verbose=True,
        proof_backend=proof_backend,
        lean_backend=lean_backend,
        resume_path=None,
        artifact_dir=None,
        equivalence_checks=True,
    )
    llm = FormalizationLLM(config.get("llm_provider", "openai"), config)
    engine = FormalizationEngine(cfg, llm)
    result = engine.run()
    print(f"Wrote: {result.output_path}")
    print(f"Typecheck: {'ok' if result.typecheck_ok else 'failed'}")
    print(f"Solved: {result.solved}, Remaining sorries: {result.remaining_sorries}")
    if result.artifact_dir:
        print(f"Artifacts: {result.artifact_dir}")


def _menu_formalize_resume(config: dict) -> None:
    artifact_dir = _find_latest_formalize_artifact()
    if not artifact_dir:
        print("No previous formalization runs found in runs/.")
        return
    snapshot = _load_formalize_snapshot(artifact_dir)
    if not snapshot:
        print(f"Could not read config.json from {artifact_dir}")
        return
    tex_path = Path(snapshot.get("tex_path", "")).expanduser()
    output_path = Path(snapshot.get("output_path", "")).expanduser()
    if not tex_path.exists():
        tex_path = Path(_prompt("Missing .tex path, enter path", default="")).expanduser()
    resume_path = _find_resume_lean_from_artifact(artifact_dir, output_path)
    if not resume_path:
        print("Could not locate a prior Lean file to resume from.")
        return
    print(f"Resuming from: {resume_path}")
    lean_project_raw = config.get("lean", {}).get("project", "")
    lean_project = Path(lean_project_raw) if lean_project_raw else None
    snapshot_project = snapshot.get("lean_project")
    if snapshot_project:
        candidate = Path(snapshot_project)
        if candidate.exists():
            lean_project = candidate
    context_files = [Path(p) for p in snapshot.get("context_files", []) if p]
    formalize_cfg = config.get("formalize", {})
    proof_backend = snapshot.get("proof_backend", formalize_cfg.get("proof_backend", "inherit"))
    if proof_backend == "inherit":
        proof_backend = config.get("prove", {}).get("mode", "tactic")
    if proof_backend == "dojo":
        proof_backend = "tactic"
    if proof_backend not in {"tactic", "lemma", "llm"}:
        proof_backend = "tactic"
    lean_backend = snapshot.get(
        "lean_backend",
        formalize_cfg.get("lean_backend", "cli" if proof_backend == "llm" else "dojo"),
    )
    dojo_timeout_s = float(config.get("lean", {}).get("dojo_timeout_s", 180))
    cfg = FormalizationConfig(
        tex_path=tex_path,
        output_path=output_path,
        context_files=context_files,
        max_rounds=max(5, int(snapshot.get("max_rounds", 5))),
        max_repairs=int(snapshot.get("max_repairs", 2)),
        max_equivalence_repairs=int(snapshot.get("max_equivalence_repairs", 2)),
        max_proof_rounds=int(snapshot.get("max_proof_rounds", 1)),
        proof_max_steps=int(snapshot.get("proof_max_steps", 64)),
        proof_beam=int(snapshot.get("proof_beam", 4)),
        proof_k=int(snapshot.get("proof_k", 8)),
        proof_timeout_s=float(snapshot.get("proof_timeout_s", 5.0)),
        proof_repair=int(snapshot.get("proof_repair", 2)),
        dojo_timeout_s=float(snapshot.get("dojo_timeout_s", dojo_timeout_s)),
        lemma_max=int(snapshot.get("lemma_max", config.get("prove", {}).get("lemma_max", 60))),
        lemma_depth=int(snapshot.get("lemma_depth", config.get("prove", {}).get("lemma_depth", 60))),
        allow_axioms=bool(
            snapshot.get("allow_axioms", config.get("prove", {}).get("allow_axioms", False))
        ),
        lean_project=lean_project,
        lean_imports=config.get("lean", {}).get("imports", []),
        verbose=True,
        proof_backend=proof_backend,
        lean_backend=lean_backend,
        resume_path=resume_path,
        artifact_dir=None,
        equivalence_checks=bool(snapshot.get("equivalence_checks", True)),
    )
    llm = FormalizationLLM(config.get("llm_provider", "openai"), config)
    engine = FormalizationEngine(cfg, llm)
    result = engine.run()
    print(f"Wrote: {result.output_path}")
    print(f"Typecheck: {'ok' if result.typecheck_ok else 'failed'}")
    print(f"Solved: {result.solved}, Remaining sorries: {result.remaining_sorries}")
    if result.artifact_dir:
        print(f"Artifacts: {result.artifact_dir}")


def _find_latest_formalize_artifact() -> Path | None:
    root = Path("runs")
    if not root.exists():
        return None
    candidates = [path for path in root.glob("formalize_*") if path.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_formalize_snapshot(artifact_dir: Path) -> dict | None:
    path = artifact_dir / "config.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _find_resume_lean_from_artifact(artifact_dir: Path, output_path: Path) -> Path | None:
    if output_path and output_path.exists():
        return output_path
    rounds = sorted(
        [p for p in artifact_dir.glob("round_*") if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    for round_dir in rounds:
        for name in ("improve.lean", "repair.lean", "start.lean"):
            candidate = round_dir / name
            if candidate.exists():
                return candidate
    return None


def _build_args_from_config(
    config: dict,
    file_path: str,
    theorem: str,
    instruction: str,
    context_files: list[Path],
):
    provider = config.get("llm_provider", "openai")
    openai = config.get("openai", {})
    ollama = config.get("ollama", {})
    anthropic = config.get("anthropic", {})
    embed = config.get("embed", {})
    lean = config.get("lean", {})
    prove = config.get("prove", {})
    openai_model = openai.get("model", "gpt-4.1")
    if provider == "codex_cli":
        openai_model = openai.get("codex_model") or openai_model or _default_codex_model(openai)
    anthropic_model = anthropic.get("model", "")
    if provider == "claude_cli":
        anthropic_model = anthropic.get("claude_model") or anthropic_model

    from argparse import Namespace

    return Namespace(
        file=Path(file_path),
        theorem=theorem,
        llm=provider,
        lean="dojo" if _infer_use_lean(lean) else "mock",
        premises=None,
        retriever="none",
        max_steps=64,
        beam=4,
        k=int(prove.get("k", 1)),
        llm_rounds=int(prove.get("llm_rounds", 4)),
        timeout=5.0,
        repair=2,
        seed=0,
        trace=Path("run.jsonl"),
        instruction=instruction,
        context=context_files,
        prove_mode=prove.get("mode", "tactic"),
        solver=prove.get("solver", "script"),
        autop=bool(prove.get("autop", True)),
        lemma_max=int(prove.get("lemma_max", 60)),
        lemma_depth=int(prove.get("lemma_depth", 60)),
        allow_axioms=bool(prove.get("allow_axioms", False)),
        openai_key=openai.get("api_key", "") or os.environ.get("ULAM_OPENAI_API_KEY", ""),
        openai_base_url=openai.get("base_url", "https://api.openai.com"),
        openai_model=openai_model,
        ollama_base_url=ollama.get("base_url", "http://localhost:11434"),
        ollama_model=ollama.get("model", "llama3.1"),
        embed_api_key=embed.get("api_key", "") or os.environ.get("ULAM_EMBED_API_KEY", ""),
        embed_base_url=embed.get("base_url", "https://api.openai.com"),
        embed_model=embed.get("model", "text-embedding-3-small"),
        embed_cache=Path(embed.get("cache", ".ulam/embeddings.json")),
        embed_batch_size=16,
        lean_project=Path(lean.get("project", "")) if lean.get("project") else None,
        lean_import=lean.get("imports", []),
        anthropic_key=anthropic.get("api_key", "") or os.environ.get("ULAM_ANTHROPIC_API_KEY", ""),
        anthropic_setup_token=anthropic.get("setup_token", "") or os.environ.get("ULAM_ANTHROPIC_SETUP_TOKEN", ""),
        anthropic_base_url=anthropic.get("base_url", "https://api.anthropic.com"),
        anthropic_model=anthropic_model,
        verbose=True,
    )


def _infer_use_lean(lean: dict) -> bool:
    project = lean.get("project", "")
    return bool(project)


def _prompt(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value if value else default


def _prompt_multiline(label: str) -> str:
    print(f"{label}")
    print("(Press Enter on an empty line to finish.)")
    lines = []
    while True:
        line = input("> ")
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _parse_paths(raw: str) -> list[Path]:
    if not raw.strip():
        return []
    return [Path(part.strip()) for part in raw.split(",") if part.strip()]


def _default_lean_output(tex_path: str) -> str:
    path = Path(tex_path)
    return str(path.with_suffix(".lean"))


def _print_lean_file_suggestions(max_items: int = 5) -> None:
    candidates: list[Path] = []
    cwd = Path.cwd()
    for root in (cwd, cwd / "ulam-lean" / "UlamAI"):
        if root.exists() and root.is_dir():
            candidates.extend(sorted(root.glob("*.lean")))
    seen: set[str] = set()
    deduped: list[Path] = []
    for path in candidates:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    if not deduped:
        example = cwd / "ulam-lean" / "UlamAI" / "proof.lean"
        print(f"Example: {example}")
        return
    print("Recent .lean files:")
    for path in deduped[:max_items]:
        rel = path
        try:
            rel = path.relative_to(cwd)
        except Exception:
            pass
        print(f"  - {rel}")


def _save_task(kind: str, payload: dict) -> None:
    runs_dir = Path("runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = runs_dir / f"{kind}_{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _generate_lean_stub(
    config: dict, theorem: str, statement: str, context_files: list[Path]
) -> Path:
    context = _read_context_files(context_files, max_chars=6000)
    llm = FormalizationLLM(config.get("llm_provider", "openai"), config)
    text = statement.strip()
    if _looks_like_lean_statement(text):
        lean_statement = _extract_lean_statement(text)
    else:
        lean_statement = llm.statement(text, context).strip()
        if lean_statement:
            lean_statement = _extract_lean_statement(lean_statement)
    if not lean_statement:
        raise RuntimeError("LLM did not return a Lean statement.")

    lean_code = _wrap_statement(theorem, lean_statement, original=statement)
    output_path = _generated_lean_path(config, theorem)
    output_path.write_text(lean_code, encoding="utf-8")
    return output_path


def _generated_lean_path(config: dict, theorem: str) -> Path:
    lean_project = config.get("lean", {}).get("project", "")
    if lean_project:
        root = Path(lean_project)
        out_dir = root / "UlamAI"
    else:
        out_dir = Path("runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = _sanitize_filename(theorem or "theorem")
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return out_dir / f"{safe}_{timestamp}.lean"


def _wrap_statement(theorem: str, statement: str, original: str | None = None) -> str:
    stmt = statement.strip().rstrip(".")
    if not stmt:
        return ""
    header = ""
    if original:
        header = "/- ULAMAI_ORIGINAL_STATEMENT\n" + original.strip() + "\n-/\n\n"
    return (
        header
        + "import Mathlib\n\n"
        + f"theorem {theorem} : {stmt} := by\n"
        + "  sorry\n"
    )


def _looks_like_lean_statement(text: str) -> bool:
    lowered = text.lower()
    if any(token in lowered for token in ("theorem ", "lemma ", "example ", ":=")):
        return True
    if any(token in text for token in ("∀", "∃", "→", "↔", "by")):
        return True
    return False


def _extract_lean_statement(text: str) -> str:
    stripped = _strip_code_fences(text.strip())
    if not stripped:
        return ""
    match = re.search(r"\b(theorem|lemma|example)\s+\w+\s*:\s*", stripped)
    if match:
        remainder = stripped[match.end():]
        if ":=" in remainder:
            remainder = remainder.split(":=", 1)[0]
        return remainder.strip()
    if ":=" in stripped:
        return stripped.split(":=", 1)[0].strip()
    return stripped


def _strip_code_fences(text: str) -> str:
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            return parts[1].strip()
    return text


def _sanitize_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return safe or "theorem"


def _read_context_files(paths: list[Path], max_chars: int = 8000) -> str:
    blocks: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        if len(content) > max_chars:
            content = content[:max_chars] + "\n-- (truncated)"
        blocks.append(f"[file: {path}]\n{content}")
    return "\n\n".join(blocks)


def _sanitize_lean_name(value: str) -> str:
    if not value:
        return "ulam_theorem"
    name = re.sub(r"[^A-Za-z0-9_']+", "_", value)
    if not re.match(r"[A-Za-z_]", name[:1]):
        name = f"theorem_{name}"
    return name or "ulam_theorem"


def _ensure_lean_backend(
    config: dict, file_path: Path, require_dojo: bool = True
) -> tuple[bool, Path | None]:
    lean = config.setdefault("lean", {})
    project_raw = lean.get("project", "")
    project_path = Path(project_raw).expanduser() if project_raw else None
    if project_path and _looks_like_lean_project(project_path):
        if require_dojo and not _lean_dojo_available():
            return _handle_missing_dojo()
        return True, project_path

    detected = _find_lean_project(file_path)
    if not detected:
        detected = _find_lean_project(Path.cwd())
    if detected:
        lean["project"] = str(detected)
        save_config(config)
        print(f"Detected Lean project: {detected}")
        if require_dojo and not _lean_dojo_available():
            return _handle_missing_dojo()
        return True, detected

    print("No Lean project detected.")
    _print_lean_install_instructions()
    choice = _prompt("Lean project path (blank to use mock backend)", default="")
    if not choice:
        return True, None
    chosen = Path(choice).expanduser()
    if not _looks_like_lean_project(chosen):
        print("No lakefile.lean or lean-toolchain found in that path. Using mock backend.")
        return True, None
    lean["project"] = str(chosen)
    save_config(config)
    if require_dojo and not _lean_dojo_available():
        return _handle_missing_dojo()
    return True, chosen


def _handle_missing_dojo() -> tuple[bool, Path | None]:
    print("LeanDojo is not installed. Install Lean + LeanDojo to enable the real backend.")
    _print_lean_install_instructions()
    choice = _prompt("Use mock backend for this run? (y/N)", default="n").strip().lower()
    if choice in {"y", "yes"}:
        return True, None
    print("Aborting run. Install LeanDojo or configure a Lean project later.")
    return False, None


def _lean_dojo_available() -> bool:
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
    if (path / "lean-toolchain").exists():
        return True
    return False


def _find_lean_project(start: Path) -> Path | None:
    root = start if start.is_dir() else start.parent
    for parent in [root, *root.parents]:
        if _looks_like_lean_project(parent):
            return parent
    return None


def _print_lean_install_instructions() -> None:
    print("Quick setup:")
    print("  ulam -lean")
    print("Lean + Lake (macOS/Linux):")
    print("  curl https://elan.lean-lang.org/elan-init.sh -sSf | sh")
    print("  source $HOME/.elan/env")
    print("Create a Mathlib project:")
    print("  lake +leanprover-community/mathlib4:lean-toolchain new MyMathlibProject math")
    print("  cd MyMathlibProject && lake build")
    print("LeanDojo + Pantograph:")
    print("  pip install lean-dojo-v2")
    print("  pip install git+https://github.com/stanford-centaur/PyPantograph")


def _prompt_model_choice(label: str, default: str, suggestions: list[str]) -> str:
    options = _dedupe([option for option in suggestions if option])
    if options:
        print("\nSuggested models:")
        for idx, model in enumerate(options, start=1):
            print(f"{idx}. {model}")
        print(f"{len(options) + 1}. Enter custom")
        choice = _prompt(f"{label} choice", default="1")
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(options):
                return options[idx - 1]
            if idx == len(options) + 1:
                return _prompt(label, default=default)
        if choice:
            return choice
    return _prompt(label, default=default)


def _default_codex_model(section: dict) -> str:
    explicit = section.get("codex_model")
    if explicit:
        return explicit
    cache = _codex_models_from_cache(limit=1)
    if cache:
        return cache[0]
    from_config = _codex_models_from_config()
    if from_config:
        return from_config[0]
    return "gpt-5.2-codex"


def _codex_model_suggestions(section: dict, default: str) -> list[str]:
    suggestions = [
        default,
        section.get("codex_model", ""),
        section.get("model", ""),
    ]
    suggestions += _split_models(os.environ.get("ULAM_CODEX_MODELS", ""))
    suggestions += _codex_models_from_cache()
    suggestions += _codex_models_from_config()
    return _dedupe(suggestions)[:8]


def _codex_models_from_cache(limit: int = 6) -> list[str]:
    path = _codex_home() / "models_cache.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    models = data.get("models")
    if not isinstance(models, list):
        return []
    ranked: list[tuple[int, str]] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        slug = model.get("slug")
        if not isinstance(slug, str):
            continue
        visibility = model.get("visibility")
        if visibility and visibility != "list":
            continue
        priority = model.get("priority", 999)
        ranked.append((priority, slug))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [slug for _, slug in ranked[:limit]]


def _codex_models_from_config() -> list[str]:
    path = _codex_home() / "config.toml"
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return []
    models = re.findall(r'(?m)^\s*model\s*=\s*"([^"]+)"', text)
    migrations: list[str] = []
    in_migrations = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_migrations = stripped == "[notice.model_migrations]"
            continue
        if not in_migrations:
            continue
        match = re.match(r'^"([^"]+)"\s*=\s*"([^"]+)"', stripped)
        if match:
            migrations.extend([match.group(1), match.group(2)])
    return _dedupe(models + migrations)


def _claude_model_suggestions(section: dict, default: str) -> list[str]:
    suggestions = [
        default,
        section.get("claude_model", ""),
        section.get("model", ""),
    ]
    suggestions += _split_models(os.environ.get("ULAM_CLAUDE_MODELS", ""))
    suggestions += _claude_models_from_stats()
    return _dedupe(suggestions)[:8]


def _default_claude_model(section: dict) -> str:
    explicit = section.get("claude_model")
    if explicit:
        return explicit
    stats = _claude_models_from_stats(limit=1)
    if stats:
        return stats[0]
    if section.get("model"):
        return section["model"]
    return "claude-3-5-sonnet-20240620"


def _claude_models_from_stats(limit: int = 6) -> list[str]:
    path = Path("~/.claude/stats-cache.json").expanduser()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    usage = data.get("modelUsage")
    if not isinstance(usage, dict):
        return []
    ranked: list[tuple[int, str]] = []
    for model_name, stats in usage.items():
        if not isinstance(model_name, str):
            continue
        total = 0
        if isinstance(stats, dict):
            for key in (
                "inputTokens",
                "outputTokens",
                "cacheReadInputTokens",
                "cacheCreationInputTokens",
            ):
                value = stats.get(key, 0)
                if isinstance(value, (int, float)):
                    total += int(value)
        ranked.append((-total, model_name))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [name for _, name in ranked[:limit]]


def _split_models(raw: str) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _codex_home() -> Path:
    base = os.environ.get("CODEX_HOME", "~/.codex")
    return Path(base).expanduser()




def _login_codex(section: dict) -> None:
    print("Launching Codex CLI login...")
    try:
        run_codex_login()
    except Exception as exc:
        print(f"Codex login failed: {exc}")
        print("Install Codex CLI first (e.g., `npm i -g @openai/codex`).")
        return
    auth_path = codex_auth_path()
    api_key = load_codex_api_key(auth_path)
    if api_key:
        section["api_key"] = api_key
        print("Codex login imported API key successfully.")
        return
    tokens = load_codex_tokens(auth_path)
    if tokens:
        print("Codex login successful (subscription token detected).")
        return
    print(f"Could not read credentials from {auth_path}.")
    print("If your Codex CLI uses a different auth file, set CODEX_HOME.")


def _login_claude_setup_token(section: dict) -> None:
    print("Launching Claude Code setup-token...")
    try:
        token = run_claude_setup_token()
    except Exception as exc:
        print(f"Claude setup-token failed: {exc}")
        print("Install Claude Code first (e.g., `npm i -g @anthropic-ai/claude-code`).")
        return
    if not token:
        token = _prompt("Paste setup-token (from `claude setup-token`)", default="")
    if not token:
        print("No setup-token captured.")
        return
    section["setup_token"] = token
    print("Claude setup-token saved.")


def _login_claude_cli() -> None:
    print("Launching Claude Code login...")
    try:
        run_claude_login()
    except Exception as exc:
        print(f"Claude login failed: {exc}")


def _print_banner() -> None:
    banner = [
        " _    _ _                          _____ ",
        "| |  | | |                   /\\   |_   _|",
        "| |  | | | __ _ _ __ ___    /  \\    | |  ",
        "| |  | | |/ _` | '_ ` _ \\  / /\\ \\   | |  ",
        "| |__| | | (_| | | | | | |/ ____ \\ _| |_ ",
        " \\____/|_|\\__,_|_| |_| |_|_/    \\_\\_____|",
    ]
    if os.isatty(1):
        color = "\033[38;2;137;207;240m"
        reset = "\033[0m"
        for line in banner:
            print(f"{color}{line}{reset}")
    else:
        for line in banner:
            print(line)


def _provider_label(config: dict) -> str:
    provider = config.get("llm_provider", "openai")
    if provider == "openai":
        key = config.get("openai", {}).get("api_key", "") or os.environ.get("ULAM_OPENAI_API_KEY", "")
        return "Codex/OpenAI" if key else "Codex/OpenAI (no API key)"
    if provider == "codex_cli":
        tokens = load_codex_tokens()
        return "Codex CLI (subscription)" if tokens else "Codex CLI (not logged in)"
    if provider == "anthropic":
        section = config.get("anthropic", {})
        key = section.get("api_key", "") or os.environ.get("ULAM_ANTHROPIC_API_KEY", "")
        token = section.get("setup_token", "") or os.environ.get("ULAM_ANTHROPIC_SETUP_TOKEN", "")
        label = "Claude (Anthropic)"
        if token:
            label += " (setup-token)"
        elif key:
            label += " (api key)"
        else:
            label += " (no credentials)"
        return label
    if provider == "ollama":
        base_url = config.get("ollama", {}).get("base_url", "")
        return "Ollama" if base_url else "Ollama (no base URL)"
    if provider == "claude_cli":
        return "Claude Code CLI"
    return provider


def _ensure_llm_ready(config: dict, allow_placeholder: bool) -> bool:
    provider = config.get("llm_provider", "openai")
    if provider == "openai":
        key = config.get("openai", {}).get("api_key", "") or os.environ.get("ULAM_OPENAI_API_KEY", "")
        if not key:
            print("No OpenAI API key configured. Choose option 1 to configure it.")
            return False
        return True
    if provider == "codex_cli":
        if not _command_exists("codex"):
            print("Codex CLI is not installed. Install it with: npm i -g @openai/codex")
            return False
        if not load_codex_tokens():
            print("Codex CLI not logged in. Choose option 1 to configure it.")
            return False
        return True
    if provider == "anthropic":
        section = config.get("anthropic", {})
        key = section.get("api_key", "") or os.environ.get("ULAM_ANTHROPIC_API_KEY", "")
        token = section.get("setup_token", "") or os.environ.get("ULAM_ANTHROPIC_SETUP_TOKEN", "")
        if not key and not token:
            print("No Claude credentials configured. Choose option 1 to configure it.")
            return False
        return True
    if provider == "claude_cli":
        if not _command_exists("claude"):
            print("Claude Code CLI is not installed. Install it with: npm i -g @anthropic-ai/claude-code")
            return False
        return True
    if provider == "ollama":
        base_url = config.get("ollama", {}).get("base_url", "")
        if not base_url:
            print("Ollama base URL is missing. Choose option 1 to configure it.")
            return False
        return True
    return False


def _command_exists(cmd: str) -> bool:
    from shutil import which

    return which(cmd) is not None
