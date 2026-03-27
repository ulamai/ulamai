from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import statistics
import hashlib
import json
import random
import os
import re
import shutil
import subprocess
import sys
import time
import fnmatch
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from . import __version__
from .lean.base import LeanRunner
from .lean.dojo import LeanDojoRunner
from .lean.lsp import lean_lsp_check, lean_lsp_diagnostics
from .lean.lsp_runner import LeanLspRunner
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
    build_premise_index,
    load_index_premises,
    load_index_stats,
)
from .search import SearchResult, best_first_search, scripted_search
from .state import state_hash
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
    prove.add_argument(
        "file",
        nargs="?",
        type=Path,
        help="path to a Lean file (required for --output-format lean)",
    )
    prove.add_argument("--theorem", required=True, help="theorem name to prove")
    prove.add_argument(
        "--output-format",
        choices=["lean", "tex"],
        default="lean",
        help="proof output format",
    )
    prove.add_argument(
        "--statement",
        default="",
        help="optional informal theorem statement (primarily for --output-format tex)",
    )
    prove.add_argument(
        "--tex-out",
        type=Path,
        default=None,
        help="output .tex path when using --output-format tex",
    )
    prove.add_argument(
        "--tex-rounds",
        type=int,
        default=None,
        help="planner/worker/judge rounds for --output-format tex",
    )
    prove.add_argument(
        "--tex-judge-repairs",
        type=int,
        default=None,
        help="max consecutive judge-directed repair rounds for --output-format tex",
    )
    prove.add_argument(
        "--tex-worker-drafts",
        type=int,
        default=None,
        help="worker drafts per round for --output-format tex",
    )
    prove.add_argument(
        "--tex-concurrency",
        dest="tex_concurrency",
        action="store_true",
        default=None,
        help="enable concurrent TeX worker evaluation for --output-format tex",
    )
    prove.add_argument(
        "--no-tex-concurrency",
        dest="tex_concurrency",
        action="store_false",
        help="disable concurrent TeX worker evaluation for --output-format tex",
    )
    prove.add_argument(
        "--tex-replan-passes",
        type=int,
        default=None,
        help="number of decomposition replan passes after stall",
    )
    prove.add_argument(
        "--tex-action-steps",
        type=int,
        default=None,
        help="bounded planner action steps for --output-format tex",
    )
    prove.add_argument(
        "--tex-planner-model",
        default="",
        help="optional provider model override for TeX planner/judge/verifier/compose",
    )
    prove.add_argument(
        "--tex-worker-model",
        default="",
        help="optional provider model override for TeX claim-draft workers",
    )
    prove.add_argument(
        "--tex-artifacts-dir",
        type=Path,
        default=None,
        help="directory root for prove-tex artifacts (snapshots/events)",
    )
    prove.add_argument(
        "--tex-resume",
        type=Path,
        default=None,
        help="resume prove-tex from a prior artifacts dir or state.json snapshot",
    )
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
    prove.add_argument("--lean", choices=["mock", "dojo", "cli", "lsp"], default="mock")
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
        help="retriever type",
    )
    prove.add_argument(
        "--retriever-source",
        choices=["local", "mathlib", "both"],
        default="local",
        help="auto-index source when --premises is not provided",
    )
    prove.add_argument(
        "--retriever-build",
        choices=["auto", "always", "never"],
        default="auto",
        help="index build policy when using auto-indexed retrieval",
    )
    prove.add_argument(
        "--retriever-index",
        type=Path,
        default=None,
        help="path to retrieval index jsonl (default: .ulam/premises_<source>.jsonl in project)",
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
    prove.add_argument(
        "--inference-profile",
        choices=["default", "balanced", "explore", "verify"],
        default="default",
        help="inference preset for generate/rank/verify budgets",
    )
    prove.add_argument(
        "--gen-k",
        type=int,
        default=0,
        help="candidate generation budget per state (0 = profile default)",
    )
    prove.add_argument(
        "--exec-k",
        type=int,
        default=0,
        help="max executed tactics per state after ranking (0 = profile default/unlimited)",
    )
    prove.add_argument(
        "--verify-level",
        choices=["auto", "none", "light", "strict"],
        default="auto",
        help="tactic verification strictness before execution",
    )
    prove.add_argument("--llm-rounds", type=int, default=4, help="LLM-only max rounds")
    prove.add_argument(
        "--llm-cycle-patience",
        type=int,
        default=None,
        help="rounds without progress before forcing a replan hint",
    )
    prove.add_argument(
        "--proof-profile",
        choices=["fast", "balanced", "strict", "normal"],
        default=None,
        help="risk/speed profile (normal is accepted as an alias of balanced)",
    )
    prove.add_argument(
        "--llm-allow-helper-lemmas",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="allow LLM mode to add helper declarations outside target theorem",
    )
    prove.add_argument(
        "--llm-edit-scope",
        choices=["full", "errors_only"],
        default=None,
        help="LLM mode edit scope (full or lock declarations without placeholders)",
    )
    prove.add_argument("--timeout", type=float, default=5.0, help="tactic timeout (seconds)")
    prove.add_argument(
        "--typecheck-timeout",
        type=float,
        default=None,
        help="Lean typecheck timeout in LLM mode (seconds)",
    )
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
        default=os.environ.get("ULAM_GEMINI_MODEL", "gemini-3.1-pro-preview"),
    )

    replay = sub.add_parser("replay", help="replay or summarize a run trace")
    replay.add_argument("trace", type=Path, help="trace jsonl path")
    replay.add_argument("--meta", type=Path, default=None, help="optional trace metadata path")
    replay.add_argument("--execute", action="store_true", help="execute trace deterministically")
    replay.add_argument("--strict", action="store_true", help="fail on any mismatch")
    replay.add_argument("--align-toolchain", action="store_true", help="attempt toolchain alignment from metadata")
    replay.add_argument("--file", type=Path, default=None, help="override Lean file for replay")
    replay.add_argument("--theorem", default="", help="override theorem name for replay")
    replay.add_argument("--lean", choices=["mock", "dojo"], default=None, help="override Lean backend")
    replay.add_argument("--lean-project", type=Path, default=None, help="override Lean project root")
    replay.add_argument("--lean-import", action="append", default=[], help="additional Lean imports")
    replay.add_argument("--timeout", type=float, default=5.0, help="tactic timeout for execute replay")

    checkpoint = sub.add_parser("checkpoint", help="read-only proof health check for a Lean file")
    checkpoint.add_argument("file", type=Path, help="path to Lean file")
    checkpoint.add_argument("--theorem", default="", help="optional theorem/lemma name to focus on")
    checkpoint.add_argument("--trace", type=Path, default=Path("run.jsonl"), help="optional run trace JSONL")
    checkpoint.add_argument("--lean-project", type=Path, default=None, help="Lean project root")
    checkpoint.add_argument("--lean-import", action="append", default=[], help="additional Lean imports")
    checkpoint.add_argument(
        "--proof-profile",
        choices=["fast", "balanced", "strict", "normal"],
        default=None,
        help="risk/speed profile for checkpoint criteria (normal = balanced)",
    )
    checkpoint.add_argument(
        "--allow-axioms",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="allow axioms/constants anywhere (default: enabled unless strict profile)",
    )
    checkpoint.add_argument(
        "--typecheck-timeout",
        type=float,
        default=None,
        help="Lean typecheck timeout in checkpoint mode (seconds)",
    )
    checkpoint.add_argument("--strict", action="store_true", help="exit non-zero when blockers are found")
    checkpoint.add_argument("--out-json", type=Path, default=None, help="write checkpoint report JSON")

    review = sub.add_parser("review", help="read-only run review with actionable next steps")
    review.add_argument("--trace", type=Path, default=Path("run.jsonl"), help="run trace JSONL")
    review.add_argument("--file", type=Path, default=None, help="optional Lean file for declaration stats")
    review.add_argument("--theorem", default="", help="optional theorem/lemma name to focus on")
    review.add_argument("--max-lines", type=int, default=800, help="max trace lines to inspect")
    review.add_argument("--out-json", type=Path, default=None, help="write review report JSON")

    auth = sub.add_parser("auth", help="authenticate with Codex, Claude, or Gemini CLI")
    auth.add_argument("provider", choices=["codex", "claude", "gemini"])

    formalize = sub.add_parser("formalize", help="formalize a .tex document to Lean")
    formalize.add_argument("tex", type=Path, help="path to .tex file")
    formalize.add_argument("--out", type=Path, default=None, help="output .lean path")
    formalize.add_argument("--context", action="append", default=[], help="context files (.lean/.tex)")
    formalize.add_argument("--max-rounds", type=int, default=None)
    formalize.add_argument(
        "--max-repairs",
        type=int,
        default=None,
        help="max typecheck repairs (default: same as --max-rounds)",
    )
    formalize.add_argument("--max-equivalence-repairs", type=int, default=2)
    formalize.add_argument("--max-proof-rounds", type=int, default=None)
    formalize.add_argument("--proof-max-steps", type=int, default=64)
    formalize.add_argument("--proof-beam", type=int, default=4)
    formalize.add_argument("--proof-k", type=int, default=1)
    formalize.add_argument("--proof-timeout", type=float, default=5.0)
    formalize.add_argument(
        "--proof-profile",
        choices=["fast", "balanced", "strict", "normal"],
        default=None,
        help="risk/speed profile (normal is accepted as an alias of balanced)",
    )
    formalize.add_argument("--proof-repair", type=int, default=None)
    formalize.add_argument(
        "--typecheck-timeout",
        type=float,
        default=None,
        help="Lean typecheck timeout in formalize mode (seconds)",
    )
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
        help="proof backend (tactic/lemma use LeanDojo, llm uses Lean typecheck loop)",
    )
    formalize.add_argument(
        "--lean-backend",
        choices=["dojo", "cli", "lsp"],
        default="dojo",
        help="typecheck backend (dojo uses Pantograph, cli/lsp use Lean tooling)",
    )
    formalize.add_argument("--no-equivalence", action="store_true", help="skip equivalence checks")
    formalize.add_argument(
        "--llm-check",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="run semantic integrity LLM check on generated Lean (default: enabled)",
    )
    formalize.add_argument(
        "--llm-check-timing",
        choices=["end", "mid+end"],
        default=None,
        help="when to run semantic integrity check",
    )
    formalize.add_argument(
        "--llm-check-repairs",
        type=int,
        default=None,
        help="max semantic integrity repair attempts",
    )
    formalize.add_argument("--artifacts-dir", type=Path, default=None)
    formalize.add_argument("--verbose", action="store_true")

    bench = sub.add_parser("bench", help="run a regression suite")
    bench.add_argument(
        "--suite",
        type=Path,
        required=True,
        help="suite alias or jsonl path (run `ulam bench-list-suites`)",
    )
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
    bench.add_argument("--lean", choices=["mock", "dojo", "lsp"], default="mock")
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
    bench.add_argument(
        "--retriever-source",
        choices=["local", "mathlib", "both"],
        default="local",
    )
    bench.add_argument(
        "--retriever-build",
        choices=["auto", "always", "never"],
        default="auto",
    )
    bench.add_argument("--retriever-index", type=Path, default=None)
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
    bench.add_argument(
        "--inference-profile",
        choices=["default", "balanced", "explore", "verify"],
        default="default",
        help="inference preset for generate/rank/verify budgets",
    )
    bench.add_argument(
        "--gen-k",
        type=int,
        default=0,
        help="candidate generation budget per state (0 = profile default)",
    )
    bench.add_argument(
        "--exec-k",
        type=int,
        default=0,
        help="max executed tactics per state after ranking (0 = profile default/unlimited)",
    )
    bench.add_argument(
        "--verify-level",
        choices=["auto", "none", "light", "strict"],
        default="auto",
        help="tactic verification strictness before execution",
    )
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
    bench.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="write structured benchmark report to JSON",
    )
    bench.add_argument(
        "--report-markdown",
        type=Path,
        default=None,
        help="write benchmark summary report to Markdown",
    )
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
        default=os.environ.get("ULAM_GEMINI_MODEL", "gemini-3.1-pro-preview"),
    )

    bench_list = sub.add_parser("bench-list-suites", help="list known benchmark suite aliases")
    bench_list.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="optional path to write suite registry snapshot JSON",
    )

    bench_validate = sub.add_parser("bench-validate", help="validate a benchmark suite jsonl")
    bench_validate.add_argument(
        "--suite",
        type=Path,
        required=True,
        help="suite alias or jsonl path (run `ulam bench-list-suites`)",
    )
    bench_validate.add_argument(
        "--no-theorem-check",
        action="store_true",
        help="skip theorem declaration existence check",
    )
    bench_validate.add_argument(
        "--max-errors",
        type=int,
        default=25,
        help="maximum number of validation errors to print",
    )
    bench_compare = sub.add_parser("bench-compare", help="compare two benchmark report JSON files")
    bench_compare.add_argument("--a", type=Path, required=True, help="baseline report JSON path")
    bench_compare.add_argument("--b", type=Path, required=True, help="candidate report JSON path")
    bench_compare.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="optional path to write comparison JSON",
    )
    bench_compare.add_argument(
        "--out-markdown",
        type=Path,
        default=None,
        help="optional path to write comparison Markdown",
    )
    bench_compare.add_argument(
        "--gate",
        action="store_true",
        help="enforce parity gate and exit non-zero on metric regressions",
    )
    bench_compare.add_argument(
        "--max-solved-drop",
        type=float,
        default=0.0,
        help="maximum allowed drop in solved count (B vs A)",
    )
    bench_compare.add_argument(
        "--max-success-rate-drop",
        type=float,
        default=0.0,
        help="maximum allowed drop in success rate percentage points (B vs A)",
    )
    bench_compare.add_argument(
        "--max-semantic-pass-rate-drop",
        type=float,
        default=0.0,
        help="maximum allowed drop in semantic pass rate percentage points (B vs A)",
    )
    bench_compare.add_argument(
        "--max-regression-rejection-rate-increase",
        type=float,
        default=0.0,
        help="maximum allowed increase in regression rejection rate percentage points (B vs A)",
    )
    bench_compare.add_argument(
        "--max-median-time-increase-pct",
        type=float,
        default=25.0,
        help="maximum allowed increase in median duration percent (B vs A)",
    )
    bench_compare.add_argument(
        "--max-planner-replan-triggers-increase",
        type=float,
        default=0.0,
        help="maximum allowed increase in planner replan triggers (B vs A)",
    )
    bench_compare.add_argument(
        "--max-planner-cached-tactic-tries-drop",
        type=float,
        default=0.0,
        help="maximum allowed drop in planner cached tactic tries (B vs A)",
    )
    bench_compare.add_argument(
        "--allow-profile-mismatch",
        action="store_true",
        help="allow parity gate to pass even when inference profile/budgets differ",
    )
    bench_compare.add_argument(
        "--allow-suite-mismatch",
        action="store_true",
        help="allow parity gate to pass even when suite SHA256 differs or is unavailable",
    )
    bench_make_minif2f = sub.add_parser(
        "bench-make-minif2f",
        help="build a miniF2F benchmark suite JSONL from a local checkout",
    )
    bench_make_minif2f.add_argument(
        "--root",
        type=Path,
        required=True,
        help="miniF2F checkout root directory",
    )
    bench_make_minif2f.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output suite JSONL path",
    )
    bench_make_minif2f.add_argument(
        "--split",
        choices=["all", "valid", "test"],
        default="all",
        help="optional split filter by path segment",
    )
    bench_make_minif2f.add_argument(
        "--glob",
        default="**/*.lean",
        help="glob pattern under --root for candidate Lean files",
    )
    bench_make_minif2f.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="exclude path pattern(s) relative to --root (repeatable)",
    )
    bench_make_minif2f.add_argument(
        "--require-sorry",
        action="store_true",
        help="only include declarations from files containing `sorry`/`admit`",
    )
    bench_make_minif2f.add_argument(
        "--shuffle",
        action="store_true",
        help="shuffle selected entries before writing",
    )
    bench_make_minif2f.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed used with --shuffle",
    )
    bench_make_minif2f.add_argument(
        "--limit",
        type=int,
        default=0,
        help="maximum number of entries to write (0 means all)",
    )
    bench_make_minif2f.add_argument(
        "--dataset",
        default="minif2f",
        help="dataset label stored in each suite row",
    )
    bench_make_minif2f.add_argument(
        "--allow-duplicate-theorems",
        action="store_true",
        help="allow same theorem name multiple times across files",
    )
    bench_make_regression100 = sub.add_parser(
        "bench-make-regression100",
        help="build a fixed-size regression suite from a source suite JSONL",
    )
    bench_make_regression100.add_argument(
        "--source",
        type=Path,
        required=True,
        help="source suite JSONL path",
    )
    bench_make_regression100.add_argument(
        "--out",
        type=Path,
        default=Path("bench/suites/regression100.jsonl"),
        help="output suite JSONL path",
    )
    bench_make_regression100.add_argument(
        "--size",
        type=int,
        default=100,
        help="number of cases to include",
    )
    bench_make_regression100.add_argument(
        "--seed",
        type=int,
        default=0,
        help="sampling seed",
    )
    bench_make_regression100.add_argument(
        "--dataset",
        default="regression100",
        help="dataset label written to each output row",
    )
    bench_make_regression100.add_argument(
        "--tag",
        action="append",
        default=["regression100"],
        help="tag to append to each row (repeatable)",
    )
    bench_make_regression100.add_argument(
        "--allow-duplicate-pairs",
        action="store_true",
        help="allow duplicate (file,theorem) rows in output",
    )

    index = sub.add_parser("index", help="build or inspect retrieval indices")
    index_sub = index.add_subparsers(dest="index_command", required=True)
    index_build = index_sub.add_parser("build", help="build premise index from Lean sources")
    index_build.add_argument("--project", type=Path, default=Path.cwd(), help="Lean project root")
    index_build.add_argument("--scope", choices=["local", "mathlib", "both"], default="both")
    index_build.add_argument(
        "--out",
        type=Path,
        default=Path(".ulam/premises_both.jsonl"),
        help="index output path",
    )
    index_stats = index_sub.add_parser("stats", help="print premise index statistics")
    index_stats.add_argument("--index", type=Path, default=Path(".ulam/premises_both.jsonl"))

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
    if args.command == "checkpoint":
        run_checkpoint(args)
        return
    if args.command == "review":
        run_review(args)
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
    if args.command == "bench-list-suites":
        run_bench_list_suites(args)
        return
    if args.command == "bench-validate":
        run_bench_validate(args)
        return
    if args.command == "bench-compare":
        run_bench_compare(args)
        return
    if args.command == "bench-make-minif2f":
        run_bench_make_minif2f(args)
        return
    if args.command == "bench-make-regression100":
        run_bench_make_regression100(args)
        return
    if args.command == "index":
        run_index(args)
        return
    if args.command == "lean-setup":
        run_lean_setup(args)
        return


def run_prove(args: argparse.Namespace) -> None:
    config_data = load_config()
    profile = _resolve_proof_profile(args, config_data)
    _apply_proof_profile_to_args(args, profile)
    if getattr(args, "verbose", False):
        print(f"[policy] profile={profile}")
    output_format = _resolve_prove_output_format(args, config_data)
    if output_format == "tex":
        run_prove_tex(args, config_data=config_data)
        return
    if getattr(args, "file", None) is None:
        print("Lean output mode requires a target file.")
        print("Usage: ulam prove path/to/File.lean --theorem MyTheorem")
        return
    inf_profile, gen_k, exec_k, verify_level = _apply_inference_runtime_to_args(args)
    if getattr(args, "verbose", False):
        exec_text = "all" if exec_k <= 0 else str(exec_k)
        print(
            f"[policy] inference_profile={inf_profile} "
            f"(gen_k={gen_k}, exec_k={exec_text}, verify={verify_level})"
        )
    if args.prove_mode != "llm" and args.lean == "cli":
        print("Lean backend `cli` is only supported with `--prove-mode llm`.")
        print("Use `--lean dojo` (default) or `--lean lsp` for tactic/lemma search modes.")
        return
    if args.prove_mode == "llm" and args.lean == "mock":
        # LLM mode always runs a Lean typecheck loop; map `mock` to CLI checks.
        args.lean = "cli"
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
        on_progress=_proof_progress_callback(args.file, args.theorem, verbose=bool(args.verbose)),
        inference_profile=inf_profile,
        generation_budget_per_state=gen_k,
        execution_budget_per_state=exec_k,
        verification_level=verify_level,
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
            exec_text = "all" if config.execution_budget_per_state <= 0 else str(config.execution_budget_per_state)
            print(
                f"[run] inference_profile={config.inference_profile} "
                f"gen_k={config.generation_budget_per_state} exec_k={exec_text} "
                f"verify={config.verification_level}"
            )

        _write_trace_metadata(
            config.trace_path,
            _trace_metadata_payload(
                args=args,
                mode="prove",
                solver=solver,
                file_path=args.file,
                theorem=args.theorem,
            ),
        )
        runner = _make_runner(args)
        llm = _make_llm(args)
        retriever = _make_retriever(args)
        trace = TraceLogger(config.trace_path)
        try:
            result = _run_with_solver(solver, runner, llm, retriever, trace, config)
            _write_trace_result_metadata(config.trace_path, result)
            return result
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


def run_prove_tex(args: argparse.Namespace, config_data: dict | None = None) -> bool:
    cfg = config_data if config_data is not None else load_config()
    profile = _resolve_proof_profile(args, cfg)
    _apply_proof_profile_to_args(args, profile)
    theorem = str(getattr(args, "theorem", "") or "").strip()
    if not theorem:
        print("`--theorem` is required for TeX output mode.")
        return False

    statement = str(getattr(args, "statement", "") or "").strip()
    file_path = getattr(args, "file", None)
    if isinstance(file_path, Path) and file_path.exists():
        extracted_stmt, original_stmt = _extract_theorem_statement(file_path, theorem)
        statement = statement or original_stmt or extracted_stmt
        if not statement:
            block = _decl_block(
                file_path.read_text(encoding="utf-8", errors="ignore"),
                theorem,
            )
            statement = " ".join(block.split())[:1200] if block else ""
    if not statement:
        print(
            "Could not infer theorem statement. Provide `--statement` "
            "or point `file` to a Lean file containing the theorem."
        )
        return False

    rounds = _resolve_tex_rounds(args, cfg)
    judge_repairs = _resolve_tex_judge_repairs(args, cfg)
    worker_drafts = _resolve_tex_worker_drafts(args, cfg)
    tex_concurrency = _resolve_tex_concurrency(args, cfg)
    replan_passes = _resolve_tex_replan_passes(args, cfg)
    action_steps = _resolve_tex_action_steps(args, cfg)
    configured_planner_model = _resolve_tex_role_model(args, "planner", cfg)
    configured_worker_model = _resolve_tex_role_model(args, "worker", cfg)
    thinker_model = _resolve_tex_primary_model(args, cfg)
    verifier_policy = str(getattr(args, "tex_verifier_policy", "promoted") or "promoted").strip().lower()
    if verifier_policy not in {"final_only", "promoted", "worker"}:
        verifier_policy = "promoted"
    compose_policy = str(getattr(args, "tex_compose_policy", "always") or "always").strip().lower()
    if compose_policy not in {"always", "on_complete"}:
        compose_policy = "always"
    instruction = str(getattr(args, "instruction", "") or "").strip()
    context_blocks = _read_context_files(list(getattr(args, "context", []) or []))
    if isinstance(file_path, Path) and file_path.exists():
        theorem_block = _decl_block(file_path.read_text(encoding="utf-8", errors="ignore"), theorem)
        if theorem_block:
            context_blocks.append(f"[theorem source: {file_path}]\n{theorem_block[:5000]}")
    base_context = "\n\n".join(context_blocks)

    out_path = _resolve_tex_output_path(args, cfg, theorem, file_path=file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts_root = _resolve_tex_artifacts_root(args, cfg, file_path)
    resume_snapshot = _resolve_tex_resume_snapshot(args)
    if getattr(args, "tex_resume", None) is not None and resume_snapshot is None:
        print("Could not find a TeX resume snapshot. Pass an artifacts directory or state.json file.")
        return False

    base_llm_config = _llm_config_from_args(args)
    thinker_llm = _make_tex_role_llm(args.llm, base_llm_config, thinker_model)
    planner_llm = _make_tex_role_llm(args.llm, base_llm_config, configured_planner_model)
    worker_llm = _make_tex_role_llm(args.llm, base_llm_config, configured_worker_model)
    if resume_snapshot is not None:
        try:
            run_state = json.loads(resume_snapshot.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Could not load TeX resume snapshot: {exc}")
            return False
        if not isinstance(run_state, dict):
            print("Could not load TeX resume snapshot: invalid state payload.")
            return False
        state_theorem = str(run_state.get("theorem", "")).strip()
        if state_theorem and state_theorem != theorem:
            print(
                f"TeX resume theorem mismatch: snapshot has `{state_theorem}`, "
                f"requested `{theorem}`."
            )
            return False
        run_dir = resume_snapshot.parent.resolve()
        statement = str(run_state.get("statement", "")).strip() or statement
        instruction = str(run_state.get("instruction", instruction))
        base_context = str(run_state.get("context", base_context))
        try:
            settings = run_state.get("settings", {}) if isinstance(run_state.get("settings", {}), dict) else {}
            rounds = max(1, int(settings.get("rounds", rounds)))
            judge_repairs = max(0, int(settings.get("judge_repairs", judge_repairs)))
            worker_drafts = max(1, int(settings.get("worker_drafts", worker_drafts)))
            tex_concurrency = bool(settings.get("concurrency", tex_concurrency))
            replan_passes = max(1, int(settings.get("replan_passes", replan_passes)))
            action_steps = max(1, int(settings.get("action_steps", action_steps)))
            verifier_policy = str(settings.get("verifier_policy", verifier_policy) or verifier_policy).strip().lower()
            compose_policy = str(settings.get("compose_policy", compose_policy) or compose_policy).strip().lower()
            thinker_model = str(
                settings.get("thinker_model", "")
                or settings.get("model", "")
                or settings.get("planner_model", "")
                or settings.get("worker_model", "")
                or thinker_model
            ).strip()
            configured_planner_model = str(settings.get("planner_model", configured_planner_model) or configured_planner_model).strip()
            configured_worker_model = str(settings.get("worker_model", configured_worker_model) or configured_worker_model).strip()
        except Exception:
            pass
        thinker_llm = _make_tex_role_llm(args.llm, base_llm_config, thinker_model)
        planner_llm = _make_tex_role_llm(args.llm, base_llm_config, configured_planner_model)
        worker_llm = _make_tex_role_llm(args.llm, base_llm_config, configured_worker_model)
        out_path = Path(str(run_state.get("out_path", str(out_path)))).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[tex] resumed snapshot: {resume_snapshot}")
    else:
        run_dir = (artifacts_root / f"tex_{_tex_timestamp()}_{_tex_slug(theorem)}").resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        run_state = {
            "version": 1,
            "status": "running",
            "theorem": theorem,
            "statement": statement,
            "file_path": str(file_path) if isinstance(file_path, Path) else "",
            "out_path": str(out_path),
            "artifacts_dir": str(run_dir),
            "instruction": instruction,
            "context": base_context,
            "settings": {
                "rounds": rounds,
                "judge_repairs": judge_repairs,
                "worker_drafts": worker_drafts,
                "concurrency": tex_concurrency,
                "replan_passes": replan_passes,
                "action_steps": action_steps,
                "verifier_policy": verifier_policy,
                "compose_policy": compose_policy,
                "thinker_model": thinker_model,
                "planner_model": configured_planner_model,
                "worker_model": configured_worker_model,
            },
            "pass_history": [],
            "action_history": [],
            "current_pass": 1,
            "current_round": 1,
            "current_claim_index": 0,
            "current_round_order": [],
            "repairs_used": 0,
            "round_progressed": False,
            "plan": None,
            "claims": [],
            "accepted_claims": {},
            "best_claim_candidates": {},
            "claim_feedback": {},
            "planner_notes": [],
            "pending_worker_guidance": "",
            "pending_repo_reads": [],
            "pending_claim_focus": [],
            "best_pass": None,
            "compose_ready": False,
            "monolithic_attempt": {},
            "best_full_candidate": {},
            "final": {},
        }

    if not isinstance(run_state.get("monolithic_attempt"), dict):
        run_state["monolithic_attempt"] = {}
    if not isinstance(run_state.get("best_full_candidate"), dict):
        run_state["best_full_candidate"] = {}

    if verifier_policy not in {"final_only", "promoted", "worker"}:
        verifier_policy = "promoted"
    if compose_policy not in {"always", "on_complete"}:
        compose_policy = "always"

    run_dir.mkdir(parents=True, exist_ok=True)
    _tex_events_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
    run_state["context"] = base_context
    if not _tex_manifest_path(run_dir).exists():
        manifest = {
            "version": 1,
            "created_at": _tex_iso_now(),
            "theorem": theorem,
            "statement": statement,
            "file_path": str(file_path) if isinstance(file_path, Path) else "",
            "out_path": str(out_path),
            "settings": {
                "rounds": rounds,
                "judge_repairs": judge_repairs,
                "worker_drafts": worker_drafts,
                "concurrency": tex_concurrency,
                "replan_passes": replan_passes,
                "action_steps": action_steps,
                "verifier_policy": verifier_policy,
                "compose_policy": compose_policy,
                "thinker_model": thinker_model,
                "planner_model": configured_planner_model,
                "worker_model": configured_worker_model,
            },
            "llm_provider": str(getattr(args, "llm", "")).strip(),
            "llm_model": str(
                getattr(args, "openai_model", "")
                or getattr(args, "anthropic_model", "")
                or getattr(args, "gemini_model", "")
                or getattr(args, "ollama_model", "")
            ).strip(),
        }
        _tex_manifest_path(run_dir).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    _sync_tex_memory_state(run_state, theorem, statement)

    def _persist_state() -> None:
        _sync_tex_memory_state(run_state, theorem, statement)
        _write_tex_state(run_dir, run_state)
        _write_tex_memory_artifacts(run_dir, run_state)

    def _finish_abstained_run(
        *,
        reason: str,
        used_pass: int,
        accepted_claims_count: int,
        total_claims_count: int,
    ) -> bool:
        run_state["status"] = "finished"
        run_state["final"] = {
            "pass": False,
            "abstained": True,
            "reason": reason,
            "used_pass": int(used_pass),
            "accepted_claims": int(accepted_claims_count),
            "total_claims": int(total_claims_count),
        }
        _persist_state()
        _tex_summary_path(run_dir).write_text(
            json.dumps(run_state["final"], indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        _append_tex_event(
            run_dir,
            {
                "kind": "final",
                "pass": False,
                "abstained": True,
                "reason": reason,
                "used_pass": int(used_pass),
                "accepted_claims": int(accepted_claims_count),
                "total_claims": int(total_claims_count),
            },
        )
        print(f"[tex] state snapshot: {_tex_state_path(run_dir)}")
        print(f"[tex] event log: {_tex_events_path(run_dir)}")
        print(f"[tex] summary: {_tex_summary_path(run_dir)}")
        return False

    _persist_state()

    print(f"[tex] theorem={theorem}")
    print(
        f"[tex] rounds={rounds} worker_drafts={worker_drafts} "
        f"judge_repairs={judge_repairs} replan_passes={replan_passes} action_steps={action_steps}"
    )
    print(
        f"[tex] profile={profile} concurrency={'on' if tex_concurrency else 'off'} verifier_policy={verifier_policy} "
        f"compose_policy={compose_policy}"
    )
    if thinker_model:
        print(f"[tex] first-round whole-proof model={thinker_model}")
    if configured_planner_model or configured_worker_model:
        print(
            f"[tex] fallback models planner={configured_planner_model or '<default>'} "
            f"worker={configured_worker_model or '<default>'}"
        )
    print(f"[tex] artifacts={run_dir}")

    if str(run_state.get("status", "")).strip().lower() == "finished":
        final = run_state.get("final", {}) if isinstance(run_state.get("final", {}), dict) else {}
        final_pass = bool(final.get("pass", False))
        print("[tex] run is already finished in snapshot.")
        return final_pass

    action_step = max(1, int(run_state.get("action_step", 1) or 1))
    run_state["action_step"] = action_step
    monolithic_attempt = run_state.get("monolithic_attempt", {})
    if not isinstance(monolithic_attempt, dict):
        monolithic_attempt = {}
    if not monolithic_attempt:
        print("[tex] whole-theorem attempt before decomposition...")
        monolithic_context = _build_tex_memory_context(base_context, run_state, max_items=8)
        monolithic = thinker_llm.tex_monolithic_attempt(
            theorem_name=theorem,
            theorem_statement=statement,
            instruction=instruction,
            context=monolithic_context,
        )
        monolithic_summary = str(monolithic.get("summary", "") or "").strip() or "whole-proof attempt"
        print(f"[tex] whole attempt: {_tex_trim(monolithic_summary, 180)}")
        _record_tex_action_history(run_state, step=0, action="monolithic", summary=monolithic_summary)
        _append_tex_event(
            run_dir,
            {
                "kind": "monolithic_attempt",
                "summary": _tex_trim(monolithic_summary, 400),
                "payload": _json_clone(monolithic),
            },
        )
        _apply_tex_planner_memory_action(run_state, monolithic)
        monolithic_eval = _evaluate_tex_final_candidate(
            thinker_llm=thinker_llm,
            theorem=theorem,
            theorem_statement=statement,
            instruction=instruction,
            plan={},
            compose_claims=[],
            draft_tex=str(monolithic.get("proof_tex", "") or ""),
            context=monolithic_context,
        )
        feedback_parts: list[str] = []
        judge_summary = str(monolithic_eval.get("judge", {}).get("summary", "") if isinstance(monolithic_eval.get("judge", {}), dict) else "").strip()
        if judge_summary:
            feedback_parts.append("Judge: " + judge_summary)
        verifier_summary = str(monolithic_eval.get("verifier", {}).get("summary", "") if isinstance(monolithic_eval.get("verifier", {}), dict) else "").strip()
        if verifier_summary:
            feedback_parts.append("Verifier: " + verifier_summary)
        for issue in list(monolithic_eval.get("static_issues", []) or [])[:6]:
            text = str(issue).strip()
            if text:
                feedback_parts.append(text)
        for question in list(monolithic.get("open_questions", []) or [])[:6]:
            text = str(question).strip()
            if text:
                feedback_parts.append("Open: " + text)
        monolithic_feedback = "\n".join(f"- {item}" for item in feedback_parts[:12])
        run_state["monolithic_attempt"] = {
            "summary": monolithic_summary,
            "pass": bool(monolithic_eval.get("pass", False)),
            "score": float(monolithic_eval.get("score", 0.0) or 0.0),
            "proof_tex": str(monolithic_eval.get("draft_tex", "") or ""),
            "feedback": monolithic_feedback,
            "open_questions": list(monolithic.get("open_questions", []) or [])[:8],
            "judge": _json_clone(monolithic_eval.get("judge", {})),
            "verifier": _json_clone(monolithic_eval.get("verifier", {})),
            "checker": _json_clone(monolithic_eval.get("checker", {})),
            "static_issues": _json_clone(monolithic_eval.get("static_issues", [])),
        }
        run_state["best_full_candidate"] = {
            "score": float(monolithic_eval.get("score", 0.0) or 0.0),
            "pass": bool(monolithic_eval.get("pass", False)),
            "pass_idx": 0,
            "draft_tex": str(monolithic_eval.get("draft_tex", "") or ""),
            "judge": _json_clone(monolithic_eval.get("judge", {})),
            "verifier": _json_clone(monolithic_eval.get("verifier", {})),
            "checker": _json_clone(monolithic_eval.get("checker", {})),
            "static_issues": _json_clone(monolithic_eval.get("static_issues", [])),
        }
        repo_items = run_state.get("repo_items", {})
        if isinstance(repo_items, dict) and str(monolithic_eval.get("draft_tex", "") or "").strip():
            attempt_body = str(monolithic_eval.get("draft_tex", "") or "").strip()
            if monolithic_feedback:
                attempt_body += "\n\nFeedback:\n" + monolithic_feedback
            _tex_repo_upsert(
                repo_items,
                "whole_problem_attempt",
                kind="attempt",
                summary="Initial whole-theorem proof attempt",
                content=attempt_body,
                status="initial",
                score=float(monolithic_eval.get("score", 0.0) or 0.0),
            )
        _persist_state()
        if bool(monolithic_eval.get("pass", False)):
            print("[tex] whole-theorem attempt passed.")
            return _finish_tex_run(
                run_state=run_state,
                run_dir=run_dir,
                out_path=out_path,
                final_eval=monolithic_eval,
                used_pass=0,
                accepted_claims=0,
                total_claims=0,
                persist_state=_persist_state,
            )
        print("[tex] whole-theorem attempt did not pass; switching to planner/worker decomposition.")

    while action_step <= action_steps and not bool(run_state.get("compose_ready", False)):
        pass_idx = max(1, int(run_state.get("current_pass", 1) or 1))
        action_state = _build_tex_action_state(
            run_state,
            rounds=rounds,
            judge_repairs=judge_repairs,
            replan_passes=replan_passes,
            action_step=action_step,
            action_limit=action_steps,
        )
        action_context = _build_tex_memory_context(
            base_context,
            run_state,
            pass_idx=pass_idx,
            extra_repo_slugs=list(run_state.get("pending_repo_reads", []) or []),
            guidance=str(run_state.get("pending_worker_guidance", "") or ""),
            max_items=6,
        )
        planner_action = planner_llm.tex_action_plan(
            theorem_name=theorem,
            theorem_statement=statement,
            instruction=instruction,
            state=action_state,
            context=action_context,
        )
        action_choice = _resolve_tex_planner_action(
            planner_action,
            run_state,
            replan_passes=replan_passes,
        )
        action_summary = str(planner_action.get("summary", "") or "").strip() or action_choice
        print(f"[tex] action {action_step}/{action_steps}: {action_choice} - {_tex_trim(action_summary, 180)}")
        _record_tex_action_history(
            run_state,
            step=action_step,
            action=action_choice,
            summary=action_summary,
        )
        _append_tex_event(
            run_dir,
            {
                "kind": "planner_action",
                "step": action_step,
                "pass": pass_idx,
                "action": action_choice,
                "summary": _tex_trim(action_summary, 400),
                "payload": _json_clone(planner_action),
            },
        )
        _apply_tex_planner_memory_action(run_state, planner_action)
        _persist_state()

        if action_choice == "write_memory":
            action_step += 1
            run_state["action_step"] = action_step
            _persist_state()
            continue

        if action_choice == "give_up":
            best_payload = run_state.get("best_pass", {}) if isinstance(run_state.get("best_pass", {}), dict) else {}
            best_claims = list(best_payload.get("claims", []) or []) if isinstance(best_payload, dict) else []
            best_accepted = dict(best_payload.get("accepted_claims", {}) or {}) if isinstance(best_payload, dict) else {}
            if not best_claims:
                best_claims = list(run_state.get("claims", []) or [])
            if not best_accepted:
                best_accepted = dict(run_state.get("accepted_claims", {}) or {})
            reason = action_summary or "planner requested give_up"
            print(f"[tex] planner gave up: {reason}")
            _clear_tex_planner_directives(run_state)
            return _finish_abstained_run(
                reason=reason,
                used_pass=int(best_payload.get("pass", pass_idx) or pass_idx) if isinstance(best_payload, dict) else pass_idx,
                accepted_claims_count=len(best_accepted),
                total_claims_count=len(best_claims),
            )

        if action_choice == "compose":
            run_state["compose_ready"] = True
            _clear_tex_planner_directives(run_state)
            _persist_state()
            break

        if action_choice == "plan":
            active_plan = isinstance(run_state.get("plan"), dict) and bool(run_state.get("plan"))
            if active_plan:
                current_plan = dict(run_state.get("plan", {}) or {})
                current_claims = list(run_state.get("claims", []) or [])
                pass_summary = _summarize_tex_pass(run_state, current_plan, current_claims, "planner_replan")
                _append_tex_event(run_dir, {"kind": "pass_summary", "summary": _json_clone(pass_summary)})
                if pass_idx >= replan_passes:
                    run_state["compose_ready"] = True
                    _clear_tex_planner_directives(run_state)
                    _persist_state()
                    break
                run_state["current_pass"] = pass_idx + 1
                _reset_tex_pass_state(run_state)
                pass_idx = int(run_state.get("current_pass", 1) or 1)
            if pass_idx > replan_passes:
                run_state["compose_ready"] = True
                _clear_tex_planner_directives(run_state)
                _persist_state()
                break
            pass_instruction = _build_tex_replan_instruction(
                instruction,
                pass_idx=pass_idx,
                pass_history=list(run_state.get("pass_history", []) or []),
            )
            print(f"[tex] generating proof plan (pass {pass_idx}/{replan_passes})...")
            plan_context = _build_tex_memory_context(
                base_context,
                run_state,
                pass_idx=pass_idx,
                extra_repo_slugs=list(run_state.get("pending_repo_reads", []) or []),
                max_items=6,
            )
            plan = planner_llm.tex_plan(
                theorem_name=theorem,
                theorem_statement=statement,
                instruction=pass_instruction,
                context=plan_context,
            )
            strategy = str(plan.get("strategy", "")).strip()
            if strategy:
                print(f"[tex] plan strategy: {strategy}")
            claims = _resolve_tex_claim_graph(plan, statement)
            print(f"[tex] planned claims={len(claims)}")
            run_state["current_instruction"] = pass_instruction
            run_state["plan"] = plan
            run_state["claims"] = claims
            run_state["accepted_claims"] = {}
            run_state["best_claim_candidates"] = {}
            run_state["claim_feedback"] = {}
            run_state["current_round"] = 1
            run_state["current_claim_index"] = 0
            run_state["current_round_order"] = []
            run_state["repairs_used"] = 0
            run_state["round_progressed"] = False
            pass_dir = run_dir / "passes" / f"pass_{pass_idx:02d}"
            pass_dir.mkdir(parents=True, exist_ok=True)
            (pass_dir / "plan.json").write_text(
                json.dumps(plan, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
            (pass_dir / "claims.json").write_text(
                json.dumps(claims, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
            _append_tex_event(
                run_dir,
                {
                    "kind": "pass_plan",
                    "pass": pass_idx,
                    "strategy": strategy,
                    "claims": len(claims),
                },
            )
            _persist_state()
            action_step += 1
            run_state["action_step"] = action_step
            _persist_state()
            continue

        solve_result = _run_tex_solve_round(
            planner_llm=planner_llm,
            worker_llm=worker_llm,
            theorem=theorem,
            theorem_statement=statement,
            instruction=instruction,
            base_context=base_context,
            run_state=run_state,
            run_dir=run_dir,
            worker_drafts=worker_drafts,
            tex_concurrency=tex_concurrency,
            verifier_policy=verifier_policy,
            judge_repairs=judge_repairs,
            rounds=rounds,
            persist_state=_persist_state,
        )
        solve_status = str(solve_result.get("status", "continue") or "continue").strip().lower()
        plan = dict(solve_result.get("plan", {}) or {})
        claims = list(solve_result.get("claims", []) or [])
        if solve_status in {"claims_complete", "stalled", "round_limit"}:
            pass_summary = _summarize_tex_pass(run_state, plan, claims, solve_status)
            _append_tex_event(run_dir, {"kind": "pass_summary", "summary": _json_clone(pass_summary)})
            if solve_status == "claims_complete" or pass_idx >= replan_passes:
                run_state["compose_ready"] = True
            else:
                run_state["current_pass"] = pass_idx + 1
            _reset_tex_pass_state(run_state)
        _clear_tex_planner_directives(run_state)
        _persist_state()
        action_step += 1
        run_state["action_step"] = action_step
        _persist_state()

    if not bool(run_state.get("compose_ready", False)) and str(run_state.get("status", "")).strip().lower() != "finished":
        print("[tex] reached action budget; composing best available draft.")
        run_state["compose_ready"] = True
        _persist_state()

    best_pass = run_state.get("best_pass", None)
    if isinstance(best_pass, dict):
        compose_plan = dict(best_pass.get("plan", {}) or {})
        compose_claim_graph = list(best_pass.get("claims", []) or [])
        compose_accepted_claims = dict(best_pass.get("accepted_claims", {}) or {})
        compose_best_candidates = dict(best_pass.get("best_claim_candidates", {}) or {})
        compose_pass_idx = int(best_pass.get("pass", int(run_state.get("current_pass", 1) or 1)))
    else:
        compose_plan = dict(run_state.get("plan", {}) or {})
        compose_claim_graph = list(run_state.get("claims", []) or [])
        compose_accepted_claims = dict(run_state.get("accepted_claims", {}) or {})
        compose_best_candidates = dict(run_state.get("best_claim_candidates", {}) or {})
        compose_pass_idx = int(run_state.get("current_pass", 1) or 1)

    compose_claims = _claims_for_composition(
        compose_claim_graph,
        compose_accepted_claims,
        compose_best_candidates,
    )
    claims_complete = len(compose_claim_graph) == 0 or (
        len(compose_accepted_claims) >= len(compose_claim_graph)
    )
    if compose_policy == "on_complete" and not claims_complete:
        reason = (
            "abstained_from_composition_due_to_unresolved_claims"
        )
        print(
            f"[tex] abstaining from final compose: accepted claims "
            f"{len(compose_accepted_claims)}/{len(compose_claim_graph)}."
        )
        return _finish_abstained_run(
            reason=reason,
            used_pass=compose_pass_idx,
            accepted_claims_count=len(compose_accepted_claims),
            total_claims_count=len(compose_claim_graph),
        )

    compose_ledger = _build_tex_claim_ledger(
        {item["id"]: item for item in compose_claims if item.get("id")}
    )
    print(
        f"[tex] composing final theorem proof (using pass {compose_pass_idx}, "
        f"accepted claims: {len(compose_accepted_claims)}/{len(compose_claim_graph)})..."
    )
    compose_context = _build_tex_memory_context(
        base_context,
        run_state,
        pass_idx=compose_pass_idx,
        max_items=8,
    )
    composed = planner_llm.tex_compose(
        theorem_name=theorem,
        theorem_statement=statement,
        instruction=instruction,
        plan=compose_plan,
        accepted_claims=compose_claims,
        ledger=compose_ledger,
        context=compose_context,
    )
    final_draft = _normalize_tex_proof(composed, theorem=theorem, theorem_statement=statement)
    if not final_draft:
        fallback = _fallback_compose_tex(compose_claim_graph, compose_claims)
        final_draft = _normalize_tex_proof(fallback, theorem=theorem, theorem_statement=statement)
    best_full_candidate = run_state.get("best_full_candidate", {})
    if not final_draft and isinstance(best_full_candidate, dict):
        final_draft = _normalize_tex_proof(
            str(best_full_candidate.get("draft_tex", "") or ""),
            theorem=theorem,
            theorem_statement=statement,
        )
    if not final_draft:
        print("Failed to produce a TeX proof draft.")
        run_state["status"] = "finished"
        run_state["final"] = {"pass": False, "error": "empty_final_draft"}
        _persist_state()
        _tex_summary_path(run_dir).write_text(
            json.dumps(run_state["final"], indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        return False

    final_eval = _evaluate_tex_final_candidate(
        thinker_llm=planner_llm,
        theorem=theorem,
        theorem_statement=statement,
        instruction=instruction,
        plan=compose_plan,
        draft_tex=final_draft,
        context=compose_context,
        compose_claims=compose_claims,
    )
    return _finish_tex_run(
        run_state=run_state,
        run_dir=run_dir,
        out_path=out_path,
        final_eval=final_eval,
        used_pass=compose_pass_idx,
        accepted_claims=len(compose_accepted_claims),
        total_claims=len(compose_claim_graph),
        persist_state=_persist_state,
    )


def run_prove_llm(args: argparse.Namespace) -> bool:
    allow_axioms = _resolve_allow_axioms(args)
    allow_helper_lemmas = _resolve_llm_allow_helper_lemmas(args)
    edit_scope = _resolve_llm_edit_scope(args)
    cycle_patience = _resolve_llm_cycle_patience(args)
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
    typecheck_timeout_s = _resolve_typecheck_timeout(args)
    llm_lean_backend = _resolve_llm_typecheck_backend(args)
    if project:
        print(f"[llm] using Lean project: {project}")
    else:
        print("[llm] no Lean project detected; attempting Lean tooling directly.")
    print(f"[llm] typecheck backend: {llm_lean_backend}")

    error: str | None = None
    error_counts: dict[str, int] = {}
    failure_cluster_counts: dict[str, int] = {}
    best_score = _llm_round_score(text, args.theorem)
    stalled_rounds = 0

    def _assign_error(reason: str) -> str:
        nonlocal error
        pivot_hint = _llm_strategy_pivot_error(
            reason=reason,
            cluster_counts=failure_cluster_counts,
        )
        error = pivot_hint if pivot_hint else reason
        return error

    def _bump_cycle(current_text: str, reason: str) -> None:
        nonlocal error, best_score, stalled_rounds
        score = _llm_round_score(current_text, args.theorem)
        if score < best_score:
            best_score = score
            stalled_rounds = 0
            return
        stalled_rounds += 1
        if stalled_rounds < cycle_patience:
            return
        stalled_rounds = 0
        replan = _llm_replan_hint(reason)
        error = replan if not error else f"{error}\n\n{replan}"
        print("[cycle] no measurable progress; forcing replan constraints.")

    for round_idx in range(1, max_rounds + 1):
        print(f"[llm] round {round_idx}/{max_rounds}")
        tex_snippet = _extract_tex_snippet(text, args.theorem)
        round_context = context
        goal_context = _maybe_query_goal_state_context(
            text=text,
            file_path=args.file,
            theorem=args.theorem,
            lean_project=project,
            lean_imports=list(getattr(args, "lean_import", []) or []),
            lean_backend=llm_lean_backend,
            timeout_s=typecheck_timeout_s,
        )
        if goal_context:
            title, snippet = goal_context
            round_context = _append_context_block(round_context, title, snippet)
            print(f"[llm] attached {title.lower()} context.")
        print("[llm] requesting proof update...")
        try:
            updated = llm.prove(
                lean_code=text,
                name=args.theorem,
                instruction=instruction,
                tex_snippet=tex_snippet,
                context=round_context,
                error=error,
                allow_helper_lemmas=allow_helper_lemmas,
                edit_scope=edit_scope,
            )
        except Exception as exc:
            print(f"[llm] error: {exc}")
            break
        if not updated.strip():
            print("LLM returned empty output.")
            break
        updated = _normalize_llm_output(updated)
        scope_error = _llm_scope_guard_error(
            before=text,
            after=updated,
            theorem=args.theorem,
            allow_helper_lemmas=allow_helper_lemmas,
            edit_scope=edit_scope,
        )
        if scope_error:
            assigned = _assign_error(scope_error)
            print(f"[scope] {scope_error}")
            if assigned != scope_error:
                print("[cycle] repeated scope-error cluster; forcing strategy pivot.")
            count = _record_error_count(error_counts, scope_error)
            if count >= 3:
                print("[stagnation] same scope error repeated multiple rounds; stopping.")
                break
            _bump_cycle(text, scope_error)
            continue
        if updated.strip() == text.strip():
            print("[stagnation] LLM returned no effective code changes.")
            break
        args.file.write_text(updated, encoding="utf-8")
        text = updated
        if _decl_has_placeholder(updated, args.theorem):
            placeholder_error = f"Declaration `{args.theorem}` still contains sorry/admit."
            assigned = _assign_error(placeholder_error)
            print(f"[typecheck] {placeholder_error}")
            if assigned != placeholder_error:
                print("[cycle] repeated placeholder-error cluster; forcing strategy pivot.")
            count = _record_error_count(error_counts, placeholder_error)
            if count >= 3:
                print("[stagnation] same error repeated multiple rounds; stopping.")
                break
            _bump_cycle(updated, placeholder_error)
            continue
        check_error = _llm_typecheck_error(
            file_path=args.file,
            project_path=project,
            timeout_s=typecheck_timeout_s,
            backend=llm_lean_backend,
        )
        if check_error:
            assigned = _assign_error(check_error)
            print(f"[typecheck] error: {check_error[:200]}")
            if assigned != check_error:
                print("[cycle] repeated typecheck-error cluster; forcing strategy pivot.")
            count = _record_error_count(error_counts, check_error)
            if count >= 3:
                print("[stagnation] same typecheck error repeated multiple rounds; stopping.")
                break
            _bump_cycle(updated, check_error)
            continue
        axiom_error = _axiom_guardrail_error(updated, allow_axioms)
        if axiom_error:
            assigned = _assign_error(axiom_error)
            print(f"[axiom] {axiom_error}")
            if assigned != axiom_error:
                print("[cycle] repeated axiom-error cluster; forcing strategy pivot.")
            count = _record_error_count(error_counts, axiom_error)
            if count >= 3:
                print("[stagnation] same axiom guardrail error repeated multiple rounds; stopping.")
                break
            _bump_cycle(updated, axiom_error)
            continue
        print("Solved.")
        print(f"Wrote proof to: {args.file}")
        return True

    print("Failed to solve with LLM-only mode.")
    return False


def _llm_round_score(text: str, theorem: str) -> tuple[int, int]:
    target = _decl_block(text, theorem)
    if not target:
        target = _any_decl_block(text, theorem)
    target_placeholders = len(re.findall(r"\b(sorry|admit)\b", _strip_comments(target)))
    total_placeholders = len(re.findall(r"\b(sorry|admit)\b", _strip_comments(text)))
    return (target_placeholders, total_placeholders)


def _llm_replan_hint(reason: str) -> str:
    short = (reason or "").strip().splitlines()[0] if reason else "unknown failure"
    return (
        "REPLAN REQUIRED:\n"
        "- Use a different proof strategy than previous attempts.\n"
        "- Edit only what is necessary for the target declaration.\n"
        "- Avoid repeating the same failing tactic/script pattern.\n"
        f"- Last blocker: {short}"
    )


def _llm_strategy_pivot_error(
    *,
    reason: str,
    cluster_counts: dict[str, int],
) -> str:
    cluster = _llm_error_cluster(reason)
    count = cluster_counts.get(cluster, 0) + 1
    cluster_counts[cluster] = count
    if count < 2:
        return reason
    return (
        reason.rstrip()
        + "\n\n"
        + "STRATEGY PIVOT REQUIRED:\n"
        + f"- Failure cluster repeated ({cluster}, count={count}).\n"
        + "- Change tactic family and intermediate lemmas.\n"
        + "- Do not repeat previous failing script shape.\n"
        + "- Prefer short compositional steps and validate each change."
    )


def _llm_error_cluster(reason: str) -> str:
    text = (reason or "").strip()
    if not text:
        return "empty"
    head = text.splitlines()[0].strip().lower()
    head = re.sub(r":[0-9]+:[0-9]+", ":#:#", head)
    head = re.sub(r"\b[0-9]{2,}\b", "#", head)
    if "still contains sorry" in head or "admit" in head:
        return "placeholder"
    if "unknown identifier" in head:
        return "unknown_identifier"
    if "type mismatch" in head:
        return "type_mismatch"
    if "unsolved goals" in head:
        return "unsolved_goals"
    if "timeout" in head or "timed out" in head:
        return "timeout"
    if "axiom" in head:
        return "axiom"
    if "scope guardrail" in head or "edit scope" in head:
        return "scope"
    return head[:120]


def _append_context_block(base: str, title: str, body: str) -> str:
    snippet = (body or "").strip()
    if not snippet:
        return base
    if len(snippet) > 3000:
        snippet = snippet[:3000].rstrip() + "\n-- (truncated)"
    block = f"{title}:\n{snippet}"
    if not base.strip():
        return block
    return base.rstrip() + "\n\n" + block


def _resolve_llm_typecheck_backend(args: argparse.Namespace) -> str:
    raw = str(getattr(args, "lean", "") or "").strip().lower()
    if raw == "lsp":
        return "lsp"
    return "cli"


def _llm_typecheck_error(
    *,
    file_path: Path,
    project_path: Path | None,
    timeout_s: float,
    backend: str,
) -> str | None:
    if backend == "lsp":
        return lean_lsp_check(file_path, project_path=project_path, timeout_s=timeout_s)
    from .lean.cli_check import lean_cli_check

    return lean_cli_check(file_path, project_path=project_path, timeout_s=timeout_s)


def _maybe_query_goal_state_context(
    text: str,
    file_path: Path,
    theorem: str,
    lean_project: Path | None,
    lean_imports: list[str],
    lean_backend: str,
    timeout_s: float,
) -> tuple[str, str] | None:
    if lean_project is None:
        return None
    if not _decl_has_placeholder(text, theorem):
        return None
    if lean_backend == "lsp":
        rows, error = lean_lsp_diagnostics(
            file_path,
            project_path=lean_project,
            timeout_s=max(5.0, timeout_s),
        )
        if error:
            return None
        errors = [row for row in rows if str(row.get("severity", "")).lower() == "error"]
        if not errors:
            return None
        lines = []
        for row in errors[:4]:
            line = int(row.get("line", 1) or 1)
            col = int(row.get("col", 1) or 1)
            msg = str(row.get("message", "")).strip()
            if msg:
                lines.append(f"{line}:{col}: {msg}")
        if not lines:
            return None
        return ("Current Lean diagnostics", "\n".join(lines))
    if importlib.util.find_spec("pantograph") is None:
        return None
    try:
        from .lean.dojo import LeanDojoRunner
    except Exception:
        return None
    runner = None
    try:
        runner = LeanDojoRunner(project_path=lean_project, imports=lean_imports or None)
        state = runner.start(file_path, theorem)
        pretty = (state.pretty or "").strip()
        if not pretty:
            return None
        return ("Current Lean goal state", pretty)
    except Exception:
        return None
    finally:
        if runner is not None:
            try:
                runner.close()
            except Exception:
                pass


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
ULAMAI_DRAFT_BEGIN = "-- ULAMAI_DRAFT_PROOF_BEGIN"
ULAMAI_DRAFT_END = "-- ULAMAI_DRAFT_PROOF_END"

_LEAN_IDENT_CHARS = "A-Za-z0-9_'"
_DECL_NAME_PATTERN = re.compile(r"^\s*(?:theorem|lemma|example)\s+([A-Za-z_][A-Za-z0-9_']*)", re.M)
_TOP_DECL_NAME_PATTERN = re.compile(
    r"^\s*(?:theorem|lemma|example|def|abbrev|structure)\s+([A-Za-z_][A-Za-z0-9_']*)",
    re.M,
)
_ANY_DECL_START_PATTERN = re.compile(r"^\s*(?:theorem|lemma|example)\s+[A-Za-z_][A-Za-z0-9_']*", re.M)
_ANY_TOP_DECL_START_PATTERN = re.compile(
    r"^\s*(?:theorem|lemma|example|def|abbrev|structure)\s+[A-Za-z_][A-Za-z0-9_']*",
    re.M,
)


def _name_token_regex(name: str) -> str:
    return rf"(?<![{_LEAN_IDENT_CHARS}]){re.escape(name)}(?![{_LEAN_IDENT_CHARS}])"


def _decl_head_regex(name: str, kinds: str = "theorem|lemma|example") -> re.Pattern[str]:
    return re.compile(rf"^\s*(?:{kinds})\s+{_name_token_regex(name)}", re.M)


def _decl_span(text: str, name: str) -> tuple[int, int] | None:
    pattern = _decl_head_regex(name)
    match = pattern.search(text)
    if not match:
        return None
    next_match = _ANY_DECL_START_PATTERN.search(text, match.end())
    end = next_match.start() if next_match else len(text)
    return match.start(), end


def _top_decl_span(text: str, name: str) -> tuple[int, int] | None:
    pattern = _decl_head_regex(name, kinds="theorem|lemma|example|def|abbrev|structure")
    match = pattern.search(text)
    if not match:
        return None
    next_match = _ANY_TOP_DECL_START_PATTERN.search(text, match.end())
    end = next_match.start() if next_match else len(text)
    return match.start(), end


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
    if not _file_has_decl(text, theorem):
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
    for match in _DECL_NAME_PATTERN.finditer(text):
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
    span = _decl_span(text, name)
    if span is None:
        return ""
    start, end = span
    return text[start:end]


def _top_decl_blocks(text: str) -> dict[str, str]:
    blocks: dict[str, str] = {}
    for match in _TOP_DECL_NAME_PATTERN.finditer(text):
        name = match.group(1)
        span = _top_decl_span(text, name)
        if span is None:
            continue
        start, end = span
        blocks[name] = text[start:end]
    return blocks


def _llm_scope_guard_error(
    before: str,
    after: str,
    theorem: str,
    allow_helper_lemmas: bool,
    edit_scope: str,
) -> str | None:
    before_blocks = _top_decl_blocks(before)
    after_blocks = _top_decl_blocks(after)
    theorem = theorem.strip()
    if theorem and theorem not in after_blocks:
        return f"target declaration `{theorem}` disappeared from file."

    before_names = set(before_blocks)
    after_names = set(after_blocks)
    if not allow_helper_lemmas:
        if before_names != after_names:
            return "helper declarations are disabled in LLM mode; keep declaration set unchanged."
        for name in before_names:
            if name == theorem:
                continue
            if before_blocks.get(name) != after_blocks.get(name):
                return f"helper declarations are disabled; declaration `{name}` was modified."

    if edit_scope == "errors_only":
        locked = {
            name
            for name, block in before_blocks.items()
            if name != theorem and re.search(r"\b(sorry|admit)\b", block) is None
        }
        for name in sorted(locked):
            if before_blocks.get(name) != after_blocks.get(name):
                return f"errors-only scope: declaration `{name}` has no placeholders and must not change."

    return None


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

    span = _decl_span(text, target_name)
    if span is None:
        return []
    insert_at = span[0]
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
    pattern = re.compile(
        r"^\s*(theorem|lemma|example|def|abbrev|structure)\s+([A-Za-z_][A-Za-z0-9_']*)",
        re.M,
    )
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
    span = _top_decl_span(text, name)
    if span is None:
        return text
    start, end = span
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
    return _decl_span(text, name) is not None


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
    inf_profile, gen_k, exec_k, verify_level = _apply_inference_runtime_to_args(args)
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
        on_progress=_proof_progress_callback(args.file, theorem, verbose=bool(args.verbose)),
        inference_profile=inf_profile,
        generation_budget_per_state=gen_k,
        execution_budget_per_state=exec_k,
        verification_level=verify_level,
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
            exec_text = "all" if config.execution_budget_per_state <= 0 else str(config.execution_budget_per_state)
            print(
                f"[run] inference_profile={config.inference_profile} "
                f"gen_k={config.generation_budget_per_state} exec_k={exec_text} "
                f"verify={config.verification_level}"
            )

        _write_trace_metadata(
            config.trace_path,
            _trace_metadata_payload(
                args=args,
                mode="prove",
                solver=solver,
                file_path=args.file,
                theorem=theorem,
            ),
        )
        runner = _make_runner(args)
        llm = _make_llm(args)
        retriever = _make_retriever(args)
        trace = TraceLogger(config.trace_path)
        try:
            result = _run_with_solver(solver, runner, llm, retriever, trace, config)
            _write_trace_result_metadata(config.trace_path, result)
            return result
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


def _resolve_inference_runtime(args: argparse.Namespace) -> tuple[str, int, int, str]:
    profile = str(getattr(args, "inference_profile", "default") or "default").strip().lower()
    if profile not in {"default", "balanced", "explore", "verify"}:
        profile = "default"
    base_k = max(1, int(getattr(args, "k", 1) or 1))
    raw_gen_k = max(0, int(getattr(args, "gen_k", 0) or 0))
    raw_exec_k = max(0, int(getattr(args, "exec_k", 0) or 0))
    verify_raw = str(getattr(args, "verify_level", "auto") or "auto").strip().lower()
    if verify_raw not in {"auto", "none", "light", "strict"}:
        verify_raw = "auto"

    defaults = {
        "default": (base_k, 0, "light"),
        "balanced": (max(base_k, 6), 3, "strict"),
        "explore": (max(base_k, 10), 5, "light"),
        "verify": (max(base_k, 5), 2, "strict"),
    }
    gen_k, exec_k, verify_level = defaults[profile]
    if raw_gen_k > 0:
        gen_k = raw_gen_k
    if raw_exec_k > 0:
        exec_k = raw_exec_k
    if verify_raw != "auto":
        verify_level = verify_raw
    return profile, int(gen_k), int(exec_k), str(verify_level)


def _apply_inference_runtime_to_args(args: argparse.Namespace) -> tuple[str, int, int, str]:
    profile, gen_k, exec_k, verify_level = _resolve_inference_runtime(args)
    args.inference_profile = profile
    args.effective_gen_k = gen_k
    args.effective_exec_k = exec_k
    args.effective_verify_level = verify_level
    return profile, gen_k, exec_k, verify_level


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
    merged_stats = _merge_search_stats(first.stats, second.stats)
    if second.solved:
        return SearchResult(True, second.proof, first.steps + second.steps, None, stats=merged_stats)
    error = second.error or first.error or "portfolio exhausted"
    return SearchResult(False, [], first.steps + second.steps, error, stats=merged_stats)


def _merge_search_stats(*stats_rows: dict[str, int] | None) -> dict[str, int]:
    merged: dict[str, int] = {}
    for row in stats_rows:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            try:
                amount = int(value)
            except Exception:
                continue
            merged[key] = merged.get(key, 0) + amount
    return merged


def _planner_case_metrics(result: SearchResult) -> dict[str, int]:
    allowed = {
        "planner_cache_hit_states",
        "planner_cached_tactic_candidates",
        "planner_cached_tactic_tries",
        "planner_replan_triggers",
        "planner_remembered_tactics",
    }
    out: dict[str, int] = {}
    for key, value in _merge_search_stats(result.stats).items():
        if key not in allowed:
            continue
        out[key] = int(value)
    return out


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


def _normalize_proof_profile(value: object) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"", "normal", "balanced", "default"}:
        return "balanced"
    if raw in {"fast", "strict"}:
        return raw
    return "balanced"


def _resolve_proof_profile(args: argparse.Namespace, config: dict | None = None) -> str:
    explicit_raw = str(getattr(args, "proof_profile", "") or "").strip()
    if explicit_raw:
        return _normalize_proof_profile(explicit_raw)
    cfg = config if config is not None else load_config()
    policy = cfg.get("policy", {})
    if isinstance(policy, dict):
        return _normalize_proof_profile(policy.get("proof_profile", "balanced"))
    return "balanced"


def _set_arg_if_unset(args: argparse.Namespace, name: str, value: object) -> None:
    if not hasattr(args, name):
        return
    if getattr(args, name) is None:
        setattr(args, name, value)


def _apply_proof_profile_to_args(args: argparse.Namespace, profile: str) -> None:
    mode = _normalize_proof_profile(profile)
    if hasattr(args, "proof_profile"):
        setattr(args, "proof_profile", mode)

    tex_profile_defaults = {
        "fast": {
            "tex_rounds": 2,
            "tex_judge_repairs": 1,
            "tex_worker_drafts": 1,
            "tex_replan_passes": 1,
            "tex_action_steps": 4,
            "tex_verifier_policy": "final_only",
            "tex_compose_policy": "on_complete",
        },
        "balanced": {
            "tex_rounds": 3,
            "tex_judge_repairs": 2,
            "tex_worker_drafts": 2,
            "tex_replan_passes": 2,
            "tex_action_steps": 10,
            "tex_verifier_policy": "promoted",
            "tex_compose_policy": "always",
        },
        "strict": {
            "tex_rounds": 4,
            "tex_judge_repairs": 3,
            "tex_worker_drafts": 3,
            "tex_replan_passes": 3,
            "tex_action_steps": 16,
            "tex_verifier_policy": "worker",
            "tex_compose_policy": "on_complete",
        },
    }
    formalize_profile_defaults = {
        "fast": {
            "max_rounds": 3,
            "max_repairs": 3,
            "max_proof_rounds": 1,
            "proof_repair": 1,
            "llm_check": True,
            "llm_check_timing": "end",
            "llm_check_repairs": 1,
        },
        "balanced": {
            "max_rounds": 5,
            "max_repairs": 5,
            "max_proof_rounds": 1,
            "proof_repair": 2,
            "llm_check": True,
            "llm_check_timing": "end",
            "llm_check_repairs": 2,
        },
        "strict": {
            "max_rounds": 6,
            "max_repairs": 6,
            "max_proof_rounds": 2,
            "proof_repair": 3,
            "llm_check": True,
            "llm_check_timing": "mid+end",
            "llm_check_repairs": 3,
        },
    }

    for key, value in tex_profile_defaults[mode].items():
        _set_arg_if_unset(args, key, value)
    for key, value in formalize_profile_defaults[mode].items():
        _set_arg_if_unset(args, key, value)

    if mode == "strict":
        if hasattr(args, "allow_axioms"):
            setattr(args, "allow_axioms", False)
        if hasattr(args, "llm_allow_helper_lemmas"):
            setattr(args, "llm_allow_helper_lemmas", False)
        if hasattr(args, "llm_edit_scope"):
            setattr(args, "llm_edit_scope", "errors_only")


def _resolve_allow_axioms(args: argparse.Namespace, config: dict | None = None) -> bool:
    explicit = getattr(args, "allow_axioms", None)
    if explicit is not None:
        return bool(explicit)
    cfg = config if config is not None else load_config()
    return bool(cfg.get("prove", {}).get("allow_axioms", True))


def _resolve_typecheck_timeout(args: argparse.Namespace, config: dict | None = None) -> float:
    explicit = getattr(args, "typecheck_timeout", None)
    if explicit is not None:
        try:
            return max(5.0, float(explicit))
        except Exception:
            return 60.0
    cfg = config if config is not None else load_config()
    raw = cfg.get("prove", {}).get("typecheck_timeout_s", 60)
    try:
        return max(5.0, float(raw))
    except Exception:
        return 60.0


def _resolve_llm_allow_helper_lemmas(
    args: argparse.Namespace,
    config: dict | None = None,
) -> bool:
    explicit = getattr(args, "llm_allow_helper_lemmas", None)
    if explicit is not None:
        return bool(explicit)
    cfg = config if config is not None else load_config()
    return bool(cfg.get("prove", {}).get("llm_allow_helper_lemmas", True))


def _resolve_llm_edit_scope(
    args: argparse.Namespace,
    config: dict | None = None,
) -> str:
    explicit = str(getattr(args, "llm_edit_scope", "") or "").strip().lower()
    if explicit in {"full", "errors_only"}:
        return explicit
    cfg = config if config is not None else load_config()
    raw = str(cfg.get("prove", {}).get("llm_edit_scope", "full")).strip().lower()
    if raw in {"full", "errors_only"}:
        return raw
    return "full"


def _resolve_llm_cycle_patience(
    args: argparse.Namespace,
    config: dict | None = None,
) -> int:
    explicit = getattr(args, "llm_cycle_patience", None)
    if explicit is not None:
        try:
            return max(1, int(explicit))
        except Exception:
            return 2
    cfg = config if config is not None else load_config()
    raw = cfg.get("prove", {}).get("llm_cycle_patience", 2)
    try:
        return max(1, int(raw))
    except Exception:
        return 2


def _resolve_prove_output_format(args: argparse.Namespace, config: dict | None = None) -> str:
    explicit = str(getattr(args, "output_format", "") or "").strip().lower()
    if explicit in {"lean", "tex"}:
        return explicit
    cfg = config if config is not None else load_config()
    raw = str(cfg.get("prove", {}).get("output_format", "lean")).strip().lower()
    if raw in {"lean", "tex"}:
        return raw
    return "lean"


def _resolve_tex_rounds(args: argparse.Namespace, config: dict | None = None) -> int:
    explicit = getattr(args, "tex_rounds", None)
    if explicit is not None:
        try:
            return max(1, int(explicit))
        except Exception:
            return 3
    cfg = config if config is not None else load_config()
    raw = cfg.get("prove", {}).get("tex_rounds", 3)
    try:
        return max(1, int(raw))
    except Exception:
        return 3


def _resolve_tex_judge_repairs(args: argparse.Namespace, config: dict | None = None) -> int:
    explicit = getattr(args, "tex_judge_repairs", None)
    if explicit is not None:
        try:
            return max(0, int(explicit))
        except Exception:
            return 2
    cfg = config if config is not None else load_config()
    raw = cfg.get("prove", {}).get("tex_judge_repairs", 2)
    try:
        return max(0, int(raw))
    except Exception:
        return 2


def _resolve_tex_worker_drafts(args: argparse.Namespace, config: dict | None = None) -> int:
    explicit = getattr(args, "tex_worker_drafts", None)
    if explicit is not None:
        try:
            return max(1, int(explicit))
        except Exception:
            return 2
    cfg = config if config is not None else load_config()
    raw = cfg.get("prove", {}).get("tex_worker_drafts", 2)
    try:
        return max(1, int(raw))
    except Exception:
        return 2


def _resolve_tex_concurrency(args: argparse.Namespace, config: dict | None = None) -> bool:
    explicit = getattr(args, "tex_concurrency", None)
    if explicit is not None:
        return bool(explicit)
    cfg = config if config is not None else load_config()
    return bool(cfg.get("prove", {}).get("tex_concurrency", False))


def _resolve_tex_replan_passes(args: argparse.Namespace, config: dict | None = None) -> int:
    explicit = getattr(args, "tex_replan_passes", None)
    if explicit is not None:
        try:
            return max(1, int(explicit))
        except Exception:
            return 2
    cfg = config if config is not None else load_config()
    raw = cfg.get("prove", {}).get("tex_replan_passes", 2)
    try:
        return max(1, int(raw))
    except Exception:
        return 2


def _resolve_tex_action_steps(args: argparse.Namespace, config: dict | None = None) -> int:
    explicit = getattr(args, "tex_action_steps", None)
    if explicit is not None:
        try:
            return max(1, int(explicit))
        except Exception:
            return 10
    cfg = config if config is not None else load_config()
    raw = cfg.get("prove", {}).get("tex_action_steps", 10)
    try:
        return max(1, int(raw))
    except Exception:
        return 10


def _resolve_tex_role_model(args: argparse.Namespace, role: str, config: dict | None = None) -> str:
    safe_role = "planner" if str(role).strip().lower() == "planner" else "worker"
    attr = f"tex_{safe_role}_model"
    explicit = str(getattr(args, attr, "") or "").strip()
    if explicit:
        return explicit
    cfg = config if config is not None else load_config()
    return str(cfg.get("prove", {}).get(attr, "") or "").strip()


def _resolve_tex_primary_model(args: argparse.Namespace, config: dict | None = None) -> str:
    planner_model = _resolve_tex_role_model(args, "planner", config)
    worker_model = _resolve_tex_role_model(args, "worker", config)
    if planner_model:
        return planner_model
    if worker_model:
        return worker_model
    return ""


def _resolve_tex_artifacts_root(
    args: argparse.Namespace,
    config: dict,
    file_path: Path | None,
) -> Path:
    explicit = getattr(args, "tex_artifacts_dir", None)
    if isinstance(explicit, Path):
        root = explicit.expanduser()
    else:
        prove_cfg = config.get("prove", {}) if isinstance(config, dict) else {}
        raw = str(prove_cfg.get("tex_artifacts_dir", "runs/prove_tex") or "runs/prove_tex").strip()
        root = Path(raw).expanduser()
    if not root.is_absolute():
        base = Path.cwd()
        if isinstance(file_path, Path):
            try:
                base = file_path.resolve().parent
            except Exception:
                base = Path.cwd()
        root = (base / root).resolve()
    return root.resolve()


def _resolve_tex_resume_snapshot(args: argparse.Namespace) -> Path | None:
    resume = getattr(args, "tex_resume", None)
    if not isinstance(resume, Path):
        return None
    path = resume.expanduser()
    if path.is_dir():
        snapshot = path / "state.json"
    else:
        snapshot = path
    if snapshot.exists():
        return snapshot.resolve()
    return None


def _resolve_tex_output_path(
    args: argparse.Namespace,
    config: dict,
    theorem: str,
    file_path: Path | None,
) -> Path:
    explicit = getattr(args, "tex_out", None)
    if isinstance(explicit, Path):
        return explicit.expanduser().resolve()
    prove_cfg = config.get("prove", {}) if isinstance(config, dict) else {}
    out_dir = str(prove_cfg.get("tex_out_dir", "proofs") or "proofs").strip()
    base = Path.cwd()
    if isinstance(file_path, Path):
        try:
            base = file_path.resolve().parent
        except Exception:
            base = Path.cwd()
    out_root = Path(out_dir).expanduser()
    if not out_root.is_absolute():
        out_root = base / out_root
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", theorem).strip("._")
    if not safe:
        safe = "proof"
    return (out_root / f"{safe}.tex").resolve()


def _tex_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._")
    return slug or "theorem"


def _tex_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _tex_iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_clone(payload):
    try:
        return json.loads(json.dumps(payload, ensure_ascii=True))
    except Exception:
        return payload


def _tex_state_path(run_dir: Path) -> Path:
    return run_dir / "state.json"


def _tex_events_path(run_dir: Path) -> Path:
    return run_dir / "events.jsonl"


def _tex_summary_path(run_dir: Path) -> Path:
    return run_dir / "summary.json"


def _tex_manifest_path(run_dir: Path) -> Path:
    return run_dir / "manifest.json"


def _write_tex_state(run_dir: Path, state: dict) -> None:
    _tex_state_path(run_dir).write_text(
        json.dumps(state, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _append_tex_event(run_dir: Path, event: dict) -> None:
    payload = dict(event)
    payload["ts"] = payload.get("ts") or _tex_iso_now()
    with _tex_events_path(run_dir).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _tex_whiteboard_path(run_dir: Path) -> Path:
    return run_dir / "WHITEBOARD.md"


def _tex_repo_dir(run_dir: Path) -> Path:
    return run_dir / "repo"


def _tex_repo_index_path(run_dir: Path) -> Path:
    return _tex_repo_dir(run_dir) / "index.json"


def _tex_trim(text: str, limit: int) -> str:
    raw = str(text or "").strip()
    if limit <= 0 or len(raw) <= limit:
        return raw
    if limit <= 3:
        return raw[:limit]
    return raw[: limit - 3].rstrip() + "..."


def _tex_repo_slug(name: str) -> str:
    return _tex_slug(str(name or "").strip().lower())


def _tex_repo_upsert(
    repo_items: dict[str, dict],
    slug: str,
    *,
    kind: str,
    summary: str,
    content: str,
    **extra,
) -> str:
    safe_slug = _tex_repo_slug(slug)
    item = {
        "slug": safe_slug,
        "kind": _tex_trim(kind, 40) or "note",
        "summary": _tex_trim(summary, 220) or "repo item",
        "content": _tex_trim(content, 20000),
        "updated_at": _tex_iso_now(),
    }
    for key, value in extra.items():
        if value is None:
            continue
        item[str(key)] = _json_clone(value)
    repo_items[safe_slug] = item
    return safe_slug


def _tex_repo_index_rows(repo_items: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for slug, item in sorted(repo_items.items()):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "slug": slug,
                "kind": str(item.get("kind", "note")).strip() or "note",
                "summary": _tex_trim(str(item.get("summary", "") or ""), 220),
                "updated_at": str(item.get("updated_at", "") or "").strip(),
            }
        )
    return rows


def _tex_system_repo_slug(slug: str) -> bool:
    safe = _tex_repo_slug(slug)
    if safe in {"theorem", "open_claims", "final_outcome", "planner_notes", "planner_guidance"}:
        return True
    if safe.startswith("claim_"):
        return True
    return bool(re.fullmatch(r"pass_\d+_(plan|summary)", safe))


def _tex_repo_delete(repo_items: dict[str, dict], slug: str) -> bool:
    safe = _tex_repo_slug(slug)
    if not safe or _tex_system_repo_slug(safe):
        return False
    return repo_items.pop(safe, None) is not None


def _tex_resolve_repo_wikilinks(
    text: str,
    repo_items: dict[str, dict],
    *,
    max_refs: int = 4,
    content_limit: int = 2400,
) -> str:
    raw = str(text or "").strip()
    if not raw or not isinstance(repo_items, dict) or not repo_items:
        return raw
    refs: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"\[\[([^\]]+)\]\]", raw):
        slug = _tex_repo_slug(match.group(1))
        if not slug or slug in seen or slug not in repo_items:
            continue
        refs.append(slug)
        seen.add(slug)
        if len(refs) >= max(1, max_refs):
            break
    if not refs:
        return raw
    sections = [raw, "", "Referenced repo materials:"]
    for slug in refs:
        item = repo_items.get(slug)
        if not isinstance(item, dict):
            continue
        summary = _tex_trim(str(item.get("summary", "") or ""), 180)
        content = _tex_trim(str(item.get("content", "") or ""), max(400, content_limit))
        header = f"- [[{slug}]] [{str(item.get('kind', 'note') or 'note').strip() or 'note'}]"
        if summary:
            header += f": {summary}"
        sections.append(header)
        if content:
            sections.append(content)
    return "\n".join(part for part in sections if part is not None).strip()


def _render_tex_repo_item_markdown(item: dict) -> str:
    slug = str(item.get("slug", "")).strip()
    kind = str(item.get("kind", "note")).strip() or "note"
    summary = str(item.get("summary", "")).strip()
    updated_at = str(item.get("updated_at", "")).strip()
    lines = [f"# {slug or 'repo_item'}", ""]
    lines.append(f"- Kind: {kind}")
    if summary:
        lines.append(f"- Summary: {summary}")
    if updated_at:
        lines.append(f"- Updated: {updated_at}")
    for key in ("status", "claim_id", "pass_idx", "round_idx", "worker", "score"):
        if key not in item:
            continue
        value = item.get(key)
        if value in (None, "", []):
            continue
        label = key.replace("_", " ").title()
        lines.append(f"- {label}: {value}")
    content = str(item.get("content", "") or "").strip()
    if content:
        lines.extend(["", content])
    return "\n".join(lines).rstrip() + "\n"


def _select_tex_repo_materials(
    run_state: dict,
    *,
    claim: dict | None = None,
    pass_idx: int | None = None,
    extra_slugs: list[str] | None = None,
    max_items: int = 6,
) -> list[dict]:
    repo_items = run_state.get("repo_items", {})
    if not isinstance(repo_items, dict):
        return []
    chosen: list[str] = []
    seen: set[str] = set()

    def _add(slug: str) -> None:
        safe = _tex_repo_slug(slug)
        if safe and safe in repo_items and safe not in seen:
            seen.add(safe)
            chosen.append(safe)

    _add("theorem")
    _add("open_claims")
    if pass_idx is not None:
        _add(f"pass_{pass_idx}_plan")
        _add(f"pass_{pass_idx}_summary")
    best_pass = run_state.get("best_pass", {})
    if isinstance(best_pass, dict):
        try:
            best_pass_idx = int(best_pass.get("pass", 0) or 0)
        except Exception:
            best_pass_idx = 0
        if best_pass_idx > 0:
            _add(f"pass_{best_pass_idx}_plan")
            _add(f"pass_{best_pass_idx}_summary")
    pass_history = run_state.get("pass_history", [])
    if isinstance(pass_history, list):
        for entry in pass_history[-2:]:
            if not isinstance(entry, dict):
                continue
            try:
                hist_pass = int(entry.get("pass", 0) or 0)
            except Exception:
                hist_pass = 0
            if hist_pass > 0:
                _add(f"pass_{hist_pass}_summary")
    if isinstance(claim, dict):
        claim_id = str(claim.get("id", "")).strip()
        if claim_id:
            _add(f"claim_{claim_id}")
        for dep in list(claim.get("depends_on", []) or []):
            dep_id = str(dep).strip()
            if dep_id:
                _add(f"claim_{dep_id}")
    for slug in list(extra_slugs or []):
        _add(str(slug))
    materials: list[dict] = []
    for slug in chosen[:max(1, max_items)]:
        item = repo_items.get(slug)
        if isinstance(item, dict):
            materials.append(item)
    return materials


def _build_tex_memory_context(
    base_context: str,
    run_state: dict,
    *,
    claim: dict | None = None,
    pass_idx: int | None = None,
    extra_repo_slugs: list[str] | None = None,
    guidance: str = "",
    max_items: int = 6,
) -> str:
    parts: list[str] = []
    raw_base = str(base_context or "").strip()
    if raw_base:
        parts.append(raw_base)
    repo_items = run_state.get("repo_items", {})
    if not isinstance(repo_items, dict):
        repo_items = {}
    guide = _tex_resolve_repo_wikilinks(guidance, repo_items, max_refs=4, content_limit=1800)
    if guide:
        parts.append("[planner guidance]\n" + _tex_trim(guide, 5000))
    whiteboard = str(run_state.get("whiteboard", "") or "").strip()
    if whiteboard:
        parts.append("[persistent whiteboard]\n" + _tex_trim(whiteboard, 6000))
    if isinstance(repo_items, dict) and repo_items:
        index_lines = []
        for row in _tex_repo_index_rows(repo_items)[:12]:
            summary = str(row.get("summary", "")).strip()
            line = f"- {row.get('slug', '')} [{row.get('kind', 'note')}]"
            if summary:
                line += f": {summary}"
            index_lines.append(line)
        if index_lines:
            parts.append("[repo index]\n" + "\n".join(index_lines))
        materials = _select_tex_repo_materials(
            run_state,
            claim=claim,
            pass_idx=pass_idx,
            extra_slugs=list(extra_repo_slugs or []),
            max_items=max_items,
        )
        if materials:
            material_blocks: list[str] = []
            for item in materials:
                slug = str(item.get("slug", "")).strip() or "repo_item"
                summary = str(item.get("summary", "")).strip()
                content = _tex_resolve_repo_wikilinks(
                    str(item.get("content", "") or ""),
                    repo_items,
                    max_refs=3,
                    content_limit=1600,
                )
                content = _tex_trim(content, 5000)
                block = f"[repo item: {slug}]"
                if summary:
                    block += f"\nSummary: {summary}"
                if content:
                    block += "\n\n" + content
                material_blocks.append(block)
            if material_blocks:
                parts.append("\n\n".join(material_blocks))
    return "\n\n".join(part for part in parts if str(part).strip())


def _render_tex_whiteboard(run_state: dict, theorem: str, theorem_statement: str) -> str:
    try:
        current_pass = int(run_state.get("current_pass", 1) or 1)
    except Exception:
        current_pass = 1
    try:
        current_round = int(run_state.get("current_round", 1) or 1)
    except Exception:
        current_round = 1
    try:
        claim_index = int(run_state.get("current_claim_index", 0) or 0)
    except Exception:
        claim_index = 0
    plan = run_state.get("plan", {})
    pass_history = run_state.get("pass_history", [])
    claims = run_state.get("claims", [])
    accepted_claims = run_state.get("accepted_claims", {})
    claim_feedback = run_state.get("claim_feedback", {})
    repo_items = run_state.get("repo_items", {})
    best_pass = run_state.get("best_pass", {})
    final = run_state.get("final", {})
    planner_notes = run_state.get("planner_notes", [])
    worker_guidance = str(run_state.get("pending_worker_guidance", "") or "").strip()
    strategy = ""
    if isinstance(plan, dict):
        strategy = str(plan.get("strategy", "") or "").strip()
    if not strategy and isinstance(pass_history, list) and pass_history:
        last = pass_history[-1]
        if isinstance(last, dict):
            strategy = str(last.get("strategy", "") or "").strip()
    unresolved_rows: list[str] = []
    if isinstance(claims, list):
        accepted_ids = set()
        if isinstance(accepted_claims, dict):
            accepted_ids = {str(key).strip() for key in accepted_claims.keys() if str(key).strip()}
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            claim_id = str(claim.get("id", "")).strip()
            if not claim_id or claim_id in accepted_ids:
                continue
            goal = _tex_trim(str(claim.get("goal", "") or ""), 180)
            unresolved_rows.append(f"- {claim_id}: {goal}")
    feedback_rows: list[str] = []
    if isinstance(claim_feedback, dict):
        for claim_id in sorted(claim_feedback.keys())[:6]:
            feedback = str(claim_feedback.get(claim_id, "") or "").strip()
            if not feedback:
                continue
            first_line = feedback.splitlines()[0].lstrip("- ").strip()
            if first_line:
                feedback_rows.append(f"- {claim_id}: {_tex_trim(first_line, 220)}")
    lines = ["# TeX Whiteboard", ""]
    lines.append(f"Theorem: {theorem}")
    if theorem_statement.strip():
        lines.append(f"Statement: {_tex_trim(theorem_statement, 260)}")
    lines.extend(["", "## Status"])
    lines.append(f"- Run status: {str(run_state.get('status', 'running') or 'running').strip()}")
    lines.append(f"- Current pass/round: {current_pass}/{current_round}")
    lines.append(f"- Current claim index: {claim_index}")
    if strategy:
        lines.append(f"- Current strategy: {_tex_trim(strategy, 220)}")
    if isinstance(best_pass, dict) and best_pass:
        quality = best_pass.get("quality", {}) if isinstance(best_pass.get("quality", {}), dict) else {}
        best_count = int(quality.get("accepted_count", 0) or 0)
        best_score = float(quality.get("score", 0.0) or 0.0)
        lines.append(
            f"- Best pass so far: {int(best_pass.get('pass', 0) or 0)} "
            f"({best_count} accepted, score={best_score:.1f})"
        )
    if isinstance(final, dict) and final:
        lines.append(f"- Final pass verdict: {bool(final.get('pass', False))}")
    lines.extend(["", "## Open Claims"])
    if unresolved_rows:
        lines.extend(unresolved_rows[:8])
    else:
        lines.append("- none")
    lines.extend(["", "## Recent Feedback"])
    if feedback_rows:
        lines.extend(feedback_rows[:8])
    else:
        lines.append("- none")
    lines.extend(["", "## Planner Notes"])
    if isinstance(planner_notes, list) and planner_notes:
        for note in planner_notes[-6:]:
            text = _tex_trim(str(note).strip(), 240)
            if text:
                lines.append(f"- {text}")
    else:
        lines.append("- none")
    lines.extend(["", "## Pending Worker Guidance"])
    if worker_guidance:
        lines.append(_tex_trim(worker_guidance, 600))
    else:
        lines.append("- none")
    lines.extend(["", "## Repo Highlights"])
    if isinstance(repo_items, dict) and repo_items:
        for row in _tex_repo_index_rows(repo_items)[:8]:
            summary = str(row.get("summary", "")).strip()
            text = f"- {row.get('slug', '')} [{row.get('kind', 'note')}]"
            if summary:
                text += f": {summary}"
            lines.append(_tex_trim(text, 240))
    else:
        lines.append("- theorem [theorem]: theorem statement and guidance")
    return "\n".join(lines).rstrip() + "\n"


def _sync_tex_memory_state(run_state: dict, theorem: str, theorem_statement: str) -> None:
    repo_items = run_state.get("repo_items")
    if not isinstance(repo_items, dict):
        repo_items = {}
        run_state["repo_items"] = repo_items
    _tex_repo_upsert(
        repo_items,
        "theorem",
        kind="theorem",
        summary=f"Informal theorem statement for {theorem}",
        content=(
            f"Theorem: {theorem}\n\n"
            "Statement:\n"
            f"{_tex_trim(theorem_statement, 4000)}\n"
        ),
        theorem=theorem,
    )
    planner_notes = run_state.get("planner_notes", [])
    if isinstance(planner_notes, list) and planner_notes:
        note_lines = [f"- {_tex_trim(str(item).strip(), 500)}" for item in planner_notes[-8:] if str(item).strip()]
        if note_lines:
            _tex_repo_upsert(
                repo_items,
                "planner_notes",
                kind="memory",
                summary="Planner-authored notes for future TeX actions",
                content="\n".join(note_lines).strip(),
                status="active",
            )
    else:
        repo_items.pop("planner_notes", None)
    worker_guidance = str(run_state.get("pending_worker_guidance", "") or "").strip()
    if worker_guidance:
        _tex_repo_upsert(
            repo_items,
            "planner_guidance",
            kind="memory",
            summary="Planner guidance queued for the next solve step",
            content=_tex_trim(worker_guidance, 10000),
            status="queued",
        )
    else:
        repo_items.pop("planner_guidance", None)
    plan = run_state.get("plan", {})
    claims = run_state.get("claims", [])
    if isinstance(plan, dict) and plan:
        try:
            pass_idx = int(run_state.get("current_pass", 1) or 1)
        except Exception:
            pass_idx = 1
        outline = [str(item).strip() for item in list(plan.get("outline", []) or []) if str(item).strip()]
        key_lemmas = [
            str(item).strip() for item in list(plan.get("key_lemmas", []) or []) if str(item).strip()
        ]
        checks = [str(item).strip() for item in list(plan.get("checks", []) or []) if str(item).strip()]
        claim_lines: list[str] = []
        if isinstance(claims, list):
            for claim in claims[:10]:
                if not isinstance(claim, dict):
                    continue
                claim_id = str(claim.get("id", "")).strip()
                goal = _tex_trim(str(claim.get("goal", "") or ""), 180)
                if claim_id:
                    claim_lines.append(f"- {claim_id}: {goal}")
        sections = [f"Strategy: {str(plan.get('strategy', '')).strip() or 'unspecified'}"]
        if outline:
            sections.extend(["", "Outline:"] + [f"- {item}" for item in outline[:8]])
        if key_lemmas:
            sections.extend(["", "Key lemmas/facts:"] + [f"- {item}" for item in key_lemmas[:8]])
        if checks:
            sections.extend(["", "Checks:"] + [f"- {item}" for item in checks[:8]])
        if claim_lines:
            sections.extend(["", "Claims:"] + claim_lines[:8])
        _tex_repo_upsert(
            repo_items,
            f"pass_{pass_idx}_plan",
            kind="plan",
            summary=str(plan.get("strategy", "") or f"Pass {pass_idx} plan").strip() or f"Pass {pass_idx} plan",
            content="\n".join(sections).strip(),
            pass_idx=pass_idx,
            strategy=str(plan.get("strategy", "") or "").strip(),
        )
    pass_history = run_state.get("pass_history", [])
    if isinstance(pass_history, list):
        for entry in pass_history:
            if not isinstance(entry, dict):
                continue
            try:
                hist_pass = int(entry.get("pass", 0) or 0)
            except Exception:
                hist_pass = 0
            if hist_pass <= 0:
                continue
            accepted_count = int(entry.get("accepted_count", 0) or 0)
            total_claims = int(entry.get("total_claims", 0) or 0)
            strategy = str(entry.get("strategy", "") or "").strip()
            body = []
            if strategy:
                body.append(f"Strategy: {strategy}")
            body.append(
                f"Status: {str(entry.get('status', 'unknown') or 'unknown').strip()} "
                f"({accepted_count}/{total_claims} claims accepted)"
            )
            unresolved = [str(item).strip() for item in list(entry.get("unresolved_claims", []) or []) if str(item).strip()]
            feedback = [str(item).strip() for item in list(entry.get("feedback", []) or []) if str(item).strip()]
            if unresolved:
                body.extend(["", "Unresolved claims:"] + [f"- {item}" for item in unresolved[:12]])
            if feedback:
                body.extend(["", "Key feedback:"] + [f"- {item}" for item in feedback[:12]])
            _tex_repo_upsert(
                repo_items,
                f"pass_{hist_pass}_summary",
                kind="summary",
                summary=(
                    f"Pass {hist_pass} {str(entry.get('status', 'unknown') or 'unknown').strip()}: "
                    f"{accepted_count}/{total_claims} accepted"
                ),
                content="\n".join(body).strip(),
                pass_idx=hist_pass,
                status=str(entry.get("status", "unknown") or "unknown").strip(),
            )
    if isinstance(claims, list):
        accepted_claims = run_state.get("accepted_claims", {})
        best_candidates = run_state.get("best_claim_candidates", {})
        claim_feedback = run_state.get("claim_feedback", {})
        if not isinstance(accepted_claims, dict):
            accepted_claims = {}
        if not isinstance(best_candidates, dict):
            best_candidates = {}
        if not isinstance(claim_feedback, dict):
            claim_feedback = {}
        try:
            pass_idx = int(run_state.get("current_pass", 1) or 1)
        except Exception:
            pass_idx = 1
        try:
            round_idx = int(run_state.get("current_round", 1) or 1)
        except Exception:
            round_idx = 1
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            claim_id = str(claim.get("id", "")).strip()
            if not claim_id:
                continue
            slug = f"claim_{claim_id}"
            accepted = accepted_claims.get(claim_id)
            if isinstance(accepted, dict):
                body = [
                    f"Goal: {str(accepted.get('goal', '')).strip()}",
                    "",
                    "Accepted proof:",
                    _tex_trim(str(accepted.get("proof_tex", "") or ""), 12000),
                ]
                assumptions = list(accepted.get("assumptions_used", []) or [])
                deps = list(accepted.get("depends_on_used", []) or [])
                citations = list(accepted.get("cited_facts", []) or [])
                if assumptions:
                    body.extend(["", "Assumptions used:"] + [f"- {str(item).strip()}" for item in assumptions[:12]])
                if deps:
                    body.extend(["", "Dependencies used:"] + [f"- {str(item).strip()}" for item in deps[:12]])
                if citations:
                    body.extend(["", "Cited facts:"] + [f"- {str(item).strip()}" for item in citations[:12]])
                _tex_repo_upsert(
                    repo_items,
                    slug,
                    kind="claim",
                    summary=f"Accepted claim {claim_id}: {_tex_trim(str(accepted.get('goal', '') or ''), 140)}",
                    content="\n".join(body).strip(),
                    claim_id=claim_id,
                    status="accepted",
                    pass_idx=pass_idx,
                    round_idx=round_idx,
                    score=float(accepted.get("score", 0.0) or 0.0),
                )
                continue
            best_candidate = best_candidates.get(claim_id)
            if not isinstance(best_candidate, dict):
                continue
            draft = best_candidate.get("draft", {}) if isinstance(best_candidate.get("draft"), dict) else {}
            proof_tex = _strip_md_fences(str(draft.get("proof_tex", "") or "")).strip()
            feedback = str(claim_feedback.get(claim_id, "") or "").strip() or _build_tex_claim_feedback(best_candidate)
            body = [f"Goal: {str(claim.get('goal', '')).strip()}"]
            if proof_tex:
                body.extend(["", "Latest best draft:", _tex_trim(proof_tex, 10000)])
            if feedback:
                body.extend(["", "Feedback to address:", _tex_trim(feedback, 5000)])
            _tex_repo_upsert(
                repo_items,
                slug,
                kind="claim",
                summary=f"Open claim {claim_id}: {_tex_trim(str(claim.get('goal', '') or ''), 140)}",
                content="\n".join(body).strip(),
                claim_id=claim_id,
                status="open",
                pass_idx=pass_idx,
                round_idx=round_idx,
                score=float(best_candidate.get("score", 0.0) or 0.0),
            )
    unresolved_lines: list[str] = []
    accepted_claims = run_state.get("accepted_claims", {})
    if not isinstance(accepted_claims, dict):
        accepted_claims = {}
    accepted_ids = {str(key).strip() for key in accepted_claims.keys() if str(key).strip()}
    if isinstance(claims, list):
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            claim_id = str(claim.get("id", "")).strip()
            if not claim_id or claim_id in accepted_ids:
                continue
            unresolved_lines.append(f"- {claim_id}: {_tex_trim(str(claim.get('goal', '') or ''), 180)}")
            feedback = str(run_state.get("claim_feedback", {}).get(claim_id, "") if isinstance(run_state.get("claim_feedback", {}), dict) else "").strip()
            if feedback:
                unresolved_lines.append(f"  feedback: {_tex_trim(feedback.splitlines()[0].lstrip('- ').strip(), 180)}")
    unresolved_count = 0
    if not unresolved_lines:
        unresolved_lines.append("- none")
    else:
        unresolved_count = len(
            [
                claim
                for claim in claims
                if isinstance(claim, dict)
                and str(claim.get("id", "")).strip()
                and str(claim.get("id", "")).strip() not in accepted_ids
            ]
        )
    _tex_repo_upsert(
        repo_items,
        "open_claims",
        kind="memory",
        summary=f"Current unresolved claims ({unresolved_count})",
        content="\n".join(unresolved_lines).strip(),
        pass_idx=int(run_state.get("current_pass", 1) or 1),
    )
    final = run_state.get("final", {})
    if isinstance(final, dict) and final:
        lines = [f"Pass verdict: {bool(final.get('pass', False))}"]
        for key in ("error", "reason"):
            text = str(final.get(key, "") or "").strip()
            if text:
                lines.append(f"{key.title()}: {text}")
        lines.append(
            f"Accepted claims: {int(final.get('accepted_claims', 0) or 0)}/"
            f"{int(final.get('total_claims', 0) or 0)}"
        )
        _tex_repo_upsert(
            repo_items,
            "final_outcome",
            kind="summary",
            summary="Final TeX run outcome",
            content="\n".join(lines).strip(),
            status="finished",
        )
    run_state["whiteboard"] = _render_tex_whiteboard(run_state, theorem, theorem_statement)


def _write_tex_memory_artifacts(run_dir: Path, run_state: dict) -> None:
    whiteboard = str(run_state.get("whiteboard", "") or "").strip()
    if whiteboard:
        _tex_whiteboard_path(run_dir).write_text(whiteboard.rstrip() + "\n", encoding="utf-8")
    repo_items = run_state.get("repo_items", {})
    if not isinstance(repo_items, dict):
        return
    repo_dir = _tex_repo_dir(run_dir)
    repo_dir.mkdir(parents=True, exist_ok=True)
    keep = {"index.json"}
    for slug, item in repo_items.items():
        if not isinstance(item, dict):
            continue
        filename = f"{slug}.md"
        keep.add(filename)
        (repo_dir / filename).write_text(
            _render_tex_repo_item_markdown(item),
            encoding="utf-8",
        )
    for path in repo_dir.glob("*.md"):
        if path.name not in keep:
            path.unlink(missing_ok=True)
    _tex_repo_index_path(run_dir).write_text(
        json.dumps(_tex_repo_index_rows(repo_items), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _build_tex_replan_instruction(base_instruction: str, pass_idx: int, pass_history: list[dict]) -> str:
    if pass_idx <= 1:
        return base_instruction
    recent = pass_history[-1] if pass_history else {}
    unresolved = list(recent.get("unresolved_claims", []) or [])
    key_feedback = list(recent.get("feedback", []) or [])
    strategy = str(recent.get("strategy", "")).strip()
    note_lines: list[str] = []
    note_lines.append(
        f"Replan pass {pass_idx}: previous decomposition stalled. Use a materially different strategy."
    )
    if strategy:
        note_lines.append(f"Previous strategy to avoid repeating: {strategy[:300]}")
    if unresolved:
        preview = ", ".join(str(item).strip() for item in unresolved[:8] if str(item).strip())
        if preview:
            note_lines.append(f"Previously unresolved claims: {preview}")
    if key_feedback:
        note_lines.append("Key failure feedback to address:")
        for item in key_feedback[:6]:
            text = str(item).strip()
            if text:
                note_lines.append(f"- {text[:400]}")
    replan_note = "\n".join(note_lines).strip()
    if base_instruction.strip():
        return base_instruction.strip() + "\n\n" + replan_note
    return replan_note


def _config_with_llm_model(provider: str, config: dict, model_name: str) -> dict:
    if not str(model_name or "").strip():
        return config
    cloned = _json_clone(config)
    if provider in {"openai", "codex_cli"}:
        section = cloned.setdefault("openai", {})
        section["model"] = model_name
        section["codex_model"] = model_name
        return cloned
    if provider in {"anthropic", "claude_cli"}:
        section = cloned.setdefault("anthropic", {})
        section["model"] = model_name
        section["claude_model"] = model_name
        return cloned
    if provider == "ollama":
        cloned.setdefault("ollama", {})["model"] = model_name
        return cloned
    if provider in {"gemini", "gemini_cli"}:
        section = cloned.setdefault("gemini", {})
        section["model"] = model_name
        section["cli_model"] = model_name
        return cloned
    return cloned


def _make_tex_role_llm(provider: str, config: dict, model_name: str) -> FormalizationLLM:
    return FormalizationLLM(provider, _config_with_llm_model(provider, config, model_name))


def _tex_claim_order(claims: list[dict], focus_ids: list[str] | None = None) -> list[str]:
    rows = [str(claim.get("id", "")).strip() for claim in claims if str(claim.get("id", "")).strip()]
    focus = {str(item).strip() for item in list(focus_ids or []) if str(item).strip()}
    if not focus:
        return rows
    prioritized = [claim_id for claim_id in rows if claim_id in focus]
    remainder = [claim_id for claim_id in rows if claim_id not in focus]
    return prioritized + remainder


def _build_tex_action_state(
    run_state: dict,
    *,
    rounds: int,
    judge_repairs: int,
    replan_passes: int,
    action_step: int,
    action_limit: int,
) -> dict:
    claims = run_state.get("claims", [])
    accepted_claims = run_state.get("accepted_claims", {})
    plan = run_state.get("plan", {})
    best_pass = run_state.get("best_pass", {})
    if not isinstance(claims, list):
        claims = []
    if not isinstance(accepted_claims, dict):
        accepted_claims = {}
    if not isinstance(plan, dict):
        plan = {}
    if not isinstance(best_pass, dict):
        best_pass = {}
    unresolved = []
    accepted_ids = {str(key).strip() for key in accepted_claims.keys() if str(key).strip()}
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        claim_id = str(claim.get("id", "")).strip()
        if not claim_id or claim_id in accepted_ids:
            continue
        unresolved.append(
            {
                "id": claim_id,
                "goal": _tex_trim(str(claim.get("goal", "") or ""), 200),
                "depends_on": [str(dep).strip() for dep in list(claim.get("depends_on", []) or []) if str(dep).strip()],
            }
        )
    best_quality = best_pass.get("quality", {}) if isinstance(best_pass.get("quality", {}), dict) else {}
    repo_items = run_state.get("repo_items", {})
    repo_index = _tex_repo_index_rows(repo_items)[:12] if isinstance(repo_items, dict) else []
    action_history = run_state.get("action_history", [])
    if not isinstance(action_history, list):
        action_history = []
    pass_history = run_state.get("pass_history", [])
    if not isinstance(pass_history, list):
        pass_history = []
    return {
        "status": str(run_state.get("status", "running") or "running").strip(),
        "compose_ready": bool(run_state.get("compose_ready", False)),
        "current_pass": int(run_state.get("current_pass", 1) or 1),
        "current_round": int(run_state.get("current_round", 1) or 1),
        "repairs_used": int(run_state.get("repairs_used", 0) or 0),
        "round_progressed": bool(run_state.get("round_progressed", False)),
        "round_limit": int(rounds),
        "judge_repair_limit": int(judge_repairs),
        "replan_pass_limit": int(replan_passes),
        "action_step": int(action_step),
        "action_limit": int(action_limit),
        "active_plan": bool(plan),
        "plan_strategy": _tex_trim(str(plan.get("strategy", "") or ""), 300),
        "plan_claim_count": len(claims),
        "accepted_claims": len(accepted_ids),
        "unresolved_claims": unresolved[:10],
        "pending_repo_reads": [
            _tex_repo_slug(item)
            for item in list(run_state.get("pending_repo_reads", []) or [])[:8]
            if _tex_repo_slug(item)
        ],
        "pending_claim_focus": [
            str(item).strip()[:24]
            for item in list(run_state.get("pending_claim_focus", []) or [])[:8]
            if str(item).strip()
        ],
        "pending_worker_guidance": _tex_trim(str(run_state.get("pending_worker_guidance", "") or ""), 1200),
        "recent_actions": [_json_clone(item) for item in action_history[-4:] if isinstance(item, dict)],
        "recent_passes": [_json_clone(item) for item in pass_history[-3:] if isinstance(item, dict)],
        "best_pass": {
            "pass": int(best_pass.get("pass", 0) or 0) if best_pass else 0,
            "accepted_count": int(best_quality.get("accepted_count", 0) or 0),
            "score": float(best_quality.get("score", 0.0) or 0.0),
        },
        "repo_index": repo_index,
    }


def _build_tex_episode_state(
    run_state: dict,
    *,
    episode_step: int,
    episode_limit: int,
    replan_passes: int,
) -> dict:
    claims = run_state.get("claims", [])
    accepted_claims = run_state.get("accepted_claims", {})
    plan = run_state.get("plan", {})
    best_pass = run_state.get("best_pass", {})
    monolithic = run_state.get("monolithic_attempt", {})
    if not isinstance(claims, list):
        claims = []
    if not isinstance(accepted_claims, dict):
        accepted_claims = {}
    if not isinstance(plan, dict):
        plan = {}
    if not isinstance(best_pass, dict):
        best_pass = {}
    if not isinstance(monolithic, dict):
        monolithic = {}
    accepted_ids = {str(key).strip() for key in accepted_claims.keys() if str(key).strip()}
    unresolved = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        claim_id = str(claim.get("id", "")).strip()
        if not claim_id or claim_id in accepted_ids:
            continue
        unresolved.append(
            {
                "id": claim_id,
                "goal": _tex_trim(str(claim.get("goal", "") or ""), 220),
                "depends_on": [
                    str(dep).strip()
                    for dep in list(claim.get("depends_on", []) or [])
                    if str(dep).strip()
                ],
                "feedback": _tex_trim(
                    str(run_state.get("claim_feedback", {}).get(claim_id, "") if isinstance(run_state.get("claim_feedback", {}), dict) else ""),
                    500,
                ),
            }
        )
    best_quality = best_pass.get("quality", {}) if isinstance(best_pass.get("quality", {}), dict) else {}
    return {
        "status": str(run_state.get("status", "running") or "running").strip(),
        "episode_step": int(episode_step),
        "episode_limit": int(episode_limit),
        "current_pass": int(run_state.get("current_pass", 1) or 1),
        "replan_pass_limit": int(replan_passes),
        "active_plan": bool(plan),
        "plan_strategy": _tex_trim(str(plan.get("strategy", "") or ""), 300),
        "plan_claim_count": len(claims),
        "accepted_claims": len(accepted_ids),
        "unresolved_claims": unresolved[:10],
        "planner_notes": [_tex_trim(str(item), 220) for item in list(run_state.get("planner_notes", []) or [])[-6:] if str(item).strip()],
        "best_pass": {
            "pass": int(best_pass.get("pass", 0) or 0) if best_pass else 0,
            "accepted_count": int(best_quality.get("accepted_count", 0) or 0),
            "score": float(best_quality.get("score", 0.0) or 0.0),
        },
        "monolithic_attempt": {
            "summary": _tex_trim(str(monolithic.get("summary", "") or ""), 500),
            "score": float(monolithic.get("score", 0.0) or 0.0),
            "pass": bool(monolithic.get("pass", False)),
            "feedback": _tex_trim(str(monolithic.get("feedback", "") or ""), 1200),
            "open_questions": [str(item).strip()[:240] for item in list(monolithic.get("open_questions", []) or [])[:8] if str(item).strip()],
        },
        "recent_passes": [
            _json_clone(item)
            for item in list(run_state.get("pass_history", []) or [])[-3:]
            if isinstance(item, dict)
        ],
        "recent_episodes": [
            _json_clone(item)
            for item in list(run_state.get("action_history", []) or [])[-4:]
            if isinstance(item, dict)
        ],
        "repo_index": _tex_repo_index_rows(run_state.get("repo_items", {}))[:12]
        if isinstance(run_state.get("repo_items", {}), dict)
        else [],
    }


def _record_tex_action_history(run_state: dict, *, step: int, action: str, summary: str) -> None:
    history = run_state.get("action_history", [])
    if not isinstance(history, list):
        history = []
    history.append(
        {
            "step": int(step),
            "action": str(action).strip().lower(),
            "summary": _tex_trim(summary, 400),
            "ts": _tex_iso_now(),
        }
    )
    run_state["action_history"] = history[-16:]


def _resolve_tex_planner_action(
    planner_action: dict,
    run_state: dict,
    *,
    replan_passes: int,
) -> str:
    requested = str(planner_action.get("action", "solve") or "solve").strip().lower()
    if requested not in {"plan", "solve", "compose", "write_memory", "give_up"}:
        requested = "solve"
    active_plan = isinstance(run_state.get("plan"), dict) and bool(run_state.get("plan"))
    compose_ready = bool(run_state.get("compose_ready", False))
    current_pass = int(run_state.get("current_pass", 1) or 1)
    if compose_ready and requested not in {"compose", "give_up", "write_memory"}:
        return "compose"
    if requested == "solve" and not active_plan:
        return "plan" if current_pass <= replan_passes else "compose"
    if requested == "plan" and current_pass > replan_passes:
        return "compose"
    return requested


def _apply_tex_planner_memory_action(run_state: dict, planner_action: dict) -> None:
    repo_items = run_state.get("repo_items")
    if not isinstance(repo_items, dict):
        repo_items = {}
        run_state["repo_items"] = repo_items
    note = str(planner_action.get("whiteboard_note", "") or "").strip()
    if note:
        notes = run_state.get("planner_notes", [])
        if not isinstance(notes, list):
            notes = []
        trimmed = _tex_trim(note, 1200)
        if not notes or str(notes[-1]).strip() != trimmed:
            notes.append(trimmed)
        run_state["planner_notes"] = notes[-8:]
    guidance = str(planner_action.get("worker_guidance", "") or "").strip()
    if guidance:
        run_state["pending_worker_guidance"] = _tex_trim(guidance, 4000)
    repo_reads = []
    for item in list(planner_action.get("repo_reads", []) or [])[:8]:
        slug = _tex_repo_slug(item)
        if slug:
            repo_reads.append(slug)
    if repo_reads:
        run_state["pending_repo_reads"] = repo_reads
    claim_focus = [str(item).strip()[:24] for item in list(planner_action.get("claim_focus", []) or [])[:8] if str(item).strip()]
    if claim_focus:
        run_state["pending_claim_focus"] = claim_focus
    for item in list(planner_action.get("repo_writes", []) or [])[:10]:
        if not isinstance(item, dict):
            continue
        op = str(item.get("op", "upsert") or "upsert").strip().lower()
        slug = _tex_repo_slug(item.get("slug", ""))
        if not slug:
            continue
        if op == "delete":
            _tex_repo_delete(repo_items, slug)
            continue
        _tex_repo_upsert(
            repo_items,
            slug,
            kind=str(item.get("kind", "note") or "note").strip()[:40] or "note",
            summary=str(item.get("summary", "") or slug).strip()[:220] or slug,
            content=str(item.get("content", "") or "").strip()[:20000],
            status="planner",
            planner_managed=True,
        )


def _clear_tex_planner_directives(run_state: dict) -> None:
    run_state["pending_worker_guidance"] = ""
    run_state["pending_repo_reads"] = []
    run_state["pending_claim_focus"] = []
    run_state["current_round_order"] = []


def _claim_spec_from_episode_draft(item: dict) -> dict:
    return {
        "id": str(item.get("claim_id", "")).strip()[:24],
        "goal": str(item.get("goal", "")).strip()[:900],
        "depends_on": [str(dep).strip()[:24] for dep in list(item.get("depends_on", []) or []) if str(dep).strip()],
        "assumptions": [str(v).strip()[:280] for v in list(item.get("assumptions", []) or []) if str(v).strip()],
        "required_facts": [str(v).strip()[:280] for v in list(item.get("required_facts", []) or []) if str(v).strip()],
        "acceptance_checks": [str(v).strip()[:280] for v in list(item.get("acceptance_checks", []) or []) if str(v).strip()],
    }


def _find_tex_claim(claims: list[dict], claim_id: str) -> dict | None:
    target = str(claim_id).strip()
    if not target:
        return None
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        if str(claim.get("id", "")).strip() == target:
            return claim
    return None


def _plan_from_episode_payload(episode: dict) -> dict:
    return {
        "strategy": str(episode.get("strategy", "") or "").strip(),
        "outline": list(episode.get("outline", []) or []),
        "key_lemmas": list(episode.get("key_lemmas", []) or []),
        "checks": list(episode.get("checks", []) or []),
        "claims": list(episode.get("claims", []) or []),
    }


def _summarize_tex_pass(run_state: dict, plan: dict, claims: list[dict], status: str) -> dict:
    accepted_claims = run_state.get("accepted_claims", {})
    best_claim_candidates = run_state.get("best_claim_candidates", {})
    claim_feedback = run_state.get("claim_feedback", {})
    if not isinstance(accepted_claims, dict):
        accepted_claims = {}
    if not isinstance(best_claim_candidates, dict):
        best_claim_candidates = {}
    if not isinstance(claim_feedback, dict):
        claim_feedback = {}
    unresolved_claims = [
        str(claim.get("id", "")).strip()
        for claim in claims
        if str(claim.get("id", "")).strip() and str(claim.get("id", "")).strip() not in accepted_claims
    ]
    feedback_lines: list[str] = []
    for item in list(claim_feedback.values())[:8]:
        if not isinstance(item, str):
            continue
        for line in item.splitlines():
            text = line.strip()
            if not text:
                continue
            feedback_lines.append(text.lstrip("- ").strip())
            if len(feedback_lines) >= 8:
                break
        if len(feedback_lines) >= 8:
            break
    quality = _tex_pass_quality(claims, accepted_claims, best_claim_candidates)
    pass_idx = int(run_state.get("current_pass", 1) or 1)
    pass_summary = {
        "pass": pass_idx,
        "strategy": str(plan.get("strategy", "")).strip(),
        "accepted_count": quality[0],
        "total_claims": len(claims),
        "quality_score": quality[1],
        "unresolved_claims": unresolved_claims[:40],
        "feedback": feedback_lines[:12],
        "status": str(status or "unknown").strip() or "unknown",
        "round_reached": int(run_state.get("current_round", 1) or 1),
    }
    pass_history = run_state.get("pass_history", [])
    if not isinstance(pass_history, list):
        pass_history = []
    pass_history.append(pass_summary)
    run_state["pass_history"] = pass_history
    current_best_payload = {
        "pass": pass_idx,
        "plan": _json_clone(plan),
        "claims": _json_clone(claims),
        "accepted_claims": _json_clone(accepted_claims),
        "best_claim_candidates": _json_clone(best_claim_candidates),
        "quality": {"accepted_count": quality[0], "score": quality[1]},
    }
    best_pass = run_state.get("best_pass", None)
    if not isinstance(best_pass, dict):
        run_state["best_pass"] = current_best_payload
    else:
        prev_quality = best_pass.get("quality", {}) if isinstance(best_pass.get("quality", {}), dict) else {}
        try:
            prev_tuple = (
                int(prev_quality.get("accepted_count", 0) or 0),
                float(prev_quality.get("score", 0.0) or 0.0),
            )
        except Exception:
            prev_tuple = (0, 0.0)
        if quality > prev_tuple:
            run_state["best_pass"] = current_best_payload
    return pass_summary


def _reset_tex_pass_state(run_state: dict) -> None:
    run_state["plan"] = None
    run_state["claims"] = []
    run_state["accepted_claims"] = {}
    run_state["best_claim_candidates"] = {}
    run_state["claim_feedback"] = {}
    run_state["current_round"] = 1
    run_state["current_claim_index"] = 0
    run_state["repairs_used"] = 0
    run_state["round_progressed"] = False
    run_state["current_round_order"] = []


def _run_tex_solve_round(
    *,
    planner_llm: FormalizationLLM,
    worker_llm: FormalizationLLM,
    theorem: str,
    theorem_statement: str,
    instruction: str,
    base_context: str,
    run_state: dict,
    run_dir: Path,
    worker_drafts: int,
    tex_concurrency: bool,
    verifier_policy: str,
    judge_repairs: int,
    rounds: int,
    persist_state,
) -> dict:
    plan = dict(run_state.get("plan", {}) or {})
    claims = list(run_state.get("claims", []) or [])
    if not plan or not claims:
        return {"status": "no_plan", "plan": plan, "claims": claims}
    pass_idx = max(1, int(run_state.get("current_pass", 1) or 1))
    round_idx = max(1, int(run_state.get("current_round", 1) or 1))
    if round_idx > rounds:
        return {"status": "round_limit", "plan": plan, "claims": claims}
    claim_index = max(0, int(run_state.get("current_claim_index", 0) or 0))
    if claim_index <= 0:
        print(f"[tex] pass {pass_idx} round {round_idx}/{rounds}")
        run_state["current_round_order"] = _tex_claim_order(
            claims,
            list(run_state.get("pending_claim_focus", []) or []),
        )
    else:
        print(
            f"[tex] pass {pass_idx} round {round_idx}/{rounds} "
            f"(resuming claim {claim_index + 1}/{len(claims)})"
        )
    order = list(run_state.get("current_round_order", []) or [])
    if not order:
        order = _tex_claim_order(claims, list(run_state.get("pending_claim_focus", []) or []))
        run_state["current_round_order"] = order
    claims_by_id = {
        str(claim.get("id", "")).strip(): claim
        for claim in claims
        if isinstance(claim, dict) and str(claim.get("id", "")).strip()
    }
    progressed = bool(run_state.get("round_progressed", False))
    while claim_index < len(order):
        claim_id = str(order[claim_index]).strip()
        claim = claims_by_id.get(claim_id)
        if not isinstance(claim, dict):
            claim_index += 1
            run_state["current_claim_index"] = claim_index
            persist_state()
            continue
        accepted_claims = run_state.get("accepted_claims", {})
        best_claim_candidates = run_state.get("best_claim_candidates", {})
        claim_feedback = run_state.get("claim_feedback", {})
        if not isinstance(accepted_claims, dict):
            accepted_claims = {}
        if not isinstance(best_claim_candidates, dict):
            best_claim_candidates = {}
        if not isinstance(claim_feedback, dict):
            claim_feedback = {}
        run_state["accepted_claims"] = accepted_claims
        run_state["best_claim_candidates"] = best_claim_candidates
        run_state["claim_feedback"] = claim_feedback

        run_state["current_claim_index"] = claim_index
        persist_state()
        if claim_id in accepted_claims:
            claim_index += 1
            run_state["current_claim_index"] = claim_index
            persist_state()
            continue
        deps = [str(dep).strip() for dep in list(claim.get("depends_on", []) or []) if str(dep).strip()]
        missing_deps = [dep for dep in deps if dep not in accepted_claims]
        if missing_deps:
            claim_index += 1
            run_state["current_claim_index"] = claim_index
            persist_state()
            continue

        ledger = _build_tex_claim_ledger(accepted_claims)
        prior_candidate = best_claim_candidates.get(claim_id, {})
        prior_draft = ""
        if isinstance(prior_candidate.get("draft"), dict):
            prior_draft = str(prior_candidate["draft"].get("proof_tex", "") or "")
        prior_feedback = str(claim_feedback.get(claim_id, "") or "")
        accepted_claims_context = _claim_context_for_prompt(claims, accepted_claims)
        claim_context = _build_tex_memory_context(
            base_context,
            run_state,
            claim=claim,
            pass_idx=pass_idx,
            extra_repo_slugs=list(run_state.get("pending_repo_reads", []) or []),
            guidance=str(run_state.get("pending_worker_guidance", "") or ""),
            max_items=8,
        )
        worker_results = _evaluate_tex_claim_workers(
            planner_llm=planner_llm,
            worker_llm=worker_llm,
            theorem=theorem,
            theorem_statement=theorem_statement,
            instruction=str(run_state.get("current_instruction", instruction)),
            plan=plan,
            claim=claim,
            accepted_claims_context=accepted_claims_context,
            ledger=ledger,
            prior_draft=prior_draft,
            prior_feedback=prior_feedback,
            prompt_context=claim_context,
            round_idx=round_idx,
            worker_drafts=worker_drafts,
            concurrent=tex_concurrency,
            verifier_policy=verifier_policy,
        )
        best_candidate: dict | None = None
        for worker_result in worker_results:
            worker_idx = int(worker_result.get("worker", 0) or 0)
            event_payload = worker_result.get("event", {})
            if isinstance(event_payload, dict):
                _append_tex_event(
                    run_dir,
                    {
                        "kind": "candidate",
                        "pass": pass_idx,
                        "round": round_idx,
                        "claim_id": claim_id,
                        **event_payload,
                    },
                )
            candidate = worker_result.get("candidate")
            if not isinstance(candidate, dict):
                status = str(worker_result.get("status", "") or "").strip().lower()
                if status == "worker_error":
                    print(
                        f"[tex] claim={claim_id} worker={worker_idx} "
                        f"error={_tex_trim(str(worker_result.get('error', '') or ''), 180)}"
                    )
                continue
            if best_candidate is None or float(candidate.get("score", -1e9) or -1e9) > float(
                best_candidate.get("score", -1e9) or -1e9
            ):
                best_candidate = candidate
            print(
                f"[tex] claim={claim_id} worker={worker_idx} "
                f"judge={candidate.get('judge', {}).get('verdict','revise')} "
                f"verifier={candidate.get('verifier', {}).get('verdict','skip')} "
                f"checker={candidate.get('checker', {}).get('status','issues')} "
                f"score={float(candidate.get('score', 0.0) or 0.0):.1f}"
            )

        if not best_candidate:
            print(f"[tex] claim={claim_id} produced no valid candidate.")
            _append_tex_event(
                run_dir,
                {
                    "kind": "claim_result",
                    "pass": pass_idx,
                    "round": round_idx,
                    "claim_id": claim_id,
                    "status": "no_candidate",
                },
            )
            claim_index += 1
            run_state["current_claim_index"] = claim_index
            persist_state()
            continue

        if verifier_policy == "promoted":
            promoted_verifier = planner_llm.tex_claim_verifier(
                theorem_name=theorem,
                theorem_statement=theorem_statement,
                instruction=str(run_state.get("current_instruction", instruction)),
                plan=plan,
                claim=claim,
                candidate=best_candidate.get("draft", {}),
                accepted_claims=accepted_claims_context,
                ledger=ledger,
                context=claim_context,
            )
            best_candidate["verifier"] = promoted_verifier
            best_candidate["verifier_mode"] = "promoted"
            best_candidate["score"] = _tex_claim_candidate_score(
                draft=best_candidate.get("draft", {}),
                judge=best_candidate.get("judge", {}),
                verifier=promoted_verifier,
                checker=best_candidate.get("checker", {}),
                static_issues=list(best_candidate.get("static_issues", []) or []),
                require_verifier=True,
            )
            best_candidate["pass_gate"] = _tex_claim_pass_gate(
                judge=best_candidate.get("judge", {}),
                verifier=promoted_verifier,
                checker=best_candidate.get("checker", {}),
                static_issues=list(best_candidate.get("static_issues", []) or []),
                require_verifier=True,
            )
            _append_tex_event(
                run_dir,
                {
                    "kind": "candidate_verifier",
                    "pass": pass_idx,
                    "round": round_idx,
                    "claim_id": claim_id,
                    "mode": "promoted",
                    "candidate": _json_clone(best_candidate),
                },
            )
            print(
                f"[tex] claim={claim_id} promoted verifier="
                f"{str(promoted_verifier.get('verdict', 'revise')).strip().lower()} "
                f"score={float(best_candidate.get('score', 0.0) or 0.0):.1f}"
            )

        best_claim_candidates[claim_id] = best_candidate
        if bool(best_candidate.get("pass_gate")):
            accepted_claims[claim_id] = _finalize_tex_claim(claim, best_candidate)
            claim_feedback.pop(claim_id, None)
            progressed = True
            run_state["round_progressed"] = True
            print(f"[tex] claim={claim_id} accepted.")
            _append_tex_event(
                run_dir,
                {
                    "kind": "claim_result",
                    "pass": pass_idx,
                    "round": round_idx,
                    "claim_id": claim_id,
                    "status": "accepted",
                    "candidate": _json_clone(best_candidate),
                },
            )
        else:
            feedback = _build_tex_claim_feedback(best_candidate)
            if feedback:
                claim_feedback[claim_id] = feedback
            print(f"[tex] claim={claim_id} needs revision.")
            _append_tex_event(
                run_dir,
                {
                    "kind": "claim_result",
                    "pass": pass_idx,
                    "round": round_idx,
                    "claim_id": claim_id,
                    "status": "revise",
                    "feedback": feedback,
                    "candidate": _json_clone(best_candidate),
                },
            )

        claim_index += 1
        run_state["current_claim_index"] = claim_index
        persist_state()

    unresolved = [
        str(claim.get("id", "")).strip()
        for claim in claims
        if str(claim.get("id", "")).strip()
        and str(claim.get("id", "")).strip() not in run_state.get("accepted_claims", {})
    ]
    run_state["current_claim_index"] = 0
    run_state["current_round_order"] = []
    if not unresolved:
        print("[tex] all planned claims accepted.")
        return {"status": "claims_complete", "plan": plan, "claims": claims}
    if round_idx >= rounds:
        run_state["round_progressed"] = False
        return {"status": "round_limit", "plan": plan, "claims": claims}
    if progressed:
        run_state["repairs_used"] = 0
    else:
        run_state["repairs_used"] = int(run_state.get("repairs_used", 0) or 0) + 1
        print(
            f"[tex] no claim accepted this round "
            f"(repair cycle {run_state['repairs_used']}/{judge_repairs})."
        )
        if int(run_state.get("repairs_used", 0) or 0) > judge_repairs:
            print("[tex] reached repair limit for this pass; triggering replan/backtrack.")
            run_state["round_progressed"] = False
            return {"status": "stalled", "plan": plan, "claims": claims}
    run_state["current_round"] = round_idx + 1
    run_state["round_progressed"] = False
    persist_state()
    return {"status": "continue", "plan": plan, "claims": claims}


def _tex_pass_quality(
    claims: list[dict],
    accepted_claims: dict[str, dict],
    best_claim_candidates: dict[str, dict],
) -> tuple[int, float]:
    accepted_count = len([c for c in claims if str(c.get("id", "")).strip() in accepted_claims])
    accepted_score = 0.0
    for claim in accepted_claims.values():
        if not isinstance(claim, dict):
            continue
        try:
            accepted_score += float(claim.get("score", 0.0) or 0.0)
        except Exception:
            continue
    best_candidate_score = 0.0
    for candidate in best_claim_candidates.values():
        if not isinstance(candidate, dict):
            continue
        try:
            best_candidate_score += float(candidate.get("score", 0.0) or 0.0)
        except Exception:
            continue
    return accepted_count, accepted_score + best_candidate_score


def _normalize_tex_proof(text: str, theorem: str, theorem_statement: str) -> str:
    cleaned = _strip_md_fences(text).strip()
    if not cleaned:
        return ""
    if "\\begin{theorem}" in cleaned and "\\begin{proof}" in cleaned:
        return cleaned.rstrip() + "\n"
    if "\\begin{proof}" in cleaned:
        theorem_header = (
            f"\\begin{{theorem}}[{theorem}]\n"
            f"{theorem_statement}\n"
            "\\end{theorem}\n\n"
        )
        return theorem_header + cleaned.rstrip() + "\n"
    return (
        f"\\begin{{theorem}}[{theorem}]\n"
        f"{theorem_statement}\n"
        "\\end{theorem}\n\n"
        "\\begin{proof}\n"
        f"{cleaned}\n"
        "\\end{proof}\n"
    )


def _resolve_tex_claim_graph(plan: dict, theorem_statement: str) -> list[dict]:
    raw_claims = plan.get("claims", [])
    if not isinstance(raw_claims, list):
        raw_claims = []
    claims: list[dict] = []
    seen: set[str] = set()
    for idx, item in enumerate(raw_claims, start=1):
        if not isinstance(item, dict):
            continue
        raw_id = str(item.get("id", "")).strip() or f"C{idx}"
        claim_id = re.sub(r"[^A-Za-z0-9_]+", "_", raw_id).strip("_")[:24] or f"C{idx}"
        if claim_id in seen:
            claim_id = f"C{idx}"
        seen.add(claim_id)
        goal = str(item.get("goal", "") or item.get("statement", "") or "").strip()
        if not goal:
            goal = theorem_statement.strip() or "Prove the theorem statement."
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
                "goal": goal[:900],
                "depends_on": [str(v).strip()[:24] for v in depends_on[:8] if str(v).strip()],
                "assumptions": [str(v).strip()[:280] for v in assumptions[:20] if str(v).strip()],
                "required_facts": [str(v).strip()[:280] for v in required_facts[:20] if str(v).strip()],
                "acceptance_checks": [
                    str(v).strip()[:280] for v in acceptance_checks[:20] if str(v).strip()
                ],
            }
        )
    if not claims:
        return [
            {
                "id": "C1",
                "goal": theorem_statement.strip() or "Prove the theorem statement.",
                "depends_on": [],
                "assumptions": [],
                "required_facts": [],
                "acceptance_checks": list(plan.get("checks", []) or []),
            }
        ]
    order = {claim["id"]: idx for idx, claim in enumerate(claims)}
    claim_ids = set(order.keys())
    for claim in claims:
        cleaned_deps: list[str] = []
        for dep in list(claim.get("depends_on", []) or []):
            dep_id = str(dep).strip()
            if not dep_id or dep_id == claim["id"]:
                continue
            if dep_id not in claim_ids:
                continue
            if order[dep_id] >= order[claim["id"]]:
                continue
            if dep_id in cleaned_deps:
                continue
            cleaned_deps.append(dep_id)
        claim["depends_on"] = cleaned_deps
    return claims


def _claim_context_for_prompt(claims: list[dict], accepted_claims: dict[str, dict]) -> list[dict]:
    context_rows: list[dict] = []
    for claim in claims:
        claim_id = str(claim.get("id", "")).strip()
        if not claim_id or claim_id not in accepted_claims:
            continue
        item = accepted_claims[claim_id]
        context_rows.append(
            {
                "id": claim_id,
                "goal": str(item.get("goal", "")).strip(),
                "depends_on_used": list(item.get("depends_on_used", []) or []),
                "assumptions_used": list(item.get("assumptions_used", []) or []),
                "cited_facts": list(item.get("cited_facts", []) or []),
                "proof_tex": str(item.get("proof_tex", "")).strip()[:5000],
                "status": str(item.get("status", "accepted")).strip() or "accepted",
            }
        )
    return context_rows


def _extract_tex_snippet_for_claim(candidate: dict, claim: dict) -> str:
    raw = str(candidate.get("proof_tex", "") or "").strip()
    cleaned = _strip_md_fences(raw).strip()
    if not cleaned:
        return ""
    claim_id = str(claim.get("id", "")).strip()
    goal = str(claim.get("goal", "")).strip()
    if "\\begin{proof}" in cleaned and "\\end{proof}" in cleaned:
        return cleaned
    if "\\begin{theorem}" in cleaned and "\\end{theorem}" in cleaned:
        return cleaned
    header = f"% claim {claim_id}\n" if claim_id else ""
    if goal:
        header += f"% goal: {goal[:240]}\n"
    return (header + cleaned).strip()


def _tex_static_claim_issues(claim: dict, candidate: dict, accepted_claims: dict[str, dict]) -> list[str]:
    issues: list[str] = []
    proof = str(candidate.get("proof_tex", "") or "").strip()
    if not proof:
        return ["missing proof text"]
    lowered = proof.lower()
    placeholders = [
        "todo",
        "tbd",
        "???",
        "to be completed",
        "left to the reader",
    ]
    for marker in placeholders:
        if marker in lowered:
            issues.append(f"placeholder text found: {marker}")

    claim_id = str(claim.get("id", "")).strip()
    used_id = str(candidate.get("claim_id", "")).strip()
    if claim_id and used_id and claim_id != used_id:
        issues.append(f"claim id mismatch (expected {claim_id}, got {used_id})")

    expected_deps = [str(dep).strip() for dep in list(claim.get("depends_on", []) or []) if str(dep).strip()]
    deps_used = [str(dep).strip() for dep in list(candidate.get("depends_on_used", []) or []) if str(dep).strip()]
    for dep in expected_deps:
        if dep not in deps_used:
            issues.append(f"missing dependency citation: {dep}")
    for dep in deps_used:
        if dep not in expected_deps and dep not in accepted_claims:
            issues.append(f"references unknown dependency: {dep}")

    required_facts = [str(v).strip() for v in list(claim.get("required_facts", []) or []) if str(v).strip()]
    cited_facts = [str(v).strip() for v in list(candidate.get("cited_facts", []) or []) if str(v).strip()]
    for fact in required_facts:
        if not _tex_fact_is_covered(fact, cited_facts, proof):
            issues.append(f"required fact not cited: {fact}")

    known_assumptions = {
        _normalize_tex_phrase(str(v))
        for v in list(claim.get("assumptions", []) or [])
        if str(v).strip()
    }
    for dep in expected_deps:
        dep_claim = accepted_claims.get(dep)
        if not isinstance(dep_claim, dict):
            continue
        for item in list(dep_claim.get("assumptions_used", []) or []):
            phrase = _normalize_tex_phrase(str(item))
            if phrase:
                known_assumptions.add(phrase)
    used_assumptions = [
        _normalize_tex_phrase(str(v))
        for v in list(candidate.get("assumptions_used", []) or [])
        if str(v).strip()
    ]
    unknown_assumptions = [
        item for item in used_assumptions if item and item not in known_assumptions and known_assumptions
    ]
    if len(unknown_assumptions) > 2:
        issues.append("uses multiple unstated assumptions")
    return issues[:20]


def _normalize_tex_phrase(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def _tex_fact_is_covered(fact: str, cited_facts: list[str], proof_tex: str) -> bool:
    target = _normalize_tex_phrase(fact)
    if not target:
        return True
    cited_norm = {_normalize_tex_phrase(item) for item in cited_facts}
    if target in cited_norm:
        return True
    proof_norm = _normalize_tex_phrase(proof_tex)
    token = target[:40]
    return bool(token and token in proof_norm)


def _tex_claim_candidate_score(
    draft: dict,
    judge: dict,
    verifier: dict,
    checker: dict,
    static_issues: list[str],
    require_verifier: bool = True,
) -> float:
    judge_score = float(judge.get("score", 0) or 0)
    verifier_score = float(verifier.get("score", 0) or 0) if require_verifier else 0.0
    checker_score = float(checker.get("score", 0) or 0)
    confidence = float(draft.get("confidence", 0) or 0)
    score = (
        0.45 * judge_score
        + (0.30 * verifier_score if require_verifier else 0.0)
        + 0.20 * checker_score
        + 0.05 * confidence
    )
    judge_verdict = str(judge.get("verdict", "revise")).strip().lower()
    verifier_verdict = str(verifier.get("verdict", "revise")).strip().lower()
    checker_status = str(checker.get("status", "issues")).strip().lower()
    if judge_verdict == "revise":
        score -= 10
    elif judge_verdict == "fail":
        score -= 28
    if require_verifier:
        if verifier_verdict == "revise":
            score -= 12
        elif verifier_verdict == "fail":
            score -= 30
    if checker_status != "ok":
        score -= 12
    score -= 7 * len(static_issues)
    return score


def _tex_claim_pass_gate(
    judge: dict,
    verifier: dict,
    checker: dict,
    static_issues: list[str],
    require_verifier: bool = True,
) -> bool:
    judge_pass = str(judge.get("verdict", "revise")).strip().lower() == "pass"
    verifier_pass = str(verifier.get("verdict", "revise")).strip().lower() == "pass"
    checker_ok = str(checker.get("status", "issues")).strip().lower() == "ok"
    if require_verifier:
        return judge_pass and verifier_pass and checker_ok and not static_issues
    return judge_pass and checker_ok and not static_issues


def _evaluate_tex_claim_candidate_from_draft(
    *,
    thinker_llm: FormalizationLLM,
    theorem: str,
    theorem_statement: str,
    instruction: str,
    plan: dict,
    claim: dict,
    draft: dict,
    accepted_claims_context: list[dict],
    ledger: dict,
    prompt_context: str,
    verifier_policy: str,
) -> dict:
    static_issues = _tex_static_claim_issues(
        claim=claim,
        candidate=draft,
        accepted_claims={row.get("id", ""): row for row in accepted_claims_context if row.get("id")},
    )
    judge = thinker_llm.tex_claim_judge(
        theorem_name=theorem,
        theorem_statement=theorem_statement,
        instruction=instruction,
        plan=plan,
        claim=claim,
        candidate=draft,
        accepted_claims=accepted_claims_context,
        ledger=ledger,
        context=prompt_context,
    )
    if verifier_policy == "worker":
        verifier = thinker_llm.tex_claim_verifier(
            theorem_name=theorem,
            theorem_statement=theorem_statement,
            instruction=instruction,
            plan=plan,
            claim=claim,
            candidate=draft,
            accepted_claims=accepted_claims_context,
            ledger=ledger,
            context=prompt_context,
        )
    else:
        verifier = {
            "verdict": "skip",
            "score": 0,
            "summary": "deferred by profile",
            "critical_issues": [],
            "counterexample_attempt": "",
            "suggested_repairs": [],
        }
    checker = thinker_llm.tex_claim_domain_check(
        theorem_name=theorem,
        theorem_statement=theorem_statement,
        plan=plan,
        claim=claim,
        candidate=draft,
        context=prompt_context,
    )
    require_verifier = verifier_policy == "worker"
    candidate_score = _tex_claim_candidate_score(
        draft=draft,
        judge=judge,
        verifier=verifier,
        checker=checker,
        static_issues=static_issues,
        require_verifier=require_verifier,
    )
    pass_gate = _tex_claim_pass_gate(
        judge=judge,
        verifier=verifier,
        checker=checker,
        static_issues=static_issues,
        require_verifier=require_verifier,
    )
    return {
        "claim_id": str(claim.get("id", "")).strip(),
        "draft": draft,
        "judge": judge,
        "verifier": verifier,
        "checker": checker,
        "static_issues": static_issues,
        "score": candidate_score,
        "pass_gate": pass_gate,
        "verifier_mode": "worker" if require_verifier else "deferred",
    }


def _evaluate_tex_final_candidate(
    *,
    thinker_llm: FormalizationLLM,
    theorem: str,
    theorem_statement: str,
    instruction: str,
    plan: dict,
    compose_claims: list[dict],
    draft_tex: str,
    context: str,
) -> dict:
    normalized = _normalize_tex_proof(draft_tex, theorem=theorem, theorem_statement=theorem_statement)
    if not normalized:
        return {
            "draft_tex": "",
            "judge": {"verdict": "revise", "score": 0, "summary": "empty draft", "required_changes": ["empty final draft"], "style_notes": [], "polished_tex": ""},
            "verifier": {"verdict": "revise", "score": 0, "summary": "empty draft", "critical_issues": ["empty final draft"], "counterexample_attempt": "", "suggested_repairs": []},
            "checker": {"status": "issues", "score": 0, "issues": ["empty final draft"], "warnings": [], "sanity_checks": []},
            "static_issues": ["missing proof text"],
            "pass": False,
            "score": -1.0,
        }
    ledger = _build_tex_claim_ledger(
        {item["id"]: item for item in compose_claims if str(item.get("id", "")).strip()}
    )
    final_judge = thinker_llm.tex_judge(
        theorem_name=theorem,
        theorem_statement=theorem_statement,
        instruction=instruction,
        plan=plan,
        draft_tex=normalized,
        context=context,
    )
    final_claim = {
        "id": "FINAL",
        "goal": theorem_statement,
        "depends_on": [
            str(item.get("id", "")).strip()
            for item in compose_claims
            if str(item.get("id", "")).strip()
        ],
        "assumptions": [],
        "required_facts": [],
        "acceptance_checks": list(plan.get("checks", []) or []),
    }
    final_candidate = {
        "claim_id": "FINAL",
        "proof_tex": normalized,
        "assumptions_used": sorted(
            {ass for item in compose_claims for ass in list(item.get("assumptions_used", []) or [])}
        ),
        "depends_on_used": [
            str(item.get("id", "")).strip()
            for item in compose_claims
            if str(item.get("id", "")).strip()
        ],
        "cited_facts": sorted(
            {fact for item in compose_claims for fact in list(item.get("cited_facts", []) or [])}
        ),
        "confidence": 90,
    }
    final_verifier = thinker_llm.tex_claim_verifier(
        theorem_name=theorem,
        theorem_statement=theorem_statement,
        instruction=instruction,
        plan=plan,
        claim=final_claim,
        candidate=final_candidate,
        accepted_claims=compose_claims,
        ledger=ledger,
        context=context,
    )
    final_checker = thinker_llm.tex_claim_domain_check(
        theorem_name=theorem,
        theorem_statement=theorem_statement,
        plan=plan,
        claim=final_claim,
        candidate=final_candidate,
        context=context,
    )
    final_static_issues = _tex_static_claim_issues(
        claim=final_claim,
        candidate=final_candidate,
        accepted_claims={
            item["id"]: item
            for item in compose_claims
            if str(item.get("id", "")).strip()
        },
    )
    final_pass = (
        str(final_judge.get("verdict", "revise")).strip().lower() == "pass"
        and str(final_verifier.get("verdict", "revise")).strip().lower() == "pass"
        and str(final_checker.get("status", "issues")).strip().lower() == "ok"
        and not final_static_issues
    )
    score = (
        0.45 * float(final_judge.get("score", 0) or 0)
        + 0.30 * float(final_verifier.get("score", 0) or 0)
        + 0.20 * float(final_checker.get("score", 0) or 0)
        - 7 * len(final_static_issues)
    )
    return {
        "draft_tex": normalized,
        "judge": final_judge,
        "verifier": final_verifier,
        "checker": final_checker,
        "static_issues": final_static_issues,
        "pass": final_pass,
        "score": score,
    }


def _finish_tex_run(
    *,
    run_state: dict,
    run_dir: Path,
    out_path: Path,
    final_eval: dict,
    used_pass: int,
    accepted_claims: int,
    total_claims: int,
    persist_state,
) -> bool:
    final_draft = str(final_eval.get("draft_tex", "") or "").strip()
    if final_draft:
        out_path.write_text(final_draft.rstrip() + "\n", encoding="utf-8")
        print(f"Wrote TeX proof draft to: {out_path}")
    final_judge = final_eval.get("judge", {}) if isinstance(final_eval.get("judge", {}), dict) else {}
    final_verifier = final_eval.get("verifier", {}) if isinstance(final_eval.get("verifier", {}), dict) else {}
    final_checker = final_eval.get("checker", {}) if isinstance(final_eval.get("checker", {}), dict) else {}
    final_static_issues = list(final_eval.get("static_issues", []) or [])
    final_pass = bool(final_eval.get("pass", False))
    print(
        f"Judge verdict: {str(final_judge.get('verdict', 'revise')).strip().lower()} "
        f"(score={int(final_judge.get('score', 0) or 0)})"
    )
    print(
        f"Verifier verdict: {str(final_verifier.get('verdict', 'revise')).strip().lower()} "
        f"(score={int(final_verifier.get('score', 0) or 0)})"
    )
    print(
        f"Checker status: {str(final_checker.get('status', 'issues')).strip().lower()} "
        f"(score={int(final_checker.get('score', 0) or 0)})"
    )
    if final_static_issues:
        print("[tex] static issues:")
        for issue in final_static_issues[:8]:
            print(f"  - {issue}")
    summary = str(final_judge.get("summary", "")).strip()
    if summary:
        print(f"Judge summary: {summary}")
    verifier_summary = str(final_verifier.get("summary", "")).strip()
    if verifier_summary:
        print(f"Verifier summary: {verifier_summary}")
    print(f"[tex] accepted claims: {accepted_claims}/{total_claims}")

    run_state["status"] = "finished"
    run_state["final"] = {
        "pass": final_pass,
        "out_path": str(out_path),
        "used_pass": int(used_pass),
        "accepted_claims": int(accepted_claims),
        "total_claims": int(total_claims),
        "judge": _json_clone(final_judge),
        "verifier": _json_clone(final_verifier),
        "checker": _json_clone(final_checker),
        "static_issues": _json_clone(final_static_issues),
    }
    persist_state()
    _tex_summary_path(run_dir).write_text(
        json.dumps(run_state["final"], indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _append_tex_event(
        run_dir,
        {
            "kind": "final",
            "pass": final_pass,
            "used_pass": int(used_pass),
            "accepted_claims": int(accepted_claims),
            "total_claims": int(total_claims),
            "judge": _json_clone(final_judge),
            "verifier": _json_clone(final_verifier),
            "checker": _json_clone(final_checker),
            "static_issues": _json_clone(final_static_issues),
        },
    )
    print(f"[tex] state snapshot: {_tex_state_path(run_dir)}")
    print(f"[tex] event log: {_tex_events_path(run_dir)}")
    print(f"[tex] summary: {_tex_summary_path(run_dir)}")
    return final_pass


def _evaluate_tex_claim_worker(
    *,
    planner_llm: FormalizationLLM,
    worker_llm: FormalizationLLM,
    theorem: str,
    theorem_statement: str,
    instruction: str,
    plan: dict,
    claim: dict,
    accepted_claims_context: list[dict],
    ledger: dict,
    prior_draft: str,
    prior_feedback: str,
    prompt_context: str,
    round_idx: int,
    worker_idx: int,
    verifier_policy: str,
) -> dict:
    try:
        draft = worker_llm.tex_claim_draft(
            theorem_name=theorem,
            theorem_statement=theorem_statement,
            instruction=instruction,
            plan=plan,
            claim=claim,
            accepted_claims=accepted_claims_context,
            ledger=ledger,
            prior_draft=prior_draft,
            prior_feedback=prior_feedback,
            context=prompt_context,
            round_idx=round_idx,
            worker_id=worker_idx,
        )
        proof_tex = _extract_tex_snippet_for_claim(draft, claim=claim)
        if not proof_tex:
            return {
                "worker": worker_idx,
                "status": "empty_proof",
                "event": {
                    "worker": worker_idx,
                    "status": "empty_proof",
                },
                "candidate": None,
            }
        draft["proof_tex"] = proof_tex
        candidate = _evaluate_tex_claim_candidate_from_draft(
            thinker_llm=planner_llm,
            theorem=theorem,
            theorem_statement=theorem_statement,
            instruction=instruction,
            plan=plan,
            claim=claim,
            draft=draft,
            accepted_claims_context=accepted_claims_context,
            ledger=ledger,
            prompt_context=prompt_context,
            verifier_policy=verifier_policy,
        )
        return {
            "worker": worker_idx,
            "status": "ok",
            "candidate": candidate,
            "event": {
                "worker": worker_idx,
                "candidate": _json_clone(candidate),
            },
        }
    except Exception as exc:
        return {
            "worker": worker_idx,
            "status": "worker_error",
            "error": str(exc),
            "candidate": None,
            "event": {
                "worker": worker_idx,
                "status": "worker_error",
                "error": str(exc),
            },
        }


def _evaluate_tex_claim_workers(
    *,
    planner_llm: FormalizationLLM,
    worker_llm: FormalizationLLM,
    theorem: str,
    theorem_statement: str,
    instruction: str,
    plan: dict,
    claim: dict,
    accepted_claims_context: list[dict],
    ledger: dict,
    prior_draft: str,
    prior_feedback: str,
    prompt_context: str,
    round_idx: int,
    worker_drafts: int,
    concurrent: bool,
    verifier_policy: str,
) -> list[dict]:
    max_workers = max(1, int(worker_drafts))
    if max_workers == 1 or not bool(concurrent):
        return [
            _evaluate_tex_claim_worker(
                planner_llm=planner_llm,
                worker_llm=worker_llm,
                theorem=theorem,
                theorem_statement=theorem_statement,
                instruction=instruction,
                plan=plan,
                claim=claim,
                accepted_claims_context=accepted_claims_context,
                ledger=ledger,
                prior_draft=prior_draft,
                prior_feedback=prior_feedback,
                prompt_context=prompt_context,
                round_idx=round_idx,
                worker_idx=worker_idx,
                verifier_policy=verifier_policy,
            )
            for worker_idx in range(1, max_workers + 1)
        ]
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _evaluate_tex_claim_worker,
                planner_llm=planner_llm,
                worker_llm=worker_llm,
                theorem=theorem,
                theorem_statement=theorem_statement,
                instruction=instruction,
                plan=plan,
                claim=claim,
                accepted_claims_context=accepted_claims_context,
                ledger=ledger,
                prior_draft=prior_draft,
                prior_feedback=prior_feedback,
                prompt_context=prompt_context,
                round_idx=round_idx,
                worker_idx=worker_idx,
                verifier_policy=verifier_policy,
            )
            for worker_idx in range(1, max_workers + 1)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: int(item.get("worker", 0) or 0))
    return results


def _finalize_tex_claim(claim: dict, best_candidate: dict) -> dict:
    draft = best_candidate.get("draft", {}) if isinstance(best_candidate.get("draft"), dict) else {}
    judge = best_candidate.get("judge", {}) if isinstance(best_candidate.get("judge"), dict) else {}
    polished = _strip_md_fences(str(judge.get("polished_proof_tex", "") or "")).strip()
    proof_tex = polished or _strip_md_fences(str(draft.get("proof_tex", "") or "")).strip()
    depends_on_used = [
        str(v).strip()
        for v in list(draft.get("depends_on_used", []) or [])
        if str(v).strip()
    ]
    assumptions_used = [
        str(v).strip()
        for v in list(draft.get("assumptions_used", []) or [])
        if str(v).strip()
    ]
    cited_facts = [
        str(v).strip()
        for v in list(draft.get("cited_facts", []) or [])
        if str(v).strip()
    ]
    if not assumptions_used:
        assumptions_used = [
            str(v).strip() for v in list(claim.get("assumptions", []) or []) if str(v).strip()
        ]
    return {
        "id": str(claim.get("id", "")).strip(),
        "goal": str(claim.get("goal", "")).strip(),
        "proof_tex": proof_tex,
        "depends_on_used": depends_on_used,
        "assumptions_used": assumptions_used,
        "cited_facts": cited_facts,
        "status": "accepted",
        "score": float(best_candidate.get("score", 0.0) or 0.0),
        "judge": judge,
        "verifier": best_candidate.get("verifier", {}),
        "checker": best_candidate.get("checker", {}),
    }


def _build_tex_claim_feedback(best_candidate: dict) -> str:
    issues: list[str] = []
    for item in list(best_candidate.get("static_issues", []) or []):
        text = str(item).strip()
        if text:
            issues.append(text)
    judge = best_candidate.get("judge", {}) if isinstance(best_candidate.get("judge"), dict) else {}
    verifier = best_candidate.get("verifier", {}) if isinstance(best_candidate.get("verifier"), dict) else {}
    checker = best_candidate.get("checker", {}) if isinstance(best_candidate.get("checker"), dict) else {}
    for key in ["required_changes", "missing_assumptions", "citation_issues"]:
        for item in list(judge.get(key, []) or []):
            text = str(item).strip()
            if text:
                issues.append(text)
    for key in ["critical_issues", "suggested_repairs"]:
        for item in list(verifier.get(key, []) or []):
            text = str(item).strip()
            if text:
                issues.append(text)
    for item in list(checker.get("issues", []) or []):
        text = str(item).strip()
        if text:
            issues.append(text)

    deduped: list[str] = []
    seen: set[str] = set()
    for item in issues:
        key = _normalize_tex_phrase(item)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return "\n".join(f"- {item}" for item in deduped[:14])


def _claims_for_composition(
    claims: list[dict],
    accepted_claims: dict[str, dict],
    best_claim_candidates: dict[str, dict],
) -> list[dict]:
    rows: list[dict] = []
    for claim in claims:
        claim_id = str(claim.get("id", "")).strip()
        if not claim_id:
            continue
        accepted = accepted_claims.get(claim_id)
        if isinstance(accepted, dict):
            rows.append(accepted)
            continue
        best = best_claim_candidates.get(claim_id)
        if not isinstance(best, dict):
            continue
        draft = best.get("draft", {}) if isinstance(best.get("draft"), dict) else {}
        judge = best.get("judge", {}) if isinstance(best.get("judge"), dict) else {}
        polished = _strip_md_fences(str(judge.get("polished_proof_tex", "") or "")).strip()
        proof_tex = polished or _strip_md_fences(str(draft.get("proof_tex", "") or "")).strip()
        rows.append(
            {
                "id": claim_id,
                "goal": str(claim.get("goal", "")).strip(),
                "proof_tex": proof_tex,
                "depends_on_used": list(draft.get("depends_on_used", []) or []),
                "assumptions_used": list(draft.get("assumptions_used", []) or []),
                "cited_facts": list(draft.get("cited_facts", []) or []),
                "status": "draft",
                "score": float(best.get("score", 0.0) or 0.0),
            }
        )
    return rows


def _fallback_compose_tex(claims: list[dict], compose_claims: list[dict]) -> str:
    if not compose_claims:
        goals = [str(claim.get("goal", "")).strip() for claim in claims if str(claim.get("goal", "")).strip()]
        if not goals:
            return ""
        return "\n\n".join(goals)
    parts: list[str] = []
    for item in compose_claims:
        claim_id = str(item.get("id", "")).strip()
        goal = str(item.get("goal", "")).strip()
        proof_tex = _strip_md_fences(str(item.get("proof_tex", "") or "")).strip()
        if claim_id:
            parts.append(f"\\paragraph{{Claim {claim_id}.}} {goal}")
        if proof_tex:
            parts.append(proof_tex)
    parts.append("Combining the established claims yields the theorem.")
    return "\n\n".join(part for part in parts if part.strip())


def _build_tex_claim_ledger(claims_by_id: dict[str, dict]) -> dict:
    claim_ids = sorted(str(cid).strip() for cid in claims_by_id.keys() if str(cid).strip())
    dependencies: dict[str, list[str]] = {}
    assumptions: dict[str, list[str]] = {}
    citations: dict[str, list[str]] = {}
    all_assumptions: set[str] = set()
    all_citations: set[str] = set()
    for claim_id in claim_ids:
        item = claims_by_id.get(claim_id, {})
        if not isinstance(item, dict):
            continue
        dep_list = [str(v).strip() for v in list(item.get("depends_on_used", []) or []) if str(v).strip()]
        asm_list = [str(v).strip() for v in list(item.get("assumptions_used", []) or []) if str(v).strip()]
        cit_list = [str(v).strip() for v in list(item.get("cited_facts", []) or []) if str(v).strip()]
        dependencies[claim_id] = dep_list[:24]
        assumptions[claim_id] = asm_list[:24]
        citations[claim_id] = cit_list[:24]
        all_assumptions.update(asm_list)
        all_citations.update(cit_list)
    return {
        "accepted_claim_ids": claim_ids,
        "dependencies": dependencies,
        "assumptions_by_claim": assumptions,
        "citations_by_claim": citations,
        "all_assumptions": sorted(all_assumptions)[:80],
        "all_citations": sorted(all_citations)[:80],
    }


def _strip_md_fences(text: str) -> str:
    raw = (text or "").strip()
    if not raw.startswith("```"):
        return raw
    lines = raw.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _resolve_formalize_max_rounds(args: argparse.Namespace, config: dict | None = None) -> int:
    explicit = getattr(args, "max_rounds", None)
    if explicit is not None:
        try:
            return max(1, int(explicit))
        except Exception:
            return 5
    cfg = config if config is not None else load_config()
    raw = cfg.get("formalize", {}).get("max_rounds", 5)
    try:
        return max(1, int(raw))
    except Exception:
        return 5


def _resolve_formalize_max_repairs(
    args: argparse.Namespace,
    max_rounds: int,
    config: dict | None = None,
) -> int:
    explicit = getattr(args, "max_repairs", None)
    if explicit is not None:
        try:
            return max(0, int(explicit))
        except Exception:
            return max(0, int(max_rounds))
    cfg = config if config is not None else load_config()
    raw = cfg.get("formalize", {}).get("max_repairs", None)
    if raw is None:
        return max(0, int(max_rounds))
    try:
        return max(0, int(raw))
    except Exception:
        return max(0, int(max_rounds))


def _resolve_formalize_max_proof_rounds(args: argparse.Namespace, config: dict | None = None) -> int:
    explicit = getattr(args, "max_proof_rounds", None)
    if explicit is not None:
        try:
            return max(1, int(explicit))
        except Exception:
            return 1
    cfg = config if config is not None else load_config()
    raw = cfg.get("formalize", {}).get("max_proof_rounds", 1)
    try:
        return max(1, int(raw))
    except Exception:
        return 1


def _resolve_formalize_proof_repair(args: argparse.Namespace, config: dict | None = None) -> int:
    explicit = getattr(args, "proof_repair", None)
    if explicit is not None:
        try:
            return max(0, int(explicit))
        except Exception:
            return 2
    cfg = config if config is not None else load_config()
    raw = cfg.get("formalize", {}).get("proof_repair", 2)
    try:
        return max(0, int(raw))
    except Exception:
        return 2


def _resolve_formalize_typecheck_timeout(
    args: argparse.Namespace, config: dict | None = None
) -> float:
    explicit = getattr(args, "typecheck_timeout", None)
    if explicit is not None:
        try:
            return max(5.0, float(explicit))
        except Exception:
            return 60.0
    cfg = config if config is not None else load_config()
    raw = cfg.get("formalize", {}).get("typecheck_timeout_s", 60)
    try:
        return max(5.0, float(raw))
    except Exception:
        return 60.0


def _resolve_formalize_llm_check(args: argparse.Namespace, config: dict | None = None) -> bool:
    explicit = getattr(args, "llm_check", None)
    if explicit is not None:
        return bool(explicit)
    cfg = config if config is not None else load_config()
    return bool(cfg.get("formalize", {}).get("llm_check", True))


def _resolve_formalize_llm_check_timing(args: argparse.Namespace, config: dict | None = None) -> str:
    explicit = str(getattr(args, "llm_check_timing", "") or "").strip().lower()
    if explicit in {"end", "mid+end"}:
        return explicit
    cfg = config if config is not None else load_config()
    raw = str(cfg.get("formalize", {}).get("llm_check_timing", "end")).strip().lower()
    if raw in {"end", "mid+end"}:
        return raw
    return "end"


def _resolve_formalize_llm_check_repairs(args: argparse.Namespace, config: dict | None = None) -> int:
    explicit = getattr(args, "llm_check_repairs", None)
    if explicit is not None:
        try:
            return max(0, int(explicit))
        except Exception:
            return 2
    cfg = config if config is not None else load_config()
    raw = cfg.get("formalize", {}).get("llm_check_repairs", 2)
    try:
        return max(0, int(raw))
    except Exception:
        return 2


def run_replay(args: argparse.Namespace) -> None:
    if not args.trace.exists():
        print(f"Trace not found: {args.trace}")
        sys.exit(1)

    trace_rows = _load_trace_rows(args.trace)
    meta = _load_trace_metadata(args.trace, getattr(args, "meta", None))

    if not getattr(args, "execute", False):
        solved = any(bool(row.get("solved")) for row in trace_rows)
        print(f"Steps: {len(trace_rows)}")
        print(f"Solved: {solved}")
        tactics = [str(row.get("tactic", "")).strip() for row in trace_rows if str(row.get("tactic", "")).strip()]
        if tactics:
            print("Tactics:")
            for tactic in tactics:
                print(f"- {tactic}")
        if meta:
            print("Metadata:")
            print(f"- Ulam: {meta.get('ulam_version', '?')}")
            print(f"- Lean backend: {meta.get('lean_backend', '?')}")
            print(f"- Toolchain: {meta.get('lean_toolchain') or '(unknown)'}")
            print(f"- Mathlib commit: {meta.get('mathlib_commit') or '(unknown)'}")
        return

    _execute_replay(args, trace_rows, meta)


def run_index(args: argparse.Namespace) -> None:
    if args.index_command == "build":
        project = Path(args.project).expanduser().resolve()
        scope = args.scope
        out = Path(args.out).expanduser()
        if not out.is_absolute():
            out = project / out
        stats = build_premise_index(project, out, scope=scope)
        print(f"Index written: {out}")
        print(f"Records: {stats.get('records', 0)}")
        print(f"Local files: {stats.get('local_files', 0)}")
        print(f"Mathlib files: {stats.get('mathlib_files', 0)}")
        return
    if args.index_command == "stats":
        index_path = Path(args.index).expanduser()
        if not index_path.is_absolute():
            index_path = Path.cwd() / index_path
        stats = load_index_stats(index_path)
        print(f"Index: {index_path}")
        print(f"Records: {stats.get('records', 0)}")
        print(f"Local records: {stats.get('local_records', 0)}")
        print(f"Mathlib records: {stats.get('mathlib_records', 0)}")
        return
    raise RuntimeError(f"unknown index command: {args.index_command}")


def _load_trace_rows(trace_path: Path) -> list[dict]:
    rows: list[dict] = []
    with trace_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _trace_meta_path(trace_path: Path) -> Path:
    if trace_path.suffix == ".jsonl":
        return trace_path.with_suffix(".meta.json")
    return trace_path.with_name(trace_path.name + ".meta.json")


def _write_trace_metadata(trace_path: Path | None, metadata: dict | None) -> None:
    if trace_path is None or metadata is None:
        return
    if str(trace_path) == "-":
        return
    meta_path = _trace_meta_path(trace_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_trace_result_metadata(trace_path: Path | None, result: SearchResult) -> None:
    if trace_path is None or str(trace_path) == "-":
        return
    meta_path = _trace_meta_path(trace_path)
    payload: dict[str, object] = {}
    if meta_path.exists():
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            loaded = {}
        if isinstance(loaded, dict):
            payload = loaded
    payload["search_result"] = {
        "solved": bool(result.solved),
        "steps": int(result.steps),
        "error": str(result.error or ""),
    }
    payload["search_stats"] = _merge_search_stats(result.stats)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _load_trace_metadata(trace_path: Path, explicit: Path | None) -> dict:
    path = explicit if explicit is not None else _trace_meta_path(trace_path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _trace_metadata_payload(
    *,
    args: argparse.Namespace,
    mode: str,
    solver: str,
    file_path: Path,
    theorem: str,
) -> dict:
    file_abs = file_path.expanduser().resolve()
    lean_project = getattr(args, "lean_project", None) or _find_lean_project_for_file(file_abs)
    lean_project_path = lean_project.expanduser().resolve() if isinstance(lean_project, Path) else None
    lean_toolchain = _read_toolchain_file(lean_project_path / "lean-toolchain") if lean_project_path else None
    mathlib_rev = _mathlib_rev_from_manifest(lean_project_path) if lean_project_path else None
    mathlib_commit = _mathlib_commit_from_checkout(lean_project_path) if lean_project_path else None
    return {
        "schema": 1,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ulam_version": __version__,
        "mode": mode,
        "file": str(file_abs),
        "file_sha256": _sha256_file(file_abs),
        "theorem": theorem,
        "lean_backend": getattr(args, "lean", "mock"),
        "lean_project": str(lean_project_path) if lean_project_path else "",
        "lean_imports": list(getattr(args, "lean_import", []) or []),
        "lean_toolchain": lean_toolchain or "",
        "mathlib_rev": mathlib_rev or "",
        "mathlib_commit": mathlib_commit or "",
        "solver": solver,
        "prove_mode": getattr(args, "prove_mode", "tactic"),
        "retriever": getattr(args, "retriever", "none"),
        "retriever_source": getattr(args, "retriever_source", "local"),
        "retriever_index": str(getattr(args, "retriever_index", "") or ""),
        "seed": int(getattr(args, "seed", 0)),
        "max_steps": int(getattr(args, "max_steps", 0)),
        "beam": int(getattr(args, "beam", 0)),
        "k": int(getattr(args, "k", 0)),
        "inference_profile": str(getattr(args, "inference_profile", "default") or "default"),
        "generation_budget_per_state": int(
            getattr(args, "effective_gen_k", getattr(args, "k", 1)) or 1
        ),
        "execution_budget_per_state": int(getattr(args, "effective_exec_k", 0) or 0),
        "verification_level": str(getattr(args, "effective_verify_level", "light") or "light"),
        "timeout_s": float(getattr(args, "timeout", 0.0)),
        "typecheck_timeout_s": float(_resolve_typecheck_timeout(args)),
        "repair_attempts": int(getattr(args, "repair", 0)),
    }


def _sha256_file(path: Path) -> str:
    try:
        data = path.read_bytes()
    except Exception:
        return ""
    return hashlib.sha256(data).hexdigest()


def _mathlib_commit_from_checkout(project_path: Path | None) -> str:
    if project_path is None:
        return ""
    repo = project_path / ".lake" / "packages" / "mathlib"
    if not repo.exists():
        return ""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _mathlib_rev_from_manifest(project_path: Path | None) -> str:
    if project_path is None:
        return ""
    manifest = project_path / "lake-manifest.json"
    if not manifest.exists():
        return ""
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return ""
    packages = payload.get("packages", [])
    if not isinstance(packages, list):
        return ""
    for package in packages:
        if not isinstance(package, dict):
            continue
        if str(package.get("name", "")).strip() != "mathlib":
            continue
        rev = str(package.get("rev", "")).strip()
        if rev:
            return rev
        version = str(package.get("version", "")).strip()
        if version:
            return version
    return ""


def _execute_replay(args: argparse.Namespace, trace_rows: list[dict], metadata: dict) -> None:
    if not trace_rows:
        print("Trace is empty.")
        if args.strict:
            sys.exit(1)
        return

    replay_file = args.file or _path_from_meta(metadata.get("file"))
    theorem = (args.theorem or str(metadata.get("theorem", "")).strip()).strip()
    lean_backend = args.lean or str(metadata.get("lean_backend", "")).strip() or "mock"
    if replay_file is None or not theorem:
        print("Replay execute requires theorem and file (from metadata or --file/--theorem).")
        sys.exit(1)
    if not replay_file.exists():
        print(f"Replay file not found: {replay_file}")
        sys.exit(1)
    if lean_backend not in {"mock", "dojo"}:
        print(f"Replay execute backend `{lean_backend}` is not supported (use mock or dojo).")
        sys.exit(1)

    replay_project = args.lean_project or _path_from_meta(metadata.get("lean_project"))
    lean_imports = list(args.lean_import) if args.lean_import else list(metadata.get("lean_imports", []) or [])
    runner_args = argparse.Namespace(
        lean=lean_backend,
        lean_project=replay_project,
        lean_import=lean_imports,
    )

    if not _check_replay_environment(
        metadata=metadata,
        replay_file=replay_file,
        replay_project=replay_project,
        strict=bool(args.strict),
        align_toolchain=bool(args.align_toolchain),
    ):
        sys.exit(1)

    runner = _make_runner(runner_args)
    states_by_logged_key: dict[str, object] = {}
    states_by_hash: dict[str, list[object]] = {}
    divergences = 0
    executed = 0

    try:
        initial = runner.start(replay_file, theorem)
        first_key = str(trace_rows[0].get("state_key", "")).strip()
        if first_key:
            states_by_logged_key[first_key] = initial
        _remember_state(states_by_hash, initial)

        for idx, payload in enumerate(trace_rows, start=1):
            state = _resolve_replay_state(payload, states_by_logged_key, states_by_hash, initial if idx == 1 else None)
            if state is None:
                print(f"[replay] step {idx}: missing state mapping.")
                divergences += 1
                if args.strict:
                    break
                continue
            tactic = str(payload.get("tactic", "")).strip()
            result = runner.apply(state, tactic, float(args.timeout))
            executed += 1

            mismatch = _compare_replay_step(payload, result)
            if mismatch:
                divergences += 1
                print(f"[replay] step {idx}: {mismatch}")
                if args.strict:
                    break

            if result.ok and result.new_state is not None:
                new_key = str(payload.get("new_state_key", "")).strip()
                if new_key:
                    states_by_logged_key[new_key] = result.new_state
                _remember_state(states_by_hash, result.new_state)
    finally:
        runner.close()

    print(f"Replayed steps: {executed}/{len(trace_rows)}")
    print(f"Divergences: {divergences}")
    if divergences == 0:
        print("Deterministic replay: OK")
        return
    if args.strict:
        sys.exit(1)


def _path_from_meta(value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    return Path(text).expanduser()


def _check_replay_environment(
    *,
    metadata: dict,
    replay_file: Path,
    replay_project: Path | None,
    strict: bool,
    align_toolchain: bool,
) -> bool:
    expected_toolchain = str(metadata.get("lean_toolchain", "")).strip()
    expected_mathlib_commit = str(metadata.get("mathlib_commit", "")).strip()
    if not expected_toolchain and not expected_mathlib_commit:
        return True
    project = replay_project or _find_lean_project_for_file(replay_file)
    if project is None:
        print("[replay] warning: no Lean project found for environment checks.")
        return not strict
    project = project.expanduser().resolve()

    if align_toolchain and expected_toolchain:
        current_toolchain = _read_toolchain_file(project / "lean-toolchain")
        if current_toolchain != expected_toolchain:
            env = _extend_path(os.environ.copy(), Path("~/.elan/bin").expanduser())
            if _which("elan", env):
                print(f"[replay] aligning toolchain to {expected_toolchain}...")
                _align_mathlib_to_toolchain(project, env, expected_toolchain)

    mismatches: list[str] = []
    current_toolchain = _read_toolchain_file(project / "lean-toolchain")
    current_mathlib_commit = _mathlib_commit_from_checkout(project)
    if expected_toolchain:
        if not current_toolchain:
            mismatches.append(f"toolchain expected {expected_toolchain}, found (missing)")
        elif expected_toolchain != current_toolchain:
            mismatches.append(f"toolchain expected {expected_toolchain}, found {current_toolchain}")
    if expected_mathlib_commit:
        if not current_mathlib_commit:
            mismatches.append(
                f"mathlib commit expected {expected_mathlib_commit[:12]}, found (missing)"
            )
        elif expected_mathlib_commit != current_mathlib_commit:
            mismatches.append(
                f"mathlib commit expected {expected_mathlib_commit[:12]}, found {current_mathlib_commit[:12]}"
            )
    if not mismatches:
        return True
    print("[replay] environment mismatch:")
    for item in mismatches:
        print(f"- {item}")
    return not strict


def _remember_state(states_by_hash: dict[str, list[object]], state) -> None:
    digest = state_hash(state.pretty)
    bucket = states_by_hash.setdefault(digest, [])
    bucket.append(state)


def _resolve_replay_state(
    payload: dict,
    states_by_logged_key: dict[str, object],
    states_by_hash: dict[str, list[object]],
    fallback_state,
):
    state_key = str(payload.get("state_key", "")).strip()
    if state_key and state_key in states_by_logged_key:
        return states_by_logged_key[state_key]
    expected_hash = _expected_state_hash(payload, key="state_hash", pretty_key="state_pretty")
    if expected_hash and expected_hash in states_by_hash and states_by_hash[expected_hash]:
        state = states_by_hash[expected_hash][0]
        if state_key:
            states_by_logged_key[state_key] = state
        return state
    if fallback_state is not None:
        if state_key:
            states_by_logged_key[state_key] = fallback_state
        return fallback_state
    return None


def _compare_replay_step(payload: dict, result) -> str:
    expected_ok = bool(payload.get("ok", False))
    expected_solved = bool(payload.get("solved", False))
    if result.ok != expected_ok:
        return f"ok mismatch (expected {expected_ok}, got {result.ok})"
    if result.is_solved != expected_solved:
        return f"solved mismatch (expected {expected_solved}, got {result.is_solved})"
    expected_error_kind = str(payload.get("error_kind") or _bench_error_kind(payload.get("error")) or "")
    actual_error_kind = _bench_error_kind(result.error) or ""
    if not expected_ok and expected_error_kind and expected_error_kind != actual_error_kind:
        return f"error kind mismatch (expected {expected_error_kind}, got {actual_error_kind or 'none'})"
    expected_new_hash = _expected_state_hash(payload, key="new_state_hash")
    if expected_new_hash:
        if result.new_state is None:
            return "expected new state but tactic produced none"
        actual_new_hash = state_hash(result.new_state.pretty)
        if actual_new_hash != expected_new_hash:
            return f"new state hash mismatch (expected {expected_new_hash[:12]}, got {actual_new_hash[:12]})"
    return ""


def _expected_state_hash(payload: dict, *, key: str, pretty_key: str | None = None) -> str:
    raw = payload.get(key)
    if raw is None:
        value = ""
    elif isinstance(raw, str):
        value = raw.strip()
    else:
        value = str(raw).strip()
    if value.lower() == "none":
        value = ""
    if value:
        return value
    if pretty_key:
        pretty = str(payload.get(pretty_key, ""))
        if pretty:
            return state_hash(pretty)
    return ""


def run_checkpoint(args: argparse.Namespace) -> None:
    cfg = load_config()
    profile = _resolve_proof_profile(args, cfg)
    _apply_proof_profile_to_args(args, profile)

    file_path = Path(args.file).expanduser()
    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"Failed to read Lean file: {exc}")
        sys.exit(1)

    allow_axioms = _resolve_allow_axioms(args, cfg)
    timeout_s = _resolve_typecheck_timeout(args, cfg)
    lean_project = args.lean_project or _find_lean_project_for_file(file_path)

    from .lean.cli_check import lean_cli_check

    typecheck_error: str | None
    try:
        typecheck_error = lean_cli_check(file_path, project_path=lean_project, timeout_s=timeout_s)
    except Exception as exc:
        typecheck_error = str(exc) or repr(exc)

    file_stats = _checkpoint_file_stats(text, theorem=str(args.theorem or "").strip())
    axiom_names = _scan_axiom_constant_names(_strip_comments(text))
    trace_stats = _checkpoint_trace_stats(args.trace)

    blockers: list[str] = []
    theorem = str(args.theorem or "").strip()
    if theorem and not file_stats["theorem_found"]:
        blockers.append(f"target declaration `{theorem}` was not found")
    if typecheck_error:
        first = _primary_typecheck_error_line(typecheck_error)
        blockers.append(first)
    if (not allow_axioms) and axiom_names:
        blockers.append(f"axioms/constants present while disallowed ({len(axiom_names)})")
    if args.strict and int(file_stats["placeholders"]) > 0:
        blockers.append(f"placeholders remain ({file_stats['placeholders']})")

    report = {
        "profile": profile,
        "file": str(file_path),
        "lean_project": str(lean_project) if lean_project else "",
        "typecheck_timeout_s": timeout_s,
        "allow_axioms": bool(allow_axioms),
        "typecheck_ok": typecheck_error is None,
        "typecheck_error": typecheck_error or "",
        "declarations_total": int(file_stats["decl_total"]),
        "placeholders": int(file_stats["placeholders"]),
        "axiom_constant_count": len(axiom_names),
        "axiom_constant_names": axiom_names[:50],
        "theorem": theorem,
        "theorem_found": bool(file_stats["theorem_found"]),
        "theorem_has_placeholder": bool(file_stats["theorem_has_placeholder"]),
        "trace": trace_stats,
        "blockers": blockers,
    }

    print(f"Checkpoint file: {file_path}")
    print(f"Policy profile: {profile}")
    print(f"Typecheck: {'ok' if typecheck_error is None else 'failed'}")
    if typecheck_error:
        preview = "\n".join((typecheck_error or "").splitlines()[:3]).strip()
        if preview:
            print(f"Typecheck error: {preview}")
    print(
        "Declarations: "
        f"{int(file_stats['decl_total'])}, placeholders: {int(file_stats['placeholders'])}, "
        f"axioms/constants: {len(axiom_names)}"
    )
    if theorem:
        print(
            f"Theorem `{theorem}`: "
            f"{'found' if file_stats['theorem_found'] else 'missing'}, "
            f"placeholder={'yes' if file_stats['theorem_has_placeholder'] else 'no'}"
        )
    if trace_stats.get("trace_found", False):
        print(
            "Trace: "
            f"{trace_stats.get('steps', 0)} steps, "
            f"ok={trace_stats.get('ok_steps', 0)}, "
            f"fail={trace_stats.get('fail_steps', 0)}"
        )
    if blockers:
        print("Blockers:")
        for item in blockers[:10]:
            print(f"- {item}")
        if len(blockers) > 10:
            print(f"- ... ({len(blockers) - 10} more)")
    else:
        print("Blockers: none")

    if args.out_json:
        out_path = Path(args.out_json).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(f"Report JSON: {out_path}")

    if args.strict and blockers:
        sys.exit(1)


def _primary_typecheck_error_line(error: str) -> str:
    lines = [line.strip() for line in str(error or "").splitlines() if line.strip()]
    for line in lines:
        if not line.lower().startswith("warning:"):
            return line
    if lines:
        return lines[0]
    return "Lean typecheck error"


def _checkpoint_file_stats(text: str, theorem: str = "") -> dict[str, object]:
    cleaned = _strip_comments(text)
    decl_pattern = re.compile(
        r"^\s*(?:theorem|lemma|example|proposition|corollary|def|abbrev|structure|class|inductive)\s+([A-Za-z_][A-Za-z0-9_']*)",
        re.M,
    )
    decl_total = len(list(decl_pattern.finditer(text)))
    placeholders = len(re.findall(r"\b(sorry|admit)\b", cleaned))
    theorem_found = False
    theorem_has_placeholder = False
    if theorem:
        theorem_found = _file_has_decl(text, theorem) or bool(
            re.search(
                rf"^\s*(?:theorem|lemma|example|proposition|corollary|def|abbrev|structure|class|inductive)\s+{_name_token_regex(theorem)}\b",
                text,
                re.M,
            )
        )
        block = _decl_block(text, theorem)
        if not block:
            block = _any_decl_block(text, theorem)
        theorem_has_placeholder = bool(re.search(r"\b(sorry|admit)\b", _strip_comments(block)))
    return {
        "decl_total": decl_total,
        "placeholders": placeholders,
        "theorem_found": theorem_found,
        "theorem_has_placeholder": theorem_has_placeholder,
    }


def _any_decl_block(text: str, name: str) -> str:
    pattern = re.compile(
        rf"^\s*(?:theorem|lemma|example|proposition|corollary|def|abbrev|structure|class|inductive)\s+{_name_token_regex(name)}\b",
        re.M,
    )
    match = pattern.search(text)
    if not match:
        return ""
    next_decl = re.compile(
        r"^\s*(?:theorem|lemma|example|proposition|corollary|def|abbrev|structure|class|inductive)\s+[A-Za-z_][A-Za-z0-9_']*\b",
        re.M,
    ).search(text, match.end())
    end = next_decl.start() if next_decl else len(text)
    return text[match.start() : end]


def _scan_axiom_constant_names(cleaned_text: str) -> list[str]:
    names: list[str] = []
    pattern = re.compile(r"^\s*(?:private\s+)?(?:axiom|constant)\s+([A-Za-z_][A-Za-z0-9_']*)", re.M)
    for match in pattern.finditer(cleaned_text):
        names.append(match.group(1))
    return names


def _checkpoint_trace_stats(trace_path: Path) -> dict[str, object]:
    path = Path(trace_path).expanduser()
    if not path.exists():
        return {"trace_found": False, "path": str(path)}
    steps = _read_trace_steps(path, max_lines=1000)
    ok_steps = 0
    fail_steps = 0
    for step in steps:
        if bool(step.get("ok", False)):
            ok_steps += 1
        else:
            fail_steps += 1
    return {
        "trace_found": True,
        "path": str(path),
        "steps": len(steps),
        "ok_steps": ok_steps,
        "fail_steps": fail_steps,
    }


def run_review(args: argparse.Namespace) -> None:
    trace_path = Path(args.trace).expanduser()
    if not trace_path.exists():
        print(f"Trace not found: {trace_path}")
        sys.exit(1)

    steps = _read_trace_steps(trace_path, max_lines=max(20, int(args.max_lines)))
    if not steps:
        print(f"Trace has no readable steps: {trace_path}")
        sys.exit(1)

    ok_steps = 0
    fail_steps = 0
    solved_steps = 0
    error_kinds: Counter[str] = Counter()
    fingerprints: Counter[str] = Counter()
    tactic_heads: Counter[str] = Counter()
    unique_states: set[str] = set()
    unique_new_states: set[str] = set()
    for row in steps:
        if bool(row.get("ok", False)):
            ok_steps += 1
        else:
            fail_steps += 1
            err = str(row.get("error") or "")
            kind = _bench_error_kind(err) or "other"
            error_kinds[kind] += 1
            if err.strip():
                fingerprints[_error_fingerprint(err)] += 1
        if bool(row.get("solved", False)):
            solved_steps += 1
        tactic = str(row.get("tactic") or "").strip()
        if tactic:
            tactic_heads[_tactic_head(tactic)] += 1
        st = str(row.get("state_hash") or "").strip()
        if st:
            unique_states.add(st)
        nst = str(row.get("new_state_hash") or "").strip()
        if nst:
            unique_new_states.add(nst)

    metadata = _load_trace_metadata(trace_path, None)
    review_file = args.file or _path_from_meta(metadata.get("file"))
    file_stats: dict[str, object] | None = None
    if isinstance(review_file, Path) and review_file.exists():
        try:
            file_text = review_file.read_text(encoding="utf-8")
            file_stats = _checkpoint_file_stats(file_text, theorem=str(args.theorem or "").strip())
            file_stats["path"] = str(review_file)
            file_stats["axiom_constant_count"] = len(_scan_axiom_constant_names(_strip_comments(file_text)))
        except Exception:
            file_stats = None

    top_errors = error_kinds.most_common(5)
    repeated_errors = [(k, v) for k, v in fingerprints.most_common(5) if v >= 2]
    repeated_heads = [(k, v) for k, v in tactic_heads.most_common(5) if v >= 3]
    suggestions = _review_suggestions(
        fail_steps=fail_steps,
        top_errors=top_errors,
        repeated_errors=repeated_errors,
        repeated_heads=repeated_heads,
        file_stats=file_stats,
    )

    report = {
        "trace": str(trace_path),
        "steps": len(steps),
        "ok_steps": ok_steps,
        "fail_steps": fail_steps,
        "solved_steps": solved_steps,
        "unique_state_hashes": len(unique_states),
        "unique_new_state_hashes": len(unique_new_states),
        "top_error_kinds": top_errors,
        "repeated_error_fingerprints": repeated_errors,
        "repeated_tactic_heads": repeated_heads,
        "file_stats": file_stats or {},
        "suggestions": suggestions,
    }

    print(f"Review trace: {trace_path}")
    print(f"Steps: {len(steps)} (ok={ok_steps}, fail={fail_steps}, solved_markers={solved_steps})")
    print(f"State coverage: unique={len(unique_states)}, progressed={len(unique_new_states)}")
    if top_errors:
        print("Top failure kinds:")
        for kind, count in top_errors:
            print(f"- {kind}: {count}")
    if repeated_errors:
        print("Repeated error fingerprints:")
        for fp, count in repeated_errors:
            print(f"- {fp}: {count}")
    if repeated_heads:
        print("Repeated tactic heads:")
        for head, count in repeated_heads:
            print(f"- {head}: {count}")
    if file_stats:
        print(
            "File stats: "
            f"declarations={file_stats.get('decl_total', 0)}, "
            f"placeholders={file_stats.get('placeholders', 0)}, "
            f"axioms/constants={file_stats.get('axiom_constant_count', 0)}"
        )
    print("Next actions:")
    for item in suggestions:
        print(f"- {item}")

    if args.out_json:
        out_path = Path(args.out_json).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(f"Report JSON: {out_path}")


def _tactic_head(tactic: str) -> str:
    stripped = tactic.strip()
    if not stripped:
        return ""
    return stripped.split(maxsplit=1)[0][:40]


def _review_suggestions(
    *,
    fail_steps: int,
    top_errors: list[tuple[str, int]],
    repeated_errors: list[tuple[str, int]],
    repeated_heads: list[tuple[str, int]],
    file_stats: dict[str, object] | None,
) -> list[str]:
    suggestions: list[str] = []
    top_kind = top_errors[0][0] if top_errors else ""
    if fail_steps == 0:
        suggestions.append("No failing steps in trace; keep current settings and expand benchmark coverage.")
    if top_kind == "timeout":
        suggestions.append("Increase tactic timeout (--timeout) or reduce search branching/step budget.")
    if top_kind in {"type_mismatch", "unsolved_goals"}:
        suggestions.append("Switch to script/LLM mode for multi-step edits and add tighter theorem-local guidance.")
    if top_kind == "unknown_identifier":
        suggestions.append("Enable retrieval/indexing and verify imports in the Lean file before next run.")
    if repeated_errors:
        suggestions.append("Run with strict profile to force alternative edits when the same error repeats.")
    if repeated_heads:
        suggestions.append("Disable autop temporarily to avoid repeating the same fallback tactic heads.")
    if file_stats:
        placeholders = int(file_stats.get("placeholders", 0) or 0)
        if placeholders > 0:
            suggestions.append("Focus next run on filling existing `sorry/admit` placeholders before broad refactors.")
        ax_count = int(file_stats.get("axiom_constant_count", 0) or 0)
        if ax_count > 0:
            suggestions.append("If semantic fidelity matters, run strict profile (`--proof-profile strict`) to disallow axioms.")
    if not suggestions:
        suggestions.append("Run `ulam checkpoint --strict` to gate the next iteration on deterministic blockers.")
    return suggestions


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
        print("1. Gemini CLI OAuth login (automatic; native fallback)")
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
    if not argv:
        return False
    return argv[0] in {"-lean", "--lean", "--lean-setup"}


def _strip_lean_setup_flags(argv: list[str]) -> list[str]:
    if not argv:
        return []
    if argv[0] in {"-lean", "--lean", "--lean-setup"}:
        return argv[1:]
    return argv


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
    base = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--timeout",
        str(args.pip_timeout),
        "--retries",
        str(args.pip_retries),
    ]
    attempts = _pip_install_attempt_flags()
    last_output = ""
    for idx, flags in enumerate(attempts):
        cmd = [*base, *flags, *packages]
        proc = subprocess.run(
            cmd,
            check=False,
            env=env,
            text=True,
            capture_output=True,
        )
        if proc.returncode == 0:
            if idx > 0:
                print(f"[pip] recovered by retrying with flags: {' '.join(flags)}")
            return True
        output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        last_output = output
        if _is_externally_managed_pip_error(output):
            continue
        print(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
        if output:
            print(output)
        return False

    final_cmd = [*base, *attempts[-1], *packages]
    print(f"Command failed: {' '.join(final_cmd)}")
    if last_output:
        print(last_output)
    return False


def _pip_install_attempt_flags() -> list[list[str]]:
    prefers_break = os.environ.get("ULAM_BREAK_SYSTEM_PACKAGES", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    in_venv = (
        getattr(sys, "base_prefix", sys.prefix) != sys.prefix
        or bool(os.environ.get("VIRTUAL_ENV"))
    )
    if in_venv:
        if prefers_break:
            return [[], ["--break-system-packages"]]
        return [[], ["--break-system-packages"]]
    default_order = [
        [],
        ["--user"],
        ["--break-system-packages"],
        ["--break-system-packages", "--user"],
    ]
    if not prefers_break:
        return default_order
    return [
        ["--break-system-packages", "--user"],
        ["--break-system-packages"],
        ["--user"],
        [],
    ]


def _is_externally_managed_pip_error(output: str) -> bool:
    text = output.lower()
    return "externally-managed-environment" in text or "externally managed" in text


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


def _proof_progress_callback(file_path: Path, theorem: str, verbose: bool = False):
    last_written: list[str] = []

    def _callback(proof: list[str]) -> None:
        nonlocal last_written
        if not proof:
            return
        if proof == last_written:
            return
        if _write_draft_proof_to_file(file_path, theorem, proof):
            last_written = list(proof)
            if verbose:
                print(f"[progress] updated draft proof for {theorem} ({len(proof)} steps)")

    return _callback


def _render_draft_block(indent: str, proof: list[str]) -> str:
    lines = [f"{indent}{ULAMAI_DRAFT_BEGIN}"]
    lines.extend(f"{indent}{line}" for line in proof)
    lines.append(f"{indent}sorry")
    lines.append(f"{indent}{ULAMAI_DRAFT_END}")
    return "\n".join(lines)


def _write_draft_proof_to_file(file_path: Path, theorem: str, proof: list[str]) -> bool:
    if not proof:
        return False
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception:
        return False
    span = _decl_span(text, theorem)
    if span is None:
        return False
    start, end = span
    decl = text[start:end]
    if ULAMAI_DRAFT_BEGIN in decl and ULAMAI_DRAFT_END in decl:
        begin_idx = decl.find(ULAMAI_DRAFT_BEGIN)
        end_idx = decl.find(ULAMAI_DRAFT_END, begin_idx)
        if end_idx < 0:
            return False
        block_start = decl.rfind("\n", 0, begin_idx) + 1
        block_end_line = decl.find("\n", end_idx)
        block_end = len(decl) if block_end_line < 0 else (block_end_line + 1)
        indent = decl[block_start:begin_idx]
        replacement = _render_draft_block(indent, proof)
        new_decl = decl[:block_start] + replacement + ("\n" if block_end_line >= 0 else "") + decl[block_end:]
    else:
        sorry_match = re.search(r"\bsorry\b", decl)
        if sorry_match is None:
            return False
        sorry_start = sorry_match.start()
        sorry_end = sorry_match.end()
        line_start = decl.rfind("\n", 0, sorry_start) + 1
        indent = decl[line_start:sorry_start]
        replacement = _render_draft_block(indent, proof)
        new_decl = decl[:line_start] + replacement + decl[sorry_end:]
    new_text = text[:start] + new_decl + text[end:]
    if new_text == text:
        return False
    try:
        file_path.write_text(new_text, encoding="utf-8")
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
    span = _decl_span(text, theorem)
    if span is None:
        return False
    start, end = span
    decl = text[start:end]
    if ULAMAI_DRAFT_BEGIN in decl and ULAMAI_DRAFT_END in decl:
        begin_idx = decl.find(ULAMAI_DRAFT_BEGIN)
        end_idx = decl.find(ULAMAI_DRAFT_END, begin_idx)
        if end_idx < 0:
            return False
        block_start = decl.rfind("\n", 0, begin_idx) + 1
        block_end_line = decl.find("\n", end_idx)
        block_end = len(decl) if block_end_line < 0 else (block_end_line + 1)
        indent = decl[block_start:begin_idx]
        proof_block = "\n".join(f"{indent}{line}" for line in proof)
        new_decl = decl[:block_start] + proof_block + ("\n" if block_end_line >= 0 else "") + decl[block_end:]
        new_text = text[:start] + new_decl + text[end:]
    else:
        sorry_match = re.search(r"\bsorry\b", decl)
        if sorry_match is None:
            return False
        sorry_start = sorry_match.start()
        sorry_end = sorry_match.end()
        line_start = decl.rfind("\n", 0, sorry_start) + 1
        indent = decl[line_start:sorry_start]
        prefix = decl[:line_start]
        has_by = re.search(r":=\s*by\s*$", prefix) is not None or re.search(r"\bby\s*$", prefix) is not None
        if has_by:
            proof_block = "\n".join(f"{indent}{line}" for line in proof)
        else:
            proof_block = "by\n" + "\n".join(f"{indent}{line}" for line in proof)
        new_decl = decl[:line_start] + proof_block + decl[sorry_end:]
        new_text = text[:start] + new_decl + text[end:]
    if new_text == text:
        return False
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
    decl_match = _decl_head_regex(theorem).search(text)
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


def _looks_like_lean_project(path: Path) -> bool:
    candidate = path.expanduser()
    if not candidate.exists() or not candidate.is_dir():
        return False
    return (
        (candidate / "lakefile.lean").exists()
        or (candidate / "lakefile.toml").exists()
        or (candidate / "lean-toolchain").exists()
    )


def _find_lean_project_for_file(file_path: Path) -> Path | None:
    root = file_path if file_path.is_dir() else file_path.parent
    for parent in [root, *root.parents]:
        if (
            (parent / "lakefile.lean").exists()
            or (parent / "lakefile.toml").exists()
            or (parent / "lean-toolchain").exists()
        ):
            return parent
    return None


def _resolve_formalize_lean_project(
    *,
    args: argparse.Namespace,
    config: dict,
    tex_path: Path,
    output_path: Path,
) -> Path | None:
    explicit = getattr(args, "lean_project", None)
    if explicit is not None:
        candidate = Path(explicit).expanduser()
        if _looks_like_lean_project(candidate):
            return candidate
        print(f"[formalize] ignoring invalid --lean-project path: {candidate}")

    configured_raw = str(config.get("lean", {}).get("project", "") or "").strip()
    if configured_raw:
        configured = Path(configured_raw).expanduser()
        if _looks_like_lean_project(configured):
            return configured

    for probe in (output_path, tex_path, Path.cwd()):
        detected = _find_lean_project_for_file(probe)
        if detected is not None:
            return detected
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
    profile = _resolve_proof_profile(args, config)
    _apply_proof_profile_to_args(args, profile)
    if getattr(args, "verbose", False):
        print(f"[policy] profile={profile}")
    tex_path = args.tex
    if not tex_path.exists():
        print(f"Tex file not found: {tex_path}")
        sys.exit(1)
    output_path = args.out if args.out else tex_path.with_suffix(".lean")
    context_files = [Path(p) for p in args.context]
    lean_project = _resolve_formalize_lean_project(
        args=args,
        config=config,
        tex_path=tex_path,
        output_path=output_path,
    )
    if lean_project is not None:
        configured = str(config.get("lean", {}).get("project", "") or "").strip()
        if configured != str(lean_project):
            config.setdefault("lean", {})["project"] = str(lean_project)
            save_config(config)
        print(f"Detected Lean project: {lean_project}")

    proof_backend = args.proof_backend
    if proof_backend == "dojo":
        proof_backend = "tactic"
    lean_backend = args.lean_backend
    if proof_backend == "llm" and lean_backend == "dojo":
        lean_backend = "cli"
    if lean_project is None and proof_backend in {"tactic", "lemma"}:
        print("[formalize] no Lean project detected; tactic/lemma proof search will be skipped.")
    max_rounds = _resolve_formalize_max_rounds(args, config)
    max_repairs = _resolve_formalize_max_repairs(args, max_rounds, config)
    max_proof_rounds = _resolve_formalize_max_proof_rounds(args, config)
    proof_repair = _resolve_formalize_proof_repair(args, config)
    dojo_timeout_s = float(config.get("lean", {}).get("dojo_timeout_s", 180))
    typecheck_timeout_s = _resolve_formalize_typecheck_timeout(args, config)
    allow_axioms = _resolve_allow_axioms(args, config)
    llm_check = _resolve_formalize_llm_check(args, config)
    llm_check_timing = _resolve_formalize_llm_check_timing(args, config)
    llm_check_repairs = _resolve_formalize_llm_check_repairs(args, config)
    cfg = FormalizationConfig(
        tex_path=tex_path,
        output_path=output_path,
        context_files=context_files,
        max_rounds=max_rounds,
        max_repairs=max_repairs,
        max_equivalence_repairs=args.max_equivalence_repairs,
        max_proof_rounds=max_proof_rounds,
        proof_max_steps=args.proof_max_steps,
        proof_beam=args.proof_beam,
        proof_k=args.proof_k,
        proof_timeout_s=args.proof_timeout,
        proof_repair=proof_repair,
        typecheck_timeout_s=typecheck_timeout_s,
        dojo_timeout_s=dojo_timeout_s,
        lemma_max=60,
        lemma_depth=60,
        allow_axioms=allow_axioms,
        lean_project=lean_project,
        lean_imports=args.lean_import,
        verbose=bool(args.verbose),
        proof_backend=proof_backend,
        lean_backend=lean_backend,
        resume_path=None,
        artifact_dir=args.artifacts_dir,
        equivalence_checks=not args.no_equivalence,
        llm_check=llm_check,
        llm_check_timing=llm_check_timing,
        llm_check_repairs=llm_check_repairs,
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


def _bench_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_bench_suite_registry() -> list[dict[str, object]]:
    return [
        {
            "name": "regression",
            "path": "bench/regression.jsonl",
            "description": "Legacy smoke regression suite.",
            "target_cases": 1,
            "kind": "fixed",
        },
        {
            "name": "internal_regression",
            "path": "bench/suites/internal_regression.jsonl",
            "description": "Internal smoke regression suite.",
            "target_cases": 1,
            "kind": "fixed",
        },
        {
            "name": "minif2f_dev",
            "path": "bench/suites/minif2f_dev.jsonl",
            "description": "Development placeholder miniF2F suite.",
            "target_cases": 1,
            "kind": "dev",
        },
        {
            "name": "putnambench_sample",
            "path": "bench/suites/putnambench_sample.jsonl",
            "description": "Optional PutnamBench placeholder sample.",
            "target_cases": 1,
            "kind": "sample",
        },
        {
            "name": "regression100",
            "path": "bench/suites/regression100.jsonl",
            "description": "Fixed-size 100-case regression suite.",
            "target_cases": 100,
            "kind": "fixed",
            "generator_hint": (
                "python3 -m ulam bench-make-regression100 --source "
                "bench/suites/minif2f_valid.jsonl --out bench/suites/regression100.jsonl "
                "--size 100 --seed 0"
            ),
        },
    ]


def _normalize_suite_alias(name: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(name or "").strip().lower())
    return text.strip("_")


def _load_bench_suite_registry() -> dict[str, dict[str, object]]:
    repo_root = _bench_repo_root()
    entries: dict[str, dict[str, object]] = {}
    for item in _default_bench_suite_registry():
        name = _normalize_suite_alias(item.get("name"))
        if not name:
            continue
        entries[name] = dict(item)

    registry_path = repo_root / "bench" / "suites" / "registry.json"
    payload = _load_json_object(registry_path)
    if not isinstance(payload, dict):
        return entries
    suite_rows = payload.get("suites")
    if not isinstance(suite_rows, list):
        return entries
    for row in suite_rows:
        if not isinstance(row, dict):
            continue
        name = _normalize_suite_alias(row.get("name"))
        path_text = str(row.get("path", "")).strip()
        if not name or not path_text:
            continue
        merged = dict(entries.get(name, {}))
        merged.update(row)
        merged["name"] = name
        entries[name] = merged
    return entries


def _resolve_suite_entry_path(path_text: str, repo_root: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _resolve_bench_suite_input(suite_arg: Path) -> tuple[Path, dict[str, object] | None]:
    raw_text = str(suite_arg).strip()
    repo_root = _bench_repo_root()
    direct = Path(raw_text).expanduser()
    if direct.exists():
        return direct.resolve(), None
    repo_relative = (repo_root / direct).resolve()
    if repo_relative.exists():
        return repo_relative, None

    registry = _load_bench_suite_registry()
    alias = _normalize_suite_alias(raw_text)
    entry = registry.get(alias)
    if entry is None:
        known = ", ".join(sorted(registry.keys()))
        raise FileNotFoundError(
            f"Suite not found: {suite_arg}. Use a JSONL path or one of: {known}"
        )
    path_text = str(entry.get("path", "")).strip()
    if not path_text:
        raise FileNotFoundError(f"Suite alias `{alias}` has no path in registry.")
    resolved = _resolve_suite_entry_path(path_text, repo_root)
    if resolved.exists():
        return resolved, dict(entry)
    hint = str(entry.get("generator_hint", "")).strip()
    if hint:
        raise FileNotFoundError(
            f"Suite alias `{alias}` maps to missing file: {resolved}\nGenerate it with:\n{hint}"
        )
    raise FileNotFoundError(f"Suite alias `{alias}` maps to missing file: {resolved}")


def _count_suite_cases_quick(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    try:
        with path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                if raw.strip():
                    count += 1
    except Exception:
        return 0
    return count


def run_bench_list_suites(args: argparse.Namespace) -> None:
    repo_root = _bench_repo_root()
    registry = _load_bench_suite_registry()
    if not registry:
        print("No suite aliases found.")
        return

    rows: list[dict[str, object]] = []
    for name in sorted(registry):
        entry = registry[name]
        path_text = str(entry.get("path", "")).strip()
        resolved = _resolve_suite_entry_path(path_text, repo_root) if path_text else Path("")
        exists = bool(path_text) and resolved.exists()
        rows.append(
            {
                "name": name,
                "path": str(resolved) if path_text else "",
                "description": str(entry.get("description", "")).strip(),
                "target_cases": int(entry.get("target_cases", 0) or 0),
                "cases": _count_suite_cases_quick(resolved) if exists else 0,
                "exists": exists,
                "generator_hint": str(entry.get("generator_hint", "")).strip(),
            }
        )

    print("Known benchmark suites:")
    for row in rows:
        status = "ok" if bool(row["exists"]) else "missing"
        target = int(row.get("target_cases", 0) or 0)
        target_text = f", target={target}" if target > 0 else ""
        print(
            f"- {row['name']}: {row['path']} [{status}, cases={row['cases']}{target_text}]"
        )
        desc = str(row.get("description", "")).strip()
        if desc:
            print(f"  {desc}")
        hint = str(row.get("generator_hint", "")).strip()
        if hint and not bool(row["exists"]):
            print(f"  generate: {hint}")

    if args.out_json:
        out_json = Path(args.out_json).expanduser().resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": 1,
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "suites": rows,
        }
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(f"Wrote suite registry snapshot: {out_json}")


def run_bench(args: argparse.Namespace) -> None:
    inf_profile, gen_k, exec_k, verify_level = _apply_inference_runtime_to_args(args)
    try:
        suite_path, suite_entry = _resolve_bench_suite_input(args.suite)
    except Exception as exc:
        print(str(exc))
        sys.exit(1)
    if suite_entry is not None:
        print(f"Resolved suite alias `{suite_entry.get('name', '')}` -> {suite_path}")

    try:
        cases = _load_bench_cases(suite_path)
    except Exception as exc:
        print(f"Failed to load suite: {exc}")
        sys.exit(1)
    if not cases:
        print(f"Suite has no cases: {suite_path}")
        return
    if bool(getattr(args, "verbose", False)):
        exec_text = "all" if exec_k <= 0 else str(exec_k)
        print(
            f"[policy] inference_profile={inf_profile} "
            f"(gen_k={gen_k}, exec_k={exec_text}, verify={verify_level})"
        )

    llm = _make_llm(args)
    started_at_epoch = time.time()
    started_perf = time.perf_counter()
    solved = 0
    results: list[dict[str, object]] = []
    error_kinds: Counter[str] = Counter()
    step_counts: list[int] = []
    durations_s: list[float] = []
    for idx, case in enumerate(cases, start=1):
        file_path = case["file_path"]
        theorem = case["theorem"]
        premises = case["premises"]
        semantic_report = case.get("semantic_report")
        artifact_dir = case.get("artifact_dir")
        dataset = str(case.get("dataset", "")).strip()
        split = str(case.get("split", "")).strip()
        tags_raw = case.get("tags", [])
        tags: list[str] = []
        if isinstance(tags_raw, list):
            tags = [str(item).strip() for item in tags_raw if str(item).strip()]
        case_args = argparse.Namespace(**vars(args))
        _apply_inference_runtime_to_args(case_args)
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
            inference_profile=str(getattr(case_args, "inference_profile", "default") or "default"),
            generation_budget_per_state=max(
                1,
                int(getattr(case_args, "effective_gen_k", getattr(case_args, "k", 1)) or 1),
            ),
            execution_budget_per_state=max(
                0,
                int(getattr(case_args, "effective_exec_k", 0) or 0),
            ),
            verification_level=str(
                getattr(case_args, "effective_verify_level", "light") or "light"
            ),
        )
        _write_trace_metadata(
            trace_path,
            _trace_metadata_payload(
                args=case_args,
                mode="bench",
                solver=solver,
                file_path=file_path,
                theorem=theorem,
            ),
        )
        runner = None
        trace = None
        result = SearchResult(False, [], 0, "unknown failure")
        start = time.perf_counter()
        try:
            runner = _make_runner(case_args)
            retriever = _make_retriever(case_args)
            trace = TraceLogger(trace_path)
            result = _run_with_solver(solver, runner, llm, retriever, trace, config)
        except Exception as exc:
            result = SearchResult(False, [], 0, f"uncaught exception: {exc}")
        finally:
            if trace is not None:
                trace.close()
            if runner is not None:
                runner.close()
        _write_trace_result_metadata(trace_path, result)
        duration_s = time.perf_counter() - start
        durations_s.append(duration_s)
        step_counts.append(result.steps)
        kind = _bench_error_kind(result.error)
        if kind:
            error_kinds[kind] += 1
        error_text = ""
        if result.error:
            error_text = str(result.error).strip()
        semantic = _collect_case_semantic_metrics(
            semantic_report=semantic_report if isinstance(semantic_report, Path) else None,
            artifact_dir=artifact_dir if isinstance(artifact_dir, Path) else None,
        )
        planner = _planner_case_metrics(result)
        results.append(
            {
                "index": idx,
                "theorem": theorem,
                "file": str(file_path),
                "premises": str(premises) if premises else "",
                "solved": result.solved,
                "steps": result.steps,
                "duration_s": duration_s,
                "error_kind": kind,
                "error": error_text,
                "trace_path": str(trace_path) if trace_path else "",
                "dataset": dataset,
                "split": split,
                "tags": tags,
                "semantic_available": bool(semantic.get("available", False)),
                "semantic_source": str(semantic.get("source", "") or ""),
                "semantic_verdict": str(semantic.get("verdict", "unknown")),
                "deterministic_issues_high": int(semantic.get("high", 0) or 0),
                "deterministic_issues_medium": int(semantic.get("medium", 0) or 0),
                "deterministic_issues_low": int(semantic.get("low", 0) or 0),
                "regression_rejections": int(semantic.get("regression_rejections", 0) or 0),
                "planner_cache_hit_states": int(planner.get("planner_cache_hit_states", 0)),
                "planner_cached_tactic_candidates": int(
                    planner.get("planner_cached_tactic_candidates", 0)
                ),
                "planner_cached_tactic_tries": int(planner.get("planner_cached_tactic_tries", 0)),
                "planner_replan_triggers": int(planner.get("planner_replan_triggers", 0)),
                "planner_remembered_tactics": int(planner.get("planner_remembered_tactics", 0)),
            }
        )
        if result.solved:
            solved += 1
        status = "solved" if result.solved else "failed"
        print(
            f"[{idx}/{len(cases)}] {theorem}: {status} "
            f"(steps={result.steps}, time={duration_s:.2f}s)",
            flush=True,
        )

    print(f"Total: {len(cases)}", flush=True)
    print(f"Solved: {solved}", flush=True)
    if results:
        success_rate = (100.0 * solved) / len(results)
        print(f"Success rate: {success_rate:.1f}%", flush=True)
        print(f"Median steps: {statistics.median(step_counts):.1f}", flush=True)
        print(f"Median time: {statistics.median(durations_s):.2f}s", flush=True)
    if error_kinds:
        summary = ", ".join(f"{name}={count}" for name, count in error_kinds.most_common(5))
        print(f"Top failure kinds: {summary}", flush=True)

    finished_at_epoch = time.time()
    summary_payload = _build_bench_summary(
        results=results,
        solved=solved,
        step_counts=step_counts,
        durations_s=durations_s,
        error_kinds=error_kinds,
    )
    dataset_rows = summary_payload.get("dataset_breakdown", [])
    if isinstance(dataset_rows, list) and dataset_rows:
        preview = []
        for item in dataset_rows[:6]:
            if not isinstance(item, dict):
                continue
            preview.append(f"{item.get('dataset', '')}={item.get('total', 0)}")
        if preview:
            print(f"Datasets: {', '.join(preview)}", flush=True)
    semantic_available = int(summary_payload.get("semantic_available_cases", 0) or 0)
    if semantic_available:
        sem_pass = int(summary_payload.get("semantic_pass_cases", 0) or 0)
        sem_fail = int(summary_payload.get("semantic_fail_cases", 0) or 0)
        sem_unknown = int(summary_payload.get("semantic_unknown_cases", 0) or 0)
        print(
            f"Semantic verdicts: pass={sem_pass}, fail={sem_fail}, unknown={sem_unknown} "
            f"(available={semantic_available})",
            flush=True,
        )
        print(
            f"Semantic pass rate: {float(summary_payload.get('semantic_pass_rate_percent', 0.0)):.1f}%",
            flush=True,
        )
    print(
        "Regression rejection rate: "
        f"{float(summary_payload.get('regression_rejection_rate_percent', 0.0)):.1f}% "
        f"({int(summary_payload.get('cases_with_regression_rejections', 0) or 0)}/{len(results)} cases)",
        flush=True,
    )
    planner_replans = int(summary_payload.get("planner_replan_triggers_total", 0) or 0)
    planner_hits = int(summary_payload.get("planner_cache_hit_states_total", 0) or 0)
    if planner_replans or planner_hits:
        print(
            "Planner stats: "
            f"replans={planner_replans}, "
            f"cache_hit_states={planner_hits}, "
            f"cached_tactic_tries={int(summary_payload.get('planner_cached_tactic_tries_total', 0) or 0)}",
            flush=True,
        )
    report_payload = {
        "schema": 1,
        "metadata": _build_bench_metadata(
            args=args,
            suite_path=suite_path,
            suite_entry=suite_entry,
            cases=cases,
            started_at_epoch=started_at_epoch,
            finished_at_epoch=finished_at_epoch,
            total_runtime_s=(time.perf_counter() - started_perf),
        ),
        "summary": summary_payload,
        "cases": results,
    }
    _write_bench_reports(
        args=args,
        report_payload=report_payload,
    )


def _load_bench_cases(suite_path: Path) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    with suite_path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Invalid JSON in suite at line {line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise RuntimeError(f"Suite line {line_no} must be a JSON object.")
            file_raw = str(payload.get("file", "")).strip()
            theorem = str(payload.get("theorem", "")).strip()
            if not file_raw:
                raise RuntimeError(f"Suite line {line_no} is missing required field `file`.")
            if not theorem:
                raise RuntimeError(f"Suite line {line_no} is missing required field `theorem`.")
            premises_raw = str(payload.get("premises", "")).strip()
            semantic_report_raw = str(payload.get("semantic_report", "")).strip()
            artifact_dir_raw = str(payload.get("artifact_dir", "")).strip()
            dataset = str(payload.get("dataset", "")).strip()
            split = str(payload.get("split", "")).strip()
            tags_raw = payload.get("tags", [])
            tags: list[str] = []
            if isinstance(tags_raw, list):
                for item in tags_raw:
                    tag = str(item).strip()
                    if tag:
                        tags.append(tag)
            cases.append(
                {
                    "line": line_no,
                    "file_path": _resolve_suite_path_entry(suite_path, file_raw),
                    "theorem": theorem,
                    "premises": _resolve_suite_path_entry(suite_path, premises_raw)
                    if premises_raw
                    else None,
                    "semantic_report": _resolve_suite_path_entry(suite_path, semantic_report_raw)
                    if semantic_report_raw
                    else None,
                    "artifact_dir": _resolve_suite_path_entry(suite_path, artifact_dir_raw)
                    if artifact_dir_raw
                    else None,
                    "dataset": dataset,
                    "split": split,
                    "tags": tags,
                }
            )
    return cases


def _resolve_suite_path_entry(suite_path: Path, entry: str) -> Path:
    path = Path(entry).expanduser()
    if path.is_absolute():
        return path
    base_relative = (suite_path.parent / path).resolve()
    if base_relative.exists():
        return base_relative
    return path


def _collect_case_semantic_metrics(
    *,
    semantic_report: Path | None,
    artifact_dir: Path | None,
) -> dict[str, object]:
    source = None
    report = None
    if semantic_report is not None:
        report_path = _resolve_case_path(semantic_report)
        report = _load_json_object(report_path)
        if report is not None:
            source = report_path
    artifact = _resolve_case_path(artifact_dir) if artifact_dir is not None else None
    if report is None and artifact is not None:
        inferred = _infer_semantic_report_from_artifact_dir(artifact)
        if inferred is not None:
            candidate = _load_json_object(inferred)
            if candidate is not None:
                report = candidate
                source = inferred

    verdict = "unknown"
    high = 0
    medium = 0
    low = 0
    if isinstance(report, dict):
        verdict = _normalize_semantic_verdict(report.get("verdict"))
        high, medium, low = _count_semantic_issue_severity(report)

    regression_rejections = 0
    if artifact is not None:
        regression_rejections = _count_regression_rejections(artifact)

    return {
        "available": isinstance(report, dict),
        "source": str(source) if source is not None else "",
        "verdict": verdict,
        "high": high,
        "medium": medium,
        "low": low,
        "regression_rejections": regression_rejections,
    }


def _normalize_semantic_verdict(value: object) -> str:
    verdict = str(value or "unknown").strip().lower()
    if verdict not in {"pass", "fail", "unknown"}:
        return "unknown"
    return verdict


def _count_semantic_issue_severity(report: dict) -> tuple[int, int, int]:
    high = 0
    medium = 0
    low = 0
    issues = report.get("deterministic_issues", [])
    if not isinstance(issues, list):
        issues = []
    for item in issues:
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity", "")).strip().lower()
        if severity == "high":
            high += 1
        elif severity == "medium":
            medium += 1
        else:
            low += 1
    return high, medium, low


def _infer_semantic_report_from_artifact_dir(artifact_dir: Path) -> Path | None:
    if not artifact_dir.exists() or not artifact_dir.is_dir():
        return None
    final_report = artifact_dir / "llm_check_final.json"
    if final_report.exists():
        return final_report
    round_reports = sorted(artifact_dir.glob("round_*/llm_check_end.json"), reverse=True)
    if round_reports:
        return round_reports[0]
    round_mid_reports = sorted(artifact_dir.glob("round_*/llm_check_mid.json"), reverse=True)
    if round_mid_reports:
        return round_mid_reports[0]
    return None


def _load_json_object(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _count_regression_rejections(artifact_dir: Path) -> int:
    payload = _load_json_object(artifact_dir / "rejection_memory.json")
    if not isinstance(payload, dict):
        return 0
    count = 0
    for reasons in payload.values():
        if not isinstance(reasons, list):
            continue
        for reason in reasons:
            text = str(reason).strip().lower()
            if not text:
                continue
            if (
                "regression" in text
                or "previously proven declaration" in text
                or "locked declaration" in text
            ):
                count += 1
    return count


def run_bench_make_minif2f(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"miniF2F root not found: {root}")
        sys.exit(1)

    out_path = args.out.expanduser().resolve()
    split = str(args.split or "all").strip().lower()
    glob_pattern = str(args.glob or "**/*.lean").strip() or "**/*.lean"
    dataset = str(args.dataset or "minif2f").strip() or "minif2f"
    require_sorry = bool(getattr(args, "require_sorry", False))
    allow_duplicate_theorems = bool(getattr(args, "allow_duplicate_theorems", False))
    do_shuffle = bool(getattr(args, "shuffle", False))
    seed = int(getattr(args, "seed", 0))
    limit = max(0, int(getattr(args, "limit", 0)))

    default_excludes = [
        ".lake/**",
        "**/.lake/**",
        "build/**",
        "**/build/**",
        ".git/**",
        "**/.git/**",
        "lake-packages/**",
        "**/lake-packages/**",
    ]
    user_excludes = [str(item).strip() for item in list(getattr(args, "exclude", [])) if str(item).strip()]
    excludes = default_excludes + user_excludes

    candidate_files = sorted(path for path in root.glob(glob_pattern) if path.is_file() and path.suffix == ".lean")
    if not candidate_files:
        print(f"No Lean files matched --glob `{glob_pattern}` under {root}")
        sys.exit(1)

    total_files = 0
    used_files = 0
    skipped_excluded = 0
    skipped_split = 0
    skipped_no_sorry = 0
    skipped_no_decl = 0
    duplicate_theorem_skips = 0
    entries: list[dict[str, object]] = []
    seen_theorems: set[str] = set()

    for file_path in candidate_files:
        total_files += 1
        rel = file_path.relative_to(root)
        rel_posix = rel.as_posix()
        if _bench_path_is_excluded(rel_posix, excludes):
            skipped_excluded += 1
            continue
        if not _bench_split_match(rel, split):
            skipped_split += 1
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if require_sorry and re.search(r"\b(sorry|admit)\b", text) is None:
            skipped_no_sorry += 1
            continue
        decl_names = _extract_decl_names(text)
        if not decl_names:
            skipped_no_decl += 1
            continue
        unique_decl_names = list(dict.fromkeys(decl_names))
        file_used = False
        suite_file = _suite_entry_path_for_output(out_path, file_path)
        for theorem in unique_decl_names:
            if not allow_duplicate_theorems and theorem in seen_theorems:
                duplicate_theorem_skips += 1
                continue
            seen_theorems.add(theorem)
            entries.append(
                {
                    "file": suite_file,
                    "theorem": theorem,
                    "dataset": dataset,
                    "split": split,
                    "source_relpath": rel_posix,
                }
            )
            file_used = True
        if file_used:
            used_files += 1

    if do_shuffle and entries:
        random.Random(seed).shuffle(entries)
    if limit > 0:
        entries = entries[:limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in entries:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote miniF2F suite: {out_path}")
    print(f"Cases: {len(entries)}")
    print(f"Files scanned: {total_files}")
    print(f"Files with included declarations: {used_files}")
    if skipped_excluded:
        print(f"Skipped by exclude patterns: {skipped_excluded}")
    if skipped_split:
        print(f"Skipped by split filter `{split}`: {skipped_split}")
    if skipped_no_sorry:
        print(f"Skipped (missing sorry/admit): {skipped_no_sorry}")
    if skipped_no_decl:
        print(f"Skipped (no theorem/lemma/example declarations): {skipped_no_decl}")
    if duplicate_theorem_skips:
        print(f"Skipped duplicate theorem names: {duplicate_theorem_skips}")
    print(f"Next: ulam bench-validate --suite {out_path}")


def _load_bench_suite_rows_raw(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Invalid JSON in suite at line {line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise RuntimeError(f"Suite line {line_no} must be a JSON object.")
            file_raw = str(payload.get("file", "")).strip()
            theorem = str(payload.get("theorem", "")).strip()
            if not file_raw or not theorem:
                raise RuntimeError(
                    f"Suite line {line_no} must include non-empty `file` and `theorem`."
                )
            rows.append(payload)
    return rows


def run_bench_make_regression100(args: argparse.Namespace) -> None:
    source_path = args.source.expanduser().resolve()
    if not source_path.exists():
        print(f"Source suite not found: {source_path}")
        sys.exit(1)
    out_path = args.out.expanduser().resolve()
    size = max(1, int(getattr(args, "size", 100)))
    seed = int(getattr(args, "seed", 0))
    dataset = str(getattr(args, "dataset", "regression100") or "regression100").strip() or "regression100"
    allow_dupes = bool(getattr(args, "allow_duplicate_pairs", False))
    raw_tags = list(getattr(args, "tag", []) or [])
    tags = [str(item).strip() for item in raw_tags if str(item).strip()]
    if not tags:
        tags = ["regression100"]

    try:
        source_rows = _load_bench_suite_rows_raw(source_path)
    except Exception as exc:
        print(f"Failed to read source suite: {exc}")
        sys.exit(1)
    if not source_rows:
        print(f"Source suite has no cases: {source_path}")
        sys.exit(1)

    unique_rows: list[dict[str, object]] = []
    seen_pairs: set[tuple[str, str]] = set()
    duplicate_skips = 0
    for row in source_rows:
        file_raw = str(row.get("file", "")).strip()
        theorem = str(row.get("theorem", "")).strip()
        key = (file_raw, theorem)
        if not allow_dupes and key in seen_pairs:
            duplicate_skips += 1
            continue
        seen_pairs.add(key)
        unique_rows.append(dict(row))

    if len(unique_rows) < size:
        print(
            f"Source suite only has {len(unique_rows)} eligible unique cases; "
            f"cannot build size {size}."
        )
        if not allow_dupes:
            print("Tip: pass --allow-duplicate-pairs if your source suite intentionally repeats pairs.")
        sys.exit(1)

    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(unique_rows)), size))
    selected_rows: list[dict[str, object]] = []
    source_hint = _suite_entry_path_for_output(out_path, source_path)
    for index in selected_indices:
        row = dict(unique_rows[index])
        for field in ("file", "premises", "semantic_report", "artifact_dir"):
            raw_value = str(row.get(field, "")).strip()
            if not raw_value:
                continue
            resolved_value = _resolve_suite_path_entry(source_path, raw_value)
            row[field] = _suite_entry_path_for_output(out_path, resolved_value)
        row["dataset"] = dataset
        tag_list = row.get("tags", [])
        merged_tags: list[str] = []
        if isinstance(tag_list, list):
            merged_tags.extend(str(item).strip() for item in tag_list if str(item).strip())
        for tag in tags:
            if tag not in merged_tags:
                merged_tags.append(tag)
        row["tags"] = merged_tags
        row["fixed_suite"] = "regression100"
        row["fixed_seed"] = seed
        row["source_suite"] = source_hint
        selected_rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in selected_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote regression suite: {out_path}")
    print(f"Cases: {len(selected_rows)} (target={size})")
    print(f"Source suite: {source_path}")
    print(f"Seed: {seed}")
    if duplicate_skips:
        print(f"Skipped duplicate (file,theorem) rows from source: {duplicate_skips}")
    print(f"Next: ulam bench-validate --suite {out_path}")


def _bench_path_is_excluded(rel_posix: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        pat = pattern.strip()
        if not pat:
            continue
        if fnmatch.fnmatch(rel_posix, pat):
            return True
    return False


def _bench_split_match(rel_path: Path, split: str) -> bool:
    if split == "all":
        return True
    if split not in {"valid", "test"}:
        return True
    aliases = {
        "valid": {"valid", "validation"},
        "test": {"test", "testing"},
    }
    wanted = aliases.get(split, {split})
    for raw_part in rel_path.parts:
        part = raw_part.lower()
        if part in wanted:
            return True
        if "." in part and part.rsplit(".", 1)[0] in wanted:
            return True
    return False


def _suite_entry_path_for_output(out_path: Path, target: Path) -> str:
    base = out_path.parent.resolve()
    resolved_target = target.resolve()
    try:
        rel = resolved_target.relative_to(base)
        return rel.as_posix()
    except Exception:
        return str(resolved_target)


def run_bench_validate(args: argparse.Namespace) -> None:
    try:
        suite_path, suite_entry = _resolve_bench_suite_input(args.suite)
    except Exception as exc:
        print(str(exc))
        sys.exit(1)
    if suite_entry is not None:
        print(f"Resolved suite alias `{suite_entry.get('name', '')}` -> {suite_path}")
    try:
        cases = _load_bench_cases(suite_path)
    except Exception as exc:
        print(f"Failed to load suite: {exc}")
        sys.exit(1)
    if not cases:
        print(f"Suite valid: 0 cases ({suite_path})")
        return

    max_errors = max(1, int(getattr(args, "max_errors", 25)))
    check_theorem = not bool(getattr(args, "no_theorem_check", False))
    errors: list[str] = []
    seen_pairs: set[tuple[str, str]] = set()
    duplicate_pairs = 0
    decl_cache: dict[Path, set[str]] = {}
    unique_files: set[str] = set()

    for idx, case in enumerate(cases, start=1):
        line = int(case.get("line", idx))
        theorem = str(case.get("theorem", "")).strip()
        file_path = case.get("file_path")
        premises_path = case.get("premises")
        if not isinstance(file_path, Path):
            errors.append(f"line {line}: invalid `file` path.")
            if len(errors) >= max_errors:
                break
            continue
        resolved_file = _resolve_case_path(file_path)
        unique_files.add(str(resolved_file))
        key = (str(resolved_file), theorem)
        if key in seen_pairs:
            duplicate_pairs += 1
        else:
            seen_pairs.add(key)

        if not resolved_file.exists():
            errors.append(f"line {line}: file not found: {resolved_file}")
            if len(errors) >= max_errors:
                break
            continue

        if premises_path:
            if not isinstance(premises_path, Path):
                errors.append(f"line {line}: invalid `premises` path.")
                if len(errors) >= max_errors:
                    break
            else:
                resolved_premises = _resolve_case_path(premises_path)
                if not resolved_premises.exists():
                    errors.append(f"line {line}: premises file not found: {resolved_premises}")
                    if len(errors) >= max_errors:
                        break
        semantic_report_path = case.get("semantic_report")
        if semantic_report_path:
            if not isinstance(semantic_report_path, Path):
                errors.append(f"line {line}: invalid `semantic_report` path.")
                if len(errors) >= max_errors:
                    break
            else:
                resolved_semantic = _resolve_case_path(semantic_report_path)
                if not resolved_semantic.exists():
                    errors.append(f"line {line}: semantic report not found: {resolved_semantic}")
                    if len(errors) >= max_errors:
                        break
        artifact_dir_path = case.get("artifact_dir")
        if artifact_dir_path:
            if not isinstance(artifact_dir_path, Path):
                errors.append(f"line {line}: invalid `artifact_dir` path.")
                if len(errors) >= max_errors:
                    break
            else:
                resolved_artifact = _resolve_case_path(artifact_dir_path)
                if not resolved_artifact.exists():
                    errors.append(f"line {line}: artifact_dir not found: {resolved_artifact}")
                    if len(errors) >= max_errors:
                        break
                elif not resolved_artifact.is_dir():
                    errors.append(f"line {line}: artifact_dir is not a directory: {resolved_artifact}")
                    if len(errors) >= max_errors:
                        break

        if not check_theorem:
            continue
        declarations = _load_decl_names_for_file(resolved_file, decl_cache)
        if declarations is None:
            errors.append(f"line {line}: failed reading Lean file: {resolved_file}")
            if len(errors) >= max_errors:
                break
            continue
        if theorem not in declarations:
            preview = ", ".join(sorted(declarations)[:8])
            suffix = " ..." if len(declarations) > 8 else ""
            errors.append(
                f"line {line}: theorem `{theorem}` not found in {resolved_file}. "
                f"Declarations: {preview}{suffix}"
            )
            if len(errors) >= max_errors:
                break

    if errors:
        print(f"Suite validation failed: {suite_path}")
        for msg in errors[:max_errors]:
            print(f"- {msg}")
        if len(errors) > max_errors:
            print(f"... and {len(errors) - max_errors} more error(s)")
        sys.exit(1)

    print(f"Suite validation OK: {suite_path}")
    print(f"Cases: {len(cases)}")
    print(f"Unique files: {len(unique_files)}")
    if duplicate_pairs:
        print(f"Duplicate (file,theorem) entries: {duplicate_pairs}")
    dataset_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    for case in cases:
        if not isinstance(case, dict):
            continue
        dataset_counts[_normalize_group_label(case.get("dataset"))] += 1
        split_counts[_normalize_group_label(case.get("split"))] += 1
    if dataset_counts:
        dataset_summary = ", ".join(f"{name}={count}" for name, count in dataset_counts.most_common())
        print(f"Datasets: {dataset_summary}")
    if split_counts:
        split_summary = ", ".join(f"{name}={count}" for name, count in split_counts.most_common())
        print(f"Splits: {split_summary}")
    if check_theorem:
        print("Theorem check: enabled")
    else:
        print("Theorem check: skipped")


def _resolve_case_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (Path.cwd() / expanded).resolve()


def _load_decl_names_for_file(path: Path, cache: dict[Path, set[str]]) -> set[str] | None:
    if path in cache:
        return cache[path]
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    names = set(_extract_decl_names(text))
    cache[path] = names
    return names


def run_bench_compare(args: argparse.Namespace) -> None:
    path_a = args.a.expanduser().resolve()
    path_b = args.b.expanduser().resolve()
    report_a = _load_json_object(path_a)
    report_b = _load_json_object(path_b)
    if report_a is None:
        print(f"Could not read report JSON: {path_a}")
        sys.exit(1)
    if report_b is None:
        print(f"Could not read report JSON: {path_b}")
        sys.exit(1)

    metrics_a = _extract_bench_report_metrics(report_a)
    metrics_b = _extract_bench_report_metrics(report_b)
    meta_a = report_a.get("metadata", {}) if isinstance(report_a.get("metadata"), dict) else {}
    meta_b = report_b.get("metadata", {}) if isinstance(report_b.get("metadata"), dict) else {}
    label_a = _report_label(meta_a)
    label_b = _report_label(meta_b)
    inference_a = _inference_signature(meta_a)
    inference_b = _inference_signature(meta_b)
    suite_a = str(meta_a.get("suite_alias") or meta_a.get("suite_path") or "").strip()
    suite_b = str(meta_b.get("suite_alias") or meta_b.get("suite_path") or "").strip()
    suite_sha_a = str(meta_a.get("suite_sha256", "") or "").strip()
    suite_sha_b = str(meta_b.get("suite_sha256", "") or "").strip()
    comparable_suite_sha = bool(suite_sha_a and suite_sha_b)
    same_suite_sha = bool(comparable_suite_sha and suite_sha_a == suite_sha_b)
    comparable_inference = bool(inference_a and inference_b)
    same_inference = bool(comparable_inference and inference_a == inference_b)

    delta = _bench_metrics_delta(metrics_a, metrics_b)
    payload = {
        "schema": 1,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "report_a": str(path_a),
        "report_b": str(path_b),
        "label_a": label_a,
        "label_b": label_b,
        "inference_a": inference_a,
        "inference_b": inference_b,
        "suite_a": suite_a,
        "suite_b": suite_b,
        "suite_sha_a": suite_sha_a,
        "suite_sha_b": suite_sha_b,
        "comparable_suite_sha": comparable_suite_sha,
        "same_suite_sha": same_suite_sha,
        "comparable_inference": comparable_inference,
        "same_inference": same_inference,
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
        "delta": delta,
    }
    gate = _evaluate_bench_parity_gate(
        metrics_a,
        metrics_b,
        args,
        comparable_suite_sha=comparable_suite_sha,
        same_suite_sha=same_suite_sha,
        comparable_inference=comparable_inference,
        same_inference=same_inference,
    )
    payload["gate"] = gate

    print(f"Benchmark comparison: A={path_a} ({label_a}) vs B={path_b} ({label_b})")
    _print_metric_delta("Solved", metrics_a["solved"], metrics_b["solved"], as_percent=False)
    _print_metric_delta("Success rate", metrics_a["success_rate_percent"], metrics_b["success_rate_percent"], as_percent=True)
    _print_metric_delta(
        "Semantic pass rate",
        metrics_a["semantic_pass_rate_percent"],
        metrics_b["semantic_pass_rate_percent"],
        as_percent=True,
    )
    _print_metric_delta(
        "Semantic fail rate",
        metrics_a["semantic_fail_rate_percent"],
        metrics_b["semantic_fail_rate_percent"],
        as_percent=True,
    )
    _print_metric_delta(
        "Median time (s)",
        metrics_a["median_duration_s"],
        metrics_b["median_duration_s"],
        as_percent=False,
    )
    _print_metric_delta(
        "Regression rejection rate",
        metrics_a["regression_rejection_rate_percent"],
        metrics_b["regression_rejection_rate_percent"],
        as_percent=True,
    )
    _print_metric_delta(
        "Planner replan triggers",
        metrics_a["planner_replan_triggers_total"],
        metrics_b["planner_replan_triggers_total"],
        as_percent=False,
    )
    _print_metric_delta(
        "Planner cached tactic tries",
        metrics_a["planner_cached_tactic_tries_total"],
        metrics_b["planner_cached_tactic_tries_total"],
        as_percent=False,
    )
    if suite_a or suite_b:
        print(f"- Suite A: {suite_a or '(unknown)'}")
        print(f"- Suite B: {suite_b or '(unknown)'}")
    if inference_a or inference_b:
        print(f"- Inference A: {inference_a or '(unknown)'}")
        print(f"- Inference B: {inference_b or '(unknown)'}")
        if inference_a and inference_b and inference_a != inference_b:
            print("- NOTE: inference profile/budgets differ between A and B.")
    if comparable_suite_sha and not same_suite_sha:
        print("- WARNING: suite SHA256 differs between A and B; parity comparison may be invalid.")
    if bool(getattr(args, "gate", False)):
        print(f"Parity gate: {'PASS' if gate['passed'] else 'FAIL'}")
        if gate["reasons"]:
            for reason in gate["reasons"]:
                print(f"- {reason}")

    if args.out_json:
        out_json = Path(args.out_json).expanduser().resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(f"Comparison JSON: {out_json}")
    if args.out_markdown:
        out_md = Path(args.out_markdown).expanduser().resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(_render_bench_compare_markdown(payload) + "\n", encoding="utf-8")
        print(f"Comparison Markdown: {out_md}")
    if bool(getattr(args, "gate", False)) and not bool(gate["passed"]):
        sys.exit(1)


def _evaluate_bench_parity_gate(
    metrics_a: dict[str, float],
    metrics_b: dict[str, float],
    args: argparse.Namespace,
    *,
    comparable_suite_sha: bool,
    same_suite_sha: bool,
    comparable_inference: bool,
    same_inference: bool,
) -> dict[str, object]:
    max_solved_drop = max(0.0, float(getattr(args, "max_solved_drop", 0.0)))
    max_success_rate_drop = max(0.0, float(getattr(args, "max_success_rate_drop", 0.0)))
    max_semantic_pass_drop = max(0.0, float(getattr(args, "max_semantic_pass_rate_drop", 0.0)))
    max_regression_increase = max(
        0.0,
        float(getattr(args, "max_regression_rejection_rate_increase", 0.0)),
    )
    max_median_time_increase_pct = max(
        0.0,
        float(getattr(args, "max_median_time_increase_pct", 25.0)),
    )
    max_planner_replan_increase = max(
        0.0,
        float(getattr(args, "max_planner_replan_triggers_increase", 0.0)),
    )
    max_planner_cached_drop = max(
        0.0,
        float(getattr(args, "max_planner_cached_tactic_tries_drop", 0.0)),
    )

    solved_drop = max(0.0, float(metrics_a.get("solved", 0.0)) - float(metrics_b.get("solved", 0.0)))
    success_rate_drop = max(
        0.0,
        float(metrics_a.get("success_rate_percent", 0.0))
        - float(metrics_b.get("success_rate_percent", 0.0)),
    )
    semantic_pass_drop = max(
        0.0,
        float(metrics_a.get("semantic_pass_rate_percent", 0.0))
        - float(metrics_b.get("semantic_pass_rate_percent", 0.0)),
    )
    regression_rejection_increase = max(
        0.0,
        float(metrics_b.get("regression_rejection_rate_percent", 0.0))
        - float(metrics_a.get("regression_rejection_rate_percent", 0.0)),
    )
    median_a = float(metrics_a.get("median_duration_s", 0.0))
    median_b = float(metrics_b.get("median_duration_s", 0.0))
    if median_a <= 0.0:
        median_time_increase_pct = 0.0 if median_b <= 0.0 else float("inf")
    else:
        median_time_increase_pct = max(0.0, ((median_b - median_a) / median_a) * 100.0)
    planner_replan_increase = max(
        0.0,
        float(metrics_b.get("planner_replan_triggers_total", 0.0))
        - float(metrics_a.get("planner_replan_triggers_total", 0.0)),
    )
    planner_cached_tries_drop = max(
        0.0,
        float(metrics_a.get("planner_cached_tactic_tries_total", 0.0))
        - float(metrics_b.get("planner_cached_tactic_tries_total", 0.0)),
    )
    allow_profile_mismatch = bool(getattr(args, "allow_profile_mismatch", False))
    allow_suite_mismatch = bool(getattr(args, "allow_suite_mismatch", False))
    gate_enabled = bool(getattr(args, "gate", False))

    reasons: list[str] = []
    if solved_drop > max_solved_drop:
        reasons.append(
            f"solved drop {solved_drop:.2f} exceeds allowed {max_solved_drop:.2f}"
        )
    if success_rate_drop > max_success_rate_drop:
        reasons.append(
            "success-rate drop "
            f"{success_rate_drop:.2f}% exceeds allowed {max_success_rate_drop:.2f}%"
        )
    if semantic_pass_drop > max_semantic_pass_drop:
        reasons.append(
            "semantic-pass-rate drop "
            f"{semantic_pass_drop:.2f}% exceeds allowed {max_semantic_pass_drop:.2f}%"
        )
    if regression_rejection_increase > max_regression_increase:
        reasons.append(
            "regression-rejection-rate increase "
            f"{regression_rejection_increase:.2f}% exceeds allowed {max_regression_increase:.2f}%"
        )
    if median_time_increase_pct > max_median_time_increase_pct:
        value = "inf" if median_time_increase_pct == float("inf") else f"{median_time_increase_pct:.2f}%"
        reasons.append(
            f"median-time increase {value} exceeds allowed {max_median_time_increase_pct:.2f}%"
        )
    if planner_replan_increase > max_planner_replan_increase:
        reasons.append(
            "planner-replan-trigger increase "
            f"{planner_replan_increase:.2f} exceeds allowed {max_planner_replan_increase:.2f}"
        )
    if planner_cached_tries_drop > max_planner_cached_drop:
        reasons.append(
            "planner-cached-tactic-tries drop "
            f"{planner_cached_tries_drop:.2f} exceeds allowed {max_planner_cached_drop:.2f}"
        )
    if gate_enabled and not allow_profile_mismatch:
        if not comparable_inference:
            reasons.append(
                "inference profile metadata missing in one or both reports "
                "(use --allow-profile-mismatch to bypass)"
            )
        elif not same_inference:
            reasons.append(
                "inference profile/budgets mismatch between A and B "
                "(use --allow-profile-mismatch to bypass)"
            )
    if gate_enabled and not allow_suite_mismatch:
        if not comparable_suite_sha:
            reasons.append(
                "suite SHA256 missing in one or both reports "
                "(use --allow-suite-mismatch to bypass)"
            )
        elif not same_suite_sha:
            reasons.append(
                "suite SHA256 mismatch between A and B "
                "(use --allow-suite-mismatch to bypass)"
            )

    return {
        "enabled": gate_enabled,
        "passed": len(reasons) == 0,
        "thresholds": {
            "max_solved_drop": max_solved_drop,
            "max_success_rate_drop": max_success_rate_drop,
            "max_semantic_pass_rate_drop": max_semantic_pass_drop,
            "max_regression_rejection_rate_increase": max_regression_increase,
            "max_median_time_increase_pct": max_median_time_increase_pct,
            "max_planner_replan_triggers_increase": max_planner_replan_increase,
            "max_planner_cached_tactic_tries_drop": max_planner_cached_drop,
            "allow_profile_mismatch": allow_profile_mismatch,
            "allow_suite_mismatch": allow_suite_mismatch,
        },
        "actual": {
            "solved_drop": solved_drop,
            "success_rate_drop": success_rate_drop,
            "semantic_pass_rate_drop": semantic_pass_drop,
            "regression_rejection_rate_increase": regression_rejection_increase,
            "median_time_increase_pct": median_time_increase_pct,
            "planner_replan_triggers_increase": planner_replan_increase,
            "planner_cached_tactic_tries_drop": planner_cached_tries_drop,
            "comparable_inference": comparable_inference,
            "same_inference": same_inference,
            "comparable_suite_sha": comparable_suite_sha,
            "same_suite_sha": same_suite_sha,
        },
        "reasons": reasons,
    }


def _report_label(metadata: dict) -> str:
    llm_backend = str(metadata.get("llm_backend", "")).strip()
    llm_model = str(metadata.get("llm_model", "")).strip()
    inference = _inference_signature(metadata)
    if llm_backend and llm_model:
        base = f"{llm_backend}:{llm_model}"
        return f"{base} [{inference}]" if inference else base
    if llm_backend:
        return f"{llm_backend} [{inference}]" if inference else llm_backend
    return f"unknown [{inference}]" if inference else "unknown"


def _inference_signature(metadata: dict) -> str:
    profile = str(metadata.get("inference_profile", "") or "").strip()
    verify = str(metadata.get("verification_level", "") or "").strip()
    try:
        gen = int(metadata.get("generation_budget_per_state", 0) or 0)
    except Exception:
        gen = 0
    try:
        exec_k = int(metadata.get("execution_budget_per_state", 0) or 0)
    except Exception:
        exec_k = 0
    if not profile and not verify and gen <= 0 and exec_k <= 0:
        return ""
    exec_text = "all" if exec_k <= 0 else str(exec_k)
    profile_text = profile if profile else "default"
    verify_text = verify if verify else "light"
    return f"profile={profile_text}, gen={max(1, gen)}, exec={exec_text}, verify={verify_text}"


def _extract_bench_report_metrics(report: dict) -> dict[str, float]:
    summary = report.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    return {
        "total": float(int(summary.get("total", 0) or 0)),
        "solved": float(int(summary.get("solved", 0) or 0)),
        "success_rate_percent": float(summary.get("success_rate_percent", 0.0) or 0.0),
        "semantic_pass_rate_percent": float(summary.get("semantic_pass_rate_percent", 0.0) or 0.0),
        "semantic_fail_rate_percent": float(summary.get("semantic_fail_rate_percent", 0.0) or 0.0),
        "median_duration_s": float(summary.get("median_duration_s", 0.0) or 0.0),
        "median_steps": float(summary.get("median_steps", 0.0) or 0.0),
        "regression_rejection_rate_percent": float(
            summary.get("regression_rejection_rate_percent", 0.0) or 0.0
        ),
        "planner_replan_triggers_total": float(
            int(summary.get("planner_replan_triggers_total", 0) or 0)
        ),
        "planner_cached_tactic_tries_total": float(
            int(summary.get("planner_cached_tactic_tries_total", 0) or 0)
        ),
    }


def _bench_metrics_delta(metrics_a: dict[str, float], metrics_b: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, a_val in metrics_a.items():
        b_val = float(metrics_b.get(key, 0.0) or 0.0)
        out[key] = b_val - float(a_val)
    return out


def _format_delta(delta: float, suffix: str = "") -> str:
    sign = "+" if delta >= 0 else ""
    if suffix:
        return f"{sign}{delta:.2f}{suffix}"
    return f"{sign}{delta:.2f}"


def _print_metric_delta(name: str, a_value: float, b_value: float, *, as_percent: bool) -> None:
    delta = b_value - a_value
    if as_percent:
        print(f"- {name}: {a_value:.2f}% -> {b_value:.2f}% ({_format_delta(delta, '%')})")
        return
    print(f"- {name}: {a_value:.2f} -> {b_value:.2f} ({_format_delta(delta)})")


def _render_bench_compare_markdown(payload: dict) -> str:
    metrics_a = payload.get("metrics_a", {})
    metrics_b = payload.get("metrics_b", {})
    delta = payload.get("delta", {})
    if not isinstance(metrics_a, dict):
        metrics_a = {}
    if not isinstance(metrics_b, dict):
        metrics_b = {}
    if not isinstance(delta, dict):
        delta = {}
    lines = [
        "# Ulam Bench Comparison",
        "",
        f"- Report A: {payload.get('report_a', '')}",
        f"- Report B: {payload.get('report_b', '')}",
        f"- Label A: {payload.get('label_a', '')}",
        f"- Label B: {payload.get('label_b', '')}",
        f"- Inference A: {payload.get('inference_a', '')}",
        f"- Inference B: {payload.get('inference_b', '')}",
        f"- Same inference profile: {'yes' if bool(payload.get('same_inference', False)) else 'no'}",
        f"- Suite A: {payload.get('suite_a', '')}",
        f"- Suite B: {payload.get('suite_b', '')}",
        f"- Same suite SHA256: {'yes' if bool(payload.get('same_suite_sha', False)) else 'no'}",
    ]
    gate = payload.get("gate", {})
    if isinstance(gate, dict):
        enabled = bool(gate.get("enabled", False))
        lines.append(f"- Parity gate enabled: {'yes' if enabled else 'no'}")
        if enabled:
            lines.append(f"- Parity gate pass: {'yes' if bool(gate.get('passed', False)) else 'no'}")
            reasons = gate.get("reasons", [])
            if isinstance(reasons, list) and reasons:
                for reason in reasons:
                    lines.append(f"- Gate reason: {reason}")
    lines.extend(
        [
            "",
            "| Metric | A | B | Delta (B-A) |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    ordered = [
        ("solved", "Solved"),
        ("total", "Total"),
        ("success_rate_percent", "Success rate (%)"),
        ("semantic_pass_rate_percent", "Semantic pass rate (%)"),
        ("semantic_fail_rate_percent", "Semantic fail rate (%)"),
        ("median_duration_s", "Median duration (s)"),
        ("median_steps", "Median steps"),
        ("regression_rejection_rate_percent", "Regression rejection rate (%)"),
        ("planner_replan_triggers_total", "Planner replan triggers"),
        ("planner_cached_tactic_tries_total", "Planner cached tactic tries"),
    ]
    for key, label in ordered:
        a_val = float(metrics_a.get(key, 0.0) or 0.0)
        b_val = float(metrics_b.get(key, 0.0) or 0.0)
        d_val = float(delta.get(key, 0.0) or 0.0)
        lines.append(f"| {label} | {a_val:.2f} | {b_val:.2f} | {_format_delta(d_val)} |")
    return "\n".join(lines).rstrip()


def _normalize_group_label(value: object, fallback: str = "unspecified") -> str:
    text = str(value or "").strip()
    return text if text else fallback


def _update_group_breakdown(
    bucket: dict[str, dict[str, int]],
    *,
    label: str,
    solved: bool,
    semantic_available: bool,
    semantic_verdict: str,
    regression_rejections: int,
) -> None:
    row = bucket.setdefault(
        label,
        {
            "total": 0,
            "solved": 0,
            "semantic_available": 0,
            "semantic_pass": 0,
            "regression_cases": 0,
        },
    )
    row["total"] += 1
    if solved:
        row["solved"] += 1
    if semantic_available:
        row["semantic_available"] += 1
        if semantic_verdict == "pass":
            row["semantic_pass"] += 1
    if regression_rejections > 0:
        row["regression_cases"] += 1


def _finalize_group_breakdown(
    bucket: dict[str, dict[str, int]],
    label_key: str,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for label in sorted(bucket):
        row = bucket[label]
        total = int(row.get("total", 0))
        solved = int(row.get("solved", 0))
        semantic_available = int(row.get("semantic_available", 0))
        semantic_pass = int(row.get("semantic_pass", 0))
        out.append(
            {
                label_key: label,
                "total": total,
                "solved": solved,
                "failed": max(0, total - solved),
                "success_rate_percent": (100.0 * solved / total) if total else 0.0,
                "semantic_available_cases": semantic_available,
                "semantic_pass_rate_percent": (
                    (100.0 * semantic_pass / semantic_available) if semantic_available else 0.0
                ),
                "cases_with_regression_rejections": int(row.get("regression_cases", 0)),
            }
        )
    return out


def _build_bench_summary(
    *,
    results: list[dict[str, object]],
    solved: int,
    step_counts: list[int],
    durations_s: list[float],
    error_kinds: Counter[str],
) -> dict[str, object]:
    total = len(results)
    success_rate = (100.0 * solved / total) if total else 0.0
    semantic_available = 0
    semantic_pass = 0
    semantic_fail = 0
    semantic_unknown = 0
    deterministic_high_total = 0
    deterministic_medium_total = 0
    deterministic_low_total = 0
    regression_cases = 0
    regression_total = 0
    dataset_breakdown: dict[str, dict[str, int]] = {}
    split_breakdown: dict[str, dict[str, int]] = {}
    tag_counts: Counter[str] = Counter()
    planner_cache_hit_states_total = 0
    planner_cached_tactic_candidates_total = 0
    planner_cached_tactic_tries_total = 0
    planner_replan_triggers_total = 0
    planner_remembered_tactics_total = 0
    for case in results:
        if not isinstance(case, dict):
            continue
        deterministic_high_total += int(case.get("deterministic_issues_high", 0) or 0)
        deterministic_medium_total += int(case.get("deterministic_issues_medium", 0) or 0)
        deterministic_low_total += int(case.get("deterministic_issues_low", 0) or 0)
        regressions = int(case.get("regression_rejections", 0) or 0)
        if regressions > 0:
            regression_cases += 1
        regression_total += regressions
        planner_cache_hit_states_total += int(case.get("planner_cache_hit_states", 0) or 0)
        planner_cached_tactic_candidates_total += int(
            case.get("planner_cached_tactic_candidates", 0) or 0
        )
        planner_cached_tactic_tries_total += int(case.get("planner_cached_tactic_tries", 0) or 0)
        planner_replan_triggers_total += int(case.get("planner_replan_triggers", 0) or 0)
        planner_remembered_tactics_total += int(case.get("planner_remembered_tactics", 0) or 0)
        solved_case = bool(case.get("solved", False))
        semantic_available_case = bool(case.get("semantic_available", False))
        semantic_verdict_case = _normalize_semantic_verdict(case.get("semantic_verdict"))
        dataset_label = _normalize_group_label(case.get("dataset"))
        split_label = _normalize_group_label(case.get("split"))
        _update_group_breakdown(
            dataset_breakdown,
            label=dataset_label,
            solved=solved_case,
            semantic_available=semantic_available_case,
            semantic_verdict=semantic_verdict_case,
            regression_rejections=regressions,
        )
        _update_group_breakdown(
            split_breakdown,
            label=split_label,
            solved=solved_case,
            semantic_available=semantic_available_case,
            semantic_verdict=semantic_verdict_case,
            regression_rejections=regressions,
        )
        tags = case.get("tags", [])
        if isinstance(tags, list):
            for raw in tags:
                tag = str(raw).strip()
                if tag:
                    tag_counts[tag] += 1
        if not bool(case.get("semantic_available", False)):
            continue
        semantic_available += 1
        verdict = _normalize_semantic_verdict(case.get("semantic_verdict"))
        if verdict == "pass":
            semantic_pass += 1
        elif verdict == "fail":
            semantic_fail += 1
        else:
            semantic_unknown += 1
    semantic_pass_rate = (100.0 * semantic_pass / semantic_available) if semantic_available else 0.0
    semantic_fail_rate = (100.0 * semantic_fail / semantic_available) if semantic_available else 0.0
    regression_rate = (100.0 * regression_cases / total) if total else 0.0
    return {
        "total": total,
        "solved": solved,
        "failed": max(0, total - solved),
        "success_rate_percent": success_rate,
        "median_steps": statistics.median(step_counts) if step_counts else 0.0,
        "median_duration_s": statistics.median(durations_s) if durations_s else 0.0,
        "mean_duration_s": (sum(durations_s) / len(durations_s)) if durations_s else 0.0,
        "error_kinds": dict(error_kinds.most_common()),
        "semantic_available_cases": semantic_available,
        "semantic_pass_cases": semantic_pass,
        "semantic_fail_cases": semantic_fail,
        "semantic_unknown_cases": semantic_unknown,
        "semantic_pass_rate_percent": semantic_pass_rate,
        "semantic_fail_rate_percent": semantic_fail_rate,
        "deterministic_issues_high_total": deterministic_high_total,
        "deterministic_issues_medium_total": deterministic_medium_total,
        "deterministic_issues_low_total": deterministic_low_total,
        "cases_with_regression_rejections": regression_cases,
        "regression_rejections_total": regression_total,
        "regression_rejection_rate_percent": regression_rate,
        "planner_cache_hit_states_total": planner_cache_hit_states_total,
        "planner_cached_tactic_candidates_total": planner_cached_tactic_candidates_total,
        "planner_cached_tactic_tries_total": planner_cached_tactic_tries_total,
        "planner_replan_triggers_total": planner_replan_triggers_total,
        "planner_remembered_tactics_total": planner_remembered_tactics_total,
        "planner_replan_triggers_per_case": (
            float(planner_replan_triggers_total) / float(total) if total else 0.0
        ),
        "dataset_breakdown": _finalize_group_breakdown(dataset_breakdown, "dataset"),
        "split_breakdown": _finalize_group_breakdown(split_breakdown, "split"),
        "top_tags": [
            {"tag": tag, "count": count}
            for tag, count in tag_counts.most_common(20)
        ],
    }


def _build_bench_metadata(
    *,
    args: argparse.Namespace,
    suite_path: Path,
    suite_entry: dict[str, object] | None,
    cases: list[dict[str, object]],
    started_at_epoch: float,
    finished_at_epoch: float,
    total_runtime_s: float,
) -> dict[str, object]:
    lean_project = _resolve_bench_lean_project(args, cases)
    lean_toolchain = _read_toolchain_file(lean_project / "lean-toolchain") if lean_project else ""
    mathlib_rev = _mathlib_rev_from_manifest(lean_project) if lean_project else ""
    mathlib_commit = _mathlib_commit_from_checkout(lean_project) if lean_project else ""
    repo_root = Path(__file__).resolve().parents[1]
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at_epoch))
    finished_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finished_at_epoch))
    return {
        "run_started_utc": started_utc,
        "run_finished_utc": finished_utc,
        "run_duration_s": total_runtime_s,
        "ulam_version": __version__,
        "ulam_git_commit": _git_rev_parse(repo_root),
        "suite_alias": str(suite_entry.get("name", "")).strip() if isinstance(suite_entry, dict) else "",
        "suite_description": (
            str(suite_entry.get("description", "")).strip() if isinstance(suite_entry, dict) else ""
        ),
        "suite_target_cases": int(suite_entry.get("target_cases", 0) or 0)
        if isinstance(suite_entry, dict)
        else 0,
        "suite_path": str(suite_path),
        "suite_sha256": _sha256_file(suite_path),
        "suite_cases": len(cases),
        "llm_backend": str(getattr(args, "llm", "")),
        "llm_model": _resolve_bench_model(args),
        "lean_backend": str(getattr(args, "lean", "")),
        "lean_project": str(lean_project) if lean_project else "",
        "lean_imports": list(getattr(args, "lean_import", []) or []),
        "lean_toolchain": lean_toolchain or "",
        "mathlib_rev": mathlib_rev or "",
        "mathlib_commit": mathlib_commit or "",
        "solver": str(getattr(args, "solver", "")),
        "retriever": str(getattr(args, "retriever", "")),
        "retriever_source": str(getattr(args, "retriever_source", "")),
        "retriever_index": str(getattr(args, "retriever_index", "") or ""),
        "trace_dir": str(getattr(args, "trace_dir", "") or ""),
        "seed": int(getattr(args, "seed", 0)),
        "max_steps": int(getattr(args, "max_steps", 0)),
        "beam": int(getattr(args, "beam", 0)),
        "k": int(getattr(args, "k", 0)),
        "inference_profile": str(getattr(args, "inference_profile", "default") or "default"),
        "generation_budget_per_state": int(
            getattr(args, "effective_gen_k", getattr(args, "k", 1)) or 1
        ),
        "execution_budget_per_state": int(getattr(args, "effective_exec_k", 0) or 0),
        "verification_level": str(getattr(args, "effective_verify_level", "light") or "light"),
        "timeout_s": float(getattr(args, "timeout", 0.0)),
        "repair_attempts": int(getattr(args, "repair", 0)),
        "autop": _autop_enabled(args),
        "instruction": str(getattr(args, "instruction", "") or ""),
    }


def _resolve_bench_model(args: argparse.Namespace) -> str:
    backend = str(getattr(args, "llm", ""))
    if backend in {"openai", "codex_cli"}:
        return str(getattr(args, "openai_model", "") or "")
    if backend in {"anthropic", "claude_cli"}:
        return str(getattr(args, "anthropic_model", "") or "")
    if backend in {"gemini", "gemini_cli"}:
        return str(getattr(args, "gemini_model", "") or "")
    if backend == "ollama":
        return str(getattr(args, "ollama_model", "") or "")
    return backend or "unknown"


def _resolve_bench_lean_project(
    args: argparse.Namespace,
    cases: list[dict[str, object]],
) -> Path | None:
    explicit = getattr(args, "lean_project", None)
    if isinstance(explicit, Path):
        return explicit.expanduser().resolve()
    for case in cases:
        file_path = case.get("file_path")
        if not isinstance(file_path, Path):
            continue
        file_abs = file_path.expanduser().resolve()
        project = _find_lean_project_for_file(file_abs)
        if project is not None:
            return project.expanduser().resolve()
    return None


def _git_rev_parse(path: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _write_bench_reports(*, args: argparse.Namespace, report_payload: dict[str, object]) -> None:
    if args.report_json:
        report_json_path = Path(args.report_json).expanduser().resolve()
        report_json_path.parent.mkdir(parents=True, exist_ok=True)
        report_json_path.write_text(
            json.dumps(report_payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        print(f"Report JSON: {report_json_path}")
    if args.report_markdown:
        report_md_path = Path(args.report_markdown).expanduser().resolve()
        report_md_path.parent.mkdir(parents=True, exist_ok=True)
        report_md_path.write_text(
            _render_bench_markdown(report_payload) + "\n",
            encoding="utf-8",
        )
        print(f"Report Markdown: {report_md_path}")


def _render_bench_markdown(report_payload: dict[str, object]) -> str:
    metadata = report_payload.get("metadata", {})
    summary = report_payload.get("summary", {})
    cases = report_payload.get("cases", [])
    if not isinstance(metadata, dict):
        metadata = {}
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(cases, list):
        cases = []
    exec_budget = "all"
    try:
        exec_budget_raw = int(metadata.get("execution_budget_per_state", 0) or 0)
        if exec_budget_raw > 0:
            exec_budget = str(exec_budget_raw)
    except Exception:
        exec_budget = str(metadata.get("execution_budget_per_state", "all") or "all")

    lines: list[str] = [
        "# Ulam Bench Report",
        "",
        "## Summary",
        f"- Total: {summary.get('total', 0)}",
        f"- Solved: {summary.get('solved', 0)}",
        f"- Failed: {summary.get('failed', 0)}",
        f"- Success rate: {float(summary.get('success_rate_percent', 0.0)):.1f}%",
        f"- Median steps: {float(summary.get('median_steps', 0.0)):.1f}",
        f"- Median time: {float(summary.get('median_duration_s', 0.0)):.2f}s",
        f"- Mean time: {float(summary.get('mean_duration_s', 0.0)):.2f}s",
        "",
        "## Anti-Cheat Metrics",
        f"- Semantic available: {summary.get('semantic_available_cases', 0)}",
        f"- Semantic pass: {summary.get('semantic_pass_cases', 0)}",
        f"- Semantic fail: {summary.get('semantic_fail_cases', 0)}",
        f"- Semantic unknown: {summary.get('semantic_unknown_cases', 0)}",
        f"- Semantic pass rate: {float(summary.get('semantic_pass_rate_percent', 0.0)):.1f}%",
        f"- Semantic fail rate: {float(summary.get('semantic_fail_rate_percent', 0.0)):.1f}%",
        f"- Deterministic issues (high/medium/low): "
        f"{summary.get('deterministic_issues_high_total', 0)}/"
        f"{summary.get('deterministic_issues_medium_total', 0)}/"
        f"{summary.get('deterministic_issues_low_total', 0)}",
        f"- Regression rejections: {summary.get('regression_rejections_total', 0)}",
        f"- Regression rejection rate: {float(summary.get('regression_rejection_rate_percent', 0.0)):.1f}%",
        "",
        "## Planner Metrics",
        f"- Planner cache-hit states: {summary.get('planner_cache_hit_states_total', 0)}",
        f"- Planner cached tactic candidates: {summary.get('planner_cached_tactic_candidates_total', 0)}",
        f"- Planner cached tactic tries: {summary.get('planner_cached_tactic_tries_total', 0)}",
        f"- Planner replan triggers: {summary.get('planner_replan_triggers_total', 0)}",
        f"- Planner remembered tactics: {summary.get('planner_remembered_tactics_total', 0)}",
        f"- Planner replan triggers per case: {float(summary.get('planner_replan_triggers_per_case', 0.0)):.2f}",
        "",
        "## Run Metadata",
        f"- Started (UTC): {metadata.get('run_started_utc', '')}",
        f"- Finished (UTC): {metadata.get('run_finished_utc', '')}",
        f"- Duration: {float(metadata.get('run_duration_s', 0.0)):.2f}s",
        f"- Ulam version: {metadata.get('ulam_version', '')}",
        f"- Ulam commit: {metadata.get('ulam_git_commit', '')}",
        f"- Suite alias: {metadata.get('suite_alias', '')}",
        f"- Suite: {metadata.get('suite_path', '')}",
        f"- Suite SHA256: {metadata.get('suite_sha256', '')}",
        f"- Suite cases (target/actual): {metadata.get('suite_target_cases', 0)}/{metadata.get('suite_cases', 0)}",
        f"- LLM: {metadata.get('llm_backend', '')} ({metadata.get('llm_model', '')})",
        f"- Lean backend: {metadata.get('lean_backend', '')}",
        f"- Lean project: {metadata.get('lean_project', '')}",
        f"- Lean toolchain: {metadata.get('lean_toolchain', '')}",
        f"- Mathlib commit: {metadata.get('mathlib_commit', '')}",
        f"- Solver: {metadata.get('solver', '')}",
        f"- Retriever: {metadata.get('retriever', '')}/{metadata.get('retriever_source', '')}",
        f"- Inference profile: {metadata.get('inference_profile', 'default')}",
        f"- Generation budget per state: {metadata.get('generation_budget_per_state', metadata.get('k', 1))}",
        f"- Execution budget per state: {exec_budget}",
        f"- Verification level: {metadata.get('verification_level', 'light')}",
        "",
    ]

    error_kinds = summary.get("error_kinds", {})
    if isinstance(error_kinds, dict) and error_kinds:
        lines.append("## Top Failure Kinds")
        for name, count in error_kinds.items():
            lines.append(f"- {name}: {count}")
        lines.append("")

    dataset_rows = summary.get("dataset_breakdown", [])
    if isinstance(dataset_rows, list) and dataset_rows:
        lines.extend(
            [
                "## Dataset Breakdown",
                "",
                "| Dataset | Total | Solved | Success (%) | Semantic pass (%) | Regression cases |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in dataset_rows:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"| {item.get('dataset', '')} | "
                f"{int(item.get('total', 0) or 0)} | "
                f"{int(item.get('solved', 0) or 0)} | "
                f"{float(item.get('success_rate_percent', 0.0) or 0.0):.1f} | "
                f"{float(item.get('semantic_pass_rate_percent', 0.0) or 0.0):.1f} | "
                f"{int(item.get('cases_with_regression_rejections', 0) or 0)} |"
            )
        lines.append("")

    split_rows = summary.get("split_breakdown", [])
    if isinstance(split_rows, list) and split_rows:
        lines.extend(
            [
                "## Split Breakdown",
                "",
                "| Split | Total | Solved | Success (%) | Semantic pass (%) | Regression cases |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in split_rows:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"| {item.get('split', '')} | "
                f"{int(item.get('total', 0) or 0)} | "
                f"{int(item.get('solved', 0) or 0)} | "
                f"{float(item.get('success_rate_percent', 0.0) or 0.0):.1f} | "
                f"{float(item.get('semantic_pass_rate_percent', 0.0) or 0.0):.1f} | "
                f"{int(item.get('cases_with_regression_rejections', 0) or 0)} |"
            )
        lines.append("")

    top_tags = summary.get("top_tags", [])
    if isinstance(top_tags, list) and top_tags:
        lines.append("## Top Tags")
        for item in top_tags[:15]:
            if not isinstance(item, dict):
                continue
            tag = str(item.get("tag", "")).strip()
            count = int(item.get("count", 0) or 0)
            if tag:
                lines.append(f"- {tag}: {count}")
        lines.append("")

    failed_cases = [case for case in cases if isinstance(case, dict) and not bool(case.get("solved", False))]
    if failed_cases:
        lines.extend(
            [
                "## Failed Cases",
                "",
                "| # | Theorem | Dataset | Split | Steps | Time (s) | Error Kind |",
                "| --- | --- | --- | --- | ---: | ---: | --- |",
            ]
        )
        for case in failed_cases[:25]:
            theorem = str(case.get("theorem", "")).replace("|", "\\|")
            dataset = str(case.get("dataset", "")).replace("|", "\\|")
            split = str(case.get("split", "")).replace("|", "\\|")
            lines.append(
                "| "
                + f"{case.get('index', '')} | {theorem} | {dataset} | {split} | {case.get('steps', 0)} | "
                + f"{float(case.get('duration_s', 0.0)):.2f} | {case.get('error_kind', '')} |"
            )
        if len(failed_cases) > 25:
            lines.append(f"| ... | ({len(failed_cases) - 25} more) |  |  |  |  |  |")
        lines.append("")
    return "\n".join(lines).rstrip()


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
    if args.lean == "lsp":
        return LeanLspRunner(project_path=args.lean_project)
    if args.lean == "cli":
        raise RuntimeError(
            "Lean backend `cli` is not a tactic runner backend. "
            "Use `--prove-mode llm` with this backend, or switch to `--lean dojo|lsp`."
        )
    raise RuntimeError(f"unknown Lean backend: {args.lean}")


def _make_llm(args: argparse.Namespace):
    llm_timeout_s, llm_heartbeat_s = _resolve_cli_llm_runtime_settings()
    if args.llm == "mock":
        return MockLLMClient()
    if args.llm == "openai":
        if not args.openai_key:
            raise RuntimeError("OpenAI API key missing. Set ULAM_OPENAI_API_KEY or --openai-key.")
        return OpenAICompatClient(
            api_key=args.openai_key,
            base_url=args.openai_base_url,
            model=args.openai_model,
            timeout_s=llm_timeout_s,
            heartbeat_s=llm_heartbeat_s,
        )
    if args.llm == "ollama":
        return OllamaClient(
            base_url=args.ollama_base_url,
            model=args.ollama_model,
            timeout_s=llm_timeout_s,
            heartbeat_s=llm_heartbeat_s,
        )
    if args.llm == "anthropic":
        token = args.anthropic_key or args.anthropic_setup_token
        return AnthropicClient(
            api_key=token,
            base_url=args.anthropic_base_url,
            model=args.anthropic_model,
            timeout_s=llm_timeout_s,
            heartbeat_s=llm_heartbeat_s,
        )
    if args.llm == "gemini":
        if not args.gemini_api_key:
            raise RuntimeError("Gemini API key missing. Set ULAM_GEMINI_API_KEY or --gemini-api-key.")
        return GeminiClient(
            api_key=args.gemini_api_key,
            base_url=args.gemini_base_url,
            model=args.gemini_model,
            timeout_s=llm_timeout_s,
            heartbeat_s=llm_heartbeat_s,
        )
    if args.llm == "codex_cli":
        model = args.openai_model or None
        return CodexCLIClient(
            model=model,
            timeout_s=llm_timeout_s,
            heartbeat_s=llm_heartbeat_s,
        )
    if args.llm == "claude_cli":
        return ClaudeCLIClient(
            model=args.anthropic_model or None,
            timeout_s=llm_timeout_s,
            heartbeat_s=llm_heartbeat_s,
        )
    if args.llm == "gemini_cli":
        return GeminiCLIClient(
            model=args.gemini_model or None,
            timeout_s=llm_timeout_s,
            heartbeat_s=llm_heartbeat_s,
        )
    raise RuntimeError(f"unknown LLM backend: {args.llm}")


def _resolve_cli_llm_runtime_settings() -> tuple[float | None, float | None]:
    timeout_s: float | None = None
    heartbeat_s: float | None = 60.0
    try:
        cfg = load_config()
    except Exception:
        return timeout_s, heartbeat_s

    llm_cfg = cfg.get("llm", {}) if isinstance(cfg, dict) else {}
    if not isinstance(llm_cfg, dict):
        return timeout_s, heartbeat_s

    raw_timeout = llm_cfg.get("timeout_s", 0)
    raw_heartbeat = llm_cfg.get("heartbeat_s", 60)
    try:
        timeout_val = float(raw_timeout)
    except Exception:
        timeout_val = 0.0
    try:
        heartbeat_val = float(raw_heartbeat)
    except Exception:
        heartbeat_val = 60.0

    if timeout_val > 0:
        timeout_s = timeout_val
    if heartbeat_val <= 0:
        heartbeat_s = None
    else:
        heartbeat_s = heartbeat_val
    return timeout_s, heartbeat_s


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
    cleaned = _extract_probable_lean_code(text)
    cleaned = _strip_known_llm_noise_lines(cleaned)
    return cleaned + "\n" if cleaned else ""


def _extract_probable_lean_code(text: str) -> str:
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    blocks = _extract_code_fence_blocks(raw)
    if blocks:
        lean_blocks = [body for lang, body in blocks if "lean" in lang]
        if lean_blocks:
            candidate = max(lean_blocks, key=_lean_likeness_score)
            return candidate.strip()
        candidate = max((body for _, body in blocks), key=_lean_likeness_score)
        if _lean_likeness_score(candidate) > 0:
            return candidate.strip()

    lines = raw.splitlines()
    start = 0
    while start < len(lines):
        line = lines[start].strip()
        if not line:
            start += 1
            continue
        if _looks_like_lean_line(line):
            break
        start += 1
    candidate = "\n".join(lines[start:]).strip() if start < len(lines) else raw.strip()
    if _lean_likeness_score(candidate) <= 0:
        return raw.strip()
    return candidate


def _extract_code_fence_blocks(text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("```"):
            i += 1
            continue
        lang = line[3:].strip().lower()
        i += 1
        body: list[str] = []
        while i < len(lines) and not lines[i].strip().startswith("```"):
            body.append(lines[i])
            i += 1
        blocks.append((lang, "\n".join(body)))
        if i < len(lines):
            i += 1
    return blocks


def _lean_likeness_score(text: str) -> int:
    score = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _looks_like_lean_line(stripped):
            score += 2
        elif stripped.startswith("--") or stripped.startswith("/-") or stripped.startswith("-/"):
            score += 1
        else:
            score -= 1
    return score


def _looks_like_lean_line(line: str) -> bool:
    pattern = (
        r"^(import\s+\S+|open\s+\S+|namespace\b|section\b|end\b|"
        r"variable\b|variables\b|theorem\b|lemma\b|example\b|def\b|abbrev\b|"
        r"inductive\b|structure\b|class\b|instance\b|axiom\b|constant\b|"
        r"set_option\b|attribute\b|noncomputable\b|private\b|protected\b|"
        r"local\b|macro\b|syntax\b|notation\b|infix[lr]?\b|prefix\b|postfix\b|"
        r"@[A-Za-z_]|#(check|eval|print|reduce|guard)\b)"
    )
    return re.match(pattern, line) is not None


def _strip_known_llm_noise_lines(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower() in {
            "data collection is disabled.",
            "data collection is disabled",
        }:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


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
    if args.retriever == "none":
        return NullRetriever()
    premises: list[str] = []
    if args.premises is not None:
        if not args.premises.exists():
            raise RuntimeError(f"Premises file not found: {args.premises}")
        with args.premises.open("r", encoding="utf-8") as fh:
            premises = [line.rstrip("\n") for line in fh]
    else:
        source = str(getattr(args, "retriever_source", "local")).strip().lower() or "local"
        build_mode = str(getattr(args, "retriever_build", "auto")).strip().lower() or "auto"
        file_path = getattr(args, "file", None)
        project = getattr(args, "lean_project", None)
        if project is None and isinstance(file_path, Path):
            project = _find_lean_project_for_file(file_path)
        if project is None:
            if args.retriever == "simple":
                if getattr(args, "verbose", False):
                    print("[retriever] no Lean project found; using null retriever.")
                return NullRetriever()
            raise RuntimeError(
                "Automatic retrieval index requires a Lean project. "
                "Set --lean-project or provide --premises."
            )
        project = Path(project).expanduser().resolve()
        index_path = _resolve_retriever_index_path(args, project, source)
        if build_mode not in {"auto", "always", "never"}:
            build_mode = "auto"
        if build_mode == "always" or (build_mode == "auto" and not index_path.exists()):
            stats = build_premise_index(project, index_path, scope=source)
            if getattr(args, "verbose", False):
                print(
                    f"[retriever] indexed {stats.get('records', 0)} premises "
                    f"({source}) -> {index_path}"
                )
        if not index_path.exists():
            raise RuntimeError(
                f"Retriever index not found: {index_path}. "
                "Run `ulam index build` or set --retriever-build auto/always."
            )
        premises = load_index_premises(index_path)
        if not premises:
            raise RuntimeError(f"Retriever index is empty: {index_path}")
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


def _resolve_retriever_index_path(args: argparse.Namespace, project: Path, source: str) -> Path:
    path = getattr(args, "retriever_index", None)
    if path is None:
        path = Path(".ulam") / f"premises_{source}.jsonl"
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = project.expanduser().resolve() / path
    return path
