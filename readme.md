# Ulam AI

A **truth-first**, reproducible, open-source **Lean 4 theorem prover CLI** that combines:

- **LLM-guided reasoning** (creative step proposals)
- **Lean verification** (zero hallucinations: only accepted if Lean checks)
- **Retrieval** (premise selection from mathlib / local repos)
- **Search + caching** (best-first / beam + transposition table)

Ulam AI is designed to plug into **Codex / Claude Code / Gemini CLI / Ollama** and produce **machine-checked Lean 4 proofs**.
Start here: [UlamAI Prover Tutorial](examples/UlamAI_Prover_Tutorial.md).

---

## Quickstart

With Homebrew:

```bash
brew tap ulamai/ulamai
brew install ulamai
```

From a fresh clone:

```bash
git clone https://github.com/ulamai/ulamai.git
cd ulamai
./install.sh
```

In the folder where you want to prove or formalize things:

```bash
ulam -lean
```

Then launch the interactive menu:

```bash
ulam
```

Suggested Codex model: `gpt-5.2-codex` (or `gpt-5.3-codex` if available).

Example for Prove with natural language guidance:

```text
prove that every PID is a UFD
```

---

## Modes
These modes apply to both `prove` and `formalize` workflows (you can set them via Settings or CLI flags).

Proof modes:
- `tactic`: LLM proposes the next tactic; LeanDojo checks each step. **Pros:** fast feedback, goal-state aware. **Cons:** brittle for long chains.
- `lemma`: LLM drafts a lemma plan; each lemma is proved sequentially. **Pros:** decomposes big proofs. **Cons:** depends on plan quality, more scaffolding.
- `llm`: LLM rewrites the Lean file; Lean CLI typechecks. **Pros:** handles multi-step edits, no Dojo required. **Cons:** slower, less guidance than goals, relies on LLM (requires Lean CLI).

Prove output formats:
- `lean` (default): current machine-checked Lean proving pipeline.
- `tex`: separate informal proving pipeline that runs a bounded planner-action loop over claim-graph proving, can split planner vs worker models, solves claims with worker drafts (serial by default, optional concurrent evaluation), maintains a persistent whiteboard + mutable `repo/` memory with planner CRUD + `[[wikilink]]` references, runs judge + adversarial verifier + domain checks, supports pass-based replan/backtrack when a plan stalls, and writes resumable per-run artifacts (`state.json`, `events.jsonl`, `summary.json`, `WHITEBOARD.md`, `repo/`) before composing a final `.tex` proof draft for later `formalize`.

Lean backends:
- `dojo`: Pantograph/LeanDojo server. **Pros:** goal-state access, tactic execution. **Cons:** extra install, toolchain pinning sensitivity.
- `cli`: `lake env lean` typecheck. **Pros:** simple, works with any toolchain. **Cons:** no goal-state feedback.
- `lsp`: Lean language server. **Pros:** goal/diagnostic loop without Pantograph, usable in `llm` and optional tactic/lemma search. **Cons:** still experimental for search parity vs Dojo.

---

## Status (v0.2.9)
This repo now contains a **working benchmark-ready proving/formalization pipeline** with reproducible reporting and optional Lean LSP loops:

- **v0.2.9 highlights:** informal `prove --output-format tex` now adds separate planner/worker model overrides, a bounded planner-action loop, and planner-managed repo CRUD + `[[wikilink]]` context expansion on top of the new persistent `WHITEBOARD.md` + `repo/` run memory and optional concurrent TeX worker evaluation.
- **Autop tactics** (aesop/simp/linarith/ring) as fallback during proof search
- **Axiom toggle** (axioms/constants allowed by default; disable with `--no-allow-axioms`)
- **Resume last formalization** in the menu + reuse prior artifacts
- **Per-lemma LaTeX proof snippets** injected into Lean comments and LLM prompts
- **Formalize proof search** now runs sequential tactic scripts (better multi-step chaining)
- **Scripted solver** (sequential tactics) enabled by default
- **Lemma-first planning** with automatic expansion on failure
- **Run summaries** appended to `.lean` on failed attempts
- **Settings** for solver choice and lemma limits
- **LLM-only mode** (Lean CLI typecheck, no Dojo)
- `ulam prove` and `ulam replay` commands
- Best-first search with beam cap + repair loop
- LLM adapters: OpenAI-compatible + Anthropic + Gemini + Ollama + CLI wrappers
- Lean runner: mock implementation + LeanDojo-v2/PyPantograph runner (external install required)
- Retrieval: token-overlap or embedding-based from a `--premises` file
- Trace logging to JSONL (`run.jsonl` by default)
- Interactive menu (`ulam`) for configuration + guided workflows

What is *not* implemented yet: robust value models and full autoformalization workflows.

Pipeline reference:
- `docs/pipeline.md`

---

## Install

Homebrew (recommended):

```bash
brew tap ulamai/ulamai
brew install ulamai
```

From a clone:

```bash
git clone https://github.com/ulamai/ulamai.git
cd ulamai
./install.sh
```

If your Python is externally managed (PEP‑668), you can force a user install:

```bash
ULAM_BREAK_SYSTEM_PACKAGES=1 ./install.sh
```

Then:

```bash
ulam --help
```

Interactive menu:

```bash
ulam
```

Login from CLI:

```bash
ulam auth codex
ulam auth claude
ulam auth gemini
```

Formalize a LaTeX document:

```bash
ulam formalize path/to/paper.tex --out path/to/paper.lean
```

Toy example:

```bash
ulam formalize examples/Formalize.tex --out examples/Formalize.lean
```

Tutorial with runnable examples:

- Markdown guide: `examples/UlamAI_Prover_Tutorial.md`
- Colab notebook: `examples/UlamAI_Prover_Tutorial.ipynb`
- Colab open link (main branch): `https://colab.research.google.com/github/ulamai/ulamai/blob/main/examples/UlamAI_Prover_Tutorial.ipynb`
- Includes both olympiad inputs: statement-only (`FormalizePolishOlympiad.tex`) and full-proof (`pol25.tex`).

Olympiad-style formalization example:

```bash
ulam formalize examples/FormalizePolishOlympiad.tex --out examples/FormalizePolishOlympiad.lean
```

Same olympiad theorem with full informal proof (recommended):

```bash
ulam formalize examples/pol25.tex --out examples/pol25.lean
```

Formalization options:
- `--no-equivalence` to skip statement equivalence checks.
- `--llm-check/--no-llm-check` to enable or disable semantic integrity checks.
- `--llm-check-timing end|mid+end` to choose when semantic checks run.
- `--llm-check-repairs N` to set semantic repair attempts.
- `--artifacts-dir` to store per-round artifacts (defaults to `runs/formalize_YYYYMMDD_HHMMSS`).
- `--proof-backend` (`tactic|lemma|llm|dojo`) to choose proof backend.
- `--lean-backend` (`dojo|cli|lsp`) to choose typecheck backend.
- `--segment` and `--segment-words` to formalize long TeX piece‑wise.
- Local declaration retrieval is enabled by default during formalize proof search;
  set `ULAM_FORMALIZE_LOCAL_RETRIEVER=0` to disable it.

One-line installer:

```bash
curl -fsSL https://raw.githubusercontent.com/ulamai/ulamai/main/install.sh | bash
```

Homebrew (tap):

```bash
brew tap ulamai/ulamai
brew install ulamai
```

Maintainers: the Homebrew tap is auto-updated on release. See `.github/workflows/update-homebrew-tap.yml` and `scripts/update_homebrew_tap.sh` (requires `TAP_PUSH_TOKEN` with push access to `ulamai/homebrew-ulamai`).

---

## Commands
Mock mode lets you smoke-test the CLI without Lean installed:

```bash
python3 -m ulam prove examples/Smoke.lean --theorem irrational_sqrt_two_smoke
```

Natural language guidance:

```bash
python3 -m ulam prove examples/Smoke.lean --theorem irrational_sqrt_two_smoke --instruction "Use a short automation tactic first."
```

Run TeX proving with replan/backtrack and artifacts:

```bash
python3 -m ulam prove --theorem infinitely_many_primes --output-format tex \
  --statement "There are infinitely many prime numbers." \
  --llm codex_cli --tex-rounds 3 --tex-judge-repairs 2 --tex-worker-drafts 2 \
  --tex-replan-passes 2 --tex-action-steps 10 \
  --tex-planner-model gpt-5.4 --tex-worker-model gpt-5.4-mini \
  --tex-artifacts-dir runs/prove_tex
```

Statement source used in that command:

```text
examples/ProveTexPrimes.txt
```

Resume a prior TeX proving run:

```bash
python3 -m ulam prove --theorem infinitely_many_primes --output-format tex \
  --statement "There are infinitely many prime numbers." \
  --llm codex_cli --tex-resume runs/prove_tex/<run_dir>
```

Verbose logs (LLM suggestions + tactic outcomes):

```bash
python3 -m ulam prove examples/Smoke.lean --theorem irrational_sqrt_two_smoke --verbose
```

Attach context files:

```bash
python3 -m ulam prove examples/Smoke.lean --theorem irrational_sqrt_two_smoke --context examples/Smoke.lean
```

Replay the run:

```bash
python3 -m ulam replay run.jsonl
```

Read-only checkpoint (typecheck/placeholders/axiom drift):

```bash
python3 -m ulam checkpoint examples/Smoke.lean --theorem irrational_sqrt_two_smoke --strict
```

Run review with next-step suggestions:

```bash
python3 -m ulam review --trace run.jsonl --file examples/Smoke.lean --theorem irrational_sqrt_two_smoke
```

Execute deterministic replay (re-run every tactic from the trace):

```bash
python3 -m ulam replay run.jsonl --execute --strict
```

Run the regression suite (mock by default):

```bash
python3 -m ulam bench --suite bench/regression.jsonl
# alias form also works
python3 -m ulam bench --suite regression
```

Run the suite with machine-readable reports:

```bash
python3 -m ulam bench --suite bench/regression.jsonl --report-json runs/bench/latest.json --report-markdown runs/bench/latest.md
```

Run with an inference profile (budgeted generate/rank/verify loop):

```bash
python3 -m ulam bench --suite internal_regression --llm codex_cli --openai-model gpt-5.3-codex --lean dojo --inference-profile balanced
# optional explicit overrides
python3 -m ulam bench --suite internal_regression --inference-profile verify --gen-k 8 --exec-k 3 --verify-level strict
```

Validate a benchmark suite before running:

```bash
python3 -m ulam bench-validate --suite bench/suites/internal_regression.jsonl
# alias form
python3 -m ulam bench-validate --suite internal_regression
```

Build a miniF2F suite from a local checkout:

```bash
python3 -m ulam bench-make-minif2f --root /path/to/miniF2F --split valid --out bench/suites/minif2f_valid.jsonl
python3 -m ulam bench-validate --suite bench/suites/minif2f_valid.jsonl
```

List known suite aliases and status:

```bash
python3 -m ulam bench-list-suites
```

Build fixed `regression100` from a larger source suite:

```bash
python3 -m ulam bench-make-regression100 \
  --source bench/suites/minif2f_valid.jsonl \
  --out bench/suites/regression100.jsonl \
  --size 100 \
  --seed 0
python3 -m ulam bench-validate --suite regression100
```

Compare two benchmark reports:

```bash
python3 -m ulam bench-compare --a runs/bench/a.json --b runs/bench/b.json --out-markdown runs/bench/compare.md
```

Compare with an automated parity gate (non-regression thresholds):

```bash
python3 -m ulam bench-compare \
  --a runs/bench/a.json \
  --b runs/bench/b.json \
  --gate \
  --max-solved-drop 0 \
  --max-success-rate-drop 0 \
  --max-semantic-pass-rate-drop 0 \
  --max-regression-rejection-rate-increase 0 \
  --max-median-time-increase-pct 25 \
  --max-planner-replan-triggers-increase 0 \
  --max-planner-cached-tactic-tries-drop 0
```

Notes:
- `--gate` now also checks comparability by default: same suite SHA256 and same inference profile/budgets.
- If you intentionally compare different suites or profile settings, pass `--allow-suite-mismatch` and/or `--allow-profile-mismatch`.

Run a reproducible benchmark campaign (timestamped artifacts + env snapshot):

```bash
scripts/run_bench_campaign.sh --suite bench/suites/internal_regression.jsonl -- --llm codex_cli --openai-model gpt-5.3-codex --lean dojo
```

Run a gated campaign against a baseline report (release/CI style):

```bash
scripts/run_bench_campaign.sh \
  --suite bench/suites/internal_regression.jsonl \
  --compare-to runs/bench_campaigns/baseline/report.json \
  --max-solved-drop 0 \
  --max-success-rate-drop 0 \
  --max-semantic-pass-rate-drop 0 \
  --max-regression-rejection-rate-increase 0 \
  --max-median-time-increase-pct 25 \
  --max-planner-replan-triggers-increase 0 \
  --max-planner-cached-tactic-tries-drop 0 \
  -- --llm codex_cli --openai-model gpt-5.3-codex --lean dojo
```

Notes:
- Campaign runs default to local repo code (`python3 -m ulam`) for version consistency.
- The wrapper now fails if report artifacts are missing or if report metadata version/commit does not match the repo checkout.
- When `--compare-to` is provided, the wrapper runs `bench-compare --gate` with strict non-regression defaults (including planner counters and profile/suite comparability checks).
- Use `--allow-profile-mismatch` and/or `--allow-suite-mismatch` only for intentional cross-profile or cross-suite comparisons.
- Bench report JSON/Markdown now include dataset/split breakdowns and top tags when suite rows provide that metadata.

Build a retrieval index from local project + mathlib declarations:

```bash
python3 -m ulam index build --project /path/to/lean-project --scope both --out .ulam/premises_both.jsonl
```

LeanDojo-v2 mode (real Lean, requires a Lean project and `sorry` placeholder):

```bash
python3 -m ulam prove path/to/File.lean --theorem MyTheorem --lean dojo --lean-project /path/to/lean-project
```

Lemma-first mode:

```bash
python3 -m ulam prove path/to/File.lean --theorem MyTheorem --prove-mode lemma
```

LLM-only mode (no Dojo, uses Lean CLI typecheck):

```bash
python3 -m ulam prove path/to/File.lean --theorem MyTheorem --prove-mode llm --llm-rounds 4
```

Formalize a .tex document:

```bash
python3 -m ulam formalize paper.tex --out paper.lean
```

Formalize with LLM-only proof attempts:

```bash
python3 -m ulam formalize paper.tex --proof-backend llm --lean-backend lsp
```

LLM-only proving with Lean LSP diagnostics:

```bash
python3 -m ulam prove path/to/File.lean --theorem MyTheorem --prove-mode llm --lean lsp
```

Informal TeX proving route (separate from Lean proving):

```bash
python3 -m ulam prove path/to/File.lean --theorem MyTheorem --output-format tex --tex-rounds 3 --tex-judge-repairs 2
```

From statement-only input (no Lean file):

```bash
python3 -m ulam prove --theorem MyTheorem --output-format tex --statement "For all n, ... "
```

TUI path: `Settings -> Prover settings -> Default prove output format (lean|tex)`.

TeX mode knobs:
- `--tex-out` explicit output path for the generated `.tex` draft.
- `--tex-rounds` max fixed orchestration rounds for claim solving.
- `--tex-worker-drafts` fixed worker candidates generated per claim per round.
- `--tex-concurrency/--no-tex-concurrency` toggle concurrent evaluation of TeX worker drafts (default: off).
- `--tex-judge-repairs` max consecutive rounds without accepted claim before composing best available draft.
- `--tex-replan-passes` max decomposition resets after a stalled pass.
- `--tex-action-steps` bounded planner-action steps for the informal solver.
- `--tex-planner-model` optional provider-model override for planning, judging, verification, and final composition.
- `--tex-worker-model` optional provider-model override for claim-draft workers.

TeX mode execution model:
- a planner action loop chooses whether to `plan`, `solve`, `write_memory`, `compose`, or `give_up`,
- the planner emits a claim graph (`claims`, dependencies, assumptions, required facts) and can attach worker guidance plus repo reads,
- workers draft each claim, with optional parallel evaluation when `--tex-concurrency` is enabled,
- each claim is gated by judge + adversarial verifier + domain checker,
- a persistent `WHITEBOARD.md` plus mutable `repo/` memory items are updated across rounds/passes, can be created/updated/deleted by the planner, and support `[[wikilink]]` expansion in later prompts,
- accepted claims are composed into the final theorem proof.
- the action budget is bounded by the settings above; when it is exhausted, the solver composes the best available draft.

Install the CLI entrypoint if you want `ulam` directly:

```bash
pip install -e .
```

---

## Lean + LeanDojo Setup (real proofs)
UlamAI runs real proofs through LeanDojo, which needs a Lean project with Mathlib.

One-command setup (interactive):

```bash
ulam -lean
# or
python3 -m ulam -lean
```

Use one of the exact command forms above.

Non-interactive:

```bash
ulam lean-setup --dir ./ulam-lean --yes
```

Flags:
- `--skip-elan` to skip installing Lean.
- `--no-build` to skip `lake build`.
- `--no-dojo` to skip LeanDojo/Pantograph.
- `--no-config` to avoid writing `.ulam/config.json`.
- `--pip-timeout` / `--pip-retries` to handle slow PyPI downloads.
- `--toolchain` to pin a specific Lean toolchain (otherwise uses Mathlib’s default).
- `--use-mathlib-toolchain` to keep the toolchain from the Mathlib template even if Pantograph differs.
- `--lakefile-lean` to generate a `lakefile.lean` mirror of `lakefile.toml`.

Note: Pantograph is anchored to a specific Lean toolchain (from its `src/lean-toolchain`). UlamAI will
align the Mathlib project to that toolchain when possible.

Install Lean + Lake (macOS/Linux):

```bash
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
source $HOME/.elan/env
```

Create a Mathlib project:

```bash
lake +leanprover-community/mathlib4:lean-toolchain new MyMathlibProject math
cd MyMathlibProject && lake build
```

Install LeanDojo + Pantograph:

```bash
pip install lean-dojo-v2
pip install git+https://github.com/stanford-centaur/PyPantograph
```

Then set `ULAM_LEAN_PROJECT` or configure the Lean project path in the menu.

---

## LLM Configuration

OpenAI-compatible (default):
- `ULAM_OPENAI_API_KEY`
- `ULAM_OPENAI_BASE_URL` (default `https://api.openai.com`)
- `ULAM_OPENAI_MODEL` (default `gpt-4.1`)
- TUI: `1. Configure LLM` -> `5. Custom API (Mistral/Z.AI/Qwen/DeepSeek/Kimi)` to save provider label + API key + base URL + model.

OpenAI (ChatGPT subscription) login:
- Run `ulam auth codex` (or `codex login`) and UlamAI can import credentials from `~/.codex/auth.json`.
- This is the same flow used by the official Codex CLI (ChatGPT sign‑in creates a key automatically).

Codex CLI provider (subscription):
- Use `codex login` and set `--llm codex_cli` (no API key required).

Ollama:
- `ULAM_OLLAMA_BASE_URL` (default `http://localhost:11434`)
- `ULAM_OLLAMA_MODEL` (default `llama3.1`)
How to set it up (example with Llama 3.1 8B):
```bash
# 1) Install Ollama
brew install ollama  # or follow https://ollama.com/download

# 2) Start the server
ollama serve

# 3) Pull a model
ollama pull llama3.1:8b

# 4) Point Ulam to it (optional; defaults shown)
export ULAM_OLLAMA_BASE_URL="http://localhost:11434"
export ULAM_OLLAMA_MODEL="llama3.1:8b"
```

Claude (Anthropic):
- `ULAM_ANTHROPIC_API_KEY` or `ULAM_ANTHROPIC_SETUP_TOKEN`
- `ULAM_ANTHROPIC_BASE_URL` (default `https://api.anthropic.com`)
- `ULAM_ANTHROPIC_MODEL` (default `claude-3-5-sonnet-20240620`)

Claude Code CLI provider (subscription):
- Run `ulam auth claude` (or `claude setup-token`) and set `--llm claude_cli` (no API key required).

Gemini API:
- `ULAM_GEMINI_API_KEY` (or `GEMINI_API_KEY`)
- `ULAM_GEMINI_BASE_URL` (default `https://generativelanguage.googleapis.com/v1beta/openai`)
- `ULAM_GEMINI_MODEL` (default `gemini-3.1-pro-preview`)

Gemini CLI provider (subscription/login):
- Run `ulam auth gemini` and complete OAuth in your browser (no API key required).
- Ulam first uses its built-in browser+callback flow and automatically falls back to Gemini CLI native login.
- If your environment is unusual and auto-discovery still fails, set:
  - `ULAM_GEMINI_OAUTH_CLIENT_ID`
  - `ULAM_GEMINI_OAUTH_CLIENT_SECRET`
  - or `ULAM_GEMINI_OAUTH2_JS` (path to Gemini CLI `oauth2.js`)

Embeddings (for retrieval):
- `ULAM_EMBED_API_KEY` (defaults to `ULAM_OPENAI_API_KEY`)
- `ULAM_EMBED_BASE_URL` (defaults to `ULAM_OPENAI_BASE_URL`)
- `ULAM_EMBED_MODEL` (default `text-embedding-3-small`)

Premises file format:
- `--premises path/to/premises.txt`
- One premise per line (e.g., `lemma_name : statement`)

Retrievers:
- `--retriever simple` (token overlap)
- `--retriever embedding` (OpenAI-compatible embeddings)
- `--retriever none`
- `--retriever-k N` to control how many premises are injected per state (default `8`)
- If `--premises` is omitted, Ulam can auto-index declarations from Lean sources:
  - `--retriever-source local|mathlib|both`
  - `--retriever-build auto|always|never`
  - `--retriever-index path/to/index.jsonl` (default `.ulam/premises_<source>.jsonl` in project)
  - Build explicitly with `ulam index build ...`

Replay metadata:
- Each run trace `*.jsonl` now writes a sidecar `*.meta.json` with pinned environment data
  (Lean backend, project path, toolchain, mathlib commit/rev, file hash, and run config).
- `ulam replay ... --execute --strict` checks and replays against that metadata.

Solver strategy:
- `--solver search` for best-first tactic search
- `--solver script` for sequential script mode
- `--solver portfolio` to run script-first then best-first fallback

Menu config file:
- Stored at `.ulam/config.json` by default (override with `ULAM_CONFIG` or `ULAM_CONFIG_DIR`).
- If no provider credentials are set, the menu will prompt you to configure them before proving.

---

## Troubleshooting

- **LLM‑only mode fails to typecheck:** make sure `lean`/`lake` is on PATH and the file is inside a Lean project (has `lakefile.lean` or `lean-toolchain`).
- **LeanDojo mismatch errors:** run `ulam -lean` in your project folder or re-run `ulam lean-setup` to align the toolchain.
- **Pantograph missing after Python/Homebrew upgrades:** run `ulam -lean` to reinstall LeanDojo/Pantograph. Ulam also attempts one-time auto-install on demand (disable with `ULAM_AUTO_INSTALL_PANTOGRAPH=0`).
- **Codex/Claude/Gemini CLI hangs:** set `LLM request timeout` in Settings or keep it `0` and use heartbeat logs to verify it’s running.

---

## Product description

### What Ulam AI is
Ulam AI is a CLI tool that:
1) opens a Lean goal (from a theorem in a file or a snippet),
2) repeatedly asks an LLM for a **single next action** (tactic line or small lemma),
3) executes it in Lean,
4) uses errors as feedback to **repair** and **backtrack**,
5) returns a final verified Lean proof (or a replayable failure trace).

### What Ulam AI is *not* (yet)
- Not an IDE replacement (but it can generate patches you apply in your editor)
- Not a fully-fledged RL system on day 1
- Not “autoformalize any paper” out of the gate (that’s a later module)

---

## Core principles

- **Verified-first:** trust Lean, not the model.
- **Search backbone:** don’t bet everything on one-shot prompting; use best-first/beam.
- **Reproducibility:** pin toolchain + mathlib commit + timeouts + seeds; log everything.
- **Small, local lemmas:** encourage `have` steps that are easy to check.
- **Exploit automation:** `simp`, `aesop`, `linarith`, `ring`, `norm_num`, etc.

---

## Architecture (MVP)

### Components
- **Lean runner (stateful):** interacts with Lean 4 / mathlib and applies tactics
- **LLM policy:** proposes candidate next tactics (k=8–16 per state)
- **Retriever:** selects relevant premises (top-k) from mathlib/local repo
- **Search engine:** best-first/beam search over proof states
- **Cache (current):** transposition table: `state_key -> best_seen_score` (avoid re-exploring)
- **Repair loop:** on tactic failure, ask LLM to fix syntax/type mismatch, bounded retries

### LeanDojo-v2 runner notes
- Requires LeanDojo-v2 + PyPantograph installed.
- Assumes the target theorem contains a `sorry` placeholder.
- The CLI loads the Lean file and selects the goal corresponding to that `sorry`.
- Use `--lean-project` (or `ULAM_LEAN_PROJECT`) to point at the Lean project root.

### Why best-first/beam first (before MCTS)
Lean execution time dominates. Best-first + caching + transpositions delivers a strong baseline
fast and provides clear debugging signals. MCTS/MCGS can be layered later.

---

## Roadmap

### v0.1 — Core baseline (shipped in v0.1.x)
- LeanDojo-v2 runner for proof-state execution
- Embedding-based retrieval (mathlib + local repo)
- Step cache `(state_key, tactic) -> result`
- Deterministic replay baseline with pinned metadata + strict execute/alignment checks (hardening ongoing)
- Minimal regression suite (`ulam bench`)

### v0.2 — “Feels powerful” baseline
- Guardrail budget profiles (`fast|balanced|strict`) for both informal `.tex` and formal `formalize` tracks, exposed in TUI settings to trade off speed vs. verification depth.
- Optional Lean LSP backend track for interactive goal/diagnostic loops in LLM/formalize mode (`Dojo` remains default for tactic search until parity gate).
- Define parity gate for any LSP promotion: runner `start/apply` semantics, replay/cache stability, and non-regression on internal + miniF2F slice.
- Better state canonicalization (stable hashing)
- Better scoring heuristic / lightweight value model (non-RL)
- More robust retrieval formatting (names + one-line statements)
- Stronger action constraints and cost controls (e.g., `aesop` budgeted)
- Regression suite management (`ulam bench-list-suites`, fixed-suite builder, and `ulam bench --suite regression100` after generation)
- Execution plan for next patch release: `docs/v0.2.1_plan.md`

Potential formalization pipeline (later v0.2.x, not active yet):
- Add pass-based replan/backtrack for `formalize` at declaration-graph level (beyond round-local repairs).
- Upgrade LLM proof filling from single-candidate loops to worker-candidate + judge/verifier selection.
- Move LLM `formalize` edits to declaration/body-scoped patches (reduce full-file collateral regressions).
- Add stateful formalization artifacts (`state.json`, `events.jsonl`, `summary.json`) for deterministic resume/debugging.
- Add dedicated formalization regression tests (stalls, equivalence repair behavior, resume determinism, guardrail non-regression).

### v0.3 — SFT training loop
- (Both tracks: informal `.tex` + formal `formalize`) Dependency-first proving loop: decompose into a concept/claim or declaration DAG first, then solve bottom-up.
- (Both tracks: informal `.tex` + formal `formalize`) Live grounding retrieval: force term/definition grounding from current Mathlib/local sources before drafting, so models don’t invent symbols/assumptions.
- (Both tracks: informal `.tex` + formal `formalize`) Compiler/typecheck-in-the-loop reflection at each node, not only at the end.
- (Both tracks: informal `.tex` + formal `formalize`) Semantic gate beyond syntax: check whether generated claims/statements match intended meanings (especially hidden conditions and weaker-theorem drift).
- (Both tracks: informal `.tex` + formal `formalize`) Reliability over forced output: add an explicit abstain/no-output or partial-safe outcome when gates fail, instead of always composing a final proof/file.
- (Both tracks: informal `.tex` + formal `formalize`) Autonomy protocol artifacts: keep fixed verification/extraction prompts, full transcripts, and an autonomy card in reports.
- Trace extraction into JSONL
- Train tactic policy (SFT) on mathlib traces
- Optional “proofstep dataset” for your chosen model family
- Evaluate on miniF2F slice + internal suite

### v0.4 — Search upgrades
- Proof-state graph reuse improvements
- MCGS/MCTS option behind a flag
- Learned value model to guide search (trained from traces)
- Parallel rollouts (Ray or multiprocessing)

### v0.5 — RL (only after stable trajectories)
- On-policy fine-tuning (PPO/GRPO-style) on successful+near-miss trajectories
- Reward shaping that correlates with solved proofs (avoid fragile heuristics)
- Curriculum scheduling (easy → harder)

### v1.0 — Autoformalization module (separate product surface)
- PDF/text → candidate formal statements
- Semantic checks (avoid “formalization drift”)
- Iterative type-check repair + theorem retrieval scaffolding
- Output: Lean files + proof attempts + TODO gaps

---

## Target theorem(s) for the MVP

### Smoke test target (retrieval allowed)
This validates end-to-end retrieval + tactic selection + Lean checking.

```lean
import Mathlib.NumberTheory.Real.Irrational

theorem irrational_sqrt_two_smoke : Irrational (Real.sqrt 2) := by
  simpa using irrational_sqrt_two
