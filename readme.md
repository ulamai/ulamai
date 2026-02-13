# Ulam Prover

A **truth-first**, reproducible, open(-ish) **Lean 4 theorem prover CLI** that combines:

- **LLM-guided reasoning** (creative step proposals)
- **Lean verification** (zero hallucinations: only accepted if Lean checks)
- **Retrieval** (premise selection from mathlib / local repos)
- **Search + caching** (best-first / beam + transposition table)

Ulam Prover is designed to plug into **Codex / Claude Code / Ollama** (or any OpenAI-compatible endpoint) and produce **machine-checked Lean 4 proofs**.

---

## Status (v0.1 scaffold)
This repo contains a **first working scaffold** of the CLI and search loop. It is intentionally thin but runnable:

- `ulam prove` and `ulam replay` commands
- Best-first search with beam cap + repair loop
- LLM adapters: OpenAI-compatible + Ollama + mock
- Lean runner: mock implementation + LeanDojo-v2/PyPantograph runner (external install required)
- Retrieval: token-overlap or embedding-based from a `--premises` file
- Trace logging to JSONL (`run.jsonl` by default)
- Interactive menu (`ulam`) for configuration + guided workflows

What is *not* implemented yet: toolchain pinning, robust value models, and full autoformalization workflows.

Pipeline reference:
- `docs/pipeline.md`

---

## Install

From a clone:

```bash
./install.sh
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
```

Formalize a LaTeX document:

```bash
ulam formalize path/to/paper.tex --out path/to/paper.lean
```

Toy example:

```bash
ulam formalize examples/Formalize.tex --out examples/Formalize.lean
```

Formalization options:
- `--no-equivalence` to skip statement equivalence checks.
- `--artifacts-dir` to store per-round artifacts (defaults to `runs/formalize_YYYYMMDD_HHMMSS`).

One-line installer (once the repo is public, replace `<ORG>/<REPO>`):

```bash
curl -fsSL https://raw.githubusercontent.com/<ORG>/<REPO>/main/install.sh | bash
```

Homebrew (planned; will require a tap repo):

```bash
brew tap <ORG>/tap
brew install ulam
```

---

## Quickstart (local)
Mock mode lets you smoke-test the CLI without Lean installed:

```bash
python -m ulam prove examples/Smoke.lean --theorem irrational_sqrt_two_smoke
```

Natural language guidance:

```bash
python -m ulam prove examples/Smoke.lean --theorem irrational_sqrt_two_smoke --instruction "Use a short automation tactic first."
```

Verbose logs (LLM suggestions + tactic outcomes):

```bash
python -m ulam prove examples/Smoke.lean --theorem irrational_sqrt_two_smoke --verbose
```

Attach context files:

```bash
python -m ulam prove examples/Smoke.lean --theorem irrational_sqrt_two_smoke --context examples/Smoke.lean
```

Replay the run:

```bash
python -m ulam replay run.jsonl
```

Run the regression suite (mock by default):

```bash
python -m ulam bench --suite bench/regression.jsonl
```

LeanDojo-v2 mode (real Lean, requires a Lean project and `sorry` placeholder):

```bash
python -m ulam prove path/to/File.lean --theorem MyTheorem --lean dojo --lean-project /path/to/lean-project
```

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
align the Mathlib project to that toolchain when possible. citeturn5search4

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

Codex (ChatGPT subscription) login:
- Run `codex login` and UlamAI can import credentials from `~/.codex/auth.json`.
- This is the same flow used by the official Codex CLI (ChatGPT sign‑in creates a key automatically).

Codex CLI provider (subscription):
- Use `codex login` and set `--llm codex_cli` (no API key required).

Ollama:
- `ULAM_OLLAMA_BASE_URL` (default `http://localhost:11434`)
- `ULAM_OLLAMA_MODEL` (default `llama3.1`)

Claude (Anthropic):
- `ULAM_ANTHROPIC_API_KEY` or `ULAM_ANTHROPIC_SETUP_TOKEN`
- `ULAM_ANTHROPIC_BASE_URL` (default `https://api.anthropic.com`)
- `ULAM_ANTHROPIC_MODEL` (default `claude-3-5-sonnet-20240620`)

Claude Code CLI provider (subscription):
- Run `claude setup-token` and set `--llm claude_cli` (no API key required).

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

Menu config file:
- Stored at `.ulam/config.json` by default (override with `ULAM_CONFIG` or `ULAM_CONFIG_DIR`).
- If no provider credentials are set, the menu will prompt you to configure them before proving.

---

## Product description

### What Ulam Prover is
Ulam Prover is a CLI tool that:
1) opens a Lean goal (from a theorem in a file or a snippet),
2) repeatedly asks an LLM for a **single next action** (tactic line or small lemma),
3) executes it in Lean,
4) uses errors as feedback to **repair** and **backtrack**,
5) returns a final verified Lean proof (or a replayable failure trace).

### What Ulam Prover is *not* (yet)
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

### v0.1 — Real LeanDojo wiring
- Wire LeanDojo-v2 runner (proof-state execution)
- Embedding-based retrieval (mathlib + local repo)
- Step cache `(state_key, tactic) -> result`
- Deterministic replay with pinned toolchain + mathlib commit
- Minimal regression suite

### v0.2 — “Feels powerful” baseline
- Better state canonicalization (stable hashing)
- Better scoring heuristic / lightweight value model (non-RL)
- More robust retrieval formatting (names + one-line statements)
- Stronger action constraints and cost controls (e.g., `aesop` budgeted)
- Regression suite management (`ulam bench --suite regression100`)

### v0.3 — SFT training loop
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
