# UlamAI Prover Tutorial with Examples

This is a practical, beginner-friendly guide.

If you only follow one file, follow this one.

## What You Will Do

1. Install UlamAI.
2. Install Lean tooling (`ulam -lean` / `ulam lean-setup`).
3. Run a small verify/prove check on Lean (`Smoke.lean`).
4. Run informal proving to `.tex` for: "There are infinitely many prime numbers."
5. Formalize the Polish olympiad theorem from two inputs:
   - statement-only input
   - full informal proof input (`pol25.tex`, recommended)
6. Learn the same flows in Terminal UI (what to click and what each step does).

## 0) Install UlamAI

### Option A: Homebrew (recommended for most users)

```bash
brew tap ulamai/ulamai
brew install ulamai
ulam --help
```

What this gives you:
- `ulam` command available globally.
- Easy upgrades via Homebrew.

### Option B: Local editable install (dev workflow)

From repo root:

```bash
python3 -m pip install -e .
python3 -m ulam --help
```

## 1) Install Lean + Mathlib + LeanDojo

Run one of these (same setup flow):

```bash
ulam -lean
```

or

```bash
ulam lean-setup
```

If you want default non-interactive setup:

```bash
ulam -lean --yes
```

What `--yes` means:
- It auto-accepts prompts and runs with defaults.
- Use it for quick setup (CI/Colab/automation).
- It is optional. If you want to choose options manually, run `ulam -lean` without `--yes`.

In simple terms, this command does the heavy lifting for you:
- installs Lean via `elan` (if missing),
- creates (or reuses) a Lean project,
- installs LeanDojo/Pantograph,
- runs `lake build`,
- saves detected Lean project path into `.ulam/config.json`.

## 2) Configure LLM (Codex Recommended)

### CLI auth (quick)

```bash
ulam auth codex
```

### TUI path

1. Run `ulam`.
2. Click `1. Configure LLM`.
3. Choose provider `OpenAI`.
4. Choose auth method `Sign in with ChatGPT (Codex CLI)`.
5. Select model (recommended: `gpt-5.2-codex` / `gpt-5.3-codex` if available).

Why this is recommended:
- Strong default quality for both proving and judging loops.
- Easy setup if you already use ChatGPT/Codex CLI.

## 3) Workflow A: Verify/Prove a Lean Theorem

### Problem used

The example theorem in `examples/Smoke.lean` is:

```lean
theorem irrational_sqrt_two_smoke : Irrational (Real.sqrt 2) := by
  simpa using irrational_sqrt_two
```

### CLI run

```bash
python3 -m ulam prove examples/Smoke.lean \
  --theorem irrational_sqrt_two_smoke \
  --prove-mode llm \
  --lean lsp \
  --llm codex_cli
```

What this does:
- opens the Lean file,
- targets the named theorem,
- uses LLM-mode prove loop,
- uses Lean LSP typecheck diagnostics.

### Terminal UI (same workflow)

1. Run `ulam`.
2. Click `2. Prove with natural language guidance`.
3. At `Enter guidance for the prover`, write optional guidance (or leave minimal).
4. At `Lean file path (optional)`, enter `examples/Smoke.lean`.
5. At `Output format (lean|tex)`, choose `lean`.
6. At `Proof mode (tactic|lemma|llm)`, choose `llm`.
7. At `Theorem name`, enter `irrational_sqrt_two_smoke`.
8. Run.

What each important choice means:
- `Output format=lean`: machine-checked Lean proof path.
- `Proof mode=llm`: iterative LLM proof-edit + Lean typecheck loop.

## 4) Workflow B: Prove to `.tex` (Informal First)

In this workflow, we are not asking Lean to verify the theorem directly yet. Instead, we run Ulam in `--output-format tex` so the model builds a structured informal proof draft that a human can read, edit, and reuse for later formalization.

The prime-number statement is a clean example because most users already know the theorem, so it is easy to judge output quality quickly. You should focus on whether the generated `.tex` is logically coherent, clearly structured, and backed by good artifact logs (`state.json`, `events.jsonl`, `summary.json`) that let you resume or debug.

### Input statement (exact)

File: `examples/ProveTexPrimes.txt`

```text
Prove that there are infinitely many prime numbers.
```

### CLI run

```bash
python3 -m ulam prove \
  --theorem infinitely_many_primes \
  --output-format tex \
  --statement "$(cat examples/ProveTexPrimes.txt)" \
  --llm codex_cli \
  --tex-rounds 3 \
  --tex-worker-drafts 2 \
  --tex-judge-repairs 2 \
  --tex-replan-passes 2 \
  --tex-artifacts-dir runs/prove_tex
```

Outputs to inspect:
- `proofs/infinitely_many_primes.tex`
- `runs/prove_tex/tex_.../state.json`
- `runs/prove_tex/tex_.../events.jsonl`
- `runs/prove_tex/tex_.../summary.json`

Resume command:

```bash
python3 -m ulam prove \
  --theorem infinitely_many_primes \
  --output-format tex \
  --statement "$(cat examples/ProveTexPrimes.txt)" \
  --llm codex_cli \
  --tex-resume runs/prove_tex/<run_dir>
```

### Terminal UI (same workflow)

1. Run `ulam`.
2. Click `2. Prove with natural language guidance`.
3. Leave `Lean file path (optional)` empty.
4. Set `Output format (lean|tex)` to `tex`.
5. Set `Proof mode (tactic|lemma|llm)` to `llm`.
6. Set `Theorem name` to `infinitely_many_primes`.
7. Paste statement:
   - `Prove that there are infinitely many prime numbers.`
8. Run.

What this route is for:
- Generate an informal proof draft in `.tex` first.
- Useful when you want a human-readable proof artifact before formalization.

## 5) Workflow C: Formalize the Polish Olympiad Problem

In this workflow, we convert natural-language math into Lean declarations and proof attempts. The goal is to measure how well Ulam can map informal mathematical reasoning into machine-checkable structure, with iterative repair when the first draft is incomplete.

We use two inputs for the same theorem on purpose: a short statement-only version and a full informal-proof version. This lets you see the practical difference between \"minimum context\" formalization and \"rich context\" formalization, and why the richer `pol25.tex` input is usually a stronger starting point.

You now have two files for the same theorem.

### Statement-only input

File: `examples/FormalizePolishOlympiad.tex`

Problem statement:

> Given positive integers $k, m, n, p$ such that $p = 2^{2^n} + 1$, $p$ is a prime number, and $2^k - m$ is divisible by $p$. Prove that there exists a positive integer $\ell$ such that the number $2^\ell - m$ is divisible by $p^2$.

CLI run:

```bash
python3 -m ulam formalize examples/FormalizePolishOlympiad.tex \
  --out examples/FormalizePolishOlympiad.lean \
  --proof-backend llm \
  --lean-backend dojo \
  --max-rounds 3 \
  --max-proof-rounds 1 \
  --artifacts-dir runs/formalize_olympiad_stmt
```

### Full informal proof input (recommended)

File: `examples/pol25.tex`

This is the same theorem, but with a full informal proof narrative.

CLI run:

```bash
python3 -m ulam formalize examples/pol25.tex \
  --out examples/pol25.lean \
  --proof-backend llm \
  --lean-backend dojo \
  --max-rounds 5 \
  --max-proof-rounds 2 \
  --artifacts-dir runs/formalize_olympiad_full
```

Why `pol25.tex` is better:
- more proof structure,
- more intermediate reasoning for the model to map into Lean declarations,
- typically better starting point than statement-only formalization.

### Terminal UI (same formalize workflow)

1. Run `ulam`.
2. Click `3. Formalize .tex to Lean`.
3. At `.tex path`, enter one of:
   - `examples/FormalizePolishOlympiad.tex` (statement-only), or
   - `examples/pol25.tex` (full proof, recommended).
4. At `Output .lean path`, use suggested or set explicit path.
5. Run.

What happens under the hood:
- Ulam segments/parses TeX,
- drafts Lean declarations,
- runs typecheck/repair loop according to your formalize settings,
- writes artifacts under `runs/formalize_*`.

Resume in TUI:
- from main menu click `4. Resume last formalization`.

## 6) Suggested Settings (TUI)

Go to `5. Settings` and set:

1. `Default proof mode` -> `llm`
2. `Default prove output format` -> `tex` (if you prefer informal-first) or `lean`
3. `Formalize proof mode` -> `llm`
4. `Formalize typecheck backend` ->
   - `lsp` if Lean project is configured and you want strict checks,
   - `dojo` for broader compatibility.

## 7) Quick Troubleshooting

- "Lean project not detected": run `ulam -lean --yes` first, or provide `--lean-project`.
- LLM not configured: run `ulam auth codex` or TUI `Configure LLM`.
- Long runs: check artifact folders (`state.json`, `events.jsonl`) to confirm progress.

## 8) Colab Version

Notebook path:
- `examples/UlamAI_Prover_Tutorial.ipynb`

Direct Colab URL pattern:

```text
https://colab.research.google.com/github/ulamai/ulamai/blob/main/examples/UlamAI_Prover_Tutorial.ipynb
```
