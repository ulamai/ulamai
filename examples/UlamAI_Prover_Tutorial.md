# UlamAI Prover Tutorial with Examples

This tutorial is a practical walkthrough for first-time users.

You will learn:

1. What to run first.
2. How to configure LLM providers (Codex recommended).
3. How to run `prove` in both Lean and `.tex` routes.
4. How to formalize `.tex` inputs, including `pol25.tex` (full informal proof).
5. How to inspect artifacts and resume runs.

## 0) What To Do First

From repo root:

```bash
python3 -m pip install -e .
python3 -m ulam --help
```

Then verify everything is wired correctly:

```bash
python3 -m ulam prove examples/Smoke.lean --theorem irrational_sqrt_two_smoke
```

If this runs, continue with LLM configuration and the richer examples.

## 1) Recommended Setup (Codex + LLM mode)

Codex is the recommended default for most users.

CLI auth:

```bash
ulam auth codex
```

TUI setup path:

1. Run `ulam`.
2. Open `Configure LLM`.
3. Select `Codex` provider.
4. Open `Settings` -> `Prover settings`.
5. Set `Default proof mode` to `llm`.
6. Set `Default prove output format` to `tex` if you want informal-first workflow.

CLI equivalent for explicit runs:

- Use `--llm codex_cli` on `prove` commands.
- Use `--prove-mode llm` when proving Lean files.

## 2) Workflow A: Prove to `.tex` (Informal Route)

Input statement file:

```text
examples/ProveTexPrimes.txt
```

Run:

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

Expected outputs:

- `proofs/infinitely_many_primes.tex`
- `runs/prove_tex/tex_.../state.json`
- `runs/prove_tex/tex_.../events.jsonl`
- `runs/prove_tex/tex_.../summary.json`

Resume a run:

```bash
python3 -m ulam prove \
  --theorem infinitely_many_primes \
  --output-format tex \
  --statement "$(cat examples/ProveTexPrimes.txt)" \
  --llm codex_cli \
  --tex-resume runs/prove_tex/<run_dir>
```

## 3) Workflow B: Formalize from `.tex`

There are two olympiad inputs for the same theorem:

- `examples/FormalizePolishOlympiad.tex`: statement-only version.
- `examples/pol25.tex`: full informal proof version (recommended for formalization quality).

Statement-only run:

```bash
python3 -m ulam formalize examples/FormalizePolishOlympiad.tex \
  --out examples/FormalizePolishOlympiad.lean \
  --proof-backend llm \
  --lean-backend dojo \
  --max-rounds 3 \
  --max-proof-rounds 1 \
  --artifacts-dir runs/formalize_olympiad_stmt
```

Full-proof run (recommended):

```bash
python3 -m ulam formalize examples/pol25.tex \
  --out examples/pol25.lean \
  --proof-backend llm \
  --lean-backend dojo \
  --max-rounds 5 \
  --max-proof-rounds 2 \
  --artifacts-dir runs/formalize_olympiad_full
```

Strict Lean typecheck variant (when you have a Lean project ready):

```bash
python3 -m ulam formalize examples/pol25.tex \
  --out examples/pol25.lean \
  --proof-backend llm \
  --lean-backend lsp \
  --lean-project /path/to/lean/project
```

## 4) Lean-file LLM Mode (Optional)

If you want LLM-mode proving directly on Lean files:

```bash
python3 -m ulam prove examples/Smoke.lean \
  --theorem irrational_sqrt_two_smoke \
  --prove-mode llm \
  --lean lsp \
  --llm codex_cli
```

This uses LLM typecheck loops for proof updates.

## 5) Artifact Inspection and Debug Loop

TeX proving artifacts:

```bash
ls -lah runs/prove_tex
```

Formalization artifacts:

```bash
ls -lah runs | rg formalize_
```

Inspect generated Lean output:

```bash
sed -n '1,200p' examples/pol25.lean
```

Useful follow-up commands:

```bash
python3 -m ulam checkpoint examples/pol25.lean --theorem <theorem_name> --strict
python3 -m ulam review --trace run.jsonl --file examples/pol25.lean --theorem <theorem_name>
```

## 6) Colab Version

Notebook path:

- `examples/UlamAI_Prover_Tutorial.ipynb`

Colab URL pattern (main branch):

```text
https://colab.research.google.com/github/ulamai/ulamai/blob/main/examples/UlamAI_Prover_Tutorial.ipynb
```
