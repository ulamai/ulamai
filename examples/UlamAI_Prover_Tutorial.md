# UlamAI Prover Tutorial with Examples

This tutorial shows the end-to-end flow on repository examples:

1. Lean proving smoke test (`Smoke.lean`)
2. Informal proving to `.tex` (infinitely many primes)
3. Formalization from `.tex` (Polish olympiad problem)
4. Artifact inspection and resume

## 0) Setup

From repo root:

```bash
python3 -m pip install -e .
python3 -m ulam --help
```

Optional: configure your LLM provider in TUI (`ulam` -> `Configure LLM`) or `.ulam/config.json`.

## 1) Lean Proving Smoke Test

```bash
python3 -m ulam prove examples/Smoke.lean \
  --theorem irrational_sqrt_two_smoke
```

This verifies the basic prove loop wiring.

## 2) Prove to `.tex` (Informal Route)

Statement source:

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

## 3) Formalize from `.tex`

Input file:

```text
examples/FormalizePolishOlympiad.tex
```

Basic run:

```bash
python3 -m ulam formalize examples/FormalizePolishOlympiad.tex \
  --out examples/FormalizePolishOlympiad.lean
```

LLM-focused loop (optional):

```bash
python3 -m ulam formalize examples/FormalizePolishOlympiad.tex \
  --out examples/FormalizePolishOlympiad.lean \
  --proof-backend llm \
  --lean-backend dojo \
  --max-rounds 3 \
  --max-proof-rounds 1 \
  --artifacts-dir runs/formalize_olympiad
```

Note: with `--lean-backend dojo` and no detected Lean project, typecheck is skipped. For strict Lean checking, provide a project and use `--lean-backend lsp` or `--lean-backend cli`.

## 4) Inspect Artifacts

TeX proving artifacts:

```bash
ls -lah runs/prove_tex
```

Formalization artifacts:

```bash
ls -lah runs | rg formalize_
```

Inspect final Lean output:

```bash
sed -n '1,160p' examples/FormalizePolishOlympiad.lean
```

## 5) Colab Version

For a notebook version of this tutorial, open:

- `examples/UlamAI_Prover_Tutorial.ipynb`

If hosted on GitHub main branch, direct Colab URL pattern:

```text
https://colab.research.google.com/github/ulamai/ulamai/blob/main/examples/UlamAI_Prover_Tutorial.ipynb
```
