# Examples

This folder contains runnable examples for the main UlamAI workflows.

## Start Here

1. Read `UlamAI_Prover_Tutorial.md` (full guide).
2. If you prefer notebooks, open `UlamAI_Prover_Tutorial.ipynb` in Colab.
3. Run the quick-start commands below.

## Files

- `Smoke.lean`: minimal Lean proving smoke test.
- `Formalize.tex`: tiny formalization toy example.
- `ProveTexPrimes.txt`: statement-only input for `prove --output-format tex`.
- `FormalizePolishOlympiad.tex`: olympiad statement-only formalization input.
- `pol25.tex`: same olympiad theorem with a full informal proof narrative (recommended formalization input).
- `UlamAI_Prover_Tutorial.md`: detailed tutorial (source of truth).
- `UlamAI_Prover_Tutorial.ipynb`: Colab-friendly runnable tutorial.

## Recommended Defaults

- LLM provider: `codex_cli` (recommended).
- Prove mode for Lean files: `llm`.
- Output format for informal proving: `tex`.

## Quick Commands

Smoke prove:

```bash
python3 -m ulam prove examples/Smoke.lean --theorem irrational_sqrt_two_smoke
```

Prove to TeX (infinitely many primes):

```bash
python3 -m ulam prove --theorem infinitely_many_primes --output-format tex \
  --statement "$(cat examples/ProveTexPrimes.txt)" \
  --llm codex_cli --tex-rounds 3 --tex-worker-drafts 2 --tex-judge-repairs 2 \
  --tex-replan-passes 2 --tex-artifacts-dir runs/prove_tex
```

Formalize olympiad statement-only file:

```bash
python3 -m ulam formalize examples/FormalizePolishOlympiad.tex \
  --out examples/FormalizePolishOlympiad.lean \
  --proof-backend llm --lean-backend dojo
```

Formalize olympiad full-proof file (recommended):

```bash
python3 -m ulam formalize examples/pol25.tex \
  --out examples/pol25.lean \
  --proof-backend llm --lean-backend dojo
```
