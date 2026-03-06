# Examples

This folder contains runnable examples for the main UlamAI workflows.

## Files

- `Smoke.lean`: minimal Lean proving smoke test.
- `Formalize.tex`: tiny formalization toy example.
- `ProveTexPrimes.txt`: statement-only input for `prove --output-format tex`.
- `FormalizePolishOlympiad.tex`: olympiad-style formalization input.
- `UlamAI_Prover_Tutorial.md`: step-by-step tutorial (source of truth).
- `UlamAI_Prover_Tutorial.ipynb`: Colab-friendly runnable version.

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

Formalize olympiad problem:

```bash
python3 -m ulam formalize examples/FormalizePolishOlympiad.tex \
  --out examples/FormalizePolishOlympiad.lean
```

For full walkthrough and artifact inspection, use:
- `examples/UlamAI_Prover_Tutorial.md`
- `examples/UlamAI_Prover_Tutorial.ipynb`
