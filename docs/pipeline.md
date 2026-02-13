# UlamAI Pipeline (Detailed)

This document describes the end-to-end pipeline for proof search and autoformalization in UlamAI, including the iterative loops that interact with Lean/LeanDojo.

## Goals

- Make the system **truth-first**: every step is validated by Lean.
- Support **robust recovery** from missing details in informal proofs.
- Make **runs reproducible** (toolchain, seeds, logs, and caches).

## Core Components

- **LLM policy**: Proposes tactic steps or Lean code fragments.
- **LeanDojo runner**: Executes tactics and returns proof states.
- **Retriever**: Fetches relevant mathlib/local premises.
- **Search engine**: Best-first / beam with transposition table.
- **Caches**: Step cache and state cache to avoid duplicate Lean calls.
- **Trace logger**: JSONL log of all states, actions, and outcomes.

## Proof Search Pipeline (Lean Proving)

### 1. Input

- Lean file path + theorem name.
- Theorem contains a `sorry` to open a goal.
- Optional natural-language instruction.
- Optional context files.

### 2. Initialize

- LeanDojo loads the file and returns the target proof state.
- A root search node is created for this state.

### 3. Iterate (Search Loop)

For each proof state in the frontier:

1. Retrieve premises (top-k).
2. Prompt the LLM for k candidate tactics.
3. Execute each tactic in Lean via LeanDojo.
4. Update caches and log results.
5. If solved, emit the proof and stop.
6. If not solved, enqueue the new state (beam capped).
7. If tactic fails, request a repair step and retry.

### 4. Output

- Final Lean script (tactic list).
- JSONL trace for replay.

## Formalization Pipeline (TeX → Lean) With Recursive Loops

Formalization is inherently incomplete: informal proofs omit details. We treat formalization as a **multi-pass, Lean-validated process**, not a one-shot prompt.

### Phase A: Ingest + Segment

1. Parse LaTeX into logical chunks:
   - Definitions
   - Lemmas
   - Theorems
   - Proof sketches
2. Extract **named entities** and **symbols**.
3. Build a document graph with dependencies.

### Phase B: Statement Drafting (Lean Skeleton)

1. For each definition/lemma/theorem:
   - Generate a Lean signature with `sorry` proofs.
2. If types are unknown:
   - Insert placeholders and annotate with TODO comments.
3. Run Lean to typecheck the skeleton.
4. Capture errors and iterate:
   - Missing imports
   - Undeclared constants
   - Type mismatches
   - Notation issues

This loop continues until the **file typechecks** with `sorry` proofs.

### Phase C: Proof Reconstruction (Per Lemma)

For each lemma with `sorry`:

1. Open its goal in LeanDojo.
2. Use the proof-search pipeline to fill steps.
3. If proof fails, attempt:
   - Retrieval of similar lemmas.
   - Reformulation of the lemma statement.
   - Generation of auxiliary lemmas.
4. If still failing, leave `sorry` and log a TODO.

### Phase D: Semantic Alignment (Recursive)

Because informal proofs skip details, we need alignment checks:

1. **Statement equivalence check**:
   - LLM judges whether Lean statement matches informal statement.
   - If mismatch, enqueue a **repair task** for that declaration.
   - Repairs run in a queue (bounded), then retypecheck.
2. **Gap detection**:
   - Identify missing assumptions or hidden lemmas.
   - Add hypotheses or intermediate lemmas.
3. **Re-run proofs**:
   - Any statement change invalidates downstream proofs.
   - Re-run Lean check + proof search recursively.

This becomes a **fixpoint loop**:

```
Draft → Typecheck → Proof Search → Validate Equivalence
   ↑                                      ↓
   └─────────────── Repair / Rewrite ─────┘
```

### Phase E: Consolidation

1. Produce final `.lean` file.
2. Record:
   - Proof success rate
   - Remaining `sorry`s
   - Error traces
3. Save artifacts (trace logs, diffs, and metadata).

## Recommended Error-Repair Strategies

- **Syntax repair**: fix parser/notation errors.
- **Type repair**: adjust arguments, coerce types, add casts.
- **Context repair**: import missing modules, open namespaces.
- **Lemma discovery**: use retrieval to find analogous statements.
- **Decomposition**: split a large lemma into smaller ones.

## Logging and Artifacts

- `run.jsonl`: step-by-step proof trace.
- `runs/*.json`: formalization tasks and inputs.
- Optional debug logs: LLM prompts, retrieved premises, tactic failures.
- `runs/formalize_*/`: per-round artifacts (Lean files, diffs, errors, equivalence reports).

## Next Implementation Steps (Suggested)

1. Add a formalization **segmenter** (TeX → structured chunks).
2. Add a **statement generator** + Lean typecheck loop.
3. Add a **lemma proof loop** with LeanDojo.
4. Add **equivalence checks** for informal vs formal statements.
5. Add **recursive repair** for statement + proof co-evolution.

## Non-Goals (For Now)

- Full proof automation for large papers.
- End-to-end RL training.
- Perfect semantic alignment without human review.
