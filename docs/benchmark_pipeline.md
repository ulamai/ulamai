# Benchmark Mini-Pipeline (Next Version)

This document defines the implementation sequence before running external benchmark campaigns with `gpt-5.3-codex`.

## Objective

Ship a benchmark-ready proving/formalization pipeline that is reproducible, comparable across models, and robust against semantic cheating/regressions.

## Current Status

- Phase 1 implemented:
  - `ulam bench` now supports `--report-json` and `--report-markdown`.
  - Reports include run metadata, per-case outcomes, and aggregate summary metrics.
- Phase 2 implemented (scaffolding):
  - Suite registry files added under `bench/suites/`.
  - `bench/README.md`, `ulam bench-validate`, and `ulam bench-make-minif2f` commands added.
- Phase 3 implemented:
  - Bench reports include semantic verdicts, deterministic issue counts, and regression rejection metrics (when semantic artifacts are provided).
- Phase 4 implemented:
  - `scripts/run_bench_campaign.sh` added for timestamped reproducible runs.
  - `ulam bench-compare` added for report-to-report comparison.

## Scope

- In scope:
  - `ulam bench` hardening and result artifacts
  - benchmark suite adapters and run scripts
  - anti-cheat reporting in benchmark summaries
  - reproducible run metadata and comparison outputs
- Out of scope:
  - SFT/RL training
  - distributed rollout infrastructure

## Phase 1: Bench Harness Hardening

Deliverables:
- Add machine-readable benchmark report output:
  - `--report-json <path>`
  - `--report-markdown <path>`
- Include run metadata in report:
  - model/provider/backend
  - toolchain and project path
  - key budgets (`max_steps`, `beam`, `k`, `timeout`, `repair`)
  - git commit hash and timestamp
- Add case-level fields:
  - solved
  - steps
  - duration
  - error kind
  - trace path

Acceptance criteria:
- Running `ulam bench` can produce deterministic structured reports without parsing stdout.
- Reports are stable enough for CI diffing and leaderboard generation.

## Phase 2: Suite Packaging

Deliverables:
- Add suite registry docs under `bench/suites/`:
  - `minif2f_dev.jsonl`
  - `internal_regression.jsonl`
  - optional `putnambench_sample.jsonl`
- Add suite alias registry (`bench/suites/registry.json`) and listing command (`ulam bench-list-suites`).
- Add deterministic fixed-suite builder (`ulam bench-make-regression100`) for large, versioned regression slices.
- Add `bench/README.md` documenting schema and curation rules.
- Add a validator command/script to check suite entries before running.

Acceptance criteria:
- Each suite passes schema validation.
- Suite entries resolve to existing files/theorems in local environments.

## Phase 3: Anti-Cheat Metrics in Bench

Deliverables:
- Extend per-case summaries to include semantic-integrity signals when available:
  - deterministic issue counts (high/medium/low)
  - semantic verdict (`pass|fail|unknown`)
  - locked-declaration regression count
- Aggregate summary metrics:
  - `semantic_pass_rate`
  - `semantic_fail_rate`
  - `regression_rejection_rate`

Acceptance criteria:
- Bench output distinguishes "Lean solved" from "semantically acceptable solved."
- Failures from private/public axioms and vacuous defs are visible in aggregate metrics.

## Phase 4: Reproducibility + Comparison

Deliverables:
- Add a run wrapper script for reproducible campaigns:
  - pins model name
  - writes reports and traces into timestamped run dirs
  - captures environment snapshot
  - optional baseline parity gate (`--compare-to`) for release checks
- Add comparison utility:
  - compare two report JSON files by solved count, semantic pass rate, and median time.

Acceptance criteria:
- Two runs with same config produce comparable reports.
- Regression checks can be automated in CI for selected suites.

Release/CI gate command (recommended):

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

## Phase 5: GPT-5.3-Codex Campaign Gate

Prerequisites to start external campaign:
- Phase 1-4 completed.
- `ulam bench` reports generated successfully on internal suite.
- no blocking semantic-regression bugs in resume/formalize flows.

Campaign kickoff config:
- LLM: `codex_cli`
- model: `gpt-5.3-codex`
- fixed budgets and fixed suite versions
- artifact retention enabled (traces + reports)

## Execution Order

1. Implement Phase 1 report outputs.
2. Implement Phase 2 suite packaging + validation.
3. Implement Phase 3 anti-cheat bench metrics.
4. Implement Phase 4 reproducibility/comparison tooling.
5. Run Phase 5 GPT-5.3-codex benchmarks.

## Definition of Done

- We can run:
  - `ulam bench --suite <suite> --llm codex_cli ... --report-json <file>`
- We can compare runs without manual log parsing.
- We can report both proof success and semantic integrity quality.
