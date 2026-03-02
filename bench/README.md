# Benchmark Suites

`ulam` benchmark suites are JSONL files with one case per line.

## Schema

Required fields:
- `file` (string): path to Lean file.
- `theorem` (string): declaration name to prove.

Optional fields:
- `premises` (string): path to premises file.
- `semantic_report` (string): path to a semantic-check JSON report (`llm_check_final.json` or compatible).
- `artifact_dir` (string): formalization artifact directory; bench will auto-read semantic report and rejection memory when present.
- any extra metadata fields are allowed and ignored by `ulam bench`.

Example:

```json
{"file":"../../examples/Smoke.lean","theorem":"irrational_sqrt_two_smoke"}
```

## Validation

Validate a suite before running:

```bash
python3 -m ulam bench-validate --suite bench/suites/internal_regression.jsonl
```

Validation checks:
- valid JSON object on each non-empty line
- required fields present
- referenced Lean files exist
- referenced premises files exist (if provided)
- theorem declaration names exist in target files (unless `--no-theorem-check`)
- semantic report/artifact paths exist (if provided)

## miniF2F Setup

Build a real miniF2F suite from a local checkout:

```bash
python3 -m ulam bench-make-minif2f --root /path/to/miniF2F --split valid --out bench/suites/minif2f_valid.jsonl
python3 -m ulam bench-validate --suite bench/suites/minif2f_valid.jsonl
```

Notes:
- `bench-make-minif2f` scans Lean files, extracts theorem/lemma/example declaration names, and writes JSONL entries.
- Use `--require-sorry` if your benchmark protocol requires explicit placeholders.
- Use `--shuffle --seed <N> --limit <K>` for fixed-size reproducible slices.

## Comparison

Compare two benchmark report JSON files:

```bash
python3 -m ulam bench-compare --a runs/bench/a.json --b runs/bench/b.json --out-markdown runs/bench/compare.md
```

## Reproducible Campaign Runs

Use the wrapper script to create timestamped run directories with report, traces, command log, and environment snapshot:

```bash
scripts/run_bench_campaign.sh --suite bench/suites/internal_regression.jsonl -- --llm codex_cli --openai-model gpt-5.3-codex --lean dojo
```

## Curation Rules

- Keep paths repository-relative and stable.
- Prefer deterministic, self-contained cases (minimal dependence on machine-local state).
- Keep theorem names exact (including apostrophes when present).
- Add metadata fields (e.g. `dataset`, `difficulty`, `tags`) as needed; they are preserved in suite files and ignored by the runner.
