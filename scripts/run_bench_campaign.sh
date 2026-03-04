#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_bench_campaign.sh --suite <suite.jsonl> [--out-root <dir>] [--use-system-ulam] [-- <ulam bench extra args...>]

Examples:
  scripts/run_bench_campaign.sh --suite bench/suites/internal_regression.jsonl -- --llm mock --lean mock
  scripts/run_bench_campaign.sh --suite bench/suites/internal_regression.jsonl --out-root runs/bench_campaigns -- --llm codex_cli --openai-model gpt-5.3-codex --lean dojo
  scripts/run_bench_campaign.sh --suite bench/suites/internal_regression.jsonl --use-system-ulam -- --llm codex_cli --openai-model gpt-5.3-codex --lean dojo
EOF
}

suite=""
out_root="runs/bench_campaigns"
use_system_ulam=0
extra_args=()
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
ulam_cmd=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --suite" >&2
        exit 1
      fi
      suite="$2"
      shift 2
      ;;
    --out-root)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --out-root" >&2
        exit 1
      fi
      out_root="$2"
      shift 2
      ;;
    --use-system-ulam)
      use_system_ulam=1
      shift
      ;;
    --)
      shift
      extra_args=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$suite" ]]; then
  echo "--suite is required" >&2
  usage >&2
  exit 1
fi

export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$use_system_ulam" -eq 1 ]]; then
  if command -v ulam >/dev/null 2>&1; then
    ulam_cmd=(ulam)
  else
    echo "Could not find `ulam` on PATH for --use-system-ulam." >&2
    exit 1
  fi
else
  ulam_cmd=(python3 -m ulam)
fi

repo_ulam_version="$(
  python3 - <<'PY'
from ulam import __version__
print(__version__)
PY
)"

repo_git_commit=""
if command -v git >/dev/null 2>&1; then
  repo_git_commit="$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || true)"
fi

stamp="$(date -u +%Y%m%d_%H%M%S)"
run_dir="${out_root%/}/bench_${stamp}"
mkdir -p "$run_dir/traces"

echo "Run dir: $run_dir"

{
  echo "created_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cwd=$(pwd)"
  echo "suite=$suite"
  echo "python=$(python3 --version 2>&1)"
  echo "uname=$(uname -a)"
  echo "repo_root=$repo_root"
  echo "repo_ulam_version=$repo_ulam_version"
  echo "ulam_cmd=${ulam_cmd[*]}"
  echo "use_system_ulam=$use_system_ulam"
  if command -v git >/dev/null 2>&1; then
    echo "git_commit=$repo_git_commit"
    echo "git_branch=$(git -C "$repo_root" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  fi
} >"$run_dir/env.txt"

"${ulam_cmd[@]}" bench-validate --suite "$suite" >"$run_dir/validate.log" 2>&1

cmd=(
  "${ulam_cmd[@]}"
  bench
  --suite "$suite"
  --report-json "$run_dir/report.json"
  --report-markdown "$run_dir/report.md"
  --trace-dir "$run_dir/traces"
)
if [[ ${#extra_args[@]} -gt 0 ]]; then
  cmd+=("${extra_args[@]}")
fi

{
  printf "command:"
  printf " %q" "${cmd[@]}"
  printf "\n"
} >"$run_dir/command.txt"

"${cmd[@]}" | tee "$run_dir/bench.log"

report_json="$run_dir/report.json"
report_md="$run_dir/report.md"
if [[ ! -s "$report_json" ]]; then
  echo "ERROR: missing report JSON artifact: $report_json" >&2
  exit 1
fi
if [[ ! -s "$report_md" ]]; then
  echo "ERROR: missing report Markdown artifact: $report_md" >&2
  exit 1
fi

python3 - "$report_json" "$repo_ulam_version" "$repo_git_commit" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
expected_version = sys.argv[2].strip()
expected_commit = sys.argv[3].strip()

try:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
except Exception as exc:
    raise SystemExit(f"ERROR: failed reading report JSON ({report_path}): {exc}")

if not isinstance(payload, dict):
    raise SystemExit(f"ERROR: report JSON root is not an object: {report_path}")

metadata = payload.get("metadata")
summary = payload.get("summary")
if not isinstance(metadata, dict):
    raise SystemExit("ERROR: report JSON missing `metadata` object")
if not isinstance(summary, dict):
    raise SystemExit("ERROR: report JSON missing `summary` object")

actual_version = str(metadata.get("ulam_version", "")).strip()
if not actual_version:
    raise SystemExit("ERROR: report metadata missing `ulam_version`")
if actual_version != expected_version:
    raise SystemExit(
        f"ERROR: report ulam_version={actual_version} does not match repo ulam_version={expected_version}"
    )

actual_commit = str(metadata.get("ulam_git_commit", "")).strip()
if expected_commit and actual_commit and actual_commit != expected_commit:
    raise SystemExit(
        f"ERROR: report ulam_git_commit={actual_commit} does not match repo commit={expected_commit}"
    )

print(
    f"Verified report metadata: ulam_version={actual_version}, "
    f"ulam_git_commit={actual_commit or 'n/a'}"
)
PY

echo "Artifacts:"
echo "- $run_dir/report.json"
echo "- $run_dir/report.md"
echo "- $run_dir/traces/"
