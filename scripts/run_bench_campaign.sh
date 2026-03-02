#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_bench_campaign.sh --suite <suite.jsonl> [--out-root <dir>] [-- <ulam bench extra args...>]

Examples:
  scripts/run_bench_campaign.sh --suite bench/suites/internal_regression.jsonl -- --llm mock --lean mock
  scripts/run_bench_campaign.sh --suite bench/suites/internal_regression.jsonl --out-root runs/bench_campaigns -- --llm codex_cli --openai-model gpt-5.3-codex --lean dojo
EOF
}

suite=""
out_root="runs/bench_campaigns"
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

if command -v ulam >/dev/null 2>&1; then
  ulam_cmd=(ulam)
else
  export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
  ulam_cmd=(python3 -m ulam)
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
  echo "ulam_cmd=${ulam_cmd[*]}"
  if command -v git >/dev/null 2>&1; then
    echo "git_commit=$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || true)"
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

echo "Artifacts:"
echo "- $run_dir/report.json"
echo "- $run_dir/report.md"
echo "- $run_dir/traces/"
