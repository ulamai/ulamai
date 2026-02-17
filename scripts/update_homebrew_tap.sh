#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-}"
if [[ -z "$TAG" ]]; then
  echo "Usage: $0 <tag>" >&2
  exit 1
fi

TAP_REPO="${TAP_REPO:-ulamai/homebrew-ulamai}"
FORMULA_PATH="${FORMULA_PATH:-Formula/ulamai.rb}"
REPO_OWNER="${REPO_OWNER:-ulamai}"
REPO_NAME="${REPO_NAME:-ulamai}"

TMP_DIR="$(mktemp -d)"
INSTALL_URL="https://raw.githubusercontent.com/${REPO_OWNER}/${REPO_NAME}/${TAG}/install.sh"

curl -fsSL "$INSTALL_URL" -o "$TMP_DIR/install.sh"
SHA="$(shasum -a 256 "$TMP_DIR/install.sh" | awk '{print $1}')"

git clone "https://github.com/${TAP_REPO}.git" "$TMP_DIR/tap"

export FORMULA="$TMP_DIR/tap/$FORMULA_PATH"
export TAG SHA REPO_OWNER REPO_NAME
python3 - <<'PY'
import os
import re
from pathlib import Path

path = Path(os.environ["FORMULA"])
tag = os.environ["TAG"]
sha = os.environ["SHA"]
owner = os.environ["REPO_OWNER"]
repo = os.environ["REPO_NAME"]

text = path.read_text(encoding="utf-8")
text = re.sub(
    r'^  url ".*"$',
    f'  url "https://raw.githubusercontent.com/{owner}/{repo}/{tag}/install.sh"',
    text,
    flags=re.M,
)
text = re.sub(r'^  sha256 ".*"$', f'  sha256 "{sha}"', text, flags=re.M)
path.write_text(text, encoding="utf-8")
PY

cd "$TMP_DIR/tap"
git config user.name "${GIT_AUTHOR_NAME:-ulamai-bot}"
git config user.email "${GIT_AUTHOR_EMAIL:-bot@ulamai.ai}"
git add "$FORMULA_PATH"
git commit -m "Update ulamai formula to ${TAG}" || exit 0

if [[ -n "${GH_TOKEN:-}" ]]; then
  git push "https://x-access-token:${GH_TOKEN}@github.com/${TAP_REPO}.git" main
else
  git push
fi
