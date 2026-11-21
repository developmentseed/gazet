#!/usr/bin/env bash
# Build the Space folder (demo files + the prompt code it shares with training) and push it.
# Usage: demo/deploy.sh [SPACE_ID]   (default yellowcap/gazet-sql-demo, created private; free Docker
# Spaces need a paid plan in the developmentseed org)
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
SPACE="${1:-yellowcap/gazet-sql-demo}"
OUT="$(mktemp -d)"
cp "$S"/demo/{Dockerfile,README.md,app.py,requirements.txt} "$S"/{harness.py,hints.py,student.py} "$OUT"/
hf repos create "$SPACE" --type space --space-sdk docker --private --exist-ok >/dev/null
hf upload "$SPACE" "$OUT" . --repo-type space --commit-message "Deploy from gazet dataset/generative/demo"
rm -rf "$OUT"
echo "https://huggingface.co/spaces/$SPACE"
