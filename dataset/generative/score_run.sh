#!/usr/bin/env bash
# Score a training run's saved answers (written by the HF job) on dev, devnew, gold and gold2, plain and hinted.
# Usage: ./score_run.sh RUN_NAME   -> runs/eval/<RUN>__<data>__<plain|hints>.jsonl, then `evaluate.py report`
set -euo pipefail
cd "$(dirname "$0")"
export PATH=$HOME/.local/bin:$PATH
RUN="$1"; REPO="${MODEL_REPO:-developmentseed/gazet-sql-qwen35}"; W="$HOME/.cache/gazet-exp/models"
files=$(hf models info "$REPO" | grep -o "\"$RUN/predictions/[^\"]*" | tr -d \")
[ -n "$files" ] || { echo "no predictions for $RUN yet"; exit 1; }
hf download "$REPO" $files --local-dir "$W" >/dev/null
for f in $W/$RUN/predictions/*.jsonl; do
  b=$(basename $f .jsonl); data=${b%_*}; mode=${b##*_}
  [ "$b" = gazet_compat ] && continue  # gazet answers: scored by gazet_compat.py (replay + score)
  h=""; [ "$mode" = hints ] && h=--hints
  .venv/bin/python -I evaluate.py run "$RUN" --preds "$f" --data "$data" $h --system minimal > "runs/eval/$RUN.$b.log" 2>&1
done
.venv/bin/python -I evaluate.py report runs/eval/${RUN}__*.jsonl
