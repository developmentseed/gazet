#!/usr/bin/env bash
# Convert a training run's merged model to GGUF (f16 -> q8_0 and q4_k_m) and upload to <run>/gguf/ in the model repo.
# Needs a llama.cpp source checkout for the converter (LLAMA_SRC, same tag as llama/) with its Python requirements
# in CONVERT_PY. Usage: ./to_gguf.sh RUN_NAME
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
RUN="$1"
REPO="${MODEL_REPO:-developmentseed/gazet-sql-qwen35}"
LLAMA_SRC="${LLAMA_SRC:-$HOME/.cache/gazet-exp/llamacpp-src/llama.cpp-b11935}"
CONVERT_PY="${CONVERT_PY:-$HOME/.cache/gazet-exp/convert-venv/bin/python}"
W="${WORK:-$HOME/.cache/gazet-exp/models}"
# --include does not fetch nested paths with this hf version, so list the files
hf download "$REPO" $(hf models info "$REPO" | grep -o "\"$RUN/merged/[^\"]*" | tr -d \") --local-dir "$W" >/dev/null
mkdir -p "$W/$RUN/gguf"
"$CONVERT_PY" "$LLAMA_SRC/convert_hf_to_gguf.py" "$W/$RUN/merged" --outtype f16 --outfile "$W/$RUN/gguf/f16.gguf"
for q in q8_0 q4_k_m; do
  "$S/llama/llama-b11935/llama-quantize" "$W/$RUN/gguf/f16.gguf" "$W/$RUN/gguf/gazet-$RUN-$q.gguf" "${q^^}" >/dev/null
done
rm "$W/$RUN/gguf/f16.gguf"
hf upload "$REPO" "$W/$RUN/gguf" "$RUN/gguf" --commit-message "Add GGUF for $RUN"
ls -la "$W/$RUN/gguf"
