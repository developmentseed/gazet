#!/usr/bin/env bash
# Serve each model with llama-server on CPU and run the zero-shot eval against it.
set -u
S="$(cd "$(dirname "$0")" && pwd)"
SERVER="$S/llama/llama-b11935/llama-server"
for m in "$@"; do
  "$SERVER" -m "$S/models/$m.gguf" -c 4096 -t 6 --port 8091 --jinja >"$S/runs/server_$m.log" 2>&1 &
  pid=$!
  for _ in $(seq 120); do curl -sf http://127.0.0.1:8091/health >/dev/null && break; sleep 1; done
  "$S/.venv/bin/python" -I "$S/eval_small.py" "$m" 8091 >"$S/runs/eval_$m.log" 2>&1
  kill "$pid"; wait "$pid" 2>/dev/null
  tail -1 "$S/runs/eval_$m.log"
done
