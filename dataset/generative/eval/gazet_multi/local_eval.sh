# Answer the held-out multi-place test (and optionally gazet_compat) with a GGUF on the laptop CPU, then score.
# Usage: local_eval.sh LABEL GGUF [compat]
cd ~/repos/gazet/dataset/generative
PY=$HOME/repos/gazet/.venv/bin/python
label=$1; model=$2; port=9011
llama/llama-b11935/llama-server -m "$model" -c 8192 -t 6 --port $port --jinja > runs/gazet_multi/server_$label.log 2>&1 &
SP=$!
until curl -sf localhost:$port/health >/dev/null; do sleep 1; done
$PY -u -I gazet_multi.py answer "$label" $port 2>&1 | grep -v "UserWarning\|cls = super"
if [ "$3" = compat ]; then
  GAZET_SQL_MODE=plan $PY -u -I gazet_compat.py gen "$label" $port 2>&1 | grep -v "UserWarning\|cls = super"
fi
kill $SP; wait $SP 2>/dev/null
$PY -u -I gazet_multi.py score runs/gazet_multi/pred_$label.jsonl "$label" 2>&1 | grep -v "UserWarning\|cls = super"
if [ "$3" = compat ]; then
  GAZET_SQL_MODE=plan $PY -u -I gazet_compat.py score "$label" 2>&1 | grep -v "UserWarning\|cls = super"
fi
echo LOCAL_EVAL_DONE $label
