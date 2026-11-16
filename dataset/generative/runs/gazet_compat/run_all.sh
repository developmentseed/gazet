# Old gazet model vs the new RL model through gazet's production prompts, then scoring.
cd ~/repos/gazet/dataset/generative
PY=$HOME/repos/gazet/.venv/bin/python
until grep -q PREP_DONE runs/gazet_compat/prep.log; do sleep 30; done
for spec in "gazet-v1:$HOME/.cache/gazet-exp/models/gazet-v1/models/ckpt-q8_0.gguf" "rl2-q3-06b:$HOME/.cache/gazet-exp/models/rl2-q3-06b/gguf/gazet-rl2-q3-06b-q8_0.gguf"; do
  label=${spec%%:*}; model=${spec#*:}
  llama/llama-b11935/llama-server -m $model -c 8192 -t 6 --port 9002 --jinja > runs/gazet_compat/server_$label.log 2>&1 &
  SP=$!
  until curl -sf localhost:9002/health >/dev/null; do sleep 1; done
  $PY -u -I gazet_compat.py gen $label 9002 2>&1 | grep -v "UserWarning\|cls = super"
  kill $SP; wait $SP 2>/dev/null
done
$PY -u -I gazet_compat.py score gazet-v1 rl2-q3-06b 2>&1 | grep -v "UserWarning\|cls = super"
echo ALL_DONE
