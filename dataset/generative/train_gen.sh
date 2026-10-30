# Generation loop (network-bound): chunks 0-9, marks each finished chunk; stops at the budget.
set -u
cd ~/repos/gazet/dataset/generative
P=.venv/bin/python
BUDGET=75
for c in 0 1 2 3 4 5 6 7 8 9; do
  [ -f runs/train/gen_c$c.done ] && continue
  used=$(curl -s https://openrouter.ai/api/v1/key -H "Authorization: Bearer $(cat ~/.config/openrouter/key)" | $P -c "import json,sys; print(json.load(sys.stdin)['data']['usage'])")
  echo "== gen chunk $c  $(date +%H:%M)  key usage \$$used"
  $P -c "import sys; sys.exit(0 if $used < $BUDGET else 1)" || { echo "budget reached, stopping"; touch runs/train/STOP; break; }
  [ -d runs/train/prompts_c$c ] || $P -I harness.py prompts 390 $((41 + c)) runs/train/prompts_c$c train
  # two passes: the second retries scenarios whose output was empty or unparsable
  for pass in 1 2; do
    $P -I or_generate.py deepseek/deepseek-v4.1-flash --prompts "runs/train/prompts_c$c/*.txt" --tag train_c$c --root runs/train --workers 16 | tail -1
  done
  touch runs/train/gen_c$c.done
done
echo GEN_DONE
