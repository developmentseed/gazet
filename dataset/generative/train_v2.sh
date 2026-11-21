# Training set v2 (second lake expansion): generate -> collect -> validate -> judge, chunk by chunk.
# Generators alternate by chunk (DeepSeek even, GLM odd); stops before a chunk when key usage passes BUDGET.
# Usage: nohup bash train_v2.sh BUDGET [CHUNKS] [PROMPTS_PER_CHUNK] [even|odd|all] > runs/v2/train_PARITY.log &
# Run "even" and "odd" side by side to use both generators at once.
set -u
cd ~/repos/gazet/dataset/generative
P=.venv/bin/python
BUDGET=${1:?usage: train_v2.sh BUDGET [CHUNKS] [PROMPTS_PER_CHUNK]}
CHUNKS=${2:-8}
N=${3:-200}
PARITY=${4:-all}
usage() { curl -s https://openrouter.ai/api/v1/key -H "Authorization: Bearer $(cat ~/.config/openrouter/key)" | $P -c "import json,sys; print(json.load(sys.stdin)['data']['usage'])"; }
for c in $(seq 0 $((CHUNKS - 1))); do
  if [ $((c % 2)) -eq 0 ]; then m=deepseek/deepseek-v4.1-flash; p=even; else m=z-ai/glm-5.3-flash; p=odd; fi
  [ "$PARITY" = all ] || [ "$PARITY" = $p ] || continue
  R=runs/v2/out_$(echo $m | sed 's|/|__|')__v2_c$c
  [ -f $R.judged.jsonl ] && continue
  used=$(usage)
  echo "== chunk $c ($m)  $(date +%H:%M)  key usage \$$used"
  $P -c "import sys; sys.exit(0 if $used < $BUDGET else 1)" || { echo "budget reached, stopping"; break; }
  [ -d runs/v2/prompts_c$c ] || $P -I harness.py prompts $N $((71 + c)) runs/v2/prompts_c$c new
  for pass in 1 2; do $P -I or_generate.py $m --prompts "runs/v2/prompts_c$c/*.txt" --tag v2_c$c --root runs/v2 --workers 16 | tail -1; done
  $P -I collect.py $R runs/v2/prompts_c$c | grep -E "pairs from|parse error"
  $P -I harness.py validate $R.jsonl | grep "executed"
  $P -I judge.py run minimax/minimax-m3 $R.validated.jsonl 16 | tail -2 && mv $R.validated.judged_minimax__minimax-m3.jsonl $R.judged.jsonl
done
echo "TRAIN_V2_DONE  key usage \$$(usage)"
