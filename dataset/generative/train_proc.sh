# Processing loop (CPU + judge): collect -> validate -> judge each chunk once its generation is marked done.
# Usage: nohup bash train_proc.sh > runs/train_proc.log &
set -u
cd ~/repos/gazet/dataset/generative
P=.venv/bin/python
for c in 0 1 2 3 4 5 6 7 8 9; do
  R=runs/train/out_deepseek__deepseek-v4.1-flash__train_c$c
  [ -f $R.judged.jsonl ] && continue
  until [ -f runs/train/gen_c$c.done ] || [ -f runs/train/STOP ]; do sleep 30; done
  [ -f runs/train/gen_c$c.done ] || break
  echo "== proc chunk $c  $(date +%H:%M)"
  if [ ! -f $R.validated.jsonl ] || [ $R.validated.jsonl -ot $R.jsonl ]; then
    $P -I collect.py $R runs/train/prompts_c$c | grep -E "pairs from|parse error"
    $P -I harness.py validate $R.jsonl | grep "executed"
  fi
  $P -I judge.py run minimax/minimax-m3 $R.validated.jsonl 16 && mv $R.validated.judged_minimax__minimax-m3.jsonl $R.judged.jsonl
done
echo PROC_DONE
