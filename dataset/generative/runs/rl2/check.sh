cd ~/repos/gazet/dataset/generative
export GAZET_SOLVER_WORKERS=32 GAZET_COMPARE_WORKERS=3 GAZET_QUERY_RSS_MB=900
for g in out_deepseek__deepseek-v4.1-flash out_z-ai__glm-5.3-flash; do
  f=runs/rl2/${g}__rl.judged.jsonl
  [ -f ${f%.jsonl}.consistency.jsonl ] || .venv/bin/python -u -I consistency.py run $f
done
echo RL2CHECK_DONE
