# Waits for the RL reference check and a clean smoke test, then builds rl/v1 and launches the full GRPO run.
cd ~/repos/gazet/dataset/generative
export PATH=$HOME/.local/bin:$PATH HF_TOKEN=$(cat ~/.cache/huggingface/token)
SMOKE=yellowcap/6af98a1323985fb9cd83318a  # passed 2026-11-15
until grep -q RLCHECK_DONE runs/rl/check.log; do sleep 60; done
for g in out_deepseek__deepseek-v4.1-flash out_z-ai__glm-5.3-flash; do
  [ -f runs/rl/${g}__rl.judged.consistency.jsonl ] || { echo "CHECK INCOMPLETE: $g has no consistency file, not launching"; exit 1; }
done
until hf jobs inspect $SMOKE 2>&1 | grep -q "'stage': '\(COMPLETED\|ERROR\|CANCELED\)'"; do sleep 60; done
if ! hf jobs inspect $SMOKE 2>&1 | grep -q "'stage': 'COMPLETED'" || ! hf jobs logs $SMOKE 2>&1 | grep -q "uploaded to"; then
  echo "SMOKE FAILED, not launching"; exit 1
fi
D=$HOME/.cache/gazet-exp/rl_v1; rm -rf $D
.venv/bin/python -I rl_data.py write $D || exit 1
n=$(wc -l < $D/train.jsonl); [ $n -ge 2000 ] || { echo "ONLY $n RL QUESTIONS, not launching"; exit 1; }
mkdir -p $D/code && cp *.py $D/code/
hf upload developmentseed/gazet-generative-sql $D rl/v1 --repo-type dataset --commit-message "RL question set v1 (solver-confirmed, training tables only)" | tail -1
cd ~/repos/gazet
hf jobs uv run --flavor a100-large --timeout 8h --secrets HF_TOKEN --detach --name gazet-rl-q3-06b \
  finetune/train_grpo_hf_jobs.py --run-name rl-q3-06b --init-run sql-v2-q3-06b --rl-data rl/v1 2>&1 | grep -o "id=[a-z0-9]*"
echo LAUNCHED
