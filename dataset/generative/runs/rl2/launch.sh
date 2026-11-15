# RL round 2: waits for the round-2 solver check, builds rl/v2 (round 1 + new-style questions), launches GRPO from
# the best RL model with the "sometimes right" prefilter.
cd ~/repos/gazet/dataset/generative
export PATH=$HOME/.local/bin:$PATH HF_TOKEN=$(cat ~/.cache/huggingface/token)
until grep -q RL2CHECK_DONE runs/rl2/check.log; do sleep 60; done
for g in out_deepseek__deepseek-v4.1-flash out_z-ai__glm-5.3-flash; do
  [ -f runs/rl2/${g}__rl.judged.consistency.jsonl ] || { echo "CHECK INCOMPLETE: $g, not launching"; exit 1; }
done
D=$HOME/.cache/gazet-exp/rl_v2; rm -rf $D
.venv/bin/python -I rl_data.py write2 $D || exit 1
n=$(wc -l < $D/train.jsonl); [ $n -ge 4000 ] || { echo "ONLY $n RL QUESTIONS, not launching"; exit 1; }
mkdir -p $D/code && cp *.py $D/code/
hf upload developmentseed/gazet-generative-sql $D rl/v2 --repo-type dataset --commit-message "RL question set v2: round 1 + new-style questions" | tail -1
hf upload developmentseed/gazet-generative-sql runs/rl2 rl/v2/checks --repo-type dataset --include "*.jsonl" --commit-message "RL v2 candidates and solver checks" | tail -1
cd ~/repos/gazet
hf jobs uv run --flavor a100-large --timeout 10h --secrets HF_TOKEN --detach --name gazet-rl2-q3-06b \
  finetune/train_grpo_hf_jobs.py --run-name rl2-q3-06b --init-run rl-q3-06b-lr2e5 --rl-data rl/v2 --lr 2e-5 --prefilter 2>&1 | grep -o "id=[a-z0-9]*"
echo LAUNCHED
