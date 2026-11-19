# After the plan-v2-smol135 job: GGUF conversion, then the same laptop eval as the baseline.
cd ~/repos/gazet/dataset/generative
export PATH=$HOME/.local/bin:$PATH
J=yellowcap/6afe74e6e41aae15936d1ba2; RUN=plan-v5-smol135
until hf jobs inspect "$J" 2>&1 | grep -q "'stage': '\(ERROR\|CANCELED\|COMPLETED\)'"; do sleep 60; done
hf jobs inspect "$J" 2>&1 | grep -q "'stage': 'COMPLETED'" || { echo "JOB FAILED"; hf jobs logs "$J" 2>&1 | tail -30; exit 1; }
hf download developmentseed/gazet-sql-qwen35 "$RUN/config.json" "$RUN/log_history.json" "$RUN/predictions/multi_test.jsonl" "$RUN/predictions/gazet_compat.jsonl" --local-dir runs/gazet_multi/job > /dev/null
cat runs/gazet_multi/job/$RUN/config.json
while pgrep -f "[l]ocal_eval.sh" > /dev/null; do sleep 30; done
./to_gguf.sh $RUN 2>&1 | tail -4
while pgrep -f "[l]ocal_eval.sh" > /dev/null; do sleep 30; done
bash runs/gazet_multi/local_eval.sh plan-v5-smol135-laptop $HOME/.cache/gazet-exp/models/$RUN/gguf/gazet-$RUN-q8_0.gguf compat
echo FINISH_DONE
