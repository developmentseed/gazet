# Usage: after_job.sh RUN JOB_ID -> waits for the HF job, scores the run, uploads scores, converts to GGUF.
cd ~/repos/gazet/dataset/generative
export PATH=$HOME/.local/bin:$PATH
RUN=$1; J=yellowcap/$2
until hf jobs inspect $J 2>&1 | grep -q "'stage': '\(ERROR\|CANCELED\|COMPLETED\)'"; do sleep 120; done
hf jobs inspect $J 2>&1 | grep -q "'stage': 'COMPLETED'" || { echo "JOB $RUN NOT COMPLETED"; exit 1; }
./score_run.sh $RUN
mkdir -p eval/$RUN && cp runs/eval/${RUN}__*.jsonl eval/$RUN/ && .venv/bin/python -I evaluate.py report eval/$RUN/*.jsonl > eval/$RUN/report.txt
hf upload developmentseed/gazet-generative-sql eval/$RUN eval/$RUN --repo-type dataset --commit-message "Add $RUN per-pair scores" | tail -1
rm eval/$RUN/*.jsonl
./to_gguf.sh $RUN 2>&1 | grep "\.gguf$"
echo AFTER_DONE $RUN
