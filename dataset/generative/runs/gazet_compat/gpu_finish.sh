# After the GPU answer job: download answers, replay through gazet.lm, score both models.
cd ~/repos/gazet/dataset/generative
export PATH=$HOME/.local/bin:$PATH
PY=$HOME/repos/gazet/.venv/bin/python
J=yellowcap/$1
until hf jobs inspect $J 2>&1 | grep -q "'stage': '\(ERROR\|CANCELED\|COMPLETED\)'"; do sleep 60; done
hf jobs inspect $J 2>&1 | grep -q "'stage': 'COMPLETED'" || { echo "GPU JOB FAILED"; hf jobs logs $J 2>&1 | tail -20; exit 1; }
A=runs/gazet_compat/answers; mkdir -p $A
hf download developmentseed/gazet-generative-sql gazet_compat/answers/gazet-v1.jsonl gazet_compat/answers/rl2-q3-06b.jsonl --repo-type dataset --local-dir $A > /dev/null
for l in gazet-v1 rl2-q3-06b; do $PY -I gazet_compat.py replay $l $A/gazet_compat/answers/$l.jsonl 2>&1 | grep -v "UserWarning\|cls = super\|generate_places: failed"; done
$PY -u -I gazet_compat.py score gazet-v1 rl2-q3-06b 2>&1 | grep -v "UserWarning\|cls = super"
echo GPU_FINISH_DONE
