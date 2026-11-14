# Laptop CPU timing: each run's GGUF answers 60 gold v1 questions via llama-server (6 threads), plain prompts.
# Results: runs/eval/<run>-<quant>__gold_v1__plain__minimal.jsonl (s/query in `evaluate.py report`).
cd ~/repos/gazet/dataset/generative
P=.venv/bin/python; W=$HOME/.cache/gazet-exp/models
for r in $(awk '{print $1}' runs/v2/train_jobs.txt); do
  for q in q4_k_m q8_0; do
    M=$W/$r/gguf/gazet-$r-$q.gguf; [ -f $M ] || { echo "missing $M"; continue; }
    llama/llama-b11935/llama-server -m $M -c 4096 -t 6 --port 8094 --jinja > runs/v2/bench_server.log 2>&1 &
    SP=$!
    until curl -sf localhost:8094/health >/dev/null; do sleep 1; kill -0 $SP 2>/dev/null || break; done
    echo "== $r $q $(date +%T) rss $(ps -o rss= -p $SP)"
    $P -I evaluate.py run $r-$q --port 8094 --data gold --system minimal --limit 60 > runs/eval/$r-$q.bench.log 2>&1
    kill $SP; wait $SP 2>/dev/null
  done
done
$P -I evaluate.py report runs/eval/sql-v2-*-q*__gold_v1__plain__minimal.jsonl
echo BENCH_DONE
