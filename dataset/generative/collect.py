"""Merge generator outputs into one JSONL and record prompt/output sizes for cost estimation."""
import json
import re
import sys
from pathlib import Path

import tiktoken

HERE = Path(__file__).parent
enc = tiktoken.get_encoding("o200k_base")
run = sys.argv[1] if len(sys.argv) > 1 else "out"
prompts = HERE / (sys.argv[2] if len(sys.argv) > 2 else "prompts")
rows, sizes = [], []
for f in sorted((HERE / run).glob("s*.json")):
    sid = f.stem
    raw = f.read_text()
    prompt = (prompts / f"{sid}.txt").read_text()
    meta = json.loads((prompts / f"{sid}.json").read_text())
    try:
        pairs = json.loads(re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE))
    except json.JSONDecodeError as e:
        print(f"{sid}: JSON parse error {e}")
        pairs = []
    sizes.append(dict(scenario=sid, prompt_tok=len(enc.encode(prompt)), output_tok=len(enc.encode(raw)), n_pairs=len(pairs)))
    for i, p in enumerate(pairs):
        rows.append(dict(scenario=sid, idx=i, tables=meta["tables"], target_skill=meta["skill"], style=meta["style"], **p))
(HERE / f"{run}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
(HERE / f"{run}.sizes.json").write_text(json.dumps(sizes, indent=1))
print(f"{len(rows)} pairs from {len(sizes)} scenarios")
for s in sizes:
    print(s)
