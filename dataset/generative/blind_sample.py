"""Blind sample for hand grading: one executing pair per (model, scenario), shuffled, model hidden.

Usage: python blind_sample.py RUN_DIR [GLOB] [MAX_PER_MODEL]   e.g. python blind_sample.py pilot "out_*__v2.validated.jsonl"
Writes RUN_DIR/grading/sheet.txt (what the grader reads) and RUN_DIR/grading/key.json (item -> model).
"""
import glob
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
RUN_DIR = HERE / sys.argv[1]
PATTERN = sys.argv[2] if len(sys.argv) > 2 else "out_*.validated.jsonl"
MAX_PER_MODEL = int(sys.argv[3]) if len(sys.argv) > 3 else None
rnd = random.Random(11)
items = []
for f in sorted(glob.glob(str(RUN_DIR / PATTERN))):
    model = Path(f).name.removeprefix("out_").removesuffix(".validated.jsonl")
    by_scn = {}
    for line in open(f):
        r = json.loads(line)
        if r["ok"]:
            by_scn.setdefault(r["scenario"], []).append(r)
    scns = sorted(by_scn)
    if MAX_PER_MODEL:
        scns = sorted(rnd.sample(scns, min(MAX_PER_MODEL, len(scns))))
    for scn in scns:
        items.append((model, rnd.choice(by_scn[scn])))
rnd.shuffle(items)
(RUN_DIR / "grading").mkdir(exist_ok=True)
key, lines = {}, []
for i, (model, r) in enumerate(items):
    key[i] = dict(model=model, scenario=r["scenario"], idx=r["idx"])
    lines.append(f"## {i} [{r['scenario']} {r['difficulty']}] rows={r['n_rows']}\nQ: {r['question']}\n{r['sql']}\n-> {r['preview'][:2]}\n")
(RUN_DIR / "grading" / "sheet.txt").write_text("\n".join(lines))
(RUN_DIR / "grading" / "key.json").write_text(json.dumps(key))
print(f"{len(items)} items")
