"""Dev split: whole scenarios held back from training, on tables the student does see in training.

Gold measures new tables; dev measures new questions on known tables. The gap between the two is the
generalisation gap. Dev pairs pass the same automatic filter as gold candidates (judge correct AND a solver
from another model family agrees AND no LIMIT tie) but, unlike gold, were not reviewed by hand.

Usage:
    python dev_split.py pick      # choose scenarios -> dev/dev_scenarios.json, runs/dev/out_<generator>__dev.judged.jsonl
    python consistency.py run runs/dev/out_<generator>__dev.judged.jsonl
    python dev_split.py write     # -> dev/dev_v1.jsonl
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
PER_BUNDLE, SEED = 5, 11
JUDGED = HERE / "runs" / "dev" / "out_deepseek__deepseek-v4.1-flash__dev.judged.jsonl"
SCENARIOS = HERE / "dev" / "dev_scenarios.json"


def train_rows():
    for f in sorted((HERE / "runs" / "train").glob("*train_c*.judged.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                yield json.loads(line)


def pick():
    by_bundle = defaultdict(lambda: defaultdict(list))
    for r in train_rows():
        by_bundle[tuple(r["tables"])][r["scenario"]].append(r)
    rnd, chosen, rows = random.Random(SEED), [], []
    for bundle in sorted(by_bundle):
        for s in rnd.sample(sorted(by_bundle[bundle]), PER_BUNDLE):
            chosen.append(s)
            rows += by_bundle[bundle][s]
    SCENARIOS.parent.mkdir(exist_ok=True)
    JUDGED.parent.mkdir(parents=True, exist_ok=True)
    SCENARIOS.write_text(json.dumps(sorted(chosen), indent=0))
    JUDGED.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"{len(chosen)} scenarios from {len(by_bundle)} bundles, {len(rows)} executed pairs, "
          f"{sum(r['judge'] == 'correct' for r in rows)} judge-correct -> {JUDGED}")


def write():
    rows = [json.loads(line) for line in JUDGED.with_suffix(".consistency.jsonl").read_text().splitlines() if line.strip()]
    dev = [dict(id=f"d{i:03d}", question=r["question"], tables=r["tables"], sql=r["sql"].strip().rstrip(";"),
                difficulty=r["difficulty"], skill=r["target_skill"], n_rows=r["n_rows"], cols=r["cols"],
                source=f"{r['scenario']}.{r['idx']}")
           for i, r in enumerate(r for r in rows if r["lenient"] and r.get("tie") is not True)]
    out = HERE / "dev" / "dev_v1.jsonl"
    out.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in dev) + "\n")
    print(f"{len(dev)} of {len(rows)} judge-correct pairs kept -> {out}")


if __name__ == "__main__":
    pick() if sys.argv[1] == "pick" else write()
