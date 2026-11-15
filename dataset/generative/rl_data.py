"""RL question set: training-table questions whose reference SQL a second model family confirms.

For GRPO the reference only supplies the target result, but a wrong reference punishes correct answers directly,
so the bar is the gold-set filter (judge correct AND cross-family solver agrees AND no LIMIT tie), not the
judge alone. Candidates come from the same pool and filters as the SFT export (export.train_pairs: training
tables only, dev scenarios excluded; no pair that needs a value the student cannot see), spread over table
bundles. Prompts use the student format with value hints on the same seeded share as the export.

Usage:
    python rl_data.py pick N          # -> runs/rl/out_<generator>__rl.judged.jsonl (one file per generator)
    python consistency.py run runs/rl/out_<generator>__rl.judged.jsonl      # for each generator
    python rl_data.py write OUT_DIR   # -> OUT_DIR/train.jsonl {"id", "messages": [system, user], "sql"}
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import export
import harness
import hints as hints_mod
import student

HERE = Path(__file__).parent
RUNS = HERE / "runs" / "rl"
GENERATORS = {"out_deepseek": "out_deepseek__deepseek-v4.1-flash", "out_z-ai": "out_z-ai__glm-5.3-flash"}
SEED = 23


def visible(c, cards, r):
    """The export's filters -> (keep, value hints or None): parseable, reads only prompt tables, and every
    filter value appears in the question, the cards or (when hinted) the hints."""
    sql = r["sql"].strip().rstrip(";").strip()
    used = export.read_tables(sql)
    known = {t.lower() for t in harness.TABLE_DOCS}
    if used is None or (used & known) - {t.lower() for t in r["tables"]}:
        return False, None
    h = hints_mod.value_hints(c, r["tables"], r["question"]) if export._hinted(r["key"], 0.5) else None
    seen = hints_mod.norm("\n".join([r["question"], *(cards[t] for t in r["tables"]), *(h or [])]))
    return not any(hints_mod.norm(v) not in seen for v in hints_mod._filter_literals(sql)), h


def pick(n):
    by_bundle = defaultdict(list)
    for r in export.train_pairs(v2=True):
        assert not set(r["tables"]) & harness.HELD_OUT
        by_bundle[tuple(r["tables"])].append(r)
    rnd = random.Random(SEED)
    for rows in by_bundle.values():
        rnd.shuffle(rows)
    # Round-robin over bundles so small bundles are not crowded out; oversample, since filters drop some.
    order, i = [], 0
    while len(order) < 1.15 * n and any(i < len(v) for v in by_bundle.values()):
        order += [v[i] for v in by_bundle.values() if i < len(v)]
        i += 1
    c = harness.con()
    cards = {t: harness.schema_card(c, t) for t in harness.TABLE_DOCS}
    out = defaultdict(list)
    kept = 0
    for r in order:
        if kept >= n:
            break
        if not visible(c, cards, r)[0]:
            continue
        out[GENERATORS[r["key"].split(":")[0]]].append(r)
        kept += 1
    RUNS.mkdir(parents=True, exist_ok=True)
    for gen, rows in out.items():
        p = RUNS / f"{gen}__rl.judged.jsonl"
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"{gen}: {len(rows)} pairs -> {p}")
    print(f"{kept} pairs from {len(by_bundle)} bundles")


def write(out_dir, unchecked=False):
    """unchecked=True skips the solver filter (judged candidates as they are), for smoke tests only."""
    c = harness.con()
    cards = {t: harness.schema_card(c, t) for t in harness.TABLE_DOCS}
    rows = []
    pattern = "out_*__rl.judged.jsonl" if unchecked else "out_*__rl.judged.consistency.jsonl"
    for f in sorted(RUNS.glob(pattern)):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if not unchecked and (not r["lenient"] or r.get("tie") is True):
                continue
            h = visible(c, cards, r)[1]
            m = export.messages(student.user_prompt(cards, r["tables"], r["question"], h), "")["messages"][:2]
            rows.append({"id": r["key"], "messages": m, "sql": r["sql"].strip().rstrip(";").strip(),
                         "tables": r["tables"], "hinted": h is not None})
    random.Random(SEED).shuffle(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} RL questions ({sum(r['hinted'] for r in rows)} hinted, "
          f"{len({tuple(r['tables']) for r in rows})} bundles) -> {out / 'train.jsonl'}")


if __name__ == "__main__":
    {"pick": lambda: pick(int(sys.argv[2])), "write": lambda: write(sys.argv[2]),
     "write-unchecked": lambda: write(sys.argv[2], unchecked=True)}[sys.argv[1]]()
