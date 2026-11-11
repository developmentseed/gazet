"""Training export: judge-correct pairs -> ChatML for finetune/train_modal_qwen35.py.

Writes OUT_DIR/sql/train.jsonl and OUT_DIR/sql/val.jsonl, one {"messages": [system, user, assistant]} per line,
in the student format of student.py (minimal system prompt, schema cards, optional value hints, question).

- train: judge-correct pairs from runs/train, minus every scenario in dev/dev_scenarios.json;
- val: dev/dev_v1.jsonl, so the eval loss is measured on unseen questions about known tables;
- half of the pairs (seeded per pair) get the value-hint block, so the student works with and without it;
- dropped: pairs that read a table missing from the prompt, and pairs that filter on a string value that is in
  neither the question, the cards nor (when hinted) the hints. The student cannot know those values.

Also writes OUT_DIR/eval/{gold,dev}_{plain,hints}.jsonl: the test prompts ({"id", "messages": [system, user]}), so a
training job without the data lake can answer them. Score the answers with `evaluate.py run NAME --preds FILE`.

Usage: python export.py OUT_DIR [--hint-share 0.5]
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import sqlglot
from sqlglot import exp

sys.path.insert(0, str(Path(__file__).parent))
import harness
import hints as hints_mod
import student

HERE = Path(__file__).parent


def read_tables(sql):
    """Base tables a query reads (CTE names excluded); None if sqlglot cannot parse it."""
    try:
        tree = sqlglot.parse_one(sql, read="duckdb")
    except Exception:
        return None
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    return {t.name.lower() for t in tree.find_all(exp.Table) if t.name and t.name.lower() not in ctes}


def _hinted(key, share):
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < share


def train_pairs():
    dev = set(json.loads((HERE / "dev" / "dev_scenarios.json").read_text()))
    for f in sorted((HERE / "runs" / "train").glob("*train_c*.judged.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["judge"] == "correct" and r["scenario"] not in dev:
                    yield r


def messages(user, sql):
    return {"messages": [{"role": "system", "content": student.SYSTEM_MINIMAL},
                         {"role": "user", "content": user},
                         {"role": "assistant", "content": sql}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--hint-share", type=float, default=0.5)
    a = ap.parse_args()
    c = harness.con()
    cards = {t: harness.schema_card(c, t) for t in harness.TABLE_DOCS}
    known = {t.lower() for t in harness.TABLE_DOCS}
    drops, train, n_hinted = Counter(), [], 0
    for r in train_pairs():
        sql = r["sql"].strip().rstrip(";").strip()
        used = read_tables(sql)
        if used is None:
            drops["sqlglot cannot parse"] += 1
            continue
        if (used & known) - {t.lower() for t in r["tables"]}:
            drops["reads a table not in the prompt"] += 1
            continue
        h = hints_mod.value_hints(c, r["tables"], r["question"]) if _hinted(f"{r['scenario']}.{r['idx']}", a.hint_share) else None
        seen = hints_mod.norm("\n".join([r["question"], *(cards[t] for t in r["tables"]), *(h or [])]))
        if any(hints_mod.norm(v) not in seen for v in hints_mod._filter_literals(sql)):
            drops["filter value the student cannot see"] += 1
            continue
        n_hinted += bool(h)
        train.append(messages(student.user_prompt(cards, r["tables"], r["question"], h), sql))
    val = []
    for line in (HERE / "dev" / "dev_v1.jsonl").read_text().splitlines():
        d = json.loads(line)
        h = hints_mod.value_hints(c, d["tables"], d["question"]) if _hinted(d["id"], a.hint_share) else None
        val.append(messages(student.user_prompt(cards, d["tables"], d["question"], h), d["sql"]))

    ev = Path(a.out_dir) / "eval"
    ev.mkdir(parents=True, exist_ok=True)
    for data in ("gold", "dev"):
        pairs = [json.loads(line) for line in (HERE / data / f"{data}_v1.jsonl").read_text().splitlines() if line.strip()]
        for mode in ("plain", "hints"):
            rows = []
            for p in pairs:
                h = hints_mod.value_hints(c, p["tables"], p["question"]) if mode == "hints" else None
                m = messages(student.user_prompt(cards, p["tables"], p["question"], h), "")["messages"][:2]
                rows.append({"id": p["id"], "messages": m})
            (ev / f"{data}_{mode}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    out = Path(a.out_dir) / "sql"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    chars = sorted(sum(len(m["content"]) for m in r["messages"]) for r in train)
    stats = dict(train=len(train), val=len(val), train_with_hints=n_hinted, dropped=dict(drops),
                 chars_median=chars[len(chars) // 2], chars_p99=chars[int(.99 * len(chars))], chars_max=chars[-1])
    (Path(a.out_dir) / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
