"""Gold eval set: blind hand-review sheet from the filtered candidates, then export the pairs graded correct.

Usage:
    python gold.py sheet OUT_DIR CONSISTENCY_JSONL...   # candidates = judge correct AND solver agrees AND no LIMIT tie
    python gold.py write OUT_DIR GOLD_JSONL             # OUT_DIR/grading/grades.json -> gold pairs (grade "C" only)

`sheet` snapshots the candidate files into OUT_DIR/candidates/ so `write` works from the committed folder.

The sheet hides the judge verdict and the solver SQL so the review is independent of both filters.
Grades follow round_v3: C correct, M minor issue, W wrong.
"""
import json
import random
import shutil
import sys
from pathlib import Path


def sheet(out_dir, paths):
    cands = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            if r["judge"] == "correct" and r["lenient"] and r.get("tie") is not True:
                cands.append(dict(r, file=Path(p).name))
    random.Random(7).shuffle(cands)
    grading = Path(out_dir) / "grading"
    grading.mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "candidates").mkdir(exist_ok=True)
    for p in paths:
        shutil.copy(p, Path(out_dir) / "candidates" / Path(p).name)
    key, lines = {}, []
    for i, r in enumerate(cands):
        key[i] = dict(file=r["file"], scenario=r["scenario"], idx=r["idx"])
        lines.append(f"## {i} [{', '.join(r['tables'])} | {r['difficulty']}] rows={r['n_rows']} cols={r['cols']}\n"
                     f"Q: {r['question']}\n{r['sql'].strip()}\n-> {r['preview'][:3]}\n")
    (grading / "sheet.txt").write_text("\n".join(lines))
    (grading / "key.json").write_text(json.dumps(key))
    print(f"{len(cands)} candidates -> {grading / 'sheet.txt'}")


def write(out_dir, gold_path):
    grading = Path(out_dir) / "grading"
    key = json.loads((grading / "key.json").read_text())
    grades = json.loads((grading / "grades.json").read_text())
    rows = {}
    for f in {k["file"] for k in key.values()}:
        for line in (Path(out_dir) / "candidates" / f).read_text().splitlines():
            r = json.loads(line)
            rows[(f, r["scenario"], r["idx"])] = r
    gold = []
    for i, k in sorted(key.items(), key=lambda kv: int(kv[0])):
        if grades.get(i) != "C":
            continue
        r = rows[(k["file"], k["scenario"], k["idx"])]
        gold.append(dict(id=f"g{len(gold):03d}", question=r["question"], tables=r["tables"], sql=r["sql"].strip().rstrip(";"),
                         difficulty=r["difficulty"], skill=r["target_skill"], n_rows=r["n_rows"], cols=r["cols"],
                         source=f"{k['file']}:{k['scenario']}.{k['idx']}"))
    Path(gold_path).write_text("\n".join(json.dumps(g, ensure_ascii=False) for g in gold) + "\n")
    counts = {g: sum(v == g for v in grades.values()) for g in "CMW"}
    print(f"graded {len(grades)}/{len(key)}: {counts}; wrote {len(gold)} gold pairs -> {gold_path}")


if __name__ == "__main__":
    if sys.argv[1] == "sheet":
        sheet(sys.argv[2], sys.argv[3:])
    elif sys.argv[1] == "write":
        write(sys.argv[2], sys.argv[3])
