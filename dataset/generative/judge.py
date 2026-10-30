"""LLM judge for generated (question, sql) pairs, calibrated against the hand labels.

Usage:
    python judge.py calibrate MODEL     # judge all hand-labelled pairs, print agreement
    python judge.py run MODEL FILE      # judge FILE (a *.validated.jsonl), write FILE.judged_<model>.jsonl

The judge sees the schema cards, the question, the SQL and a result preview (re-executed, up to 8 rows).
"""
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import eval_small
import harness
from or_generate import api_key, call

HERE = Path(__file__).parent
VERDICTS = ("correct", "minor", "wrong")

PROMPT = """You review training data for a small text-to-SQL model. The SQL runs in DuckDB with the spatial extension
(`SET geometry_always_xy = true`, geometries in EPSG:4326 lon/lat). Decide whether the SQL correctly answers the question.

{cards}

QUESTION: {question}

SQL:
{sql}

RESULT: {n_rows} rows ({more}), columns {cols}
{rows}

Verdicts:
- wrong: the query answers a different question; wrong units or conversion factors; distances, buffers or areas
  measured where scale is distorted (EPSG:3857, EPSG:6933, a UTM zone far outside the data, or plain degrees)
  when real units are asked for; join fan-out that double counts; a proxy the question did not ask for
  (e.g. centroid-to-centroid distance for "nearest" between shapes); a logic error (part of an area missing,
  wrong filter, wrong ordering or ranking); geography that is not in the data invented as if it were;
  or the question itself is nonsensical.
- minor: answers the question, but with something a careful analyst would flag: an ambiguous interpretation,
  a trivial or degenerate result (all zeros, a constant), a rough approximation of a named place, raw binary
  geometry in the output.
- correct: otherwise. Do not penalise style, verbosity or harmless redundant steps.

Return ONLY JSON: {{"verdict": "correct|minor|wrong", "reason": "<one sentence>"}}"""


def _cards():
    c = harness.con()
    return {t: harness.schema_card(c, t) for t in harness.TABLE_DOCS}


def build_prompt(cards, pair):
    rows = "\n".join(" | ".join(map(str, r)) for r in (pair.get("preview") or []))
    n = pair.get("n_rows") or 0
    return PROMPT.format(
        cards="\n\n".join(cards[t] for t in pair["tables"]), question=pair["question"], sql=pair["sql"],
        n_rows=n, more="1001 means at least 1001" if n >= 1001 else "showing up to 8",
        cols=pair.get("cols"), rows=rows or "(no rows)",
    )


def parse_verdict(text):
    m = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    try:
        out = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        out = {}
    v = str(out.get("verdict", "")).lower()
    return (v if v in VERDICTS else None), out.get("reason")


def judge_pairs(model, pairs, workers=8):
    key, cards = api_key(), _cards()

    def one(pair):
        resp = call(model, build_prompt(cards, pair), ["fp8", "bf16", "fp16"], None, key, temperature=0)
        if "error" in resp:
            return dict(pair, judge=None, judge_reason=resp["error"][:200], judge_cost=0)
        verdict, reason = parse_verdict(resp["choices"][0]["message"].get("content"))
        return dict(pair, judge=verdict, judge_reason=reason, judge_cost=(resp.get("usage") or {}).get("cost") or 0)

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, pairs))


def labelled_pairs():
    """All hand-labelled pairs (Haiku pilot, bake-off v2, round v3), re-executed for a fresh 8-row preview."""
    out = []
    for line in (HERE / "pilot" / "out.validated.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["ok"]:
            k = f"{r['scenario']}.{r['idx']}"
            out.append(dict(r, label="W" if k in eval_small.WRONG else "M" if k in eval_small.MINOR else "C", source="haiku"))
    for folder in ("pilot", "round_v3"):
        key = json.loads((HERE / folder / "grading" / "key.json").read_text())
        grades = json.loads((HERE / folder / "grading" / "grades.json").read_text())
        files = {}
        for i, k in key.items():
            if k["model"] not in files:
                files[k["model"]] = {(r["scenario"], r["idx"]): r for r in map(json.loads, (HERE / folder / f"out_{k['model']}.validated.jsonl").read_text().splitlines())}
            out.append(dict(files[k["model"]][(k["scenario"], k["idx"])], label=grades[i], source=f"{folder}:{k['model']}"))
    with ThreadPoolExecutor(4) as ex:
        return list(ex.map(harness._validate_row, out))


def report(judged):
    """Agreement with hand labels. Keep = judge says correct."""
    rows = [r for r in judged if r.get("judge")]
    conf = Counter((r["label"], r["judge"]) for r in rows)
    print(f"judged {len(rows)}/{len(judged)}; cost ${sum(r['judge_cost'] for r in judged):.3f}")
    print("label \\ judge   correct  minor  wrong")
    for lab in "CMW":
        print(f"  {lab:12s} " + "".join(f"{conf[(lab, v)]:>8d}" for v in VERDICTS))
    w = [r for r in rows if r["label"] == "W"]
    c = [r for r in rows if r["label"] == "C"]
    kept = [r for r in rows if r["judge"] == "correct"]
    print(f"wrong caught (judge says wrong):        {sum(r['judge'] == 'wrong' for r in w)}/{len(w)}")
    print(f"wrong caught (judge says wrong/minor):  {sum(r['judge'] != 'correct' for r in w)}/{len(w)}")
    print(f"correct kept:                           {sum(r['judge'] == 'correct' for r in c)}/{len(c)}")
    print(f"kept set: {len(kept)} pairs, {sum(r['label'] == 'C' for r in kept) / max(1, len(kept)):.0%} hand-labelled correct, "
          f"{sum(r['label'] == 'W' for r in kept)} wrong slipped through "
          f"(vs {sum(r['label'] == 'C' for r in rows) / max(1, len(rows)):.0%} correct before judging)")


def main():
    cmd, model = sys.argv[1], sys.argv[2]
    slug = model.replace("/", "__")
    if cmd == "calibrate":
        pairs = [p for p in labelled_pairs() if p["ok"]]
        judged = judge_pairs(model, pairs)
        (HERE / "runs").mkdir(exist_ok=True)
        (HERE / "runs" / f"judge_calibration_{slug}.jsonl").write_text("\n".join(json.dumps(r, default=str) for r in judged))
        report(judged)
    elif cmd == "run":
        path = Path(sys.argv[3])
        pairs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        workers = int(sys.argv[4]) if len(sys.argv) > 4 else 8
        judged = judge_pairs(model, [p for p in pairs if p["ok"]], workers)
        path.with_suffix(f".judged_{slug}.jsonl").write_text("\n".join(json.dumps(r, default=str) for r in judged))
        print(Counter(r["judge"] for r in judged))


if __name__ == "__main__":
    main()
