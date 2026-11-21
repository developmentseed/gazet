"""Zero-shot eval of small local models on question + schema -> DuckDB spatial SQL.

Gold = pilot pairs that executed and were hand-labelled correct. Metric = execution match:
predicted SQL's result set vs gold result set (row order ignored, numbers to 3 significant figures).

Usage: python eval_small.py MODEL_NAME [PORT]   (llama-server must be running on PORT)
"""
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
import harness

HERE = Path(__file__).parent
# Hand labels from reviewing the pilot (scenario.idx); everything else that executed is "correct".
WRONG = {"s7_001.1", "s7_002.3", "s7_004.0", "s7_004.3", "s7_004.5"}
MINOR = {"s7_001.4", "s7_001.5", "s7_002.0", "s7_002.5", "s7_006.2", "s7_008.0", "s7_004.4"}

SYSTEM = """You write DuckDB SQL. The spatial extension is loaded and `SET geometry_always_xy = true` is active (lon/lat order). Geometries are EPSG:4326 lon/lat degrees; for metres use spheroid functions (ST_Distance_Sphere, ST_Area_Spheroid, ST_Length_Spheroid) or ST_Transform to a local metric CRS with always_xy := true. Return geometries with ST_AsGeoJSON only when the user wants shapes.
Answer with a single read-only SELECT statement and nothing else."""


def norm_val(v):
    if isinstance(v, float):
        if math.isnan(v) or v == 0:
            return 0.0
        return float(f"{v:.3g}")
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return float(f"{v:.3g}") if abs(v) > 999 else float(v)
    return str(v)


def result(c, sql, limit=2000):
    cur = c.execute(sql)
    cols = [d[0] for d in cur.description]
    return cols, cur.fetchmany(limit)


def match(gold, pred):
    """exact: same multiset of rows (values only); lenient: every gold column appears in pred."""
    (gc, gr), (pc, pr) = gold, pred
    g_rows = Counter(tuple(sorted(map(str, map(norm_val, r)))) for r in gr)
    p_rows = Counter(tuple(sorted(map(str, map(norm_val, r)))) for r in pr)
    exact = g_rows == p_rows
    p_cols = [Counter(norm_val(r[i]) for r in pr) for i in range(len(pc))]
    lenient = len(gr) == len(pr) and all(
        Counter(norm_val(r[i]) for r in gr) in p_cols for i in range(len(gc))
    )
    return exact, exact or lenient


def extract_sql(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    sql = m.group(1) if m else text
    return sql.strip().rstrip(";").strip()


def main():
    model, port = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8080
    c = harness.con()
    cards = {}
    gold = []
    for line in (HERE / "pilot" / "out.validated.jsonl").read_text().splitlines():
        r = json.loads(line)
        key = f"{r['scenario']}.{r['idx']}"
        if r["ok"] and key not in WRONG and key not in MINOR:
            gold.append(r)
    out = []
    for r in gold:
        for t in r["tables"]:
            cards.setdefault(t, harness.schema_card(c, t))
        user = "\n\n".join(cards[t] for t in r["tables"]) + f"\n\nQuestion: {r['question']}"
        t0 = time.time()
        resp = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=dict(
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            temperature=0, max_tokens=320, chat_template_kwargs={"enable_thinking": False},
        ), timeout=600).json()
        wall = time.time() - t0
        text = resp["choices"][0]["message"]["content"] or ""
        sql = extract_sql(text)
        tm = resp.get("timings", {})
        rec = dict(key=f"{r['scenario']}.{r['idx']}", difficulty=r["difficulty"], skill=r["target_skill"][:40],
                   question=r["question"], gold_sql=r["sql"], pred_sql=sql, wall_s=round(wall, 2),
                   prompt_n=tm.get("prompt_n"), prompt_ms=tm.get("prompt_ms"),
                   predicted_n=tm.get("predicted_n"), predicted_ms=tm.get("predicted_ms"))
        try:
            g = result(c, r["sql"].strip().rstrip(";"))
            p = result(c, sql)
            rec["executes"] = True
            rec["exact"], rec["lenient"] = match(g, p)
        except Exception as e:
            rec.update(executes=False, exact=False, lenient=False, error=str(e).splitlines()[0][:200])
        out.append(rec)
        print(f"{rec['key']:10s} exec={rec['executes']!s:5} exact={rec['exact']!s:5} lenient={rec['lenient']!s:5} {wall:5.1f}s", flush=True)
    (HERE / "runs" / f"eval_{model}.jsonl").write_text("\n".join(json.dumps(o) for o in out))
    n = len(out)
    s = lambda k: sum(o[k] for o in out)
    pp = sum(o["prompt_n"] or 0 for o in out) / max(1, sum(o["prompt_ms"] or 0 for o in out)) * 1000
    dd = sum(o["predicted_n"] or 0 for o in out) / max(1, sum(o["predicted_ms"] or 0 for o in out)) * 1000
    print(f"\n{model}: n={n} executes={s('executes')/n:.0%} exact={s('exact')/n:.0%} lenient={s('lenient')/n:.0%} "
          f"median_wall={sorted(o['wall_s'] for o in out)[n//2]}s prefill={pp:.0f} tok/s decode={dd:.0f} tok/s")


if __name__ == "__main__":
    main()
