"""Self-consistency check: an independent model writes SQL from the question alone; keep pairs whose results agree.

Usage:
    python consistency.py calibrate     # on all hand-labelled pairs, combined with the judge calibration
    python consistency.py run JUDGED    # solve + compare the judge-`correct` pairs of a judged file, plus a tie check
    python consistency.py compare       # worker: stdin {"gold": sql, "pred": sql} -> stdout match flags
    python consistency.py ties          # worker: stdin sql -> stdout {"tie": true|false|null}

The solver sees exactly what the student model will see (schema cards + question), never the generated SQL.
Pairs from DeepSeek are solved by GLM and everything else by DeepSeek, so agreement is always cross-family.
"""
import json
import math
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sqlglot
from sqlglot import exp

sys.path.insert(0, str(Path(__file__).parent))
import harness
import judge
from or_generate import api_key, call

HERE = Path(__file__).parent
SOLVERS = {"deepseek": "z-ai/glm-5.3-flash", "default": "deepseek/deepseek-v4.1-flash"}

SYSTEM = """You write DuckDB SQL. The spatial extension is loaded and `SET geometry_always_xy = true` is active (lon/lat order).
Geometries are EPSG:4326 lon/lat degrees. DuckDB spatial is not PostGIS: no SRID arguments, ST_Transform takes CRS strings
(ST_Transform(geom, 'EPSG:4326', 'EPSG:3763', always_xy := true)), no GEOGRAPHY type, ST_Collect takes a list.
For metres use ST_Distance_Sphere, ST_Area_Spheroid, ST_Length_Spheroid, or a local metric CRS; ST_Distance_Spheroid and
ST_DWithin_Spheroid need geometry::GEOMETRY casts. List lambdas are written lambda x: x + 1, not x -> x + 1. Return geometries with ST_AsGeoJSON only when the user wants shapes.
Answer with a single read-only SELECT statement and nothing else."""


def solver_for(source):
    return SOLVERS["deepseek"] if "deepseek" in source else SOLVERS["default"]


def solve(pairs):
    key, cards = api_key(), judge._cards()

    def one(pair):
        model = solver_for(pair.get("source", ""))
        prompt = SYSTEM + "\n\n" + "\n\n".join(cards[t] for t in pair["tables"]) + f"\n\nQuestion: {pair['question']}"
        resp = call(model, prompt, ["fp8", "bf16", "fp16"], None, key, temperature=0)
        if "error" in resp:
            return dict(pair, solver=model, solver_sql=None, solver_cost=0)
        text = resp["choices"][0]["message"].get("content") or ""
        m = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        sql = (m.group(1) if m else text).strip().rstrip(";").strip()
        return dict(pair, solver=model, solver_sql=sql, solver_cost=(resp.get("usage") or {}).get("cost") or 0)

    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(one, pairs))


def _norm(v, sig):
    if isinstance(v, (list, tuple)):  # lists/structs: normalise inside, or float summation noise breaks matches
        return "[" + ", ".join(_norm(x, sig) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_norm(x, sig)}" for k, x in v.items()) + "}"
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, (int, float)):
        if isinstance(v, float) and math.isnan(v):
            return "nan"
        return f"{float(v):.{sig}g}"
    return str(v)


def compare_results(gold_sql, pred_sql, limit=2000):
    """Run both queries (in this process) and compare result sets, ignoring row and column order."""
    c = harness.con()
    c.execute("SET threads=2")
    c.execute("SET memory_limit='2GB'")
    g = c.execute(gold_sql).fetchmany(limit)
    try:
        cur = c.execute(pred_sql)
        p, p_ncols = cur.fetchmany(limit), len(cur.description)
    except Exception as e:
        return dict(pred_error=f"{type(e).__name__}: {str(e).splitlines()[0][:150]}", exact=False, lenient=False)

    def rows(data, sig):
        return Counter(tuple(sorted(_norm(v, sig) for v in r)) for r in data)

    def cols(data, n, sig):
        return [Counter(_norm(r[i], sig) for r in data) for i in range(n)]

    exact = rows(g, 3) == rows(p, 3)
    g_cols = cols(g, len(g[0]) if g else 0, 2)
    p_cols = cols(p, p_ncols, 2)
    lenient = exact or (len(g) == len(p) and all(gc in p_cols for gc in g_cols))
    return dict(pred_error=None, exact=exact, lenient=lenient, gold_rows=len(g), pred_rows=len(p))


def _compare_isolated(pair):
    if not pair.get("solver_sql"):
        return dict(pair, exact=False, lenient=False, pred_error="no solver sql")
    try:
        proc = subprocess.run([sys.executable, "-I", __file__, "compare"], capture_output=True, text=True, timeout=90,
                              input=json.dumps({"gold": pair["sql"].strip().rstrip(";"), "pred": pair["solver_sql"]}))
        out = json.loads(proc.stdout) if proc.returncode == 0 and proc.stdout.strip() else dict(exact=False, lenient=False, pred_error=f"crash (exit {proc.returncode})")
    except subprocess.TimeoutExpired:
        out = dict(exact=False, lenient=False, pred_error="timeout")
    return dict(pair, **out)


def limit_tie(sql):
    """True if an outer ORDER BY ... LIMIT k cuts through tied rows (result depends on execution order).

    Rewrites the outer SELECT to also project its ORDER BY keys, fetches k+1 rows and compares rows k and k+1.
    None when the query shape is not checked (set operations, DISTINCT, OFFSET, non-literal LIMIT).
    """
    try:
        tree = sqlglot.parse_one(sql, read="duckdb")
    except Exception:
        return None
    if not isinstance(tree, exp.Select) or not tree.args.get("limit"):
        return None if not isinstance(tree, exp.Select) else False
    if tree.args.get("distinct") or tree.args.get("offset"):
        return None
    try:
        k = int(tree.args["limit"].expression.name)
    except (ValueError, AttributeError):
        return None
    c = harness.con()
    c.execute("SET threads=2")
    order = tree.args.get("order")
    if not order:  # LIMIT without ORDER BY is only deterministic if it does not cut anything
        q = tree.copy()
        q.set("limit", None)
        return len(c.execute(f"SELECT * FROM ({q.sql('duckdb')}) LIMIT {k + 1}").fetchall()) > k
    q = tree.copy()
    projections = q.expressions[:]
    keys = []
    for i, o in enumerate(order.expressions):
        e = o.this
        if isinstance(e, exp.Literal) and e.is_int:
            e = projections[int(e.name) - 1]
            e = exp.column(e.alias) if e.alias else e
        keys.append(exp.alias_(e.copy(), f"__k{i}"))
    q = q.select(*keys).limit(k + 1)
    rows = c.execute(q.sql("duckdb")).fetchall()
    if len(rows) <= k:
        return False
    nk = len(keys)
    return [_norm(v, 9) for v in rows[k - 1][-nk:]] == [_norm(v, 9) for v in rows[k][-nk:]]


def _isolated(mode, payload, timeout=90):
    try:
        proc = subprocess.run([sys.executable, "-I", __file__, mode], capture_output=True, text=True, timeout=timeout, input=payload)
        return json.loads(proc.stdout) if proc.returncode == 0 and proc.stdout.strip() else None
    except subprocess.TimeoutExpired:
        return None


def _tie_isolated(pair):
    out = _isolated("ties", pair["sql"].strip().rstrip(";"))
    return dict(pair, tie=out["tie"] if out else None, tie_checked=out is not None)


def run(path):
    """Solver agreement and tie check for the judge-`correct` pairs of a judged file (gold-set filter)."""
    path = Path(path)
    judged = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    # Generated files carry the generator only in their name (out_<model>__<tag>); the solver must be another family.
    cands = [dict(r, source=r.get("source") or path.name) for r in judged if r["judge"] == "correct"]
    with ThreadPoolExecutor(4) as ex:
        compared = list(ex.map(_compare_isolated, solve(cands)))
        compared = list(ex.map(_tie_isolated, compared))
    outp = path.with_suffix(".consistency.jsonl")
    outp.write_text("\n".join(json.dumps(r, default=str) for r in compared))
    kept = [r for r in compared if r["lenient"] and r["tie"] is not True]
    print(f"{len(cands)}/{len(judged)} judge-correct; solver cost ${sum(r['solver_cost'] for r in compared):.3f}")
    print(f"agree lenient {sum(r['lenient'] for r in compared)}, exact {sum(r['exact'] for r in compared)}; "
          f"ties {sum(r['tie'] is True for r in compared)}, unchecked {sum(r['tie'] is None for r in compared)}")
    print(f"gold candidates (correct AND agree AND no tie): {len(kept)} -> {outp}")


def calibrate():
    # Reuse the judge calibration file: same 215 labelled pairs, already re-executed, with MiniMax verdicts.
    judged = [json.loads(line) for line in (HERE / "runs" / "judge_calibration_minimax__minimax-m3.jsonl").read_text().splitlines()]
    solved = solve(judged)
    with ThreadPoolExecutor(4) as ex:
        compared = list(ex.map(_compare_isolated, solved))
    (HERE / "runs" / "consistency_calibration.jsonl").write_text("\n".join(json.dumps(r, default=str) for r in compared))
    print(f"solver cost ${sum(r['solver_cost'] for r in compared):.3f}; solver SQL failed to run: "
          f"{sum(bool(r.get('pred_error')) for r in compared)}/{len(compared)}")
    print("agreement by hand label:")
    for lab in "CMW":
        rs = [r for r in compared if r["label"] == lab]
        print(f"  {lab}: exact {sum(r['exact'] for r in rs)}/{len(rs)}  lenient {sum(r['lenient'] for r in rs)}/{len(rs)}")
    policies = {
        "judge correct": lambda r: r["judge"] == "correct",
        "agree (lenient)": lambda r: r["lenient"],
        "judge correct AND agree": lambda r: r["judge"] == "correct" and r["lenient"],
        "judge correct OR agree": lambda r: r["judge"] == "correct" or r["lenient"],
        "judge not wrong AND agree": lambda r: r["judge"] != "wrong" and r["lenient"],
    }
    n_c = sum(r["label"] == "C" for r in compared)
    print(f"\n{'keep policy':28s} {'kept':>5s} {'C kept':>9s} {'W kept':>7s} {'kept %C':>8s}")
    for name, keep in policies.items():
        kept = [r for r in compared if keep(r)]
        kc = sum(r["label"] == "C" for r in kept)
        print(f"{name:28s} {len(kept):>5d} {kc:>4d}/{n_c:<4d} {sum(r['label'] == 'W' for r in kept):>7d} {kc / max(1, len(kept)):>8.0%}")


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        args = json.loads(sys.stdin.read())
        print(json.dumps(compare_results(args["gold"], args["pred"]), default=str))
    elif sys.argv[1] == "ties":
        print(json.dumps({"tie": limit_tie(sys.stdin.read())}))
    elif sys.argv[1] == "run":
        run(sys.argv[2])
    elif sys.argv[1] == "calibrate":
        calibrate()
